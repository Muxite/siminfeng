"""Offline CUDA ELF inspection; no Docker or host ELF tools required.

Layer streaming, digest checks and whiteout merging follow
cc/engine_handoff/engine_handoff.py:observe_archive. Only the resolved CUDA
backend is copied out, into a caller-owned temporary directory.
"""
import gzip
import hashlib
import json
import mmap
from pathlib import Path
import posixpath
import re
import shutil
import struct
import tarfile
import tempfile

SYMBOLS = ("cudaMalloc", "cudaFree", "cudaMallocAsync", "cudaFreeAsync",
           "cudaMemGetInfo", "cuMemCreate")


class HashReader:
    # Same streaming adapter as engine_handoff; avoid importing its yaml dependency.
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        return data

    def drain(self):
        while self.read(1024 * 1024):
            pass
        return self.hash.hexdigest()


def archive_name(name):
    if not isinstance(name, str) or ".." in name.split("/"):
        raise ValueError(f"unsafe archive path: {name!r}")
    return posixpath.normpath(name.lstrip("/"))


def extract_backend(path, destination, config_id=None):
    """Merge layer metadata, resolve links, then stream only the chosen file out.

    Whiteouts apply to the lower view before same-layer entries, irrespective of
    tar order. Never extract tar paths or links onto the host filesystem.
    """
    with tarfile.open(path, "r:*") as outer:
        members = {}
        for member in outer:
            name = archive_name(member.name)
            if name in members:
                raise ValueError("duplicate archive member: " + name)
            members[name] = member

        def stream(name):
            member = members.get(archive_name(name))
            if member is None or not member.isfile():
                raise ValueError("missing regular archive member: " + name)
            return outer.extractfile(member)

        def metadata(name, expected=None):
            with stream(name) as src:
                raw = src.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("archive metadata too large: " + name)
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            if expected is not None and expected != digest:
                raise ValueError("archive metadata SHA-256 mismatch: " + name)
            return json.loads(raw), digest

        def blob(digest):
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("unsupported OCI digest: " + str(digest))
            return metadata("blobs/sha256/" + digest[7:], digest)[0]

        images, seen = {}, set()

        def visit(desc, depth=0):
            if depth > 8:
                raise ValueError("OCI index nesting too deep")
            if desc.get("annotations", {}).get("vnd.docker.reference.type") == "attestation-manifest":
                return
            platform = desc.get("platform", {})
            if platform and (platform.get("os"), platform.get("architecture")) != ("linux", "amd64"):
                return
            digest = desc["digest"]
            if digest in seen:
                return
            seen.add(digest)
            doc = blob(digest)
            if "manifests" in doc:
                for child in doc["manifests"]:
                    visit(child, depth + 1)
            else:
                identity = doc["config"]["digest"]
                config = blob(identity)
                if (config.get("os"), config.get("architecture")) == ("linux", "amd64"):
                    layers = [("blobs/sha256/" + d["digest"][7:], d["digest"]) for d in doc["layers"]
                              if re.fullmatch(r"sha256:[0-9a-f]{64}", d["digest"])]
                    if len(layers) != len(doc["layers"]):
                        raise ValueError("unsupported OCI layer digest")
                    if identity in images and images[identity] != (config, layers):
                        raise ValueError("ambiguous image layers")
                    images[identity] = (config, layers)

        if "index.json" in members:
            catalog, _ = metadata("index.json")
            for descriptor in catalog["manifests"]:
                visit(descriptor)
        else:
            manifest, _ = metadata("manifest.json")
            for entry in manifest:
                config, identity = metadata(entry["Config"])
                if (config.get("os"), config.get("architecture")) == ("linux", "amd64"):
                    images[identity] = (config, [(name, None) for name in entry["Layers"]])
        if len(images) != 1 or (config_id is not None and config_id not in images):
            raise ValueError("archive must identify exactly one matching linux/amd64 config")
        identity, (config, layers) = next(iter(images.items()))
        diff_ids = config.get("rootfs", {}).get("diff_ids")
        if diff_ids is not None and len(diff_ids) != len(layers):
            raise ValueError("config rootfs.diff_ids count differs from layer count")

        def entries(index, consume):
            name, expected = layers[index]
            with stream(name) as src:
                magic = src.read(2)
                src.seek(0)
                raw = HashReader(src)
                decoded = gzip.GzipFile(fileobj=raw) if magic == b"\x1f\x8b" else raw
                unpacked = HashReader(decoded)
                with tarfile.open(fileobj=unpacked, mode="r|") as layer:
                    for number, entry in enumerate(layer):
                        consume(layer, number, entry)
                        layer.members.clear()
                diff_sha = unpacked.drain()
                blob_sha = raw.drain()
                if magic == b"\x1f\x8b":
                    decoded.close()
            if expected is not None and expected != "sha256:" + blob_sha:
                raise ValueError("layer blob SHA-256 mismatch: " + name)
            if diff_ids is not None and diff_ids[index] != "sha256:" + diff_sha:
                raise ValueError("layer rootfs.diff_ids SHA-256 mismatch: " + name)

        view = {}

        def remove(tree, target, children_only=False):
            for key in list(tree):
                if (key == target and not children_only) or key.startswith(target.rstrip("/") + "/"):
                    del tree[key]

        for index in range(len(layers)):
            changes, removals = {}, []

            def merge(layer, number, entry):
                p = "/" + archive_name(entry.name)
                parent, leaf = posixpath.split(p)
                if leaf == ".wh..wh..opq":
                    removals.append((parent, True))
                elif leaf.startswith(".wh."):
                    removals.append((posixpath.join(parent, leaf[4:]), False))
                else:
                    # Replacing a non-directory also hides lower descendants.
                    if not entry.isdir():
                        removals.append((p, False))
                        remove(changes, p)
                    kind = ("file" if entry.isfile() else "symlink" if entry.issym()
                            else "hardlink" if entry.islnk() else "dir" if entry.isdir() else "other")
                    changes[p] = (kind, entry.linkname, index, number)

            entries(index, merge)
            for target, children_only in removals:
                remove(view, target, children_only)
            view.update(changes)

        def resolve(path):
            visited = set()
            for _ in range(64):
                if path in visited:
                    raise ValueError("library symlink cycle: " + path)
                visited.add(path)
                parts = path.strip("/").split("/")
                for count in range(1, len(parts) + 1):
                    prefix = "/" + "/".join(parts[:count])
                    node = view.get(prefix)
                    if node is None:
                        if count < len(parts):  # tar layers may omit directory headers
                            continue
                        return None
                    kind, target, _, _ = node
                    if kind in ("symlink", "hardlink"):
                        base = "/" if kind == "hardlink" else posixpath.dirname(prefix)
                        path = posixpath.normpath(posixpath.join(base, target, *parts[count:]))
                        break
                    if count == len(parts):
                        return (path, node) if kind == "file" else None
                    if kind != "dir":
                        return None
                else:
                    return None
            raise ValueError("library symlink chain too deep")

        candidates = sorted(p for p in view if re.fullmatch(r"/app/libggml-cuda\.so(?:\..+)?", p))
        for candidate in candidates:
            resolved = resolve(candidate)
            if resolved is not None:
                resolved_path, node = resolved
                output = Path(destination) / "libggml-cuda.so"

                def copy(layer, number, entry):
                    if number == node[3]:
                        with layer.extractfile(entry) as src, output.open("wb") as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)

                entries(node[2], copy)
                return output, {"library": candidate, "resolved_library": resolved_path,
                                "image_config_id": identity}
        raise ValueError("CUDA backend library not found in merged /app")


def inspect_elf(path):
    """Read SHT_DYNSYM and SHT_DYNAMIC in ELF32/64 of either byte order.

    Section headers survive normal stripping. An ELF without usable dynamic
    sections is explicitly unavailable, never reported as missing imports.
    """
    with Path(path).open("rb") as src:
        if Path(path).stat().st_size < 16:
            raise ValueError("truncated ELF identification")
        with mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as data:
            def unpack(fmt, offset):
                size = struct.calcsize(fmt)
                if offset < 0 or offset + size > len(data):
                    raise ValueError("ELF structure outside file")
                return struct.unpack_from(fmt, data, offset)

            if data[:4] != b"\x7fELF" or data[4] not in (1, 2) or data[5] not in (1, 2) or data[6] != 1:
                raise ValueError("unsupported ELF identification")
            endian = "<" if data[5] == 1 else ">"
            wide = data[4] == 2
            header = unpack(endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH"), 16)
            shoff, shsize, shnum = header[5], header[10], header[11]
            section_fmt = endian + ("IIQQQQIIQQ" if wide else "IIIIIIIIII")
            if not shoff or shsize < struct.calcsize(section_fmt):
                raise ValueError("ELF section headers unavailable")
            if shnum == 0:
                shnum = unpack(section_fmt, shoff)[5]
            if shnum < 1 or shoff + shsize * shnum > len(data):
                raise ValueError("invalid ELF section table")
            sections = [unpack(section_fmt, shoff + i * shsize) for i in range(shnum)]

            def string(section, offset):
                start, size = section[4], section[5]
                if section[1] != 3 or start + size > len(data) or not 0 <= offset < size:
                    raise ValueError("invalid ELF string table reference")
                end = data.find(b"\0", start + offset, start + size)
                if end < 0:
                    raise ValueError("unterminated ELF string")
                return data[start + offset:end].decode("utf-8", errors="strict")

            imports, needed, found = set(), [], set()
            for section in sections:
                kind, offset, size, link, stride = section[1], section[4], section[5], section[6], section[9]
                if kind not in (6, 11):  # SHT_DYNAMIC, SHT_DYNSYM
                    continue
                found.add(kind)
                fmt = endian + (("qQ" if wide else "iI") if kind == 6 else
                                ("IBBHQQ" if wide else "IIIBBH"))
                if stride < struct.calcsize(fmt) or size % stride or offset + size > len(data) or link >= shnum:
                    raise ValueError("invalid ELF dynamic section")
                for pos in range(offset, offset + size, stride):
                    row = unpack(fmt, pos)
                    if kind == 6:
                        if row[0] == 0:
                            break
                        if row[0] == 1:  # DT_NEEDED
                            needed.append(string(sections[link], row[1]))
                    elif row[3 if wide else 5] == 0:  # SHN_UNDEF only
                        imports.add(string(sections[link], row[0]).split("@", 1)[0])
            if found != {6, 11}:
                raise ValueError("ELF dynamic symbols/dependencies unavailable")
            return {"dt_needed": needed, "undefined_imports": {name: name in imports for name in SYMBOLS},
                    "libcudart_dynamic": any(re.fullmatch(r"libcudart\.so(?:\..+)?", name) for name in needed)}


def collect_host_linkage(archive, config_id=None, temp_parent=None):
    result = {"status": "unavailable", "library": None, "resolved_library": None,
              "dt_needed": None, "libcudart_dynamic": None,
              "undefined_imports": dict.fromkeys(SYMBOLS),
              "symbols": {"tool": "python-elf", "location": "host", "returncode": None, "reason": None}}
    try:
        with tempfile.TemporaryDirectory(prefix="linkage-", dir=temp_parent) as temporary:
            library, evidence = extract_backend(archive, temporary, config_id)
            result.update(evidence)
            result.update(inspect_elf(library))
        result["status"] = "passed" if result["libcudart_dynamic"] and result["undefined_imports"]["cudaMalloc"] else "failed"
        result["symbols"]["returncode"] = 0
    except Exception as exc:
        result["reason"] = str(exc)
        result["symbols"].update(returncode=1, reason=str(exc))
    return result
