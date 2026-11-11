#!/usr/bin/env python3
"""Offline receipt inspection. Never imports an image or launches a workload."""
import argparse
import hashlib
import json
import gzip
import posixpath
import tarfile
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parent
GGUF_PY = ROOT.parents[1] / "ref/llama.cpp/gguf-py"
FORMAT = "engine-lab-handoff-draft/0"
SHA = re.compile(r"[0-9a-f]{64}\Z")
CLAIMS = ("image_config_id", "archive_sha256", "archive_bytes", "executable_sha256",
          "platform", "base_image_digests", "source_commit", "patch_sha256", "status")


def digest(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def fact(value, tag="observed", source=None):
    return {"value": value, "tag": tag, "source": source}


def gguf_header(path):
    """Use the bundled gguf-py parser, stopping before ALL tensor processing.

    Its read-only mmap reserves address space; only metadata pages are accessed.
    NumPy is an existing transitive dependency of gguf-py, not of the core CLI.
    """
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(GGUF_PY))
    from gguf import GGUFReader

    class MetadataComplete(Exception):
        pass

    class HeaderReader(GGUFReader):
        def _build_tensor_info(self, offs, count):
            raise MetadataComplete

    reader = HeaderReader.__new__(HeaderReader)
    try:
        try:
            reader.__init__(str(path), mode="r")
        except MetadataComplete:
            pass
        arch_field = reader.get_field("general.architecture")
        arch = arch_field.contents() if arch_field else None
        keys = {"arch": "general.architecture", "file_type": "general.file_type"}
        keys.update({name: f"{arch}.{name}" for name in (
            "expert_count", "expert_used_count", "block_count", "context_length")})
        return {name: fact(reader.get_field(key).contents() if reader.get_field(key) else None,
                           tag="observed" if reader.get_field(key) else "missing",
                           source=f"{path}#{key}") for name, key in keys.items()}
    finally:
        # Drop views before closing the mapping; no tensor arrays were constructed.
        if hasattr(reader, "fields"):
            reader.fields.clear()
        if hasattr(reader, "data"):
            reader.data._mmap.close()


def normalize_build_record(record):
    """Accept the Build CC's nested record (cc/build_engine, schema_version 1) as well as the flat claim form.
    Adds flat claim keys derived ONLY from fields the Build CC actually recorded; flat inputs pass through unchanged."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        return record
    out = dict(record)
    image, archive, tools, source = (record.get(k) or {} for k in ("image", "archive", "tools", "source"))
    derived = {
        "image_config_id": image.get("config_digest") or image.get("Id"),   # prefer the archive-derived config digest; Id may be an index digest
        "archive_sha256": archive.get("sha256"), "archive_bytes": archive.get("bytes"),
        "executable_sha256": dict(sorted((record.get("app_files") or tools).items())) or None,   # all /app files when recorded
        "source_commit": source.get("commit"),
        "base_image_digests": {k: (v or {}).get("reference") for k, v in (record.get("base_images") or {}).items()} or None,
    }
    for key, value in derived.items():
        out.setdefault(key, value)
    return out


class HashReader:
    """Hash a stream as it is consumed, without buffering its contents."""
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
    # No extraction occurs, but ambiguous traversal names must not gain identity.
    if not isinstance(name, str) or ".." in name.split("/"):
        raise ValueError(f"unsafe archive path: {name!r}")
    return posixpath.normpath(name.removeprefix("/"))


def observe_archive(path):
    """Observe a single image without extraction; memory holds only metadata/hashes.

    Outer tar members are seeked, layer tars are streamed, and both compressed
    blob digests and uncompressed config rootfs diff_ids are checked when present.
    Multi-image archives are rejected rather than choosing an arbitrary image.
    """
    try:
        with tarfile.open(path, "r:*") as outer:
            members = {}
            for member in outer:
                name = archive_name(member.name)
                if name in members:
                    raise ValueError(f"duplicate archive member: {name}")
                members[name] = member

            def stream(name):
                name = archive_name(name)
                member = members.get(name)
                if member is None or not member.isfile():
                    raise ValueError(f"missing regular archive member: {name}")
                return outer.extractfile(member)

            def metadata(name, expected=None):
                with stream(name) as src:
                    data = src.read(8 * 1024 * 1024 + 1)
                if len(data) > 8 * 1024 * 1024:
                    raise ValueError(f"archive metadata too large: {name}")
                actual = hashlib.sha256(data).hexdigest()
                named = (name.split("/")[-1] if name.startswith("blobs/sha256/")
                         else posixpath.basename(name).removesuffix(".json"))
                if (SHA.fullmatch(named) and actual != named) or (expected and expected != "sha256:" + actual):
                    raise ValueError(f"config/metadata SHA-256 mismatch: {name}")
                return json.loads(data), actual

            layer_digests = {}
            if "manifest.json" in members:
                manifest, _ = metadata("manifest.json")
                if not isinstance(manifest, list) or len(manifest) != 1:
                    raise ValueError("archive must contain exactly one image")
                config_name = manifest[0]["Config"]
                layers = manifest[0]["Layers"]
            else:
                index, _ = metadata("index.json")
                def images(node, depth=0):
                    if depth > 8:
                        raise ValueError("OCI index nesting too deep")
                    if "manifests" not in node:
                        return [node]
                    found = []
                    for descriptor in node["manifests"]:
                        if descriptor.get("annotations", {}).get("vnd.docker.reference.type") == "attestation-manifest":
                            continue
                        dg = descriptor["digest"]
                        if not re.fullmatch(r"sha256:[0-9a-f]{64}", dg):
                            raise ValueError("unsupported OCI digest: " + dg)
                        child, _ = metadata("blobs/sha256/" + dg[7:], dg)
                        found.extend(images(child, depth + 1))
                    return found
                manifests = images(index)
                if len(manifests) != 1:
                    raise ValueError("OCI archive must contain exactly one runnable image")
                manifest = manifests[0]
                config_name = "blobs/sha256/" + manifest["config"]["digest"].removeprefix("sha256:")
                layers = []
                for descriptor in manifest["layers"]:
                    dg = descriptor["digest"]
                    if not re.fullmatch(r"sha256:[0-9a-f]{64}", dg):
                        raise ValueError("unsupported OCI layer digest: " + dg)
                    name = "blobs/sha256/" + dg[7:]
                    layers.append(name)
                    layer_digests[name] = dg[7:]
            config, config_sha = metadata(config_name)
            named = posixpath.basename(config_name).removesuffix(".json")
            if not SHA.fullmatch(named):
                raise ValueError("config blob name must be its SHA-256: " + config_name)
            diff_ids = config.get("rootfs", {}).get("diff_ids")
            if diff_ids is not None and len(diff_ids) != len(layers):
                raise ValueError("config rootfs.diff_ids count differs from layer count")
            executables = {}
            for i, name in enumerate(layers):
                changes, removals, directory_replacements = {}, [], set()
                with stream(name) as src:
                    raw = HashReader(src)
                    # BufferedReader needs a richer IO interface; use the seekable
                    # outer member only to sniff the gzip magic before streaming.
                    magic = src.read(2)
                    src.seek(0)
                    decoded = gzip.GzipFile(fileobj=raw) if magic == b"\x1f\x8b" else raw
                    unpacked = HashReader(decoded)
                    with tarfile.open(fileobj=unpacked, mode="r|") as layer:
                        for entry in layer:
                            p = archive_name(entry.name)
                            parent, leaf = posixpath.split(p)
                            if leaf == ".wh..wh..opq":
                                removals.append(parent + "/" if parent else "")
                            elif leaf.startswith(".wh."):
                                removals.append(posixpath.join(parent, leaf[4:]))
                            elif p == "app" or p.startswith("app/"):
                                if entry.isdir():
                                    directory_replacements.add("/" + p)
                                    changes.pop("/" + p, None)
                                else:
                                    removals.append(p)
                                    for key in list(changes):
                                        if key == "/" + p or key.startswith("/" + p + "/"):
                                            del changes[key]
                                    if entry.isfile():   # every regular file under app/ (binaries AND libraries), matching the Build CC policy
                                        h = HashReader(layer.extractfile(entry))
                                        changes["/" + p] = h.drain()
                            # TarFile otherwise retains every layer header.
                            layer.members.clear()
                    diff_sha = unpacked.drain()
                    blob_sha = raw.drain()
                    if magic == b"\x1f\x8b":
                        decoded.close()
                expected = layer_digests.get(name)
                if name.startswith("blobs/sha256/"):
                    expected = name.rsplit("/", 1)[1]
                if expected and blob_sha != expected:
                    raise ValueError(f"layer blob SHA-256 mismatch: {name}")
                if diff_ids is not None and diff_ids[i] != "sha256:" + diff_sha:
                    raise ValueError(f"layer uncompressed SHA-256 mismatch (rootfs.diff_ids[{i}]): {name}")
                # Whiteouts affect the lower view regardless of order in the tar.
                for p in removals:
                    target = "/" + p
                    for key in list(executables):
                        if key == target or key.startswith(target.rstrip("/") + "/"):
                            del executables[key]
                for key in directory_replacements:
                    executables.pop(key, None)
                executables.update(changes)
            return {"image_config_id": "sha256:" + config_sha,
                    "platform": f"{config['os']}/{config['architecture']}",
                    "layer_count": len(layers), "executables": dict(sorted(executables.items()))}
    except (tarfile.TarError, OSError, EOFError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid image archive: {exc}") from exc


def collect_facts(record_path, archive_path, config_path, gguf=None):
    record_path, archive_path, config_path = [Path(p).resolve() for p in
                                             (record_path, archive_path, config_path)]
    record, config = normalize_build_record(read_json(record_path)), read_json(config_path)
    if not isinstance(record, dict) or not isinstance(config, dict):
        raise ValueError("build record and HF config must be JSON objects")
    archive_sha, archive_bytes = digest(archive_path)
    config_sha, _ = digest(config_path)
    record_sha, _ = digest(record_path)
    result = {
        "format": "engine-lab-facts/0",
        "archive": {"artifact": fact(str(archive_path)), "sha256": fact(archive_sha),
                    "bytes": fact(archive_bytes)},
        "build_record": {"artifact": fact(str(record_path)), "sha256": fact(record_sha),
                         "claims": {k: fact(v, "declared", str(record_path))
                                    for k, v in record.items()}},
        "model_config": {"artifact": fact(str(config_path)), "sha256": fact(config_sha),
                         "values": {k: fact(v, source=str(config_path)) for k, v in config.items()}},
        "gguf": {}, "unresolved": [],
    }
    try:
        observation = observe_archive(archive_path)
        result["archive"].update({k: fact(v, source=str(archive_path)) for k, v in observation.items()})
    except ValueError as exc:
        result["unresolved"].append(f"archive observation failed: {exc}")
    for name in CLAIMS:
        result["build_record"]["claims"].setdefault(name, fact(None, "missing", source=str(record_path)))
    if gguf:
        result["gguf"] = gguf_header(Path(gguf).resolve())
    else:
        result["gguf"] = {k: fact(None, "missing") for k in (
            "arch", "expert_count", "expert_used_count", "block_count", "file_type", "context_length")}
    for claim, key in (("image_config_id", "image_config_id"), ("platform", "platform"),
                       ("executable_sha256", "executables")):
        actual = observed(result["archive"].get(key))
        if record.get(claim) != actual:
            result["unresolved"].append(f"build_record.{claim}: differs from observed archive")
    # Ancillary build metadata stays declared; it is not an admission identity.
    for name, value in result["gguf"].items():
        if value["tag"] == "missing":
            result["unresolved"].append(f"gguf.{name}: missing")
    for key, actual in (("archive_sha256", archive_sha), ("archive_bytes", archive_bytes)):
        if record.get(key) != actual:
            result["unresolved"].append(f"build_record.{key}: differs from observed archive")
    return result


def observed(node):
    return node.get("value") if isinstance(node, dict) and node.get("tag") == "observed" else None


def generate(facts, workload, argv, requirements=None, requirements_source=None):
    """Copy archive observations and explicitly tagged requester declarations."""
    archive = facts["archive"]
    model = facts["model_config"]
    workload = Path(workload).resolve() if workload else None
    workload_sha = digest(workload)[0] if workload else None
    handoff = {
        "format": FORMAT, "status": "received", "handoff_id": None,
        "release": {
            "role": None, "platform": observed(archive.get("platform")),
            "transport": {"kind": "docker-archive", "artifact": observed(archive["artifact"]),
                          "sha256": observed(archive["sha256"]), "bytes": observed(archive["bytes"])},
            "expected_image_config_id": observed(archive.get("image_config_id")),
            "build_record": observed(facts["build_record"]["artifact"]),
            "executable_sha256": observed(archive.get("executables")),
        },
        "inputs": {"model": {"repository": None, "revision": None,
                             "manifest": observed(model["artifact"]),
                             "manifest_sha256": observed(model["sha256"])},
                   "workload": {"artifact": str(workload) if workload else None, "sha256": workload_sha}},
        "invocation": {"argv": argv or None,
                       "mounts": [{"input": "model", "access": "ro"},
                                  {"input": "workload", "access": "ro"},
                                  {"role": "results", "access": "rw"}]},
        "sandbox_requirements": {"os": "linux", "gpu": {"vendor": None, "count": None},
                                 "required_runtime_compatibility": None,
                                 "network_after_staging": "none",
                                 "isolation_and_provisioning_owner": "lab-team",
                                 "resource_limits": {"cpus": None, "memory": None, "pids": None, "timeout_s": None}},
        "tests": {"requested": None, "policy_reference": None,
                  "profiling": {"requested": None, "tool_and_version": None}},
        "evidence": {"control_and_durable_storage_outside_test_sandbox": "required", "outputs": [],
                     "crash_outcome": "interrupted-attempt-with-partial-coverage",
                     "resume": "new-attempt-after-confirmed-stop"},
        "unresolved": list(facts.get("unresolved", [])),
    }

    handoff["provenance"] = {}
    for key, identity in (("image_config_id", "release.expected_image_config_id"),
                          ("platform", "release.platform"),
                          ("executables", "release.executable_sha256")):
        handoff["provenance"][identity] = archive.get(key, fact(None, "missing"))
    allowed = ("handoff_id", "release.role", "inputs.model.repository", "inputs.model.revision",
               "inputs.model.manifest", "inputs.model.manifest_sha256", "inputs.workload.artifact",
               "sandbox_requirements.gpu.vendor", "sandbox_requirements.gpu.count",
               "sandbox_requirements.required_runtime_compatibility",
               "sandbox_requirements.network_after_staging",
               "sandbox_requirements.resource_limits.cpus", "sandbox_requirements.resource_limits.memory",
               "sandbox_requirements.resource_limits.pids", "sandbox_requirements.resource_limits.timeout_s",
               "tests.requested", "tests.policy_reference", "tests.profiling.requested",
               "tests.profiling.tool_and_version")
    if requirements is not None:
        if not isinstance(requirements, dict):
            raise ValueError("requirements must be a mapping")
        def supplied(path):
            node = requirements
            for key in path.split("."):
                if not isinstance(node, dict) or key not in node:
                    return False
                node = node[key]
            return True
        for path in allowed:
            if supplied(path):
                parent, _, key = path.rpartition(".")
                target = at(handoff, parent) if parent else handoff
                target[key] = at(requirements, path)
                handoff["provenance"][path] = fact(target[key], "declared", requirements_source)
        if supplied("inputs.workload.artifact") and at(requirements, "inputs.workload.artifact"):
            artifact = Path(at(requirements, "inputs.workload.artifact"))
            if not artifact.is_absolute() and requirements_source:
                artifact = Path(requirements_source).resolve().parent / artifact
            artifact = artifact.resolve()
            workload_sha, _ = digest(artifact)
            handoff["inputs"]["workload"] = {"artifact": str(artifact), "sha256": workload_sha}
    handoff["provenance"]["inputs.workload.sha256"] = fact(
        handoff["inputs"]["workload"]["sha256"],
        tag="observed" if handoff["inputs"]["workload"]["sha256"] else "missing",
        source=handoff["inputs"]["workload"]["artifact"])

    def missing(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k not in {"unresolved", "provenance"}:
                    missing(v, f"{prefix}.{k}" if prefix else k)
        elif node is None:
            if prefix == "tests.profiling.tool_and_version" and at(handoff, "tests.profiling.requested") is False:
                return
            handoff["unresolved"].append(prefix + ": missing")

    missing(handoff)
    return handoff


def at(node, path):
    for name in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(name)
    return node


def validate(handoff, base):
    errors, unknown = [], []

    def reject(reason):
        errors.append(reason)

    if not isinstance(handoff, dict):
        return {"state": "rejected", "reasons": ["handoff must be a mapping"]}
    if handoff.get("format") != FORMAT:
        reject("unsupported handoff format")
    if handoff.get("status") not in {"received", "accepted", "rejected", "needs-information"}:
        reject("status must be a receipt state")

    # Scan recursively, including unused extensions. Nothing here is executed.
    def forbidden(node, path="handoff", ancestors=()):
        if isinstance(node, (dict, list)):
            if id(node) in ancestors:
                reject(path + ": cyclic YAML is forbidden")
                return
            ancestors = ancestors + (id(node),)
        if isinstance(node, dict):
            for key, value in node.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if any(word in normalized for word in ("script", "shell", "cmd", "hostpath", "privileged", "dockersocket")):
                    reject(f"{path}.{key}: forbidden execution key")
                forbidden(value, f"{path}.{key}", ancestors)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                forbidden(value, f"{path}[{i}]", ancestors)
        elif isinstance(node, str):
            if ("docker.sock" in node.lower() or "--privileged" in node.lower()
                    or node.lower() in {"script", "shell", "cmd"}):
                reject(path + ": forbidden execution string")

    forbidden(handoff)
    identities = ("release.transport.sha256", "release.transport.bytes",
                  "release.expected_image_config_id", "release.executable_sha256",
                  "inputs.model.manifest_sha256", "inputs.workload.sha256", "invocation.argv")
    for path in identities:
        if at(handoff, path) is None:
            reject(path + ": null or missing required identity")
    for path in ("release.transport.sha256", "inputs.model.manifest_sha256", "inputs.workload.sha256"):
        value = at(handoff, path)
        if not isinstance(value, str) or not SHA.fullmatch(value):
            reject(path + ": expected lowercase SHA-256 hex")
    image = at(handoff, "release.expected_image_config_id")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        reject("release.expected_image_config_id: expected sha256:<64 hex>")
    size = at(handoff, "release.transport.bytes")
    if type(size) is not int or size <= 0:
        reject("release.transport.bytes: expected positive integer")
    executables = at(handoff, "release.executable_sha256")
    if (not isinstance(executables, dict) or not executables
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not SHA.fullmatch(v)
                   for k, v in (executables.items() if isinstance(executables, dict) else []))):
        reject("release.executable_sha256: expected nonempty executable-to-SHA-256 map")
    argv = at(handoff, "invocation.argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or "\0" in v for v in argv) or not argv[0]:
        reject("invocation.argv: expected nonempty argument vector")
    elif isinstance(executables, dict) and argv[0] not in executables:
        reject("invocation.argv[0]: executable identity missing")
    if isinstance(argv, list):
        for arg in argv:
            if isinstance(arg, str) and (Path(arg).name.lower() in {
                "sh", "bash", "dash", "zsh", "ksh", "fish", "csh", "tcsh", "cmd", "cmd.exe",
                "powershell", "pwsh", "shell", "script"} or arg.endswith((".sh", ".bat", ".cmd", ".ps1"))
                    or re.search(r"(?:^|[\s/])(?:sh|bash|dash|zsh|cmd(?:\.exe)?|pwsh|powershell)\s+(?:-c|/c|-[Cc]ommand)\b", arg)
                    or re.match(r"--(?:shell|script|cmd|host-path)(?:=|$)", arg)):
                reject("invocation.argv: shell/script execution is forbidden")
    platform = at(handoff, "release.platform")
    if platform is None:
        unknown.append("release.platform: missing")
    elif platform != "linux/amd64":
        reject("release.platform: only linux/amd64 is admitted")
    if at(handoff, "release.transport.kind") != "docker-archive":
        reject("release.transport.kind must be docker-archive")
    mounts = at(handoff, "invocation.mounts")
    if not isinstance(mounts, list):
        reject("invocation.mounts must be an allowlisted list")
    else:
        seen = set()
        for mount in mounts:
            if not isinstance(mount, dict) or set(mount) not in ({"input", "access"}, {"role", "access"}):
                reject("invocation.mounts: only input/role and access are allowed")
                continue
            name = mount.get("input", mount.get("role"))
            access = mount.get("access")
            if (not isinstance(name, str) or not isinstance(access, str)
                    or (name, access) not in {("model", "ro"), ("workload", "ro"), ("results", "rw")}
                    or ("role" in mount) != (name == "results")):
                reject("invocation.mounts: mount is outside allowlist")
            elif name in seen:
                reject("invocation.mounts: duplicate role")
            else:
                seen.add(name)
    for path, value in {
        "sandbox_requirements.os": "linux",
        "sandbox_requirements.network_after_staging": "none",
        "sandbox_requirements.isolation_and_provisioning_owner": "lab-team",
        "evidence.control_and_durable_storage_outside_test_sandbox": "required",
        "evidence.crash_outcome": "interrupted-attempt-with-partial-coverage",
        "evidence.resume": "new-attempt-after-confirmed-stop",
    }.items():
        if at(handoff, path) != value:
            reject(path + ": must be " + value)

    def local_file(path):
        value = at(handoff, path)
        if not isinstance(value, str) or not value:
            reject(path + ": missing local artifact")
            return None
        target = Path(value)
        return target if target.is_absolute() else Path(base) / target

    actual_archive = None
    for path, hash_path in (("release.transport.artifact", "release.transport.sha256"),
                            ("inputs.model.manifest", "inputs.model.manifest_sha256"),
                            ("inputs.workload.artifact", "inputs.workload.sha256")):
        target = local_file(path)
        if target is not None:
            try:
                actual, nbytes = digest(target)
                if actual != at(handoff, hash_path):
                    reject(path + ": SHA-256 mismatch vs recomputed bytes")
                if path == "release.transport.artifact":
                    actual_archive = (actual, nbytes)
                    try:
                        inspection = observe_archive(target)
                        for key, identity in (("image_config_id", "release.expected_image_config_id"),
                                              ("platform", "release.platform"),
                                              ("executables", "release.executable_sha256")):
                            if inspection[key] != at(handoff, identity):
                                reject(identity + ": differs from re-observed archive")
                    except ValueError as exc:
                        reject(f"release.transport.artifact: archive observation failed: {exc}")
                    if nbytes != size:
                        reject(path + ": size mismatch vs recomputed bytes")
            except OSError as exc:
                unknown.append(f"{path}: cannot read artifact ({exc})")
    record_path = local_file("release.build_record")
    if record_path is not None:
        try:
            record = normalize_build_record(read_json(record_path))
            if not isinstance(record, dict):
                reject("release.build_record: expected JSON object")
            else:
                if actual_archive and (record.get("archive_sha256"), record.get("archive_bytes")) != actual_archive:
                    reject("build-record archive hash/bytes mismatch vs recomputed bytes")
                for name, path in (("image_config_id", "release.expected_image_config_id"),
                                   ("executable_sha256", "release.executable_sha256"),
                                   ("platform", "release.platform")):
                    if record.get(name) != at(handoff, path):
                        reject("build-record claim differs from handoff: " + name)
        except OSError as exc:
            unknown.append(f"release.build_record: cannot read ({exc})")
        except (ValueError, UnicodeError) as exc:
            reject(f"release.build_record: invalid JSON ({exc})")
    unresolved = handoff.get("unresolved")
    if not isinstance(unresolved, list) or any(not isinstance(v, str) for v in unresolved):
        reject("unresolved must be a list of strings")
    elif unresolved:
        unknown.extend(unresolved)
    return {"state": "rejected" if errors else "needs-information" if unknown else "accepted",
            "reasons": list(dict.fromkeys(errors + unknown))}


def markdown(facts):
    lines = ["# Engine handoff facts", "", "Values describe bytes or claims; no execution was performed.", ""]

    def walk(node, prefix=""):
        if isinstance(node, dict) and "tag" in node and "value" in node:
            lines.append(f"- `{prefix}` [{node['tag']}]: {json.dumps(node['value'], ensure_ascii=True)}")
        elif isinstance(node, dict):
            for key, value in node.items():
                if key not in {"format", "unresolved"}:
                    walk(value, f"{prefix}.{key}" if prefix else key)
    walk(facts)
    lines.extend(["", "## Unresolved", ""] + ["- " + v for v in facts["unresolved"]])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    facts = subs.add_parser("facts")
    for name in ("build-record", "archive", "model-config"):
        facts.add_argument("--" + name, required=True)
    facts.add_argument("--gguf")
    facts.add_argument("--out-dir", default=str(ROOT))
    gen = subs.add_parser("generate")
    gen.add_argument("--facts", required=True)
    gen.add_argument("--workload")
    gen.add_argument("--requirements")
    gen.add_argument("--out", required=True)
    gen.add_argument("--invocation-argv", nargs=argparse.REMAINDER, required=True)
    val = subs.add_parser("validate")
    val.add_argument("handoff")
    args = parser.parse_args(argv)
    try:
        if args.command == "facts":
            result = collect_facts(args.build_record, args.archive, args.model_config, args.gguf)
            dest = Path(args.out_dir)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "facts.json").write_text(json.dumps(result, indent=2) + "\n")
            (dest / "facts.md").write_text(markdown(result))
        elif args.command == "generate":
            requirements = None
            if args.requirements:
                with Path(args.requirements).open() as stream:
                    requirements = yaml.load(stream, Loader=UniqueLoader)
            workload = args.workload or at(requirements, "inputs.workload.artifact")
            if workload and not args.workload and args.requirements and not Path(workload).is_absolute():
                workload = str(Path(args.requirements).resolve().parent / workload)
            result = generate(read_json(args.facts), workload, args.invocation_argv,
                              requirements, str(Path(args.requirements).resolve()) if args.requirements else None)
            Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False))
        else:
            with Path(args.handoff).open() as stream:
                handoff = yaml.load(stream, Loader=UniqueLoader)
            verdict = validate(handoff, Path(args.handoff).resolve().parent)
            print(json.dumps(verdict))
            return 0 if verdict["state"] == "accepted" else 2
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError, yaml.YAMLError) as exc:
        if args.command == "validate":
            print(json.dumps({"state": "rejected", "reasons": [f"invalid handoff: {exc}"]}))
        else:
            print(str(exc), file=sys.stderr)
        return 2
    return 0


class UniqueLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate keys rather than accepting the last value."""


def unique_mapping(loader, node):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)

if __name__ == "__main__":
    sys.exit(main())
