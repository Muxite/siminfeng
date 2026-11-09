#!/usr/bin/env python3
"""Build a local CUDA engine and record evidence; uses only the Python stdlib."""
import argparse
import hashlib
import json
import shutil
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone

RECIPES = Path(__file__).resolve().parent / "recipes"
TOOLS = ("llama-bench", "llama-perplexity", "llama-completion", "llama-server",
         "llama-eval-callback", "llama-cli", "llama-tokenize")
CMAKE_FLAGS = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
               "-DCMAKE_CUDA_ARCHITECTURES=120", "-DLLAMA_CURL=OFF",
               "-DBUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF",
               "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_UI=OFF",
               "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF",
               "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
               "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
               "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"]
DEFAULT_DEVEL = ("nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:"
                 "520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66")
DEFAULT_RUNTIME = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"


def validate_defines(items):
    """Return canonical definitions; allow only literal, single shell words."""
    builtins = {flag[2:].split("=", 1)[0] for flag in CMAKE_FLAGS}
    defines = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Z0-9_]+", key):
            raise ValueError("--cmake-define requires KEY=VALUE with KEY matching ^[A-Z0-9_]+$")
        if not re.fullmatch(r"[A-Za-z0-9_./:+,@%=-]+", value):
            raise ValueError("--cmake-define VALUE must be nonempty and contain only letters, digits, _./:+,@%=-")
        if key in builtins or key in defines:
            raise ValueError("Duplicate or built-in CMake definition: " + key)
        defines[key] = value
    return dict(sorted(defines.items()))


def recipe_identity(defines, extra_targets=()):
    recipes = {name: file_digest(RECIPES / name)
               for name in ("Dockerfile", "docker-bake.hcl")}
    inputs = {"recipes": recipes, "cmake_defines": defines,
              "cmake_flags": CMAKE_FLAGS, "extra_targets": list(extra_targets)}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return recipes, digest


LINKAGE_SYMBOLS = ("cudaMalloc", "cudaFree", "cudaMemGetInfo")
LINKAGE_SCRIPT = r'''
library=""
for candidate in /app/libggml-cuda.so /app/libggml-cuda.so.*; do
    if [ -f "$candidate" ]; then library="$candidate"; break; fi
done
printf '__CAMPAIGN_LIBRARY__%s\n' "$library"
printf '__CAMPAIGN_LDD_BEGIN__\n'
if [ -z "$library" ]; then
    echo 'CUDA backend library not found in /app'; ldd_rc=125
elif command -v ldd >/dev/null 2>&1; then
    ldd "$library" 2>&1; ldd_rc=$?
else
    echo 'ldd unavailable in runtime image'; ldd_rc=127
fi
printf '__CAMPAIGN_LDD_END__%s\n' "$ldd_rc"
symbol_tool=unavailable
if command -v readelf >/dev/null 2>&1; then symbol_tool=readelf
elif command -v nm >/dev/null 2>&1; then symbol_tool=nm; fi
printf '__CAMPAIGN_SYMBOLS_BEGIN__%s\n' "$symbol_tool"
if [ -z "$library" ]; then
    echo 'CUDA backend library not found in /app'; symbols_rc=125
elif [ "$symbol_tool" = readelf ]; then
    readelf -Ws "$library" 2>&1; symbols_rc=$?
elif [ "$symbol_tool" = nm ]; then
    nm -D "$library" 2>&1; symbols_rc=$?
else
    echo 'readelf and nm unavailable in runtime image'; symbols_rc=127
fi
printf '__CAMPAIGN_SYMBOLS_END__%s\n' "$symbols_rc"
'''


def parse_linkage(ldd_text, symbols_text, *, symbol_tool="readelf",
                  ldd_returncode=0, symbols_returncode=0):
    dynamic = None if ldd_returncode else bool(re.search(r"\blibcudart\.so(?:\.[\w.]+)?\b", ldd_text))
    imports = {name: None for name in LINKAGE_SYMBOLS}
    if not symbols_returncode and symbol_tool in ("readelf", "nm"):
        for name in imports:
            if symbol_tool == "readelf":
                pattern = r"\bUND\s+" + name + r"(?:@\S+)?(?:\s|$)"
            else:
                pattern = r"^\s*(?:[0-9a-fA-F]+\s+)?[Uwv]\s+" + name + r"(?:@\S+)?(?:\s|$)"
            imports[name] = bool(re.search(pattern, symbols_text, re.MULTILINE))
    unavailable = bool(ldd_returncode or symbols_returncode or symbol_tool not in ("readelf", "nm"))
    passed = dynamic is True and all(value is True for value in imports.values())
    return {"status": "unavailable" if unavailable else "passed" if passed else "failed",
            "libcudart_dynamic": dynamic, "undefined_imports": imports,
            "ldd": {"returncode": ldd_returncode,
                    "reason": ldd_text.strip() if ldd_returncode else None},
            "symbols": {"tool": symbol_tool, "returncode": symbols_returncode,
                        "reason": symbols_text.strip() if symbols_returncode or symbol_tool == "unavailable" else None}}


def collect_linkage(run, out, image_id):
    result = {"status": "unavailable", "image_config_id": image_id,
              "libcudart_dynamic": None, "undefined_imports": dict.fromkeys(LINKAGE_SYMBOLS),
              "raw_output": "linkage.raw.txt", "stderr_output": "linkage.stderr.txt"}
    try:
        run(["docker", "run", "--rm", "--network=none", "--read-only",
             "--entrypoint", "/bin/sh", image_id, "-c", LINKAGE_SCRIPT],
            stdout_path=out / "linkage.raw.txt", stderr_path=out / "linkage.stderr.txt", timeout=60)
        raw = (out / "linkage.raw.txt").read_text()
        match = re.fullmatch(r"__CAMPAIGN_LIBRARY__([^\n]*)\n__CAMPAIGN_LDD_BEGIN__\n(.*?)"
                             r"__CAMPAIGN_LDD_END__(\d+)\n__CAMPAIGN_SYMBOLS_BEGIN__(\w+)\n(.*?)"
                             r"__CAMPAIGN_SYMBOLS_END__(\d+)\n", raw, re.DOTALL)
        if not match:
            raise ValueError("Malformed linkage probe output; see linkage.raw.txt and linkage.stderr.txt")
        library, ldd, ldd_rc, tool, symbols, symbols_rc = match.groups()
        (out / "linkage.ldd.txt").write_text(ldd)
        (out / "linkage.symbols.txt").write_text(symbols)
        result.update(parse_linkage(ldd, symbols, symbol_tool=tool,
                                   ldd_returncode=int(ldd_rc), symbols_returncode=int(symbols_rc)))
        result.update(library=library or None, ldd_output="linkage.ldd.txt",
                      symbols_output="linkage.symbols.txt")
    except Exception as exc:
        result.update(status="unavailable", reason=str(exc))
    return result


def file_digest(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def archive_identities(path):
    """Read the linux/amd64 config through OCI indexes or a classic Docker manifest.

    index.json is an archive catalog, not necessarily the original image index.
    Its descriptors may include both an image index and a platform manifest.
    """
    out = {"config_digest": None, "index_digest": None, "index_manifests": [],
           "index_entry_digests": [], "manifest_digest": None}
    configs, manifests, indexes = set(), set(), set()
    with tarfile.open(path) as archive:
        names = {member.name.removeprefix("./"): member for member in archive}

        def read(name, digest=None):
            member = names.get(name)
            if member is None or not member.isfile():
                raise ValueError("Missing image archive object: " + name)
            raw = archive.extractfile(member).read()
            actual = "sha256:" + hashlib.sha256(raw).hexdigest()
            if digest is not None and actual != digest:
                raise ValueError("Image archive digest mismatch: " + name)
            return json.loads(raw), actual

        def blob(digest):
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("Invalid image archive digest: " + str(digest))
            return read("blobs/sha256/" + digest.split(":")[1], digest)[0]

        seen = set()

        def visit(desc):
            if desc.get("annotations", {}).get("vnd.docker.reference.type") == "attestation-manifest":
                return
            platform = desc.get("platform", {})
            if platform and (platform.get("os"), platform.get("architecture")) != ("linux", "amd64"):
                return
            digest = desc.get("digest")
            if digest in seen:
                return
            seen.add(digest)
            doc = blob(digest)
            if "manifests" in doc:
                indexes.add(digest)
                for child in doc["manifests"]:
                    visit(child)
            elif "config" in doc:
                config_id = doc["config"]["digest"]
                config = blob(config_id)
                if (config.get("os"), config.get("architecture")) == ("linux", "amd64"):
                    configs.add(config_id)
                    manifests.add(digest)

        if "index.json" in names:
            catalog, _ = read("index.json")
            out["index_manifests"] = [{"digest": d.get("digest"), "mediaType": d.get("mediaType")}
                                      for d in catalog.get("manifests", [])]
            out["index_entry_digests"] = [d["digest"] for d in out["index_manifests"]]
            for descriptor in catalog.get("manifests", []):
                visit(descriptor)
        elif "manifest.json" in names:
            entries, _ = read("manifest.json")
            for entry in entries:
                config, digest = read(entry["Config"])
                if (config.get("os"), config.get("architecture")) == ("linux", "amd64"):
                    configs.add(digest)
        if len(configs) != 1:
            raise ValueError("Saved archive must identify exactly one linux/amd64 image config")
        out["config_digest"] = configs.pop()
        out["manifest_digest"] = next(iter(manifests)) if len(manifests) == 1 else None
        out["index_digest"] = next(iter(indexes)) if len(indexes) == 1 else None
    return out


def archive_provenance(path):
    """Follow only manifests reachable from the saved archive's OCI index."""
    with tarfile.open(path, "r:*") as archive:
        names = {member.name.removeprefix("./"): member for member in archive}

        def read_json(name):
            member = names.get(name)
            if member is None:
                raise ValueError("missing OCI manifest: " + name)
            with archive.extractfile(member) as stream:
                return json.load(stream)

        if "index.json" not in names:
            return {"retained": False, "reason": "No OCI index.json in docker save archive",
                    "attestation_manifests": []}
        found, seen = [], set()

        def visit(desc, attestation=False):
            digest = desc.get("digest", "")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or digest in seen:
                return
            seen.add(digest)
            name = "blobs/sha256/" + digest.split(":")[1]
            doc = read_json(name)
            marked = (attestation or desc.get("annotations", {}).get(
                "vnd.docker.reference.type") == "attestation-manifest")
            for layer in doc.get("layers", []):
                if layer.get("mediaType") != "application/vnd.in-toto+json":
                    continue
                statement = read_json("blobs/sha256/" + layer["digest"].split(":")[1])
                if statement.get("predicateType", "").startswith("https://slsa.dev/provenance/"):
                    # The in-toto predicate itself proves this is provenance, not just an SBOM.
                    found.append({"manifest": digest, "statement": layer["digest"],
                                  "predicate_type": statement["predicateType"],
                                  "attestation_annotation": marked})
            for child in doc.get("manifests", []):
                visit(child, marked)

        for descriptor in read_json("index.json").get("manifests", []):
            visit(descriptor)
        return {"retained": bool(found), "reason": "Inspected reachable OCI manifests and predicates",
                "attestation_manifests": found}


class Runner:
    def __init__(self, out):
        self.out = out

    def __call__(self, command, *, cwd=None, env=None, stdout_path=None, capture=True,
                 stderr_path=None, timeout=None):
        # Spool process output to files: full compiler logs never accumulate in RAM.
        with tempfile.TemporaryFile(dir=self.out) as stdout, tempfile.TemporaryFile(dir=self.out) as stderr:
            expired = None
            try:
                result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                        **({"timeout": timeout} if timeout is not None else {}))
            except subprocess.TimeoutExpired as exc:
                expired = exc
            stdout.seek(0)
            stderr.seek(0)
            with (self.out / "build.stdout").open("ab") as log:
                if stdout_path:
                    with Path(stdout_path).open("wb") as target:
                        copy_stream(stdout, target)
                else:
                    copy_stream(stdout, log)
            with (self.out / "build.stderr").open("ab") as log:
                copy_stream(stderr, log)
            if stderr_path:
                stderr.seek(0)
                with Path(stderr_path).open("wb") as target:
                    copy_stream(stderr, target)
            if expired:
                raise expired
            if result.returncode:
                raise RuntimeError(f"Command exited {result.returncode}: {command!r}")
            if stdout_path or not capture:
                return ""
            stdout.seek(0)
            return stdout.read().decode("utf-8", errors="replace")


def copy_stream(source, target):
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        target.write(chunk)


def resolve_base(run, value, kind):
    expected = f"nvidia/cuda:12.8.1-{kind}-ubuntu24.04"
    name, separator, digest = value.partition("@")
    if name not in (expected, "docker.io/" + expected):
        raise ValueError(f"Expected {expected}, got {name}")
    if not separator:
        info = json.loads(run(["docker", "image", "inspect", name]))[0]
        candidates = [item.partition("@")[2] for item in info.get("RepoDigests", [])
                      if item.partition("@")[0] in ("nvidia/cuda", "docker.io/nvidia/cuda")]
        if not candidates:
            raise ValueError(f"No local RepoDigest for {name}; set BASE_{kind.upper()}=name@sha256:...")
        digest = candidates[0]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid base digest: " + digest)
    return {"reference": name + "@" + digest, "digest": digest}


def build(args, runner=None):
    defines = validate_defines(getattr(args, "cmake_define", []) or [])
    define_flags = [f"-D{key}={value}" for key, value in defines.items()]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Never overwrite evidence from an earlier attempt.
    if any(out.iterdir()):
        raise ValueError("--out must be an empty directory")
    for name in ("build.stdout", "build.stderr"):
        (out / name).touch()
    run = runner or Runner(out)
    started = time.monotonic()
    record = {
        "schema_version": 2, "status": "failed", "failure_stage": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(Path(args.source).resolve()), "requested_commit": args.commit,
                   "commit": None, "tree": None, "dirty": None, "status_porcelain": None},
        "patch_sha256": None, "extra_targets": list(getattr(args, "extra_target", []) or []), "tag": args.tag, "platform": "linux/amd64",
        "cmake_flags": CMAKE_FLAGS + define_flags, "cmake_defines": defines,
        "recipe_sha256": None,
        "linkage": {"status": "not_requested"},
        "recipes": {}, "base_images": {}, "versions": {},
        "image": None, "tools": {}, "app_files": {}, "archive": None, "buildx_metadata": None,
        "provenance": {"retained": False, "reason": "Archive not inspected"},
        "build_wall_seconds": None,
    }
    stage = "recipes"
    try:
        record["recipes"], record["recipe_sha256"] = recipe_identity(defines, record["extra_targets"])
        stage = "verify_source"
        source = Path(args.source).resolve()
        git_env = os.environ.copy()
        git_env["GIT_OPTIONAL_LOCKS"] = "0"

        def git(*items):
            return run(["git", "-C", str(source), *items], env=git_env).strip()

        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.commit):
            raise ValueError("--commit must be a hexadecimal SHA (7-40 characters)")
        if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
            raise ValueError("--source must be the checkout root")
        head = git("rev-parse", "HEAD")
        expected = git("rev-parse", "--verify", args.commit + "^{commit}")
        record["source"].update(commit=head, tree=git("rev-parse", "HEAD^{tree}"))
        dirty = git("status", "--porcelain=v1", "--untracked-files=all")
        record["source"].update(dirty=bool(dirty), status_porcelain=dirty)
        if head != expected:
            raise ValueError("Source HEAD does not match --commit")
        if dirty:
            raise ValueError("Source is dirty; commit changes or supply --patch")
        if any(line.startswith("160000 ") for line in git("ls-files", "--stage").splitlines()):
            raise ValueError("Submodules are unsupported; git archive would omit their contents")

        with tempfile.TemporaryDirectory(prefix="staging-", dir=out) as temporary:
            temporary = Path(temporary)
            stage = "stage_source"
            snapshot = temporary / "source.tar"
            run(["git", "-C", str(source), "archive", "--format=tar", head],
                env=git_env, stdout_path=snapshot)
            staged = temporary / "source"
            staged.mkdir()
            with tarfile.open(snapshot) as archive:
                archive.extractall(staged, filter="data")
            if args.patch:
                stage = "patch_check"
                patch = temporary / "input.patch"
                patch.write_bytes(Path(args.patch).read_bytes())
                record["patch_sha256"] = file_digest(patch)["sha256"]
                patch_env = git_env.copy()
                patch_env["GIT_CEILING_DIRECTORIES"] = str(temporary)
                run(["git", "apply", "--check", str(patch)], cwd=staged, env=patch_env)
                stage = "patch_apply"
                run(["git", "apply", str(patch)], cwd=staged, env=patch_env)
            # User-supplied ignore rules must not silently remove tracked source.
            for ignore in (".dockerignore", "Dockerfile.dockerignore"):
                ignore_path = staged / ignore
                if ignore_path.is_symlink():
                    ignore_path.unlink()
                ignore_path.write_text("")

            stage = "versions"
            record["versions"]["docker"] = run(["docker", "version", "--format", "{{json .}}"])
            record["versions"]["buildx"] = run(["docker", "buildx", "version"]).strip()
            stage = "base_images"
            record["base_images"]["devel"] = resolve_base(
                run, os.environ.get("BASE_DEVEL", DEFAULT_DEVEL), "devel")
            record["base_images"]["runtime"] = resolve_base(
                run, os.environ.get("BASE_RUNTIME", DEFAULT_RUNTIME), "runtime")
            env = os.environ.copy()
            env.update(TAG=args.tag, SOURCE_DIR=str(staged), SOURCE_COMMIT=head, EXTRA_TARGETS=" ".join(getattr(args, "extra_target", []) or []),
                       CMAKE_DEFINES=" ".join(define_flags),
                       PATCH_SHA256=record["patch_sha256"] or "",
                       BASE_DEVEL=record["base_images"]["devel"]["reference"],
                       BASE_RUNTIME=record["base_images"]["runtime"]["reference"],
                       BUILDX_METADATA_PROVENANCE="max")
            stage = "build"
            build_started = time.monotonic()
            try:
                # Stage the recipe INSIDE the build context so Bake needs no filesystem entitlements (no fs.read=/).
                recipe_dir = staged / ".campaign-recipe"; recipe_dir.mkdir(exist_ok=True)
                for name in ("Dockerfile", "docker-bake.hcl"):
                    shutil.copy2(RECIPES / name, recipe_dir / name)
                env["SOURCE_DIR"] = "."
                run(["docker", "buildx", "bake", "--file", str(recipe_dir / "docker-bake.hcl"),
                     "--metadata-file", str(out / "buildx-metadata.json"), "--progress", "plain", "engine"],
                    cwd=staged, env=env, capture=False)
            finally:
                record["build_wall_seconds"] = time.monotonic() - build_started
                if (out / "buildx-metadata.json").exists():
                    record["buildx_metadata"] = json.loads((out / "buildx-metadata.json").read_text())

        stage = "inspect_image"
        info = json.loads(run(["docker", "image", "inspect", args.tag]))[0]
        record["image"] = {key: info.get(key) for key in
                           ("Id", "RepoDigests", "Created", "Architecture", "Os")}
        if info.get("Architecture") != "amd64" or info.get("Os") != "linux":
            raise ValueError("Built image is not linux/amd64")
        # Save the inspected immutable local object before any runtime probe. With
        # containerd this can be an index ID; it must never be used as a config ID.
        reported_id = info["Id"]
        if not isinstance(reported_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", reported_id):
            raise ValueError("Docker inspect lacks an immutable image ID")
        record["image"]["docker_reported_id"] = reported_id
        stage = "save_image"
        image_tar = out / "engine-image.docker.tar"
        run(["docker", "save", "--output", str(image_tar), reported_id])
        record["archive"] = {"path": image_tar.name, **file_digest(image_tar)}
        stage = "inspect_image_identity"
        ids = archive_identities(image_tar)
        record["image"].update({key: ids[key] for key in
                               ("config_digest", "index_digest", "manifest_digest", "index_manifests")})
        record["image"]["docker_id_in_archive_index"] = reported_id in ids["index_entry_digests"]
        metadata = (record["buildx_metadata"] or {}).get("engine", {})
        metadata_config = metadata.get("containerimage.config.digest")
        image_id = ids["config_digest"]
        if metadata_config is not None and metadata_config != image_id:
            raise ValueError("Saved archive config digest differs from Buildx metadata")
        descriptor = metadata.get("containerimage.descriptor") or {}
        metadata_digest = metadata.get("containerimage.digest") or descriptor.get("digest")
        if metadata_digest is not None:
            if not isinstance(metadata_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", metadata_digest):
                raise ValueError("Invalid Buildx image digest")
            if descriptor.get("digest") and descriptor["digest"] != metadata_digest:
                raise ValueError("Buildx image descriptor and digest disagree")
            media_type = descriptor.get("mediaType", "")
            is_index = "index" in media_type or "manifest.list" in media_type
            expected = ids["index_digest"] if is_index or not media_type else ids["manifest_digest"]
            if expected and metadata_digest != expected:
                kind = "index" if is_index or not media_type else "manifest"
                raise ValueError("Saved archive " + kind + " digest differs from Buildx metadata")
            if is_index:
                record["image"]["index_digest"] = metadata_digest
        record["image"]["id_note"] = (
            "Id/docker_reported_id is the Docker inspect ID (possibly an OCI index digest). "
            "config_digest is the archive-derived linux/amd64 config ID; run_ref (= Docker ID) is what local docker run accepts; "
            "manifest_digest and index_digest identify distinct OCI objects.")
        # Observed (novmm-2/shot-1, containerd store): `docker run <config digest>` exits 125;
        # Docker only resolves its own reported ID locally. Probes run by that ID, which must
        # be the index/manifest that the saved archive contains; config_digest stays the identity.
        run_ref = reported_id
        record["image"]["run_ref"] = run_ref
        stage = "hash_tools"
        hashes = run(["docker", "run", "--rm", "--network=none", "--read-only",
                      "--entrypoint", "/bin/sh", run_ref, "-c",
                      "find /app -type f -print0 | sort -z | xargs -0 /usr/bin/sha256sum"])
        for line in hashes.splitlines():
            digest, name = line.split(maxsplit=1)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid file sha256")
            record["app_files"][name.strip()] = digest   # every regular file shipped in /app (binaries AND shared libraries)
        required = {"/app/" + tool for tool in TOOLS + tuple(getattr(args, "extra_target", []) or [])}
        if not required <= set(record["app_files"]):
            raise ValueError("Missing tool hashes: " + ", ".join(sorted(required - set(record["app_files"]))))
        record["tools"] = {k: v for k, v in record["app_files"].items() if k in required}
        stage = "inspect_provenance"
        record["provenance"] = archive_provenance(image_tar)
        if getattr(args, "linkage", False) or getattr(args, "linkage_optional", False):
            stage = "linkage"
            record["linkage"] = collect_linkage(run, out, run_ref)
            record["linkage"].update(image_config_id=image_id, run_ref=run_ref)
            record["linkage"]["optional"] = bool(getattr(args, "linkage_optional", False))
            if record["linkage"]["status"] != "passed" and not getattr(args, "linkage_optional", False):
                raise RuntimeError("Linkage check " + record["linkage"]["status"] + "; see linkage evidence")
        record.update(status="success", failure_stage=None)
    except Exception as exc:
        record.update(status="failed", failure_stage=stage, error=str(exc))
        with (out / "build.stderr").open("a") as log:
            log.write(f"\n{stage}: {exc}\n")
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out / "build-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["status"] == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--patch")
    parser.add_argument("--extra-target", action="append", default=[], help="extra CMake executable target to build and ship in /app (e.g. llama-routing-trace added by a patch)")
    parser.add_argument("--cmake-define", action="append", default=[], metavar="KEY=VALUE",
                        help="additional CMake definition (repeatable; built-in flags cannot be overridden)")
    parser.add_argument("--linkage", action="store_true", help="check dynamic cudart linkage and CUDA imports after building; failures are fatal")
    parser.add_argument("--linkage-optional", action="store_true", help="enable linkage check but allow failed/unavailable checks")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return build(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
