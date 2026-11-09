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

RECIPES = Path(__file__).resolve().parents[2] / "builds" / "llamacpp"
TOOLS = ("llama-bench", "llama-perplexity", "llama-completion", "llama-server",
         "llama-eval-callback", "llama-cli", "llama-tokenize")
CMAKE_FLAGS = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
               "-DCMAKE_CUDA_ARCHITECTURES=120", "-DLLAMA_CURL=OFF",
               "-DBUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF",
               "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_UI=OFF",
               "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF"]
DEFAULT_DEVEL = ("nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:"
                 "520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66")
DEFAULT_RUNTIME = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"


def file_digest(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def archive_identities(path):
    """Distinct identities from a `docker save` archive: config digest (image config), top-level index digest (what the containerd store
    reports as `Id`), and image-manifest digests. Never conflate them: only the config digest is the classic image ID."""
    out = {"config_digest": None, "index_digest": None, "index_manifests": []}
    with tarfile.open(path) as t:
        names = t.getnames()
        if "manifest.json" in names:
            m = json.load(t.extractfile("manifest.json"))
            if m and m[0].get("Config", "").startswith("blobs/sha256/"):
                out["config_digest"] = "sha256:" + m[0]["Config"].split("/")[-1]
        if "index.json" in names:
            raw = t.extractfile("index.json").read()
            idx = json.loads(raw)
            out["index_manifests"] = [{"digest": d.get("digest"), "mediaType": d.get("mediaType")} for d in idx.get("manifests", [])]
            out["index_digest"] = out["index_manifests"][0]["digest"] if len(out["index_manifests"]) == 1 else None
            out["index_entry_digests"] = [d["digest"] for d in out["index_manifests"]]   # may include an attestation manifest next to the image manifest
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

    def __call__(self, command, *, cwd=None, env=None, stdout_path=None, capture=True):
        # Spool process output to files: full compiler logs never accumulate in RAM.
        with tempfile.TemporaryFile(dir=self.out) as stdout, tempfile.TemporaryFile(dir=self.out) as stderr:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
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
        "schema_version": 1, "status": "failed", "failure_stage": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(Path(args.source).resolve()), "requested_commit": args.commit,
                   "commit": None, "tree": None, "dirty": None, "status_porcelain": None},
        "patch_sha256": None, "extra_targets": list(getattr(args, "extra_target", []) or []), "tag": args.tag, "platform": "linux/amd64",
        "cmake_flags": CMAKE_FLAGS,
        "recipes": {}, "base_images": {}, "versions": {},
        "image": None, "tools": {}, "app_files": {}, "archive": None, "buildx_metadata": None,
        "provenance": {"retained": False, "reason": "Archive not inspected"},
        "build_wall_seconds": None,
    }
    stage = "recipes"
    try:
        record["recipes"] = {name: file_digest(RECIPES / name)
                             for name in ("Dockerfile", "docker-bake.hcl")}
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
        # Use the immutable config ID after resolving the tag exactly once.
        image_id = info["Id"]
        stage = "hash_tools"
        hashes = run(["docker", "run", "--rm", "--network=none", "--read-only",
                      "--entrypoint", "/bin/sh", image_id, "-c",
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
        stage = "save_image"
        image_tar = out / "engine-image.docker.tar"
        run(["docker", "save", "--output", str(image_tar), image_id])
        record["archive"] = {"path": image_tar.name, **file_digest(image_tar)}
        record["image"].update({"docker_reported_id": info["Id"], **{"config_digest": None, "index_digest": None}})
        ids = archive_identities(image_tar)
        record["image"]["config_digest"], record["image"]["index_digest"] = ids["config_digest"], ids["index_digest"]
        record["image"]["index_manifests"] = ids["index_manifests"]
        record["image"]["docker_id_in_archive_index"] = info["Id"] in ids.get("index_entry_digests", [])
        record["image"]["id_note"] = "Id/docker_reported_id is whatever `docker image inspect` returned (containerd store: an index digest). config_digest is the image config digest. They are different objects."
        stage = "inspect_provenance"
        record["provenance"] = archive_provenance(image_tar)
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
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return build(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
