"""No Docker daemon, CUDA toolkit, GPU, or network needed."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

import build_engine as engine


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / "input.txt").write_text("original\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "-c", "core.hooksPath=/dev/null",
                    "commit", "--quiet", "-m", "fixture"], check=True)
    return repo


def arguments(source, tmp_path, patch=None):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return argparse.Namespace(source=str(source), commit=commit, patch=patch,
                              tag="campaign/test:unit", out=str(tmp_path / "out"))


def record(args):
    return json.loads((Path(args.out) / "build-record.json").read_text())


def make_tar(path, objects):
    with tarfile.open(path, "w") as archive:
        for name, obj in objects.items():
            data = json.dumps(obj).encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_sha256_bytes(tmp_path):
    path = tmp_path / "payload"
    data = b"hello\x00world" * 200000
    path.write_bytes(data)
    assert engine.file_digest(path) == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    path.write_bytes(b"")
    assert engine.file_digest(path) == {"sha256": hashlib.sha256(b"").hexdigest(), "bytes": 0}


def test_patch_check_failure(source, tmp_path):
    patch = tmp_path / "bad.patch"
    patch.write_text("diff --git a/input.txt b/input.txt\n--- a/input.txt\n+++ b/input.txt\n@@ -1 +1 @@\n-missing\n+changed\n")
    args = arguments(source, tmp_path, str(patch))
    assert engine.build(args) == 1
    result = record(args)
    assert result["failure_stage"] == "patch_check"
    assert result["patch_sha256"] == engine.file_digest(patch)["sha256"]
    assert result["source"]["dirty"] is False
    assert (source / "input.txt").read_text() == "original\n"
    assert "patch failed" in (Path(args.out) / "build.stderr").read_text()
    assert (Path(args.out) / "build.stdout").exists()
    assert not list(Path(args.out).glob("staging-*"))


@pytest.fixture
def mock_docker(monkeypatch):
    real_run = subprocess.run
    calls = []
    state = {"fail_build": False, "staged_content": None}

    def fake(command, **kwargs):
        if command[0] != "docker":
            return real_run(command, **kwargs)
        calls.append(command)
        stdout, stderr = kwargs["stdout"], kwargs["stderr"]
        result = ""
        code = 0
        if command[1] == "version":
            result = '{"Client":{"Version":"test"},"Server":{"Version":"test"}}'
        elif command[1:3] == ["buildx", "version"]:
            result = "buildx unit-test"
        elif command[1:3] == ["buildx", "bake"]:
            env = kwargs["env"]
            staged = Path(kwargs["cwd"])
            assert env["SOURCE_DIR"] == "."
            assert (staged / ".campaign-recipe" / "Dockerfile").exists()
            assert Path(command[command.index("--file") + 1]).parent == staged / ".campaign-recipe"
            assert not any(a.startswith("--allow") for a in command)  # no filesystem entitlements requested
            state["staged_content"] = (staged / "input.txt").read_text()
            assert not (staged / ".git").exists()
            assert "@sha256:" in env["BASE_RUNTIME"]
            result = "full compiler stdout\n"
            stderr.write(b"full compiler stderr\n")
            metadata = {"engine": {"containerimage.config.digest": "sha256:" + "1" * 64}}
            Path(command[command.index("--metadata-file") + 1]).write_text(json.dumps(metadata))
            code = 42 if state["fail_build"] else 0
        elif command[1:3] == ["image", "inspect"]:
            if command[-1].startswith("nvidia/"):
                result = json.dumps([{"RepoDigests": ["nvidia/cuda@sha256:" + "2" * 64]}])
            else:
                result = json.dumps([{"Id": "sha256:" + "1" * 64, "RepoDigests": [],
                                      "Created": "2026-01-01T00:00:00Z", "Architecture": "amd64", "Os": "linux"}])
        elif command[1] == "run":
            assert "--network=none" in command
            assert "--rm" in command
            assert "/bin/sh" in command and "sha256sum" in command[-1]
            assert "sha256:" + "1" * 64 in command   # hashed by immutable id, not by tag
            result = "".join("a" * 64 + "  /app/" + tool + "\n" for tool in engine.TOOLS) + "b" * 64 + "  /app/libggml.so.0\n"
        elif command[1] == "save":
            assert command[-1] == "sha256:" + "1" * 64
            make_tar(command[command.index("--output") + 1], {"manifest.json": []})
        else:
            pytest.fail("Unexpected Docker command: " + repr(command))
        stdout.write(result.encode())
        return subprocess.CompletedProcess(command, code)

    monkeypatch.delenv("BASE_DEVEL", raising=False)
    monkeypatch.delenv("BASE_RUNTIME", raising=False)
    monkeypatch.setattr(subprocess, "run", fake)
    return calls, state


def test_record_schema_and_success(source, tmp_path, mock_docker):
    args = arguments(source, tmp_path)
    assert engine.build(args) == 0
    result = record(args)
    assert result["schema_version"] == 1
    assert result["status"] == "success"
    assert result["failure_stage"] is None
    assert result["source"]["commit"] == args.commit
    assert len(result["source"]["tree"]) == 40
    assert result["source"]["dirty"] is False
    assert result["patch_sha256"] is None
    assert result["platform"] == "linux/amd64"
    assert result["cmake_flags"] == engine.CMAKE_FLAGS
    assert {"Id", "RepoDigests", "Created", "Architecture", "Os", "docker_reported_id", "config_digest", "index_digest"} <= set(result["image"])
    assert set(result["tools"]) == {"/app/" + tool for tool in engine.TOOLS}
    assert "/app/libggml.so.0" in result["app_files"] and "/app/libggml.so.0" not in result["tools"]   # libraries hashed too
    # identities are distinct objects: the record must not present the docker-reported Id as the config digest
    assert result["image"]["docker_reported_id"] == result["image"]["Id"]
    assert result["archive"]["sha256"] == engine.file_digest(Path(args.out) / result["archive"]["path"])["sha256"]
    assert result["archive"]["bytes"] > 0
    assert result["base_images"]["runtime"]["digest"] == "sha256:" + "2" * 64
    assert set(result["versions"]) == {"docker", "buildx"}
    assert result["buildx_metadata"]["engine"]
    assert result["build_wall_seconds"] >= 0
    assert result["wall_seconds"] >= result["build_wall_seconds"]
    assert result["provenance"]["retained"] is False
    assert "full compiler stdout" in (Path(args.out) / "build.stdout").read_text()
    assert "full compiler stderr" in (Path(args.out) / "build.stderr").read_text()
    calls, _ = mock_docker
    assert not any("--push" in command or "prune" in command for command in calls)


def test_build_failure_preserves_logs_and_metadata(source, tmp_path, mock_docker):
    mock_docker[1]["fail_build"] = True
    args = arguments(source, tmp_path)
    assert engine.build(args) == 1
    result = record(args)
    assert result["failure_stage"] == "build"
    assert result["status"] == "failed"
    assert result["buildx_metadata"]
    assert result["build_wall_seconds"] >= 0
    assert "full compiler stdout" in (Path(args.out) / "build.stdout").read_text()
    assert "full compiler stderr" in (Path(args.out) / "build.stderr").read_text()
    assert not any(command[1] == "save" for command in mock_docker[0])


def test_valid_patch_changes_staging_only(source, tmp_path, mock_docker):
    patch = tmp_path / "good.patch"
    patch.write_text("diff --git a/input.txt b/input.txt\n--- a/input.txt\n+++ b/input.txt\n@@ -1 +1 @@\n-original\n+changed\n")
    args = arguments(source, tmp_path, str(patch))
    assert engine.build(args) == 0
    assert mock_docker[1]["staged_content"] == "changed\n"
    assert (source / "input.txt").read_text() == "original\n"
    assert record(args)["patch_sha256"] == engine.file_digest(patch)["sha256"]


@pytest.mark.parametrize("change", ["dirty", "wrong_commit"])
def test_source_rejected(source, tmp_path, change):
    args = arguments(source, tmp_path)
    if change == "dirty":
        (source / "input.txt").write_text("dirty\n")
    else:
        args.commit = "0" * 40
    assert engine.build(args) == 1
    assert record(args)["failure_stage"] == "verify_source"
    if change == "dirty":
        assert record(args)["source"]["dirty"] is True


def test_runner_captures_stdout_stderr_on_error(tmp_path):
    for name in ("build.stdout", "build.stderr"):
        (tmp_path / name).touch()
    with pytest.raises(RuntimeError):
        engine.Runner(tmp_path)([sys.executable, "-c",
                                "import sys; print('out'); print('err', file=sys.stderr); sys.exit(9)"])
    assert (tmp_path / "build.stdout").read_text() == "out\n"
    assert (tmp_path / "build.stderr").read_text() == "err\n"


@pytest.mark.parametrize("predicate,retained", [
    ("https://slsa.dev/provenance/v0.2", True),
    ("https://spdx.dev/Document", False),
])
def test_oci_provenance_requires_actual_predicate(tmp_path, predicate, retained):
    path = tmp_path / "oci.tar"
    manifest_digest, layer_digest = "3" * 64, "4" * 64
    make_tar(path, {
        "index.json": {"manifests": [{"digest": "sha256:" + manifest_digest,
                        "annotations": {"vnd.docker.reference.type": "attestation-manifest"}}]},
        "blobs/sha256/" + manifest_digest: {"layers": [{"mediaType": "application/vnd.in-toto+json",
                                                          "digest": "sha256:" + layer_digest}]},
        "blobs/sha256/" + layer_digest: {"predicateType": predicate},
    })
    assert engine.archive_provenance(path)["retained"] is retained


def test_empty_output_required(source, tmp_path):
    args = arguments(source, tmp_path)
    out = Path(args.out)
    out.mkdir()
    (out / "build-record.json").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        engine.build(args)
    assert (out / "build-record.json").read_text() == "keep"
