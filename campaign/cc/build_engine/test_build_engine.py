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

FIXTURES = Path(__file__).parent / "fixtures"


def linkage_output(ldd=None, symbols=None, tool="readelf", ldd_rc=0, symbols_rc=0):
    return ("__CAMPAIGN_LIBRARY__/app/libggml-cuda.so.0\n__CAMPAIGN_LDD_BEGIN__\n"
            + (ldd if ldd is not None else (FIXTURES / "ldd.txt").read_text())
            + f"__CAMPAIGN_LDD_END__{ldd_rc}\n__CAMPAIGN_SYMBOLS_BEGIN__{tool}\n"
            + (symbols if symbols is not None else (FIXTURES / "readelf.txt").read_text())
            + f"__CAMPAIGN_SYMBOLS_END__{symbols_rc}\n")


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
            state["cmake_defines"] = env["CMAKE_DEFINES"]
            state["dockerfile"] = (staged / ".campaign-recipe" / "Dockerfile").read_text()
            state["bake"] = (staged / ".campaign-recipe" / "docker-bake.hcl").read_text()
            result = "full compiler stdout\n"
            stderr.write(b"full compiler stderr\n")
            metadata = {"engine": {"containerimage.config.digest": "sha256:" + "1" * 64}}
            Path(command[command.index("--metadata-file") + 1]).write_text(json.dumps(metadata))
            code = 42 if state["fail_build"] else 0
        elif command[1:3] == ["image", "inspect"]:
            if command[-1].startswith("nvidia/"):
                result = json.dumps([{"RepoDigests": ["nvidia/cuda@sha256:" + "2" * 64]}])
            else:
                result = json.dumps([{"Id": "sha256:" + "9" * 64, "RepoDigests": [],
                                      "Created": "2026-01-01T00:00:00Z", "Architecture": "amd64", "Os": "linux"}])
        elif command[1] == "run":
            assert "--network=none" in command
            assert "--rm" in command
            assert "--read-only" in command
            assert "/bin/sh" in command
            assert "sha256:" + "1" * 64 in command   # hashed by immutable id, not by tag
            if command[-1] == engine.LINKAGE_SCRIPT:
                assert kwargs["timeout"] == 60
                result = state.get("linkage_output", linkage_output())
                code = state.get("linkage_code", 0)
                stderr.write(b"probe diagnostic\n")
                if state.get("linkage_timeout"):
                    stdout.write(b"partial probe output\n")
                    raise subprocess.TimeoutExpired(command, 60)
            else:
                assert "sha256sum" in command[-1]
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
    assert result["schema_version"] == 2
    assert result["status"] == "success"
    assert result["failure_stage"] is None
    assert result["source"]["commit"] == args.commit
    assert len(result["source"]["tree"]) == 40
    assert result["source"]["dirty"] is False
    assert result["patch_sha256"] is None
    assert result["platform"] == "linux/amd64"
    assert result["cmake_flags"] == engine.CMAKE_FLAGS
    assert result["cmake_defines"] == {}
    assert result["recipe_sha256"] == engine.recipe_identity({})[1]
    assert result["linkage"] == {"status": "not_requested"}
    assert result["image"]["config_digest"] == "sha256:" + "1" * 64
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


@pytest.mark.parametrize("items", [
    ["NO_EQUALS"], ["lower=ON"], ["A-B=ON"], ["=ON"], ["A="],
    ["A=two words"], ["A=tab\tvalue"], ["A=line\nbreak"],
    ["A=x;y"], ["A=$(id)"], ["A=`id`"], ["A=x&y"], ["A=x|y"],
    ["A=x>y"], ["A=x<y"], ['A="ON"'], ["A='ON'"], ["A=*"],
    ["A=?"], ["A=[x]"], ["A=x\\y"], ["A=~root"], ["A=x#y"],
    ["A=ON", "A=ON"], ["A=ON", "A=OFF"],
    *[[flag[2:]] for flag in engine.CMAKE_FLAGS],
])
def test_invalid_defines(items):
    with pytest.raises(ValueError):
        engine.validate_defines(items)


def test_valid_defines_are_canonical():
    assert engine.validate_defines(["Z_123=a/b:c+2.0,@%=-", "GGML_CUDA_NO_VMM=ON"]) == {
        "GGML_CUDA_NO_VMM": "ON", "Z_123": "a/b:c+2.0,@%=-"}
    assert engine.validate_defines(["123=OFF"]) == {"123": "OFF"}


def test_recipe_hash_includes_defines_and_recipe_bytes(tmp_path, monkeypatch):
    default = engine.recipe_identity({})[1]
    variant = engine.recipe_identity({"GGML_CUDA_NO_VMM": "ON"})[1]
    assert default != variant
    assert variant != engine.recipe_identity({"GGML_CUDA_NO_VMM": "OFF"})[1]
    assert variant != engine.recipe_identity({"GGML_CUDA_NO_VMM": "ON"}, ["extra"])[1]
    assert engine.recipe_identity({"A": "ON", "B": "OFF"}) == engine.recipe_identity({"B": "OFF", "A": "ON"})
    for name in ("Dockerfile", "docker-bake.hcl"):
        (tmp_path / name).write_bytes((engine.RECIPES / name).read_bytes())
    monkeypatch.setattr(engine, "RECIPES", tmp_path)
    (tmp_path / "Dockerfile").write_text((tmp_path / "Dockerfile").read_text() + "\n# variant\n")
    assert engine.recipe_identity({})[1] != default


def test_variant_build_args_record_and_label(source, tmp_path, mock_docker):
    args = arguments(source, tmp_path)
    args.cmake_define = ["GGML_CUDA_NO_VMM=ON", "ZZ_TEST=2"]
    assert engine.build(args) == 0
    result = record(args)
    assert result["cmake_defines"] == {"GGML_CUDA_NO_VMM": "ON", "ZZ_TEST": "2"}
    assert result["cmake_flags"] == engine.CMAKE_FLAGS + ["-DGGML_CUDA_NO_VMM=ON", "-DZZ_TEST=2"]
    assert result["recipe_sha256"] == engine.recipe_identity(result["cmake_defines"])[1]
    state = mock_docker[1]
    assert state["cmake_defines"] == "-DGGML_CUDA_NO_VMM=ON -DZZ_TEST=2"
    assert 'campaign.cmake.defines="${CMAKE_DEFINES}"' in state["dockerfile"]
    assert '${CMAKE_DEFINES} \\\n    && cmake --build' in state["dockerfile"]
    assert 'CMAKE_DEFINES = CMAKE_DEFINES' in state["bake"]


def test_invalid_define_before_output_creation(source, tmp_path):
    args = arguments(source, tmp_path)
    args.cmake_define = ["GGML_CUDA=OFF"]
    with pytest.raises(ValueError, match="built-in"):
        engine.build(args)
    assert not Path(args.out).exists()


@pytest.mark.parametrize("tool", ["readelf", "nm"])
def test_linkage_parser_captured_elf(tool):
    result = engine.parse_linkage((FIXTURES / "ldd.txt").read_text(),
                                  (FIXTURES / (tool + ".txt")).read_text(), symbol_tool=tool)
    assert result["status"] == "passed"
    assert result["libcudart_dynamic"] is True
    assert result["undefined_imports"] == dict.fromkeys(engine.LINKAGE_SYMBOLS, True)


def test_linkage_parser_versioned_imports_and_exact_names():
    text = "  1: 000000 0 FUNC GLOBAL DEFAULT UND cudaMalloc@libcudart.so.12 (2)\n"
    text += "  2: 000000 0 FUNC GLOBAL DEFAULT UND cudaFreeAsync\n"
    text += "  3: 000001 8 FUNC GLOBAL DEFAULT 12 cudaMemGetInfo\n"
    result = engine.parse_linkage("libcudart.so.12 => not found\n", text)
    assert result["libcudart_dynamic"] is True
    assert result["undefined_imports"] == {"cudaMalloc": True, "cudaFree": False, "cudaMemGetInfo": False}
    assert result["status"] == "failed"
    assert engine.parse_linkage("libc.so.6 => /lib/libc.so.6\n", "")["libcudart_dynamic"] is False


@pytest.mark.parametrize("missing", ["ldd", "symbols", "both"])
def test_linkage_parser_unavailable_is_not_false(missing):
    result = engine.parse_linkage("ldd unavailable\n", "readelf and nm unavailable\n",
                                  ldd_returncode=127 if missing in ("ldd", "both") else 0,
                                  symbols_returncode=127 if missing in ("symbols", "both") else 0,
                                  symbol_tool="unavailable" if missing in ("symbols", "both") else "readelf")
    assert result["status"] == "unavailable"
    if missing in ("ldd", "both"):
        assert result["libcudart_dynamic"] is None
        assert "unavailable" in result["ldd"]["reason"]
    if missing in ("symbols", "both"):
        assert all(value is None for value in result["undefined_imports"].values())
        assert "unavailable" in result["symbols"]["reason"]


def test_linkage_build_success_and_raw_outputs(source, tmp_path, mock_docker):
    args = arguments(source, tmp_path)
    args.linkage = True
    assert engine.build(args) == 0
    linkage = record(args)["linkage"]
    assert linkage["status"] == "passed"
    assert linkage["image_config_id"] == "sha256:" + "1" * 64
    assert linkage["library"] == "/app/libggml-cuda.so.0"
    assert (Path(args.out) / linkage["ldd_output"]).read_text() == (FIXTURES / "ldd.txt").read_text()
    assert (Path(args.out) / linkage["symbols_output"]).read_text() == (FIXTURES / "readelf.txt").read_text()
    assert (Path(args.out) / "linkage.stderr.txt").read_text() == "probe diagnostic\n"


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("failure", ["missing_tools", "missing_imports", "docker", "malformed", "timeout"])
def test_linkage_failure_policy(source, tmp_path, mock_docker, optional, failure):
    args = arguments(source, tmp_path)
    args.linkage = not optional
    args.linkage_optional = optional  # this flag alone also enables the check
    state = mock_docker[1]
    if failure == "missing_tools":
        state["linkage_output"] = linkage_output(symbols="readelf and nm unavailable\n",
                                                 tool="unavailable", symbols_rc=127)
    elif failure == "missing_imports":
        state["linkage_output"] = linkage_output(symbols="no CUDA imports\n")
    elif failure == "docker":
        state["linkage_code"] = 125
    elif failure == "timeout":
        state["linkage_timeout"] = True
    else:
        state["linkage_output"] = "unexpected probe output\n"
    assert engine.build(args) == (0 if optional else 1)
    result = record(args)
    assert result["status"] == ("success" if optional else "failed")
    assert result["failure_stage"] == (None if optional else "linkage")
    assert result["linkage"]["status"] == ("failed" if failure == "missing_imports" else "unavailable")
    assert result["archive"]  # all other evidence survives even a fatal linkage failure
    assert (Path(args.out) / "linkage.raw.txt").exists()
    if failure == "timeout":
        assert (Path(args.out) / "linkage.raw.txt").read_text() == "partial probe output\n"
        assert "timed out" in result["linkage"]["reason"]


@pytest.mark.parametrize("tool", ["readelf", "nm", "unavailable"])
def test_probe_shell_tool_selection_and_versioned_library(tmp_path, tool):
    import os
    import shlex
    app = tmp_path / "app"
    app.mkdir()
    (app / "libggml-cuda.so.0").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, output in [("ldd", (FIXTURES / "ldd.txt").read_text())] + (
            [] if tool == "unavailable" else [(tool, (FIXTURES / (tool + ".txt")).read_text())]):
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\nprintf '%s' " + shlex.quote(output) + "\n")
        executable.chmod(0o755)
    script = engine.LINKAGE_SCRIPT.replace("/app", str(app))
    result = subprocess.run(["/bin/sh", "-c", script], env={**os.environ, "PATH": str(bin_dir)},
                            capture_output=True, text=True, check=True)
    assert f"__CAMPAIGN_LIBRARY__{app}/libggml-cuda.so.0\n" in result.stdout
    assert f"__CAMPAIGN_SYMBOLS_BEGIN__{tool}\n" in result.stdout
    assert f"__CAMPAIGN_SYMBOLS_END__{127 if tool == 'unavailable' else 0}\n" in result.stdout


@pytest.mark.parametrize("library_exists", [False, True])
def test_probe_shell_missing_library_or_all_tools(tmp_path, library_exists):
    app = tmp_path / "app"
    app.mkdir()
    if library_exists:
        (app / "libggml-cuda.so").touch()
    script = engine.LINKAGE_SCRIPT.replace("/app", str(app))
    result = subprocess.run(["/bin/sh", "-c", script], env={"PATH": str(tmp_path / "no-tools")},
                            capture_output=True, text=True, check=True)
    code = 127 if library_exists else 125
    assert f"__CAMPAIGN_LDD_END__{code}\n" in result.stdout
    assert f"__CAMPAIGN_SYMBOLS_END__{code}\n" in result.stdout
    assert ("ldd unavailable" if library_exists else "CUDA backend library not found") in result.stdout
