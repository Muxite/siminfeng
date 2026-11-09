"""CPU-only input identity regressions; all job writes stay in the fixture lab."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lab(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB", str(tmp_path))
    monkeypatch.delenv("LAB_FAULT", raising=False)
    (tmp_path / "tests").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "images/fake.json").write_text(json.dumps(dict(image_id="fake-id")))
    models = tmp_path / "models"
    models.mkdir()
    for name in ("bf16.gguf", "mix.gguf", "q8.gguf"):
        (models / name).write_bytes(name.encode())
    prompts = tmp_path / "prompts"
    (prompts / "sub").mkdir(parents=True)
    for name in ("chat.txt", "sub/prose.txt", "unused.txt"):
        (prompts / name).write_text(name)
    tests = {
        "default": dict(tool="fake", parser="exit0", args=[]),
        "mix": dict(tool="fake", parser="exit0", model_file="mix.gguf",
                    docker_args=["-v", f"{prompts}:/tin:ro"], args=["-f", "/tin/chat.txt"]),
        "q8": dict(tool="fake", parser="exit0", model_file="q8.gguf",
                   docker_args=["--volume=" + f"{prompts}:/tin:ro"],
                   args=["--file=/tin/sub/prose.txt"]),
    }
    (tmp_path / "tests/fake.json").write_text(json.dumps(tests))
    with monkeypatch.context() as importing:
        importing.setattr(argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(f=lambda a: None))
        loader = importlib.machinery.SourceFileLoader("lab_inputs_under_test", str(ROOT / "bin/lab"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    # Any unmocked external process is a test failure (no Docker/GPU).
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected subprocess")
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    monkeypatch.setattr(module, "sh", forbidden)
    return module


def submit(lab, **overrides):
    args = dict(image="fake", model=str(Path(lab.LAB) / "models/bf16.gguf"),
                model_sha256=None, data=None, tests="default,mix,q8,mix", key="key", res=None)
    args.update(overrides)
    lab.cmd_submit(SimpleNamespace(**args))
    return Path(lab.LAB) / "jobs" / ("job-" + lab.sha_s(args["key"])[:10])


def job_spec(jdir):
    return json.loads((jdir / "job.json").read_text())["spec"]


def test_manifest_tracks_actual_models_and_referenced_files(lab):
    jdir = submit(lab)
    inputs = job_spec(jdir)["inputs"]
    root = Path(lab.LAB)
    assert inputs["model_files"] == {name: lab.sha(root / "models" / name)
                                      for name in ("bf16.gguf", "mix.gguf", "q8.gguf")}
    assert inputs["mounts"] == {str(root / "prompts"): {
        name: lab.sha(root / "prompts" / name) for name in ("chat.txt", "sub/prose.txt")}}
    assert inputs["tests"] == {t: lab.test_definition_sha(lab.TESTS[t]) for t in lab.TESTS}
    assert lab.verify_inputs(job_spec(jdir)) is None
    # Override-only jobs must not claim the unused default is a test model.
    other = submit(lab, key="overrides", tests="mix,q8")
    assert set(job_spec(other)["inputs"]["model_files"]) == {"mix.gguf", "q8.gguf"}


@pytest.mark.parametrize("name", ["models/bf16.gguf", "models/mix.gguf", "models/q8.gguf",
                                  "prompts/chat.txt", "prompts/sub/prose.txt", "data.txt"])
@pytest.mark.parametrize("change", ["modify", "delete"])
def test_each_file_fails_preparing_before_execution(lab, name, change):
    root = Path(lab.LAB)
    (root / "data.txt").write_text("data")
    jdir = submit(lab, data=str(root / "data.txt"))
    path = root / name
    if change == "delete": path.unlink()
    else: path.write_bytes(b"different bytes")
    with pytest.raises(SystemExit) as exc:
        lab.cmd_run(SimpleNamespace(job=jdir.name))
    assert exc.value.code == 4
    state = lab.last_state(str(jdir / "att-1"))
    assert state["state"] == "terminal"
    assert state["execution"] == "not_started"
    assert str(path) in state["reason"]
    assert not (jdir / "att-1/result.json").exists()


@pytest.mark.parametrize("change", ["args", "model_file", "env", "delete"])
def test_definition_change_fails_preparing(lab, change):
    jdir = submit(lab)
    if change == "delete": del lab.TESTS["mix"]
    else: lab.TESTS["mix"][change] = {"args": [], "model_file": "q8.gguf", "env": {"X": "1"}}[change]
    with pytest.raises(SystemExit, match="4"):
        lab.cmd_run(SimpleNamespace(job=jdir.name))
    assert "test definition hash mismatch: mix" in lab.last_state(str(jdir / "att-1"))["reason"]


def test_canonical_definition_hash(lab):
    td = lab.TESTS["mix"]
    assert lab.test_definition_sha(td) == lab.test_definition_sha(dict(reversed(list(td.items()))))
    path = Path(lab.LAB) / "tests/fake.json"
    path.write_text(json.dumps(lab.TESTS, indent=4, sort_keys=True))
    assert lab.load_tests() == lab.TESTS
    assert lab.test_definition_sha(lab.load_tests()["mix"]) == lab.test_definition_sha(td)


@pytest.mark.parametrize("name", ["models/mix.gguf", "prompts/chat.txt", "prompts/sub/prose.txt"])
def test_missing_referenced_file_rejected_without_job(lab, name):
    path = Path(lab.LAB) / name
    path.unlink()
    with pytest.raises(SystemExit) as exc:
        submit(lab)
    assert str(path) in str(exc.value)
    assert not (Path(lab.LAB) / "jobs").exists()


def test_idempotency_and_unreferenced_files(lab):
    jdir = submit(lab)
    before = (jdir / "job.json").read_bytes()
    (Path(lab.LAB) / "prompts/unused.txt").write_text("changed")
    submit(lab)
    assert (jdir / "job.json").read_bytes() == before
    assert lab.verify_inputs(job_spec(jdir)) is None
    (Path(lab.LAB) / "prompts/chat.txt").write_text("changed")
    with pytest.raises(SystemExit, match="DIFFERENT spec"):
        submit(lab)
    assert (jdir / "job.json").read_bytes() == before


@pytest.mark.parametrize("change", ["model", "definition", "resources"])
def test_new_key_rejects_other_identity_changes(lab, change):
    jdir = submit(lab)
    before = (jdir / "job.json").read_bytes()
    kwargs = {}
    if change == "model": (Path(lab.LAB) / "models/mix.gguf").write_bytes(b"changed")
    elif change == "definition": lab.TESTS["mix"]["env"] = {"X": "1"}
    else: kwargs["res"] = "cpus=2"
    with pytest.raises(SystemExit, match="DIFFERENT spec"):
        submit(lab, **kwargs)
    assert (jdir / "job.json").read_bytes() == before


def test_mount_matching_uses_path_boundaries_and_nested_mount(lab):
    root = Path(lab.LAB)
    nested = root / "nested"
    nested.mkdir()
    (nested / "prose.txt").write_text("nested file")
    lab.TESTS["mix"]["docker_args"] += ["-v" + f"{nested}:/tin/sub:ro"]
    lab.TESTS["mix"]["args"] += ["--file=/tin/sub/prose.txt", "/tin-other/not-an-input"]
    jdir = submit(lab, tests="mix")
    assert job_spec(jdir)["inputs"]["mounts"] == {
        str(root / "prompts"): {"chat.txt": lab.sha(root / "prompts/chat.txt")},
        str(nested): {"prose.txt": lab.sha(nested / "prose.txt")}}


def test_missing_required_data_is_rejected(lab):
    lab.TESTS["default"]["needs_data"] = True
    with pytest.raises(SystemExit, match="requires --data"):
        submit(lab)


def make_legacy(jdir):
    job = json.loads((jdir / "job.json").read_text())
    del job["spec"]["inputs"]
    job["spec_sha256"] = "legacy-preserved"
    (jdir / "job.json").chmod(0o644)
    (jdir / "job.json").write_text(json.dumps(job))


def test_legacy_idempotency_keeps_immutable_spec(lab):
    jdir = submit(lab)
    make_legacy(jdir)
    before = (jdir / "job.json").read_bytes()
    # Missing override/prompt inputs must not invalidate the old key comparison.
    (Path(lab.LAB) / "models/mix.gguf").unlink()
    (Path(lab.LAB) / "prompts/chat.txt").unlink()
    submit(lab)
    assert (jdir / "job.json").read_bytes() == before
    with pytest.raises(SystemExit, match="DIFFERENT spec"):
        submit(lab, tests="default")


@pytest.mark.parametrize("legacy", [False, True])
def test_unchanged_inputs_run_to_completion(lab, monkeypatch, legacy):
    jdir = submit(lab)
    if legacy:
        make_legacy(jdir)
        (Path(lab.LAB) / "models/mix.gguf").write_bytes(b"legacy not checked")
    calls = []

    class FakeProcess:
        def __init__(self, cmd, **kwargs): calls.append(cmd)
        def poll(self): return 0
        def wait(self): return 0

    monkeypatch.setattr(lab.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(lab, "sh", lambda *a, **kw: SimpleNamespace(stdout="0"))
    monkeypatch.setattr(lab, "gpu_state", lambda: dict(compute_apps=[]))
    monkeypatch.setattr(lab, "owned_containers", lambda *a: [])
    lab.cmd_run(SimpleNamespace(job=jdir.name))
    assert len(calls) == 4
    assert "/m/mix.gguf" in calls[1]
    states = [json.loads(s) for s in (jdir / "att-1/state.jsonl").read_text().splitlines()]
    preparing = next(s for s in states if s["state"] == "preparing")
    assert preparing.get("inputs_unverified", False) is legacy
    assert states[-1]["state"] == "terminal"
    assert json.loads((jdir / "att-1/result.json").read_text())["execution"]["status"] == "completed"


def test_legacy_model_is_still_verified(lab):
    jdir = submit(lab)
    make_legacy(jdir)
    (Path(lab.LAB) / "models/bf16.gguf").unlink()
    with pytest.raises(SystemExit, match="4"):
        lab.cmd_run(SimpleNamespace(job=jdir.name))
    state = lab.last_state(str(jdir / "att-1"))
    assert state["inputs_unverified"] is True
    assert "bf16.gguf" in state["reason"]
