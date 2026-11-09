"""CPU-only regression tests for raw-byte assessment and runner recovery."""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lab(monkeypatch, tmp_path):
    # lab is an executable without a suffix or an __main__ guard. Suppress only
    # CLI dispatch while importing its actual functions through importlib.
    monkeypatch.setenv("LAB", str(tmp_path))
    monkeypatch.delenv("LAB_FAULT", raising=False)
    with monkeypatch.context() as importing:
        importing.setattr(argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(f=lambda a: None))
        loader = importlib.machinery.SourceFileLoader("lab_under_test", str(ROOT / "bin/lab"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    module.TESTS = {"completion": {"parser": "completion"}, "gen": {"parser": "gen"}}
    return module


TIMING = b"eval time = 100 ms / 2 tokens (20 tokens per second)\n"


def test_actual_crashing_output(lab):
    path = ROOT / "lab/jobs/job-8df3c7bb27/att-1/st_prose_n18#8.stdout"
    # Keep this test portable when the read-only job evidence is not shipped.
    raw = path.read_bytes() if path.exists() else b"f Du Fu 's l\xc7\n\n"
    before = hashlib.sha256(raw).hexdigest()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")  # The original strict text read failed here.
    result = lab.assess("completion", raw, TIMING, 0)
    # Policy (orchestrator review): generated text may split a multibyte char; flag, don't fail.
    assert result["ok"] is True
    assert result["invalid_utf8"] is True
    assert result["decode_replacements"] >= 1
    assert result["decode_tps"] == 20.0
    assert "\ufffd" in result["text_head"]
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("raw", [b"abc\xe2\x82", b"abc\xff\xfedef"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_invalid_bytes_and_truncated_multibyte(lab, raw, stream):
    out, err = (raw, TIMING) if stream == "stdout" else (b"output", raw + b"\n" + TIMING)
    result = lab.assess("completion", out, err, 0)
    assert result["ok"] is True
    assert result["invalid_utf8"] is True
    assert result["decode_replacements"] >= 1
    assert result["decode_tokens"] == 2
    # Parsers that extract numbers from text fail closed on replaced bytes.
    lab.TESTS["ppl"] = {"parser": "ppl"}
    bad = lab.assess("ppl", b"", raw + b"\nFinal estimate: PPL = 2.5 +/- 0.1", 0)
    assert bad["ok"] is False and bad["invalid_utf8"] is True
    assert "assessment_error" not in result


@pytest.mark.parametrize("parser,out,err,expected", [
    ("ppl", "", "Final estimate: PPL = 2.5 +/- 0.1", dict(ok=True, ppl=2.5, ppl_stderr=0.1)),
    ("kl", "Mean KLD: 0.1", "", dict(ok=True, kld=0.1, kld_p99=None, top1_pct=None, ppl_ratio=None, ppl=None)),
    ("completion", "café \ufffd\r\n", TIMING.decode(), dict(ok=True, decode_tps=20.0, decode_tokens=2, prefill_tps=None, prefill_tokens=None, text_head="café \ufffd")),
    ("gen", "une réponse française assez longue", "", dict(ok=True, chars=34, head="une réponse française assez longue")),
    ("bench_json", '[{"n_prompt":1,"n_gen":2,"avg_ts":3}]', "", dict(ok=True, samples=[dict(n_prompt=1, n_gen=2, avg_ts=3, samples_ts=None)])),
    ("other", "", "", dict(ok=True)),
])
def test_valid_utf8_unchanged(lab, parser, out, err, expected):
    lab.TESTS[parser] = {"parser": parser}
    assert lab.assess(parser, out.encode(), err.encode(), 0) == expected
    assert lab.assess(parser, out, err, 0) == expected
    assert lab.assess(parser, out.encode(), err.encode(), 7)["ok"] is False


@pytest.mark.parametrize("parser,raw,error_type", [
    ("ppl", b"Final estimate: PPL = 1.2.3 +/- 0.1", "ValueError"),
    ("bench_json", b"[broken]", "JSONDecodeError"),
    ("bench_json", b"[{}]", "KeyError"),
])
def test_malformed_parser_output(lab, parser, raw, error_type):
    lab.TESTS[parser] = {"parser": parser}
    result = lab.assess(parser, raw, b"", 0)
    assert result["ok"] is False
    assert result["assessment_error"].startswith(error_type + ": ")


def test_files_are_preserved_and_read_errors_contained(lab, tmp_path):
    out, err = tmp_path / "raw.stdout", tmp_path / "raw.stderr"
    out.write_bytes(b"abc\xe2\x82")
    err.write_bytes(TIMING)
    before = (out.read_bytes(), err.read_bytes())
    assert lab.assess_files("completion", out, err, 0)["invalid_utf8"] is True
    assert (out.read_bytes(), err.read_bytes()) == before
    missing = lab.assess_files("completion", tmp_path / "absent", err, 0)
    assert missing["ok"] is False
    assert missing["assessment_error"].startswith("FileNotFoundError: ")


def test_assess_exception_is_contained_at_call_boundary(lab, monkeypatch, tmp_path):
    out, err = tmp_path / "out", tmp_path / "err"
    out.write_bytes(b"valid output")
    err.write_bytes(b"")

    def broken_assess(*args):
        raise ValueError("assessment failed")

    monkeypatch.setattr(lab, "assess", broken_assess)
    assert lab.assess_files("gen", out, err, 0) == dict(
        ok=False, assessment_error="ValueError: assessment failed")


def test_runner_records_parser_error_and_continues(lab, monkeypatch, tmp_path):
    jdir = tmp_path / "jobs/job-test"
    jdir.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake model")
    spec = dict(image="fake", image_id="fake-id", model=str(model), model_sha256=lab.sha(model),
                tests=["completion", "gen"], resources=dict(cpus="1", memory="1g", pids="10", timeout_s="10"))
    (jdir / "job.json").write_text(json.dumps(dict(spec=spec)))
    (tmp_path / "images/fake.json").write_text(json.dumps(dict(image_id="fake-id")))
    for td in lab.TESTS.values():
        td.update(tool="fake", args=[])
    calls = []
    raw = b"raw engine output with invalid byte \xff\n"
    clean = b"valid engine output long enough to pass generation\n"

    class FakeProcess:
        def __init__(self, cmd, stdout, stderr, **kwargs):
            calls.append(cmd)
            # Emulate subprocess fd writes: bypass TextIOWrapper encoding.
            lab.os.write(stdout.fileno(), raw if len(calls) == 1 else clean)
            lab.os.write(stderr.fileno(), TIMING)

        def poll(self):
            return 0

        def wait(self):
            return 0

    parse = lab._parse_assessment

    def faulty_parser(test, *args):
        if test == "completion":
            raise RuntimeError("injected parser failure")
        return parse(test, *args)

    monkeypatch.setattr(lab, "_parse_assessment", faulty_parser)
    monkeypatch.setattr(lab.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(lab, "sh", lambda *a, **kw: SimpleNamespace(stdout="42"))
    monkeypatch.setattr(lab, "gpu_state", lambda: dict(compute_apps=[]))
    monkeypatch.setattr(lab, "owned_containers", lambda *a: [])
    # Exercise real result writing, export, manifest hashing and terminal state.
    lab.cmd_run(SimpleNamespace(job="job-test"))
    adir = jdir / "att-1"
    result = json.loads((adir / "result.json").read_text())
    tests = result["quality"]["tests"]
    failed = tests["completion#0"]
    assert failed["assessment_error"] == "RuntimeError: injected parser failure"
    assert failed["ok"] is False
    assert failed["invalid_utf8"] is True
    assert failed["exit_code"] == 0
    assert failed["elapsed_s"] >= 0
    assert failed["mem"]["gpu_baseline_mib"] == 42
    assert failed["mem"]["samples"] == 0
    assert "assessment_error" not in tests["gen#1"]
    assert tests["gen#1"]["ok"] is True
    assert len(calls) == 2
    assert result["quality"]["all_ok"] is False
    assert lab.last_state(str(adir))["state"] == "terminal"
    assert (tmp_path / "export/job-test/att-1/result.json").read_bytes() == (adir / "result.json").read_bytes()
    for name in tests:
        assert (adir / (name + ".stdout")).read_bytes() == (raw if name == "completion#0" else clean)
        assert (adir / (name + ".stderr")).read_bytes() == TIMING
