import json
from pathlib import Path
import struct

import pytest
import yaml

import engine_handoff as eh

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def receipt(tmp_path):
    for name in ("engine.tar", "build-record.json", "config.json", "workload.json"):
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    facts = eh.collect_facts(tmp_path / "build-record.json", tmp_path / "engine.tar", tmp_path / "config.json")
    handoff = eh.generate(facts, tmp_path / "workload.json", ["/app/run", "--tokens", "8"])
    # Admission also checks the independently observed archive identities.
    record = eh.read_json(tmp_path / "build-record.json")
    handoff["release"].update(platform=record["platform"],
                               expected_image_config_id=record["image_config_id"],
                               executable_sha256=record["executable_sha256"])
    handoff["unresolved"] = []
    return handoff, tmp_path


def test_happy_path(receipt):
    handoff, base = receipt
    assert eh.validate(handoff, base) == {"state": "accepted", "reasons": []}


@pytest.mark.parametrize("path", ["release.transport.sha256", "release.transport.bytes",
                                    "release.expected_image_config_id", "release.executable_sha256",
                                    "inputs.model.manifest_sha256", "inputs.workload.sha256", "invocation.argv"])
def test_null_required_identity(receipt, path):
    handoff, base = receipt
    parent, key = path.rsplit(".", 1)
    eh.at(handoff, parent)[key] = None
    assert eh.validate(handoff, base)["state"] == "rejected"


def test_tampered_archive_same_size(receipt):
    handoff, base = receipt
    archive = base / "engine.tar"
    data = bytearray(archive.read_bytes())
    data[-1] ^= 1
    archive.write_bytes(data)
    verdict = eh.validate(handoff, base)
    assert verdict["state"] == "rejected"
    assert any("recomputed bytes" in reason for reason in verdict["reasons"])


def test_wrong_platform(receipt):
    handoff, base = receipt
    handoff["release"]["platform"] = "linux/arm64"
    assert eh.validate(handoff, base)["state"] == "rejected"


@pytest.mark.parametrize("mount", [{"role": "secrets", "access": "ro"},
                                   {"input": "model", "access": "rw"},
                                   {"input": "model", "access": "ro", "source": "/etc"}])
def test_extra_mount(receipt, mount):
    handoff, base = receipt
    handoff["invocation"]["mounts"].append(mount)
    assert eh.validate(handoff, base)["state"] == "rejected"


@pytest.mark.parametrize("key,value", [("privileged", False), ("shell", "false"),
                                       ("script", "anything"), ("cmd", ["true"]),
                                       ("host_path", "/etc"), ("docker_socket", None)])
def test_forbidden_key_even_nested(receipt, key, value):
    handoff, base = receipt
    handoff["tests"]["extension"] = [{key: value}]
    assert eh.validate(handoff, base)["state"] == "rejected"


@pytest.mark.parametrize("argv", [["/bin/sh", "-c", "true"], ["/app/run.sh"],
                                  "echo hello", [], ["/app/run", None],
                                  ["/app/run", "/var/run/docker.sock"],
                                  ["/app/run", "--privileged"],
                                  ["/app/run", "bash -c true"],
                                  ["/app/run", "--shell=/bin/bash"]])
def test_execution_escape(receipt, argv):
    handoff, base = receipt
    handoff["invocation"]["argv"] = argv
    assert eh.validate(handoff, base)["state"] == "rejected"


@pytest.mark.parametrize("network", [None, "host", "bridge", "default"])
def test_network(receipt, network):
    handoff, base = receipt
    handoff["sandbox_requirements"]["network_after_staging"] = network
    assert eh.validate(handoff, base)["state"] == "rejected"


def test_missing_artifact_needs_information(receipt):
    handoff, base = receipt
    (base / "engine.tar").unlink()
    assert eh.validate(handoff, base)["state"] == "needs-information"


@pytest.mark.parametrize("name", ["config.json", "workload.json"])
def test_tampered_inputs(receipt, name):
    handoff, base = receipt
    (base / name).write_text("tampered")
    assert eh.validate(handoff, base)["state"] == "rejected"


def test_build_record_mismatch(receipt):
    handoff, base = receipt
    record = eh.read_json(base / "build-record.json")
    record["archive_bytes"] += 1
    (base / "build-record.json").write_text(json.dumps(record))
    assert eh.validate(handoff, base)["state"] == "rejected"


def test_size_mismatch(receipt):
    handoff, base = receipt
    handoff["release"]["transport"]["bytes"] += 1
    assert eh.validate(handoff, base)["state"] == "rejected"


def test_relative_artifacts(receipt):
    handoff, base = receipt
    for path in ["release.transport.artifact", "release.build_record", "inputs.model.manifest", "inputs.workload.artifact"]:
        parent, key = path.rsplit(".", 1)
        eh.at(handoff, parent)[key] = Path(eh.at(handoff, path)).name
    assert eh.validate(handoff, base)["state"] == "accepted"


def test_facts_and_conservative_generation(receipt):
    _, base = receipt
    facts = eh.collect_facts(base / "build-record.json", base / "engine.tar", base / "config.json")
    assert facts["archive"]["sha256"]["tag"] == "observed"
    assert facts["build_record"]["claims"]["image_config_id"]["tag"] == "declared"
    assert facts["model_config"]["values"]["num_local_experts"]["value"] == 32
    assert facts["gguf"]["arch"]["tag"] == "missing"
    draft = eh.generate(facts, base / "workload.json", ["/app/run"])
    assert draft["release"]["expected_image_config_id"] == eh.observe_archive(base / "engine.tar")["image_config_id"]
    assert draft["release"]["executable_sha256"] == eh.observe_archive(base / "engine.tar")["executables"]
    assert draft["release"]["platform"] == "linux/amd64"
    assert eh.validate(draft, base)["state"] == "needs-information"
    assert "[declared]" in eh.markdown(facts)


def test_facts_preserves_all_claims_and_mismatch(receipt):
    _, base = receipt
    record = eh.read_json(base / "build-record.json")
    record["archive_sha256"] = "0" * 64
    record["extra_build_detail"] = "not verified"
    (base / "build-record.json").write_text(json.dumps(record))
    facts = eh.collect_facts(base / "build-record.json", base / "engine.tar", base / "config.json")
    assert facts["build_record"]["claims"]["extra_build_detail"]["tag"] == "declared"
    assert any("differs from observed" in v for v in facts["unresolved"])


def test_explicit_null_is_observed_not_absent(receipt):
    _, base = receipt
    config = eh.read_json(base / "config.json")
    config["rope_scaling"] = None
    (base / "config.json").write_text(json.dumps(config))
    facts = eh.collect_facts(base / "build-record.json", base / "engine.tar", base / "config.json")
    assert facts["model_config"]["values"]["rope_scaling"]["tag"] == "observed"
    assert facts["build_record"]["claims"]["patch_sha256"]["tag"] == "declared"


def test_unresolved_prevents_acceptance(receipt):
    handoff, base = receipt
    handoff["unresolved"] = ["runtime compatibility not established"]
    assert eh.validate(handoff, base)["state"] == "needs-information"


def test_cli_round_trip(receipt, capsys):
    handoff, base = receipt
    assert eh.main(["facts", "--build-record", str(base / "build-record.json"), "--archive", str(base / "engine.tar"),
                    "--model-config", str(base / "config.json"), "--out-dir", str(base)]) == 0
    assert (base / "facts.md").is_file()
    assert eh.main(["generate", "--facts", str(base / "facts.json"), "--workload", str(base / "workload.json"),
                    "--out", str(base / "draft.yaml"), "--invocation-argv", "/app/run", "--tokens", "8"]) == 0
    draft = yaml.safe_load((base / "draft.yaml").read_text())
    assert draft["invocation"]["argv"] == ["/app/run", "--tokens", "8"]
    (base / "complete.yaml").write_text(yaml.safe_dump(handoff))
    assert eh.main(["validate", str(base / "complete.yaml")]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "accepted"
    assert eh.main(["validate", str(base / "draft.yaml")]) == 2
    assert json.loads(capsys.readouterr().out)["state"] == "needs-information"


@pytest.mark.parametrize("text", ["[]", "status: received\nstatus: accepted", "!!python/object:subprocess.Popen {}", "a: [unclosed"])
def test_bad_yaml_json_verdict(tmp_path, capsys, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    assert eh.main(["validate", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["state"] == "rejected"


def test_header_stops_before_tensor_descriptors(tmp_path):
    # Deliberately no tensor descriptors despite tensor_count=1: an ordinary
    # GGUFReader would fail; header-only inspection must succeed.
    def string(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    values = {"general.architecture": "granitemoe", "general.file_type": 32,
              "granitemoe.expert_count": 32, "granitemoe.expert_used_count": 8,
              "granitemoe.block_count": 24, "granitemoe.context_length": 131072}
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(values))
    for key, value in values.items():
        data += string(key)
        data += struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack("<I", value)
    path = tmp_path / "header-only.gguf"
    path.write_bytes(data)
    result = eh.gguf_header(path)
    assert result["arch"]["value"] == "granitemoe"
    assert result["expert_used_count"]["value"] == 8
    assert all(v["tag"] == "observed" for v in result.values())
