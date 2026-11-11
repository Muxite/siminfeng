"""Small, offline Docker-save and OCI fixtures; no image runtime required."""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest
import yaml

import engine_handoff as eh


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tar_bytes(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, payload, mode in entries:
            member = tarfile.TarInfo(name)
            member.mode = mode
            if payload is None:
                member.type = tarfile.SYMTYPE
                member.linkname = "elsewhere"
            else:
                member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload) if payload is not None else None)
    return buf.getvalue()


def image(tmp_path, layout="classic", layers=None, compressed=False):
    layers = layers or [[("app/run", b"executable-one", 0o755),
                         ("app/plain", b"data", 0o644), ("usr/bin/ignored", b"outside", 0o755)]]
    unpacked = [tar_bytes(entries) for entries in layers]
    blobs = [gzip.compress(data, mtime=0) if compressed else data for data in unpacked]
    config = json.dumps({"os": "linux", "architecture": "amd64", "rootfs": {
        "type": "layers", "diff_ids": ["sha256:" + sha(data) for data in unpacked]}}).encode()
    entries = []
    if layout == "classic":
        config_name = sha(config) + ".json"
        names = [f"layer-{i}/layer.tar" for i in range(len(blobs))]
        manifest = [{"Config": config_name, "Layers": names, "RepoTags": ["test:only"]}]
        entries.append(("manifest.json", json.dumps(manifest).encode(), 0o644))
    else:
        config_name = "blobs/sha256/" + sha(config)
        names = ["blobs/sha256/" + sha(data) for data in blobs]
        manifest = json.dumps({"schemaVersion": 2,
            "config": {"digest": "sha256:" + sha(config), "size": len(config)},
            "layers": [{"digest": "sha256:" + sha(data), "size": len(data)} for data in blobs]}).encode()
        descriptor = {"digest": "sha256:" + sha(manifest), "size": len(manifest)}
        # Include an index indirection, as in the real Docker 29 save archive.
        index_blob = json.dumps({"manifests": [descriptor]}).encode()
        entries.extend([("blobs/sha256/" + sha(manifest), manifest, 0o644),
                        ("blobs/sha256/" + sha(index_blob), index_blob, 0o644),
                        ("index.json", json.dumps({"manifests": [{"digest": "sha256:" + sha(index_blob)}]}).encode(), 0o644)])
    entries.append((config_name, config, 0o644))
    entries.extend((name, data, 0o644) for name, data in zip(names, blobs))
    path = tmp_path / "image.tar"
    path.write_bytes(tar_bytes(entries))
    return path, "sha256:" + sha(config)


@pytest.mark.parametrize("layout,compressed", [("classic", False), ("classic", True), ("oci", False), ("oci", True)])
def test_layout_and_compression(tmp_path, layout, compressed):
    path, config = image(tmp_path, layout, compressed=compressed)
    assert eh.observe_archive(path) == {"image_config_id": config, "platform": "linux/amd64",
                                       "layer_count": 1, "executables": {"/app/run": sha(b"executable-one"), "/app/plain": sha(b"data")}}


@pytest.mark.parametrize("compressed", [False, True])
def test_whiteouts_and_replacements(tmp_path, compressed):
    path, _ = image(tmp_path, layers=[
        [("app/remove", b"remove", 0o755), ("app/dir/old", b"old", 0o755),
         ("app/run", b"old", 0o755), ("app/plain", b"old", 0o755), ("app/link", b"old", 0o755)],
        [("app/new", b"new", 0o755), ("app/.wh.remove", b"", 0o644),
         ("app/dir/fresh", b"fresh", 0o755), ("app/dir/.wh..wh..opq", b"", 0o644),
         ("app/run", b"new", 0o755), ("app/plain", b"plain", 0o644), ("app/link", None, 0o755)]],
         compressed=compressed)
    assert eh.observe_archive(path)["executables"] == {
        "/app/new": sha(b"new"), "/app/dir/fresh": sha(b"fresh"), "/app/run": sha(b"new"), "/app/plain": sha(b"plain")}   # every regular file under app/, executable or not


def complete(tmp_path, layout="oci"):
    archive, _ = image(tmp_path, layout)
    observation = eh.observe_archive(archive)
    record = {"schema_version": 1, "image": {"Id": observation["image_config_id"]},
              "archive": dict(zip(("sha256", "bytes"), eh.digest(archive))),
              "tools": observation["executables"], "platform": observation["platform"]}
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record))
    config_path = tmp_path / "config.json"
    config_path.write_text('{}')
    workload = tmp_path / "workload.json"
    workload.write_text('{}')
    facts = eh.collect_facts(record_path, archive, config_path)
    handoff = eh.generate(facts, workload, ["/app/run"])
    handoff["unresolved"] = []
    assert eh.validate(handoff, tmp_path)["state"] == "accepted"
    return facts, handoff, record_path


@pytest.mark.parametrize("layout", ["classic", "oci"])
def test_tampered_layer_precise_rejection(tmp_path, layout):
    _, handoff, _ = complete(tmp_path, layout)
    path = Path(handoff["release"]["transport"]["artifact"])
    data = path.read_bytes().replace(b"executable-one", b"executable-ony")
    path.write_bytes(data)
    verdict = eh.validate(handoff, tmp_path)
    assert verdict["state"] == "rejected"
    assert any("layer" in reason and "SHA-256 mismatch" in reason for reason in verdict["reasons"])


@pytest.mark.parametrize("key,value", [("expected_image_config_id", "sha256:" + "0" * 64),
                                       ("executable_sha256", {"/app/run": "0" * 64}),
                                       ("platform", "linux/arm64")])
def test_only_handoff_edited(tmp_path, key, value):
    _, handoff, _ = complete(tmp_path)
    transport = copy.deepcopy(handoff["release"]["transport"])
    handoff["release"][key] = value
    verdict = eh.validate(handoff, tmp_path)
    assert verdict["state"] == "rejected"
    assert any("re-observed archive" in reason for reason in verdict["reasons"])
    assert handoff["release"]["transport"] == transport


def test_record_archive_discrepancy(tmp_path):
    facts, _, record_path = complete(tmp_path)
    record = eh.read_json(record_path)
    record["tools"]["/app/run"] = "0" * 64
    record["image"]["Id"] = "sha256:" + "0" * 64
    record["platform"] = "linux/arm64"
    record_path.write_text(json.dumps(record))
    facts = eh.collect_facts(record_path, tmp_path / "image.tar", tmp_path / "config.json")
    for name in ("image_config_id", "platform", "executable_sha256"):
        assert f"build_record.{name}: differs from observed archive" in facts["unresolved"]
    draft = eh.generate(facts, tmp_path / "workload.json", ["/app/run"])
    assert draft["release"]["executable_sha256"] == {"/app/run": sha(b"executable-one"), "/app/plain": sha(b"data")}
    assert facts["archive"]["executables"]["tag"] == "observed"
    assert facts["build_record"]["claims"]["executable_sha256"]["tag"] == "declared"


def test_requirements_cli_and_provenance(tmp_path):
    facts, _, _ = complete(tmp_path)
    requirements = yaml.safe_load((eh.ROOT / "example/requirements.granite.yaml").read_text())
    requirements["inputs"]["workload"]["artifact"] = "workload.json"
    requirements["inputs"]["model"].update(manifest=str(tmp_path / "config.json"),
                                            manifest_sha256=eh.digest(tmp_path / "config.json")[0])
    path = tmp_path / "requirements.yaml"
    path.write_text(yaml.safe_dump(requirements))
    (tmp_path / "facts.json").write_text(json.dumps(facts))
    assert eh.main(["generate", "--facts", str(tmp_path / "facts.json"), "--requirements", str(path),
                    "--out", str(tmp_path / "handoff.yaml"), "--invocation-argv", "/app/run"]) == 0
    result = yaml.safe_load((tmp_path / "handoff.yaml").read_text())
    assert result["handoff_id"] == "granite-b0-smoke-1"
    assert result["inputs"]["workload"]["sha256"] == eh.digest(tmp_path / "workload.json")[0]
    assert result["provenance"]["sandbox_requirements.gpu.count"] == eh.fact(1, "declared", str(path))
    assert result["provenance"]["inputs.model.manifest_sha256"]["tag"] == "declared"
    assert result["provenance"]["release.expected_image_config_id"]["tag"] == "observed"
    assert not any("tool_and_version" in reason for reason in result["unresolved"])
    assert not any("gpu" in reason for reason in result["unresolved"])
    assert result["tests"]["profiling"]["tool_and_version"] is None


def test_config_digest_checked(tmp_path):
    path, _ = image(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"architecture": "amd64"', b'"architecture": "arm64"'))
    with pytest.raises(ValueError, match="config/metadata SHA-256 mismatch"):
        eh.observe_archive(path)


def test_invalid_archive_not_accepted(tmp_path):
    _, handoff, record_path = complete(tmp_path)
    path = Path(handoff["release"]["transport"]["artifact"])
    path.write_bytes(b"not a tar")
    handoff["release"]["transport"].update(zip(("sha256", "bytes"), eh.digest(path)))
    verdict = eh.validate(handoff, tmp_path)
    assert verdict["state"] == "rejected"
    assert any("archive observation failed" in reason for reason in verdict["reasons"])


def test_root_whiteout(tmp_path):
    path, _ = image(tmp_path, layers=[[("app/old", b"old", 0o755)],
                                     [("app/new", b"new", 0o755), (".wh..wh..opq", b"", 0o644)]])
    assert eh.observe_archive(path)["executables"] == {"/app/new": sha(b"new")}


def test_missing_requirements_remain_null(tmp_path):
    facts, _, _ = complete(tmp_path)
    draft = eh.generate(facts, None, ["/app/run"], {"handoff_id": "partial"}, "request.yaml")
    assert draft["inputs"]["workload"] == {"artifact": None, "sha256": None}
    assert "inputs.workload.artifact: missing" in draft["unresolved"]
    assert "sandbox_requirements.resource_limits.cpus: missing" in draft["unresolved"]
    assert draft["provenance"]["handoff_id"]["tag"] == "declared"


def test_directory_replacement_and_same_layer_override(tmp_path):
    path, _ = image(tmp_path, layers=[[("app/dir/old", b"old", 0o755)],
                                     [("app/dir/new", b"new", 0o755), ("app/dir", b"file", 0o755)]])
    assert eh.observe_archive(path)["executables"] == {"/app/dir": sha(b"file")}
