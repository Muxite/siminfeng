#!/usr/bin/env python3
"""Demo-only record display, assertions, hashing, and journal polling.

Execution, admission, result projection, and fault injection remain existing CLIs.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def load(path):
    return json.loads(Path(path).read_text())


def dump(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(live, exported):
    return {label: {str(p.relative_to(root)): sha(p)
                    for p in sorted(root.rglob("*")) if p.is_file()}
            for label, root in (("live", Path(live)), ("export", Path(exported)))}


def main():
    action, *args = sys.argv[1:]
    if action == "build":
        b = load(args[0])
        dump({"status": b["status"], "config_digest": b["image"]["config_digest"],
              "run_ref": b["image"]["run_ref"], "archive": b["archive"],
              "linkage": {k: b["linkage"][k] for k in
                          ("status", "libcudart_dynamic", "undefined_imports", "dt_needed")}})
        assert b["status"] == "success" and b["linkage"]["status"] == "passed"
    elif action == "setup":
        out, source = map(Path, args)
        out.mkdir(parents=True, exist_ok=False)
        (out / "lab" / "images").mkdir(parents=True)
        shutil.copy2(source / "images/llamacpp-novmm4.json", out / "lab/images")
        shutil.copytree(source / "tests", out / "lab/tests")
        print("Existing image registration and test definitions copied to", out / "lab")
    elif action == "policy":
        path, destination, build = args
        save(path, {"campaign_id": "c1", "experiment_id": "DEMO", "attempt_id": "att-1",
                    "release_role": "engine", "build_record_ref": build,
                    "wall_time_s": 900, "capture_bytes": 8388608,
                    "memory": {"domain": "host-ram", "budget_bytes": 17179869184,
                               "guarantee": "cgroup-host", "accounting_method": "cgroup-v2-memory"},
                    "evidence": {"controller_outside_test_sandbox": True,
                                 "durable_destination_ref": destination,
                                 "outputs": ["stdout", "stderr", "state.jsonl", "result.json"]}})
        dump(load(path))
    elif action == "proposal":
        p = load(args[0])
        dump(p)
        assert p["dispatchable"] is True and p["unresolved"] == []
    elif action == "admission":
        receipt = load(args[0])
        dump(receipt)
        assert receipt["decision"] == args[1]
        if args[1] == "rejected":
            assert receipt["reasons"]
    elif action == "wait-running":
        path = Path(args[0])
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A concurrent append may not yet be complete.
                    if row.get("state") == "running" and row.get("container"):
                        print(row["container"])
                        return
                    if row.get("state") == "terminal":
                        raise SystemExit("Attempt terminated before a test container was recorded")
            time.sleep(0.05)
        raise SystemExit("Timed out waiting for the test-container journal event")
    elif action == "verify":
        root = Path(args[0])
        manifest = load(root / "MANIFEST.json")
        assert manifest, "Empty export manifest"
        assert {"owner.json", "result.json", "state.jsonl"} <= manifest.keys()
        assert any(name.endswith(".stdout") for name in manifest)
        assert any(name.endswith(".stderr") for name in manifest)
        for name, expected in sorted(manifest.items()):
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            assert (root / name).is_file(), f"Missing exported file: {name}"
            assert sha(root / name) == expected, f"SHA256 FAILED: {name}"
            print(f"{name}: OK (sha256 {expected})")
    elif action == "result":
        r = load(args[0])
        dump(r)
        assert r["execution_outcome"]["status"] == args[1]
        c = r["evidence_completeness"]
        assert c["manifest_present"] and not c["journal_errors"]
        assert c["expected"] == 2 and r["stop_confirmed"] is True
        assert r["last_durable_sequence"]["line_count"] > 0
        if args[1] == "completed":
            assert c["clean"] == 2 and c["missing"] == []
        else:
            assert c["clean"] < 2 and c["missing"]
            assert r["test_assessment"]["assessed"] is False
    elif action in ("snapshot", "compare"):
        live, exported, path = args
        observed = snapshot(live, exported)
        assert observed["live"] and observed["export"]
        if action == "snapshot":
            save(path, observed)
            print("Saved att-1 live/export SHA256 snapshot:", path)
        else:
            assert observed == load(path), "att-1 evidence changed during retry"
            print("att-1 live and exported file sets + every SHA256 unchanged")
    elif action == "summary":
        job, out = args
        # Assertions in preceding steps establish these statements before display.
        print("1. novmm-4: successful build; config, run_ref, and archive identities are distinct fields.")
        print(f"2. {job}: new immutable submission with two pinned test definitions and inputs.")
        print("3. Compatibility receipt: x86 accepted; Jetson template rejected with reasons.")
        print("4. att-1: sandbox interrupted, partial evidence exported and hashes verified; stop confirmed.")
        print(f"5. att-2: 2/2 clean completions; att-1 unchanged; records: {out}")
    else:
        raise SystemExit(f"Unknown display action: {action}")


if __name__ == "__main__":
    main()
