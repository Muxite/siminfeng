#!/usr/bin/env python3
"""Offline shared envelopes over Build CC and bin/lab records; never executes jobs.

Values copied from records are declarations, not archive observations. Admission
is a compatibility/identity receipt, not execution or proof of isolation. See
ENVELOPES.md for the small adapter input contract and evidence limitations.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

SHA = re.compile(r"[0-9a-f]{64}\Z")
GUARANTEES = {"none", "cooperative-intercept", "cgroup-host", "hard"}


def read_json(path):
    return json.loads(Path(path).read_text())


def at(node, path):
    for key in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def positive(value):
    return type(value) in (int, float) and value > 0


def canonical_sha(value):
    return sha_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def readonly_binds(args):
    """Translate only bin/lab's read-only -v forms; other Docker flags stay unresolved."""
    args = iter(args)
    binds = []
    for arg in args:
        if arg in ("-v", "--volume"):
            value = next(args, "")
        elif arg.startswith("--volume="):
            value = arg.split("=", 1)[1]
        elif arg.startswith("-v") and len(arg) > 2:
            value = arg[2:]
        else:
            raise ValueError("unsupported Docker argument")
        parts = value.split(":")
        if len(parts) != 3 or parts[2] != "ro" or not all(p.startswith("/") for p in parts[:2]):
            raise ValueError("only absolute read-only binds can map to roles")
        binds.append(parts[:2])
    return binds


def _issues(p):
    """Recompute missing identity/policy information; never trust dispatchable."""
    missing = []
    for key in ("campaign_id", "request_id", "experiment_id", "attempt_id",
                "target.host_id", "target.platform", "release.role",
                "release.source.commit", "release.source.tree", "release.build_record_ref",
                "policy.memory.domain", "policy.memory.accounting_method",
                "evidence.durable_destination_ref"):
        if not at(p, key):
            missing.append(key + ": missing")
    for key in ("release.transport.sha256",):
        if not valid_sha(at(p, key)):
            missing.append(key + ": missing or invalid SHA-256")
    config = at(p, "release.image_config_id")
    if not isinstance(config, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", config):
        missing.append("release.image_config_id: missing or invalid config identity")
    exes = at(p, "release.executables_sha256")
    if not isinstance(exes, dict) or not exes or not all(valid_sha(v) for v in exes.values()):
        missing.append("release.executables_sha256: missing or invalid")
    for key in ("release.transport.bytes", "policy.wall_time_s", "policy.capture_bytes",
                "policy.memory.budget_bytes"):
        if not positive(at(p, key)):
            missing.append(key + ": missing or invalid positive budget/size")
    if at(p, "policy.memory.guarantee") is None:
        missing.append("policy.memory.guarantee: missing")
    if at(p, "evidence.controller_outside_test_sandbox") is None:
        missing.append("evidence.controller_outside_test_sandbox: missing")
    if not at(p, "evidence.outputs"):
        missing.append("evidence.outputs: missing")
    inputs = p.get("inputs") or {}
    if not isinstance(inputs, dict):
        inputs = {}
    for section in ("model_files", "tests"):
        values = inputs.get(section)
        if not isinstance(values, dict) or not values or not all(valid_sha(v) for v in values.values()):
            missing.append(f"inputs.{section}: missing or invalid hashes")
    for section in ("data_files", "mounts"):
        if section not in inputs:
            missing.append(f"inputs.{section}: missing (explicit empty mapping allowed)")
    for section in ("data_files", "mounts"):
        values = inputs.get(section, {})
        if not isinstance(values, dict):
            missing.append(f"inputs.{section}: invalid mapping")
            continue
        hashes = values.values() if section == "data_files" else [
            v for manifest in values.values() if isinstance(manifest, dict) for v in manifest.values()]
        if not all(valid_sha(v) for v in hashes) or (section == "mounts" and
                any(not isinstance(v, dict) for v in values.values())):
            missing.append(f"inputs.{section}: invalid hashes")
    tests = at(p, "invocation.tests")
    if not isinstance(tests, list) or not tests:
        missing.append("invocation.tests: missing")
    else:
        for i, test in enumerate(tests):
            argv = test.get("argv") if isinstance(test, dict) else None
            if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and '\0' not in a for a in argv):
                missing.append(f"invocation.tests.{i}.argv: missing or invalid")
            test_hashes = inputs.get("tests") or {}
            if not isinstance(test_hashes, dict):
                test_hashes = {}
            if not isinstance(test, dict) or not valid_sha(test_hashes.get(test.get("test"))):
                missing.append(f"invocation.tests.{i}: missing test-definition hash")
    return missing


def make_proposal(build_record, job_spec, host_profile, policy):
    """Project nested schema 1/2 records; supplement legacy gaps explicitly.

    job_spec may be job.json or its spec. policy supplies campaign/experiment/
    attempt IDs, build_record_ref, release_role, memory/evidence policy. To bind
    argv, supply spec.test_definitions (the bin/lab definitions) or explicit
    spec.invocation. No live lab definitions, host paths, or archive are read.
    """
    b, wrapper, policy = deepcopy(build_record), deepcopy(job_spec), deepcopy(policy)
    s = wrapper.get("spec", wrapper)
    image, source = b.get("image") or {}, b.get("source") or {}
    defines = {}
    for flag in b.get("cmake_flags", []):
        if flag.startswith("-D") and "=" in flag:
            key, value = flag[2:].split("=", 1)
            defines[key] = value
    defines.update(b.get("cmake_defines") or {})
    arch = defines.get("CMAKE_CUDA_ARCHITECTURES")
    compat = {"jetpack_l4t": None, "cuda_arch": ["sm_" + a for a in str(arch).split(";")] if arch else None,
              "driver_min": None}
    compat.update(s.get("platform_compat") or {})
    inputs = deepcopy(s.get("inputs") or {})
    # Replace host locators with stable roles; hashes and basenames remain intact.
    inputs["mounts"] = {f"prompt-{i}": manifest for i, (_, manifest) in enumerate(
        sorted(inputs.get("mounts", {}).items()))} if "mounts" in inputs else None
    if inputs["mounts"] is None:
        del inputs["mounts"]
    if "data_files" in inputs:
        inputs["data_files"] = {Path(k).name: v for k, v in inputs["data_files"].items()}
    inv = deepcopy(s.get("invocation") or {"tests": [], "mounts": [
        {"role": "model", "access": "ro"}, {"role": "results", "access": "rw"}]})
    unresolved = []
    known_image_ids = {image.get("config_digest"), image.get("Id"), image.get("docker_reported_id"), image.get("index_digest")}
    known_image_ids.update(entry.get("digest") for entry in image.get("index_manifests", []))
    if s.get("image_id") and s["image_id"] not in known_image_ids:
        unresolved.append("job_spec.image_id: not bound to this build record's config or index identities")
    if not s.get("invocation"):
        if s.get("data"):
            inv["mounts"].append({"role": "data", "access": "ro"})
        for role in inputs.get("mounts", {}):
            inv["mounts"].append({"role": role, "access": "ro"})
        mount_roles = {host: f"prompt-{i}" for i, host in enumerate(sorted((s.get("inputs") or {}).get("mounts", {})))}
        for i, name in enumerate(s.get("tests", [])):
            td = (s.get("test_definitions") or {}).get(name)
            argv = None
            if td:
                if canonical_sha(td) != inputs.get("tests", {}).get(name):
                    unresolved.append(f"invocation.tests.{i}: definition differs from spec.inputs hash")
                try:
                    for host, target in readonly_binds(td.get("docker_args", [])):
                        if host not in mount_roles:
                            unresolved.append(f"invocation.tests.{i}: mounted input absent from spec.inputs")
                            continue
                        role = mount_roles[host]
                        mount = next(m for m in inv["mounts"] if m["role"] == role)
                        if mount.get("target", target) != target:
                            raise ValueError("one role has conflicting container targets")
                        mount["target"] = target
                    argv = ["/app/" + td["tool"], "-m", "/m/" + td.get("model_file", Path(s.get("model", "")).name)]
                    if td.get("needs_data"):
                        argv += ["-f", "/d/" + Path(s.get("data") or "").name]
                    argv += td.get("args", [])
                except ValueError as exc:
                    unresolved.append(f"invocation.tests.{i}: {exc}")
            inv["tests"].append({"test": name, "occurrence": i, "argv": argv,
                                 "environment": (td or {}).get("env", {}),
                                 "cold_cache": (td or {}).get("cold_cache")})
    p = {"format": "campaign-job-proposal/0",
         **{key: s.get(key, wrapper.get(key, policy.get(key))) for key in
            ("campaign_id", "request_id", "experiment_id", "attempt_id")},
         "target": {"host_id": host_profile.get("host_id"), "platform": b.get("platform"),
                    "required_capabilities": s.get("required_capabilities", ["docker", "nvidia-runtime"]),
                    "platform_compat": compat},
         "release": {"role": policy.get("release_role"), "source": {"commit": source.get("commit"), "tree": source.get("tree")},
                     "patch_sha256": b.get("patch_sha256"), "cmake_defines": defines,
                     "transport": {"kind": "docker-archive", "sha256": at(b, "archive.sha256"), "bytes": at(b, "archive.bytes")},
                     "image_config_id": image.get("config_digest"), "index_digest": image.get("index_digest"),
                     "executables_sha256": b.get("app_files") or b.get("tools"),
                     "build_record_ref": policy.get("build_record_ref")},
         "inputs": inputs, "invocation": inv,
         "policy": {"wall_time_s": policy.get("wall_time_s"), "capture_bytes": policy.get("capture_bytes"),
                    "memory": policy.get("memory", {}), "gpu_jobs_on_host": 1},
         "evidence": policy.get("evidence", {})}
    p["request_id"] = p["request_id"] or wrapper.get("job") or s.get("key")
    p["unresolved"] = list(dict.fromkeys(unresolved + _issues(p)))
    # Include structural input gaps (e.g. a referenced prompt with no hash) and
    # declared receiver incompatibilities in dispatchability, too.
    p["unresolved"] = check_admission(p, host_profile)["reasons"]
    p["dispatchable"] = not p["unresolved"]
    return p


def check_admission(proposal, host_profile):
    """Compatibility receipt only. Does not reserve capacity or inspect bytes."""
    p, h = proposal, host_profile
    if not isinstance(p, dict) or not isinstance(h, dict):
        return {"decision": "rejected", "reasons": ["proposal and host_profile must be objects"]}
    errors, unknown = [], _issues(p) + list(p.get("unresolved") or [])
    # Missing input identities cannot describe an admissible workload. Missing
    # transport identity instead asks the sender to complete delivery metadata.
    errors.extend(reason for reason in unknown if reason.startswith("inputs."))
    if p.get("format") != "campaign-job-proposal/0":
        errors.append("unsupported proposal format")
    for label, requested, offered in (
        ("platform", at(p, "target.platform"), h.get("platform")),
        ("host_id", at(p, "target.host_id"), h.get("host_id"))):
        if requested is not None and offered is not None and requested != offered:
            errors.append(f"{label} mismatch: requested {requested}, host {offered}")
        elif offered is None:
            unknown.append(f"host.{label}: missing")
    compat = at(p, "target.platform_compat") or {}
    for key in ("cuda_arch", "jetpack_l4t"):
        req, available = compat.get(key), at(h, "platform_compat." + key)
        if req is None:
            if key == "cuda_arch" or (key == "jetpack_l4t" and h.get("memory_model") == "shared-dram"):
                unknown.append(f"target.platform_compat.{key}: missing")
        elif available is None:
            unknown.append(f"host.platform_compat.{key}: missing")
        elif key == "cuda_arch":
            reqs = req if isinstance(req, list) else [req]
            avail = available if isinstance(available, list) else [available]
            if not set(reqs).intersection(avail):
                errors.append("cuda_arch mismatch")
        elif req != available:
            errors.append(f"{key} mismatch")
    driver = compat.get("driver_min")
    if driver:
        version = h.get("driver_version")
        if not version:
            unknown.append("host.driver_version: missing")
        else:
            def numeric(v):
                return tuple(int(x) for x in v.split("."))
            try:
                if numeric(version) < numeric(driver):
                    errors.append("driver below driver_min")
            except (ValueError, AttributeError):
                errors.append("invalid driver version")
    caps = h.get("capabilities")
    if caps is None:
        unknown.append("host.capabilities: missing")
    elif any(c not in caps for c in at(p, "target.required_capabilities") or []):
        errors.append("required capabilities unavailable")
    guarantee = at(p, "policy.memory.guarantee")
    if guarantee is not None:
        if guarantee not in GUARANTEES:
            errors.append("unknown memory guarantee")
        elif guarantee != "none":
            offers = h.get("memory_guarantees")
            if offers is None:
                unknown.append("host.memory_guarantees: missing")
            elif not any(o.get("guarantee") == guarantee and o.get("domain") == at(p, "policy.memory.domain")
                         and o.get("accounting_method") == at(p, "policy.memory.accounting_method") for o in offers):
                errors.append("unsupported memory guarantee/domain/accounting_method")
    if at(p, "policy.gpu_jobs_on_host") != 1:
        errors.append("gpu_jobs_on_host must be 1")
    if at(p, "release.transport.kind") != "docker-archive":
        errors.append("unsupported transport kind")
    if at(p, "evidence.controller_outside_test_sandbox") is False:
        errors.append("controller must be outside test sandbox")
    mounts = at(p, "invocation.mounts")
    if not isinstance(mounts, list) or not mounts:
        unknown.append("invocation.mounts: missing")
    else:
        for m in mounts:
            if (not isinstance(m, dict) or not {"role", "access"} <= set(m) or set(m) - {"role", "access", "target"}
                    or not isinstance(m["role"], str) or not re.fullmatch(r"[a-z][a-z0-9-]*", m["role"])
                    or m["access"] != ("rw" if m["role"] == "results" else "ro")
                    or ("target" in m and (not isinstance(m["target"], str) or not m["target"].startswith("/") or ".." in Path(m["target"]).parts))):
                errors.append("mounts must use roles with ro inputs / rw results, without host paths")
    for test in at(p, "invocation.tests") or []:
        if not isinstance(test, dict):
            continue
        argv = test.get("argv") or []
        if argv and argv[0] not in (at(p, "release.executables_sha256") or {}):
            errors.append("invocation executable has no release identity")
        if not isinstance(argv, list):
            continue
        for arg in argv[1:]:
            if not isinstance(arg, str):
                continue
            value = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
            if not value.startswith("/"):
                continue
            matched = False
            for m in mounts if isinstance(mounts, list) else []:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                target = m.get("target", {"model": "/m", "data": "/d", "results": "/results"}.get(role))
                if not isinstance(target, str) or not value.startswith(target.rstrip("/") + "/"):
                    continue
                rel = value[len(target.rstrip("/")) + 1:]
                if ".." in Path(rel).parts:
                    errors.append("input path escapes its mount role")
                    matched = True
                    break
                if role == "results":
                    matched = True
                    break
                section = {"model": "model_files", "data": "data_files"}.get(role)
                manifest = at(p, "inputs." + section) if section else (at(p, "inputs.mounts") or {}).get(role)
                if not isinstance(manifest, dict) or not valid_sha(manifest.get(rel)):
                    errors.append(f"missing input identity for {role}:{rel}")
                matched = True
                break
            if not matched:
                errors.append("invocation contains an absolute path without an input/output role")
    return {"decision": "rejected" if errors else "needs-information" if unknown else "accepted",
            "reasons": list(dict.fromkeys(errors + unknown))}


def make_result(job_dir, attempt, claim_category="unclassified", *, manifest_path=None):
    """Read one attempt only. No scoring, process probing, or token disclosure.

    Default MANIFEST lookup follows bin/lab's jobs/../export layout. A manifest
    contains declared artifact identities; no claim of current byte verification.
    """
    root = Path(job_dir)
    if not re.fullmatch(r"att-[1-9][0-9]*", str(attempt)):
        raise ValueError("attempt must be att-N")
    adir = root / attempt
    job = read_json(root / "job.json")
    result = read_json(adir / "result.json") if (adir / "result.json").exists() else {}
    owner = read_json(adir / "owner.json") if (adir / "owner.json").exists() else {}
    if result.get("attempt", attempt) != attempt or owner.get("attempt", attempt) != attempt or result.get("job", job["job"]) != job["job"]:
        raise ValueError("result/owner job or attempt mismatch")
    rows, line_count, journal_errors = [], 0, []
    if (adir / "state.jsonl").exists():
        with (adir / "state.jsonl").open() as stream:
            for line in stream:
                line_count += 1
                try:
                    row = json.loads(line)
                    if row.get("attempt") != attempt or row.get("job") != job["job"]:
                        raise ValueError("journal job or attempt mismatch")
                    rows.append(row)
                except (ValueError, AttributeError):
                    journal_errors.append(f"state.jsonl:{line_count}")
    token_sha = sha_text(owner["token"]) if owner.get("token") else None
    if result.get("token_sha") and token_sha and token_sha != result["token_sha"]:
        raise ValueError("result owner token mismatch")
    spec = job["spec"]
    expected = [f"{name}#{i}" for i, name in enumerate(spec.get("tests", []))]
    tests = at(result, "quality.tests") or {}
    started = [f"{r['test']}#{i}" for i, r in enumerate(r for r in rows if r.get("test") and r.get("state") == "running")]
    started = list(dict.fromkeys(started + list(tests)))
    clean = [key for key, value in tests.items() if value.get("exit_code") == 0 and value.get("partial") is False]
    missing = [key for key in expected if key not in clean]
    manifest = Path(manifest_path) if manifest_path else root.parent.parent / "export" / root.name / attempt / "MANIFEST.json"
    artifacts = []
    if manifest.exists():
        for path, sha in sorted(read_json(manifest).items()):
            if Path(path).is_absolute() or ".." in Path(path).parts or not valid_sha(sha):
                raise ValueError("invalid MANIFEST artifact identity")
            artifacts.append({"path": str(manifest.parent / path), "sha256": sha})
    assessment = deepcopy(result.get("quality") or {"assessed": False})
    # Legacy interrupted results have no per-test clean evidence. Never score gaps.
    if missing or journal_errors:
        assessment = {"assessed": False, "reason": "incomplete durable test evidence", "recorded_quality": assessment}
    last = rows[-1] if rows else {}
    return {"format": "campaign-job-result/0", "request_id": job["job"], "attempt_id": attempt,
            "campaign_id": spec.get("campaign_id"), "experiment_id": spec.get("experiment_id"),
            "execution_outcome": result.get("execution") or {"status": last.get("execution", "unknown")},
            "test_assessment": assessment, "claim_category": claim_category,
            "evidence_completeness": {"expected": len(expected), "started": len(started), "clean": len(clean), "missing": missing,
                                      "manifest_present": manifest.exists(), "journal_errors": journal_errors},
            "last_durable_sequence": {"line_count": line_count, "last_ts": last.get("ts")},
            "resource_generation": {"attempt": attempt, "owner_token_sha256": token_sha or result.get("token_sha")},
            "stop_confirmed": at(result, "cleanup.confirmed") is True and at(result, "cleanup.owned_containers_left") == [],
            "artifact_refs": artifacts, "manifest_ref": str(manifest) if manifest.exists() else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proposal = commands.add_parser("proposal")
    proposal.add_argument("--build-record", required=True)
    proposal.add_argument("--job-spec", required=True)
    proposal.add_argument("--host-profile", required=True)
    proposal.add_argument("--policy", required=True)
    admit = commands.add_parser("admit")
    admit.add_argument("--proposal", required=True)
    admit.add_argument("--host-profile", required=True)
    result = commands.add_parser("result")
    result.add_argument("--job-dir", required=True)
    result.add_argument("--attempt", required=True)
    result.add_argument("--claim-category", default="unclassified")
    result.add_argument("--manifest")
    for cmd in (proposal, admit, result):
        cmd.add_argument("--out", help="JSON destination; defaults to stdout")
    args = parser.parse_args(argv)
    try:
        if args.command == "proposal":
            value = make_proposal(read_json(args.build_record), read_json(args.job_spec), read_json(args.host_profile), read_json(args.policy))
        elif args.command == "admit":
            value = check_admission(read_json(args.proposal), read_json(args.host_profile))
        else:
            value = make_result(args.job_dir, args.attempt, args.claim_category, manifest_path=args.manifest)
        output = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(output)
        else:
            print(output, end="")
        return 2 if args.command == "admit" and value["decision"] != "accepted" else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
