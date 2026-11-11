#!/usr/bin/env python3
"""CC kernel screen: interleaved test-backend-ops perf comparisons (stdlib only)."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess

OPS = ("MUL_MAT", "MUL_MAT_ID", "MUL_MAT_VEC_FUSION", "MUL_MAT_ID_FUSION", "SWIGLU")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
LINE = re.compile(r"^\s*(\w+(?:\([^)]*\))?)\((.*)\):\s*(.*)$")


def parse_output(text):
    """Console printer at 31288c9c6: FLOPS OR bandwidth, never both."""
    rows = []
    for line in ANSI.sub("", text).splitlines():
        match = LINE.match(line)
        if not match:
            continue
        op, params, tail = match.groups()
        # Split at the next key rather than commas inside array values.
        fields = dict(re.findall(r"(?:^|,)(\w+)=(.*?)(?=,\w+=|$)", params))
        base = {"op": op, "params": params, "fields": fields,
                "case": f"{op}({params})"}
        if tail.startswith("not supported"):
            rows.append(dict(base, supported=False, us_run=None, gflops=None, gb_s=None, runs=None))
            continue
        timing = re.search(rf"(\d+) runs\s*-\s*({NUMBER}) us/run", tail)
        if not timing:
            continue
        rate = re.search(rf"({NUMBER})\s*([TGMk])FLOPS", tail)
        bandwidth = re.search(rf"({NUMBER})\s*GB/s", tail)
        us = float(timing[2])
        if us <= 0:
            raise ValueError("Nonpositive rounded timing: " + base["case"])
        rows.append(dict(base, supported=True, runs=int(timing[1]), us_run=us,
                         gflops=float(rate[1]) * {"T": 1000, "G": 1, "M": .001, "k": .000001}[rate[2]] if rate else None,
                         gb_s=float(bandwidth[1]) if bandwidth else None))
    return rows


# ggml type names are case-sensitive in test-backend-ops vars() (q4_K, not q4_k); KS-20261110-1620 and
# KS-q4K-20261111 both failed on q4_K because names were lower()/upper()-cased. Canonicalize instead.
GGML_TYPES = ["f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", "q2_K", "q3_K", "q4_K",
              "q5_K", "q6_K", "q8_K", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s",
              "iq4_xs", "iq1_m", "tq1_0", "tq2_0", "mxfp4"]
_CANON = {t.lower(): t for t in GGML_TYPES}


def ggml_type(name):
    return _CANON.get(name.lower(), name)


def param_regex(shape, op, weight):
    """ECMAScript-compatible lookaheads against vars(), passed before allocation."""
    e, f, x, u, t = (shape[k] for k in ("n_embd", "n_ff", "n_expert", "n_used", "n_tokens"))
    down = shape.get("direction", "up") == "down"
    k, width = (f, e) if down else (e, f)
    if op == "MUL_MAT_VEC_FUSION":
        wanted = dict(type=ggml_type(weight), k=k, n=width, m=t, use_id=1,
                      n_mats=x, n_used=u, with_gate=1)
    elif op in ("MUL_MAT_ID", "MUL_MAT_ID_FUSION"):
        wanted = dict(type_a=ggml_type(weight), k=k, m=width, n=t, n_mats=x, n_used=u)
    elif op == "MUL_MAT":
        wanted = dict(type_a=ggml_type(weight), k=k, m=width, n=t)
    elif op in ("SWIGLU", "GEGLU", "REGLU", "GLU"):
        # Activation dtype, not quantized weight dtype; use F32/F16 in --types.
        return (f"^(?=.*(?:^|,)type={re.escape(ggml_type(weight))}(?:,|$))"
                f"(?=.*(?:^|,)ne_a=\\[{2 * width},{t},1,1\\](?:,|$)).*$")
    else:
        raise ValueError("Unsupported op: " + op)
    return "^" + "".join(f"(?=.*(?:^|,){key}={re.escape(str(value))}(?:,|$))" for key, value in wanted.items()) + ".*$"


def command(args, op, pattern, arm, managed):
    perf = ["perf", "-o", op, "-b", args.backend, "-p", pattern]
    env = os.environ.copy()
    for key in managed:
        env.pop(key, None)
    env.update({k: str(v) for k, v in arm.items() if v is not None})
    if args.binary:
        return [args.binary, *perf], env
    # env -u also removes values baked into the image's ENV, not just host values.
    inside = ["/usr/bin/env"]
    for key in sorted(managed):
        inside += ["-u", key]
    inside += [f"{k}={v}" for k, v in sorted(arm.items()) if v is not None]
    inside += ["/app/test-backend-ops", *perf]
    cmd = ["docker", "run", "--rm", "--runtime", "nvidia", "--network=none",
           "-e", "NVIDIA_VISIBLE_DEVICES=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
           "--entrypoint", "/usr/bin/env", args.image, *inside[1:]]
    prefix = shlex.split(os.environ.get("DOCKER_CMD_PREFIX", ""))
    return (prefix + [shlex.join(cmd)] if prefix else cmd), env


def summarize(runs, arm_names):
    cases = sorted({row["case"] for run in runs for row in run["rows"]})
    result = []
    for case in cases:
        arms = {}
        for name in arm_names:
            samples = [r["us_run"] for run in runs if run["arm"] == name
                       for r in run["rows"] if r["case"] == case and r["supported"]]
            arms[name] = {"samples_us": samples, "median_us": statistics.median(samples) if samples else None,
                          "spread_pct": (max(samples) / min(samples) - 1) * 100 if samples else None}
        baseline = arms[arm_names[0]]["median_us"]
        ratios = {name: baseline / arms[name]["median_us"] if baseline and arms[name]["median_us"] else None
                  for name in arm_names[1:]}
        result.append({"case": case, "arms": arms, "ratios_A_over_arm": ratios})
    return result


def markdown(report):
    names = list(report["arms"])
    lines = ["# CC kernel screen", "", "Ratios are A/arm time; >1 means that arm is faster. Spread = 100*(max/min - 1).",
             "A2 duplicates A for an interleaved A/A control when using default arms.", "",
             "| Case | " + " | ".join(f"{n} median us / spread %" for n in names) + " | " + " | ".join(f"A/{n}" for n in names[1:]) + " |",
             "|---|" + "---|" * (2 * len(names) - 1)]
    for row in report["summary"]:
        cells = [row["case"].replace("|", "\\|")]
        for name in names:
            a = row["arms"][name]
            cells.append(f"{a['median_us']:.2f} / {a['spread_pct']:.2f}" if a["median_us"] else "missing/unsupported")
        cells += [f"{v:.4f}" if v else "—" for v in row["ratios_A_over_arm"].values()]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Status: " + report["status"], "", *["- " + e for e in report["errors"]]]
    return "\n".join(lines) + "\n"


def execute(args, runner=subprocess.run):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("--out must be empty")
    shapes = json.loads(Path(args.preset).read_text())["shapes"] if args.preset else [dict(zip(
        ("n_embd", "n_ff", "n_expert", "n_used", "n_tokens"), values), direction=d)
        for values in itertools.product(args.n_embd, args.n_ff, args.n_expert, args.n_used, args.n_tokens)
        for d in args.direction]
    if not shapes:
        raise ValueError("No shapes requested")
    for shape in shapes:
        if any(not isinstance(shape.get(k), int) or shape[k] < 1
               for k in ("n_embd", "n_ff", "n_expert", "n_used", "n_tokens")):
            raise ValueError("Shapes require five positive integer dimensions")
        if shape["n_used"] > shape["n_expert"] or shape.get("direction", "up") not in ("up", "down"):
            raise ValueError("Invalid expert count or direction")
    if args.filter:
        re.compile(args.filter)
    arms = dict(args.arm) if args.arm else {"A": {}, "A2": {}, "B": {"GGML_CUDA_DISABLE_FUSION": "1"}}
    if len(arms) < 2:
        raise ValueError("At least two distinct arms required")
    managed = {"GGML_CUDA_J_UNFUSE_IDS_TYPES", "GGML_CUDA_DISABLE_FUSION"} | {k for a in arms.values() for k in a}
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "inspected_source_reference": "31288c9c6", "image": args.image, "binary": args.binary,
              "backend": args.backend, "shapes": shapes, "types": args.types, "ops": args.ops,
              "arms": arms, "managed_env": sorted(managed), "repeats": args.repeats,
              "runs": [], "errors": [], "summary": [], "status": "running"}

    def save():
        report["summary"] = summarize(report["runs"], list(arms))
        for row in report["summary"]:
            if any(len(a["samples_us"]) != args.repeats for a in row["arms"].values()):
                row["ratios_A_over_arm"] = dict.fromkeys(list(arms)[1:])
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "results.md").write_text(markdown(report))

    save()
    for shape, op, weight in itertools.product(shapes, args.ops, args.types):
        pattern = param_regex(shape, op, weight)
        if args.filter:
            pattern = pattern[:-3] + f"(?=.*(?:{args.filter})).*$"
        for repeat in range(args.repeats):
            names = list(arms) if repeat % 2 == 0 else list(reversed(arms))
            for name in names:
                cmd, env = command(args, op, pattern, arms[name], managed)
                index = len(report["runs"])
                run = {"arm": name, "repeat": repeat, "shape": shape, "op": op, "weight_type": weight,
                       "command": cmd, "params_filter": pattern, "rows": [], "log": f"{index:05d}.log"}
                try:
                    proc = runner(cmd, env=env, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
                    (out / run["log"]).write_text(proc.stdout)
                    run["returncode"] = proc.returncode
                    run["rows"] = parse_output(proc.stdout)
                    for row in run["rows"]:
                        row["case"] += " [" + ",".join(f"{k}={v}" for k, v in sorted(shape.items())) + "]"
                    if proc.returncode:
                        raise ValueError(f"exit {proc.returncode}")
                    if not any(r["supported"] for r in run["rows"]):
                        raise ValueError("no supported cases; requested shape/type may not exist in binary")
                    if any(not re.search(pattern, r["params"]) for r in run["rows"]):
                        raise ValueError("binary returned cases outside requested filter")
                except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    if isinstance(exc, subprocess.TimeoutExpired):
                        partial = exc.stdout or ""
                        if isinstance(partial, bytes):
                            partial = partial.decode("utf-8", errors="replace")
                        (out / run["log"]).write_text(partial)
                    run["error"] = str(exc)
                    report["errors"].append(f"run {index} ({name}, {op}, {weight}, {shape}): {exc}")
                report["runs"].append(run)
                save()
    # Every case must have one sample per repeat in every arm. Never compare partial coverage.
    for row in report["summary"]:
        if any(len(a["samples_us"]) != args.repeats for a in row["arms"].values()):
            report["errors"].append("Incomplete or duplicate case coverage: " + row["case"])
    report["status"] = "failed" if report["errors"] else "success"
    save()
    return 1 if report["errors"] else 0


def csv_int(value):
    values = [int(x) for x in value.split(",")]
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def env_arm(value):
    name, sep, raw = value.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError('expected NAME={"ENV":"value", "UNSET":null}')
    try:
        env = json.loads(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(env, dict) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or
                                        (v is not None and not isinstance(v, str)) for k, v in env.items()):
        raise argparse.ArgumentTypeError("arm must map env names to strings or null")
    return name, env


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--binary")
    p.add_argument("--backend", default="CUDA0")
    p.add_argument("--preset", help="preset JSON path; replaces CLI shapes")
    for name in ("n-embd", "n-ff", "n-expert", "n-used", "n-tokens"):
        p.add_argument("--" + name, type=csv_int, default=None)
    p.add_argument("--direction", type=lambda s: s.split(","), default=["up", "down"])
    p.add_argument("--types", type=lambda s: s.upper().split(","), default=["Q4_K", "Q5_K", "Q6_K"])
    p.add_argument("--ops", type=lambda s: s.split(","), default=list(OPS[:3]))
    p.add_argument("--filter", help="additional params regex, ANDed with shape/type filter")
    p.add_argument("--arm", action="append", type=env_arm)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--out", required=True)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.repeats < 2 or args.timeout <= 0:
        p.error("need --repeats >= 2 and --timeout > 0")
    if not args.preset and any(getattr(args, k) is None for k in ("n_embd", "n_ff", "n_expert", "n_used", "n_tokens")):
        p.error("provide --preset or all five --n-* shape options")
    if any(d not in ("up", "down") for d in args.direction):
        p.error("--direction must be up, down or up,down")
    if args.arm and len(dict(args.arm)) != len(args.arm):
        p.error("duplicate arm names")
    try:
        return execute(args)
    except (OSError, ValueError) as exc:
        p.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
