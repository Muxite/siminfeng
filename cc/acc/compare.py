"""Compare two saved summaries using paired item correctness and exact McNemar."""
import argparse
import json
from pathlib import Path

try:
    from .common import indexed, read_jsonl, wilson
except ImportError:
    from common import indexed, read_jsonl, wilson


def mcnemar_exact(b, c):
    if not isinstance(b, int) or not isinstance(c, int) or min(b, c) < 0:
        raise ValueError("discordant counts must be nonnegative integers")
    n = b + c
    if not n:
        return 1.0
    coefficient = total = 1
    for k in range(1, min(b, c) + 1):
        coefficient = coefficient * (n - k + 1) // k
        total += coefficient
    return min(1., (2 * total) / (1 << n))


def compare(base, candidate):
    if base.keys() != candidate.keys() or not base:
        raise ValueError("paired comparison requires identical nonempty sets of unique item IDs")
    for key in base:
        if base[key].get("answer") != candidate[key].get("answer"):
            raise ValueError(f"answer differs for {key}")
    b = sum(bool(base[k]["correct"]) and not candidate[k]["correct"] for k in base)
    c = sum(not base[k]["correct"] and bool(candidate[k]["correct"]) for k in base)
    n = len(base)
    a_correct = sum(bool(v["correct"]) for v in base.values())
    b_correct = sum(bool(v["correct"]) for v in candidate.values())
    return dict(n=n, base_correct=a_correct, candidate_correct=b_correct,
                base_acc=a_correct / n, candidate_acc=b_correct / n,
                delta_acc=(b_correct - a_correct) / n, b=b, c=c,
                mcnemar_exact_p=mcnemar_exact(b, c),
                base_wilson95=wilson(a_correct, n), candidate_wilson95=wilson(b_correct, n))


def load(path):
    summary = json.loads(path.read_text())
    if summary.get("complete") is False:
        raise ValueError("incomplete run: finish or resume before comparing")
    rows = indexed(read_jsonl(path.parent / summary.get("items_file", "items.jsonl")))
    if summary["n"] != len(rows) or summary["correct"] != sum(bool(r["correct"]) for r in rows.values()):
        raise ValueError("summary does not match per-item records")
    return summary, rows


def markdown(result):
    lines = ["| Run | Correct / n | Accuracy | Wilson 95% CI |",
             "|---|---:|---:|---:|"]
    for key, label in (("base", "Reference"), ("candidate", "Candidate")):
        lo, hi = result[key + "_wilson95"]
        lines.append(f"| {label} | {result[key + '_correct']} / {result['n']} | "
                     f"{result[key + '_acc']:.2%} | {lo:.2%}–{hi:.2%} |")
    lines += ["", f"Candidate − reference: {result['delta_acc'] * 100:+.3f} percentage points. "
              f"Exact paired McNemar p = {result['mcnemar_exact_p']:.9g} "
              f"(b={result['b']}, c={result['c']})."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    base_summary, base = load(args.base)
    candidate_summary, candidate = load(args.candidate)
    for key in ("dataset_sha256", "mode", "protocol"):
        if base_summary.get(key) != candidate_summary.get(key):
            raise ValueError(f"incomparable run settings: {key}")
    result = compare(base, candidate)
    print(json.dumps(result, indent=2) if args.json else markdown(result))


if __name__ == "__main__":
    main()
