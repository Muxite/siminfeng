#!/usr/bin/env python3
"""Create the next decisions/D0xx-<slug>.md skeleton (never overwrites).
Usage (from campaign/): new_note.py 'short-slug' 'Headline with verdict' [--category CAT] [--dir decisions] [--protocol]
Prints the path. --protocol uses the protocol-frozen template (written BEFORE results)."""
import argparse, glob, os, re, sys
CATS = ["full-engine code gain", "runtime-placement trade-off", "representation trade-off", "memory-fit",
        "layer-component result", "infrastructure validation", "failed-inconclusive"]
RESULT = """# D{n:03d} — {title}
Claim category: {cat}   (one of: {cats})
Evidence: <job-ids, build-record paths, evidence/*.jsonl op ids / tasks, artifacts/... paths>
Conditions: <image run_ref + config digest, llama.cpp commit, patch sha256, model file (+sha), flags (-c, -ngl, --n-cpu-moe), prompt/seed, host-load monitor summary (cc=0? dl=0?), rounds, clean N/N, GPU RTX 5070 Ti>
Protocol: <D0xx frozen rule, quoted threshold; amendments>
Numbers: <median [range] per arm, n; GPU MiB; PPL/KLD/top-1; text sha; deltas in %>
Verdict: <ACCEPTED / REJECTED / INCONCLUSIVE / PROVISIONAL (why)> against the frozen rule: <criterion met/unmet, by how much>
Interpretation: <mechanism, what the numbers do and do not show; competing explanations>
Limits: <single model/workload, one host, spread vs effect, untimed/provisional flags, not demonstrated items>
Next: <discriminating experiment / follow-up task ids; factory rule if any>
"""
PROTO = """# D{n:03d} — {title} — protocol frozen BEFORE results
Candidate: <patch/path + sha256 | model recipe | runtime flags>; hypothesis: <mechanism and predicted size>.
Builds/models: <image names, commit, bases, extra targets; model files>.
Primary comparison: <same binary OFF vs ON | same-K controls>; tests: lab/tests/<x>.json; order: artifacts/x2/<x>-order.txt (<rounds>, alternating); host-load monitored, no concurrent compiles/downloads.
Acceptance (frozen): <numeric thresholds beyond observed spread> AND <identity / quality gates> AND <correctness tests>. Otherwise rejected / inconclusive, reported as such.
Classification if accepted: {cat}; <hardware, model, workload scope>.
"""
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("slug"); ap.add_argument("title"); ap.add_argument("--category", default="<category>")
    ap.add_argument("--dir", default="decisions"); ap.add_argument("--protocol", action="store_true"); a = ap.parse_args()
    if a.category != "<category>" and a.category not in CATS: sys.exit(f"category must be one of {CATS}")
    nums = [int(m.group(1)) for f in glob.glob(os.path.join(a.dir, "D[0-9][0-9][0-9]-*.md")) if (m := re.match(r"D(\d+)-", os.path.basename(f)))]
    n = max(nums, default=0) + 1; p = os.path.join(a.dir, f"D{n:03d}-{re.sub(r'[^a-z0-9-]+', '-', a.slug.lower())}.md")
    if os.path.exists(p): sys.exit("exists: " + p)
    open(p, "w").write((PROTO if a.protocol else RESULT).format(n=n, title=a.title, cat=a.category, cats=" | ".join(CATS))); print(p)
main()
