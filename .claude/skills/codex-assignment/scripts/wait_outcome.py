#!/usr/bin/env python3
"""Wait for / show the run_op outcome record of an agent (evidence/<agent>.jsonl). Run from the campaign dir.
Usage: wait_outcome.py codex-MI [--no-wait] [--poll 20] [--max 7200]
Exit 0 = outcome status ok; 1 = failed/timeout/signaled/spawn_error; 2 = still running (--no-wait) or gave up.
Reports the LAST intent's op, status, exit code, elapsed, survivors, and where stdout/stderr/last-message are.
A 'failed' outcome after ~2 minutes with no deliverables = suspect a provider refusal (see SKILL.md): read the
stdout tail and record it as a failed attempt in decisions/gaps.md; do not silently drop it."""
import argparse, json, os, sys, time
ROOT = os.environ.get("CAMPAIGN_ROOT", os.path.expanduser("~/projects/siminfeng/campaign"))
def last(agent):
    p = os.path.join(ROOT, "evidence", agent + ".jsonl")
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    ops = [r["op"] for r in rows if r["kind"] == "intent"]
    if not ops: return None, None
    op = ops[-1]; return op, [r for r in rows if r.get("op") == op]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("agent"); ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--poll", type=float, default=20); ap.add_argument("--max", type=float, default=7200)
    a = ap.parse_args(); t0 = time.time()
    while True:
        op, rows = last(a.agent)
        if op is None: sys.exit(f"no intent record for {a.agent}")
        out = next((r for r in rows if r["kind"] == "outcome"), None)
        if out or a.no_wait or time.time() - t0 > a.max: break
        time.sleep(a.poll)
    if not out:
        print(f"{a.agent} {op}: no outcome yet (running or wrapper killed; check `bin/lab`-style reconcile: ps for the pgid)"); sys.exit(2)
    print(f"{a.agent} {op}: status={out['status']} exit={out['exit_code']} signal={out['signal']} elapsed={out['elapsed_s']}s survivors={len(out['survivors'])}")
    print("stdout:", out["stdout"]["path"], out["stdout"]["bytes"], "B | stderr:", out["stderr"]["path"], out["stderr"]["bytes"], "B")
    task = out.get("task", "")[2:]
    lm = os.path.join(ROOT, "artifacts", "codex" + task, "last-message.txt")
    print("last-message:", lm, "(exists)" if os.path.exists(lm) else "(MISSING)")
    if out["survivors"]: print("WARNING: surviving processes:", out["survivors"])
    sys.exit(0 if out["status"] == "ok" else 1)
main()
