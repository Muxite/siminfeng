#!/usr/bin/env python3
"""Alternating-order generator for lab test lists (comma-separated, as `bin/lab submit --tests` wants).

Semantics (reproduces artifacts/x2/{pap1,c1,match1,kv1,hsplit1,plan1}-order.txt):
  per group: `rounds` rounds, round i runs the group forward if i is even, reversed if odd
  (2 rounds on a pair = ABBA); the concatenation of all groups is repeated `--repeat` times;
  `--once` tests (deterministic quality tests: ppl/kld) are appended a single time at the end.
Usage:
  make_order.py --group a,b,c --rounds 3                      # one group, F R F
  make_order.py --group a,b --group c,d --rounds 2 --repeat 2 # c1 style
  make_order.py --group a,b --rounds 3 --once kl_a,kl_b --out order.txt
Prints the list to stdout (and to --out if given); prints a count summary to stderr.
"""
import argparse, collections, sys

def make(groups, rounds=3, repeat=1, once=()):
    seq = []
    for _ in range(repeat):
        for g in groups:
            for i in range(rounds):
                seq += g if i % 2 == 0 else g[::-1]
    return seq + list(once)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="comma-separated test names; repeatable")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--once", default="", help="comma-separated tests appended once (quality tests)")
    ap.add_argument("--out")
    a = ap.parse_args()
    groups = [[t for t in g.split(",") if t] for g in a.group]
    seq = make(groups, a.rounds, a.repeat, [t for t in a.once.split(",") if t])
    s = ",".join(seq)
    if a.out: open(a.out, "w").write(s)
    print(s)
    c = collections.Counter(seq)
    print(f"# {len(seq)} runs, {len(c)} distinct tests, per-test count min/max {min(c.values())}/{max(c.values())}", file=sys.stderr)

if __name__ == "__main__":
    main()
