#!/usr/bin/env python3
"""Greedy-text fingerprint (D024): sha256 of each llama-completion stdout in a lab job, grouped by test name.
Run from the campaign dir:  text_fp.py JOB [--filter gen]   (LAB env overrides lab dir)
Same fingerprint across variants = identical greedy text (required for computation-preserving candidates).
Known references (Granite, chat.txt, -n 160, temp 0, seed 1): Q8_0 all-GPU 65894776; mixE4K all-GPU 950b46b1;
mixE4K experts on CPU (--n-cpu-moe 24) 76dc9422; static-hot 'never engaged' also shows 76dc9422 (D024)."""
import collections, glob, hashlib, os, re, sys
a = [x for x in sys.argv[1:] if not x.startswith("--")]; flt = sys.argv[sys.argv.index("--filter") + 1] if "--filter" in sys.argv else ""
if not a: sys.exit(__doc__)
a = [x for x in a if x != flt]
lab = os.environ.get("LAB", "lab"); g = collections.defaultdict(list)
for f in sorted(glob.glob(f"{lab}/jobs/{a[0]}/att-*/*.stdout")):
    t = os.path.basename(f)[:-7].split("#")[0]
    if flt in t: g[t].append(hashlib.sha256(open(f, "rb").read()).hexdigest()[:8])
for t, h in g.items(): print(f"{t:30s}", "IDENTICAL" if len(set(h)) == 1 else "DIFFER   ", " ".join(h))
