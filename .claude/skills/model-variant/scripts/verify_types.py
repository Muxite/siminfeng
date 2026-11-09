#!/usr/bin/env python3
"""Verify a quantised GGUF's per-tensor types against a recipe, using cc/model_inspect (header-only, never loads payloads).
Run from campaign/:
  verify_types.py MODEL.gguf --expect 'expert_ffn@0-11=Q3_K' --expect 'expert_ffn@12-23=Q4_K' --expect 'attn_qkv=Q8_0' --expect 'norm=F32'
  --expect ROLE[@LO-HI]=TYPE : every tensor of that role (optionally only layers LO..HI inclusive) must have exactly that ggml type.
Also prints bytes by type in MiB (2**20) AND MB (1e6) - D035 quotes file sizes in MB, PAP README in MiB; never mix them.
Exit 0 only if all expectations hold and unknown_tensors is empty."""
import argparse, json, os, re, subprocess, sys
ROOT = os.environ.get("CAMPAIGN_ROOT", os.path.expanduser("~/projects/siminfeng/campaign"))
sys.path.insert(0, os.path.join(ROOT, "cc", "model_inspect"))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("--expect", action="append", default=[]); a = ap.parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1"); sys.dont_write_bytecode = True
    import inspect_model
    d = inspect_model.inspect(a.gguf); bad = 0
    for e in a.expect:
        m = re.fullmatch(r"(\w+)(?:@(\d+)-(\d+))?=(\w+)", e)
        if not m: sys.exit(f"bad --expect {e}")
        role, lo, hi, ty = m.group(1), m.group(2), m.group(3), m.group(4)
        sel = [t for t in d["tensors"] if t["role"] == role and (lo is None or (t["layer"] is not None and int(lo) <= t["layer"] <= int(hi)))]
        wrong = [t for t in sel if t["ggml_type"].lower() != ty.lower()]
        ok = bool(sel) and not wrong; bad += not ok
        print(("OK  " if ok else "FAIL"), e, f"({len(sel)} tensors" + (f", wrong: {[(t['name'], t['ggml_type']) for t in wrong[:3]]}" if wrong else "") + ")")
    for ty, b in sorted(d["totals"]["by_type"].items()): print(f"{ty:6s} {b:>12,d} B  {b/2**20:9.2f} MiB  {b/1e6:9.2f} MB")
    print(f"weights {d['totals']['weight_bytes']/2**20:.2f} MiB; file {os.path.getsize(a.gguf)/2**20:.2f} MiB (file includes metadata); unknown tensors: {len(d['unknown_tensors'])}")
    sys.exit(1 if bad or d["unknown_tensors"] else 0)
main()
