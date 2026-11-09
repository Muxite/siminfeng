#!/usr/bin/env python3
"""Aggregate lab job results into per-test medians (generalised from artifacts/x2/agg.py).

Run from the campaign dir (or set LAB=/path/to/lab):
  agg.py job-ad883810b3 [job-...]            # table per job
  agg.py JOB --compare base=treat ...        # % delta of decode/prefill medians between two test names
  agg.py JOB --strip _r0,_r1                 # merge variants by stripping suffixes from test names
  agg.py JOB --json                          # machine-readable
Rows: test name (the part before '#<index>'), n samples, gpu delta MiB (max), host cgroup peak MiB (max),
median of decode_tps/prefill_tps/ppl/kld/top1_pct, [min-max] spread of decode, and BAD count when ok=false.
A job is only scorable if tests_completed_cleanly == tests_expected; the header says so.
"""
import argparse, collections, json, os, statistics as st, sys

METRICS = (("decode_tps", "dec"), ("prefill_tps", "pre"), ("ppl", "ppl"), ("kld", "kld"), ("top1_pct", "top1"), ("tg_ts", "tg"), ("pp_ts", "pp"))
# llama-bench tests (parser bench_json) carry v["samples"]=[{n_prompt,n_gen,avg_ts}]; map them to tg (n_gen>0) / pp metrics.
def bench(v):
    out = {}
    for s in v.get("samples") or []:
        if "avg_ts" in s: out["tg_ts" if s.get("n_gen", 0) > 0 else "pp_ts"] = s["avg_ts"]
    return out

def load(job, lab):
    p = job if os.path.isfile(job) else os.path.join(lab, "jobs", job, "att-1", "result.json")
    if os.path.isdir(job): p = os.path.join(job, "result.json")
    return json.load(open(p))

def group(res, strip=()):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in res["quality"]["tests"].items():
        n = k.split("#")[0]
        for s in strip: n = n.replace(s, "")
        m = g[n]; mem = v.get("mem") or {}
        m["n"].append(1)
        if "gpu_delta_mib" in mem: m["mib"].append(mem["gpu_delta_mib"])
        if "host_cgroup_peak_mib" in mem: m["host"].append(mem["host_cgroup_peak_mib"])
        vv = dict(v, **bench(v))
        for x, y in METRICS:
            if vv.get(x) is not None: m[y].append(vv[x])
        if not v.get("ok", False): m["bad"].append(k)
    return g

def summarize(g):
    out = {}
    for n, m in g.items():
        d = dict(n=len(m["n"]), gpu_mib=max(m["mib"]) if m["mib"] else None, host_mib=max(m["host"]) if m["host"] else None, bad=len(m["bad"]))
        for _, y in METRICS:
            if m[y]: d[y] = st.median(m[y]); d[y + "_min"] = min(m[y]); d[y + "_max"] = max(m[y])
        out[n] = d
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+"); ap.add_argument("--compare", action="append", default=[], help="BASE=TREAT test names")
    ap.add_argument("--strip", default=""); ap.add_argument("--json", action="store_true")
    ap.add_argument("--lab", default=os.environ.get("LAB", "lab"))
    a = ap.parse_args(); strip = [s for s in a.strip.split(",") if s]
    allj = {}
    for j in a.jobs:
        r = load(j, a.lab); cov = r["evidence"]["coverage"]; s = summarize(group(r, strip)); allj[j] = s
        if a.json: continue
        flag = "" if cov["tests_completed_cleanly"] == cov["tests_expected"] else "  ** INCOMPLETE: do not score **"
        print(f"== {j} {r['execution']} {cov['tests_completed_cleanly']}/{cov['tests_expected']}{flag}")
        for n, d in s.items():
            cols = []
            for _, y in METRICS:
                if y in d:
                    f = f"{d[y]:.3f}" if y in ("kld", "ppl") else f"{d[y]:.1f}"
                    cols.append(f"{y} {f}" + (f" [{d[y+'_min']:.1f}-{d[y+'_max']:.1f}]" if y in ("dec", "tg") and d["n"] > 1 else ""))
            gm = "-" if d["gpu_mib"] is None else d["gpu_mib"]; hm = "-" if d["host_mib"] is None else f"{d['host_mib']:.0f}"
            print(f"{n:34s} n{d['n']:<3d} gpu {gm!s:>5} host {hm:>7}", *cols, f"BAD {d['bad']}" if d["bad"] else "")
        for c in a.compare:
            b, t = c.split("=")
            for _, y in (METRICS[0], METRICS[1], METRICS[5], METRICS[6]):
                if b in s and t in s and y in s[b] and y in s[t]:
                    print(f"  {t} vs {b} {y}: {s[t][y]:.1f} vs {s[b][y]:.1f} = {100*(s[t][y]/s[b][y]-1):+.2f}%")
    if a.json: print(json.dumps(allj, indent=1, sort_keys=True))

if __name__ == "__main__":
    main()
