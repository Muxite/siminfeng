#!/usr/bin/env python3
"""Summarise hybrid_moe_bench results JSON: miss-cost fit, overlap efficiency, host launch overhead, H2D cost, correctness, and
comparison with the linear model of experiments/routing/predict_slots.py (33.3 us per CPU-computed expert activation;
serial bound t0 + misses*c, perfect-overlap bound max(t0, misses*c)). Pure stdlib.  Usage: analyze.py results.json [...]"""
import json, sys, statistics as st

PRED_MISS_US = (1000 / 128.0 - 1000 / 704.0) / (24 * 8) * 1000   # identical to predict_slots.py calibration (33.3 us)

def fit(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs); return my - b * mx, b

def main(path):
    R = json.load(open(path)); C = {c["name"]: c for c in R["configs"]}; m = R["meta"]; NL = len(m["layers"])
    print(f"## {path}\n\nlayers {m['layers'][0]}..{m['layers'][-1]} ({NL}), threads {m['threads']}, CPU weights {m['cpu_weights']}, "
          f"types {m['types']}, expert {m['expert_bytes']} B ({m['expert_bytes']/2**20:.3f} MiB), {m['stream_positions']} decode positions x "
          f"{m['passes']} pass(es), synth {m['synth_n']} x {m['passes']}; {m['label']}. All times: sum over the {NL} layer(s) per decode step, us.\n")
    e = R["endpoint_rel_l2_vs_fp64"]
    print(f"- correctness: all_checks_ok = **{R['all_checks_ok']}**; endpoint rel-L2 vs fp64: gpu_ref median {e['gpu_ref_median']:.2e} "
          f"(max {e['gpu_ref_max']:.2e}), cpu_all median {e['cpu_all_median']:.2e} (max {e['cpu_all_max']:.2e})")
    mi = R["micro"]
    print(f"- H2D admission of one expert: pinned median {mi['h2d_one_expert_pinned_us_median']:.1f} us (p95 {mi['h2d_one_expert_pinned_us_p95']:.1f}) "
          f"= {m['expert_bytes']/mi['h2d_one_expert_pinned_us_median']/1e3:.1f} GB/s; pageable {mi['h2d_one_expert_pageable_us_median']:.1f} us; "
          f"bare GPU round trip (4 KB H2D + 4 KB D2H + sync) {mi['gpu_roundtrip_4KB_us_median']:.1f} us\n")
    print("### Controlled misses per layer (K=16 static slots, synthetic ids with exactly m misses per layer, real routing weights)\n")
    print("| m/layer | overlap med | p95 | serial med | serial GPU part | serial CPU part | overlap host launch+combine | cpu-only(m) | gpu-only(8-m) | overlap saving / min(gpu,cpu) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    xs, yo, ys, cx, cy = [], [], [], [], []
    for k in range(9):
        o, s = C[f"synth_overlap_m{k}"], C[f"synth_serial_m{k}"]; co, go = C.get(f"cpu_only_n{k}"), C.get(f"gpu_full_only_n{8-k}")
        g, c = s["gpu_us_median"], s["cpu_us_median"]
        launch = o["total_us_median"] - o["cpu_us_median"] - o["wait_us_median"]
        eff = (s["total_us_median"] - o["total_us_median"]) / min(g, c) if 0 < k < 8 else float("nan")
        print(f"| {k} | {o['total_us_median']:.0f} | {o['total_us_p95']:.0f} | {s['total_us_median']:.0f} | {g:.0f} | {c:.0f} | {launch:.0f} | "
              f"{co['total_us_median'] if co else 0:.0f} | {go['total_us_median'] if go else 0:.0f} | {eff:.2f} |")
        xs.append(k); yo.append(o["total_us_median"]); ys.append(s["total_us_median"])
        if co: cx.append(k); cy.append(co["total_us_median"])
    a, b = fit(cx, cy); a1, b1 = fit(xs[1:], yo[1:]); a2, b2 = fit(xs[1:], ys[1:])
    print(f"\n- CPU-only miss path (m = 1..8 per layer): t = {a/NL:.1f} us/layer + **{b/NL:.1f} us per CPU expert** (least squares on medians)")
    print(f"- hybrid overlap total: {a1/NL:.1f} us/layer + {b1/NL:.1f} us per miss; serial: {a2/NL:.1f} us/layer + {b2/NL:.1f} us per miss")
    print(f"- predict_slots.py assumed {PRED_MISS_US:.1f} us per miss and no per-layer fixed cost\n")
    g = C["gpu_ref"]; t0 = g["total_us_median"]
    print(f"### Trace-driven ({m['stream_positions']} real decode positions; static = prefill-profile top-K, LRU cold start)\n")
    print(f"gpu_ref (all experts on GPU) median {t0:.0f} us, cpu_all median {C['cpu_all']['total_us_median']:.0f} us.\n")
    print("| config | hit % | misses/step | med | p95 | pred serial (t0+33.3us*miss) | pred perfect overlap | measured/pred serial | admit between steps (mean) | check |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in R["configs"]:
        if c["stream"] != "trace": continue
        mus = [NL * 8 - h for h in c["hits"]]
        ps = st.median([t0 + x * PRED_MISS_US for x in mus]); po = st.median([max(t0, x * PRED_MISS_US) for x in mus])
        print(f"| {c['name']} | {100*c['hit_rate']:.1f} | {st.mean(mus):.1f} | {c['total_us_median']:.0f} | {c['total_us_p95']:.0f} | {ps:.0f} | {po:.0f} | "
              f"{c['total_us_median']/ps:.2f} | {c['admit_between_steps_us_mean']:.0f} | {c['check']} |")
    print()

if __name__ == "__main__":
    for p in sys.argv[1:]: main(p)
