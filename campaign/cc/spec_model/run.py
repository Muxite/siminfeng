"""Reproduce T-SPECM artifacts from frozen local evidence (CPU only)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from model import decode_runs, implied_length, pool_curves, predict, recommend, union_curves

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/codexSPECM'
DOMAINS = ('prose', 'chat', 'code')
KS = (18, 30, 42, 48)
DS = (1, 2, 3, 4, 8, 16)
ACCEPTANCE = (.5, .8, 1.)


def read(path):
    return json.loads((ROOT/path).read_text())


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = read('artifacts/codexPL4/model-stats.json')['qwen']
    constants = read('cc/moe_planner/calibration.json')['physical_decode']['constants']
    host = read('artifacts/codexPL3/factory-host.json')
    curves, sources = {}, {}
    for domain in DOMAINS:
        path = ROOT/f'artifacts/codexQRT/{domain}.jsonl'
        runs = decode_runs(path, stats['n_layer'], stats['n_expert'], stats['n_expert_used'])
        curves[domain] = union_curves(runs)
        sources[domain] = dict(path=str(path.relative_to(ROOT)),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               run_lengths=[len(r) for r in runs])
    curves['pooled'] = pool_curves([curves[d] for d in DOMAINS])
    union_rows, layer_rows = [], []
    for domain, curve in curves.items():
        for d, entry in curve.items():
            union_rows.append(dict(domain=domain, d=d, windows=entry['windows'],
                                   mean_distinct=entry['mean'], naive=d*8,
                                   union_over_top_k=entry['mean']/8,
                                   fraction_of_naive=entry['mean']/(d*8)))
            layer_rows.extend(dict(domain=domain, d=d, layer=l, windows=entry['windows'],
                                   mean_distinct=u) for l, u in enumerate(entry['per_layer']))
    write_csv('union-curves.csv', union_rows)
    write_csv('union-per-layer.csv', layer_rows)
    rows, all_predictions, baseline = [], {}, {}
    for k in KS:
        baseline[k] = predict(stats, constants, k, 1, 1., curves['pooled'][1]['per_layer'])
    for domain, curve in curves.items():
        all_predictions[domain] = {}
        for k in KS:
            all_predictions[domain][k] = {}
            for d in range(1, 17):
                p = predict(stats, constants, k, d, 1., curve[d]['per_layer'])
                all_predictions[domain][k][d] = p
                if d not in DS:
                    continue
                sensitivity = predict(stats, constants, k, d, 1., curve[d]['per_layer'],
                                      scale_nonexpert=True)
                for a in ACCEPTANCE:
                    rows.append(dict(domain=domain, k=k, d=d, a=a, accepted_tokens=a*d,
                                     step_us=p['step_us'], tps=p['tps']*a,
                                     speedup=p['tps']*a/baseline[k]['tps'],
                                     cpu_us=p['cpu_us'], gpu_us=p['gpu_us'],
                                     handoff_us=p['handoff_us'],
                                     cpu_bytes=sum(p['cpu_bytes'].values()), gpu_bytes=p['gpu_bytes'],
                                     whole_gpu_scaled_tps=sensitivity['tps']*a,
                                     break_even_acceptance=baseline[k]['tps']/p['tps']))
    write_csv('predictions.csv', rows)
    comparisons = []
    for domain in curves:
        times = {d:p['step_us'] for d, p in all_predictions[domain][30].items()}
        for name, speedup, measured_base, measured_tps in (
                ('rewrite', 2.4, 35.6, 85.2), ('repeat', 3., 35.2, 106.6)):
            comparisons.append(dict(domain=domain, workload=name,
                rounded_speedup=speedup, measured_base=measured_base, measured_tps=measured_tps,
                relative=implied_length(speedup, times, baseline[30]['step_us']),
                absolute=implied_length(measured_tps/baseline[30]['tps'], times, baseline[30]['step_us'])))
    rules = [dict(k=k, a=a, **recommend(stats, constants, k,
                  [curves[x] for x in DOMAINS], a)) for k in KS for a in ACCEPTANCE]
    inputs = ['cc/moe_planner/calibration.json', 'cc/moe_planner/decode_model.py',
              'artifacts/codexPL4/model-stats.json', 'artifacts/codexPL3/factory-host.json',
              'decisions/D042-external-research-probes-1.md']
    manifest = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    (OUT/'results.json').write_text(json.dumps(dict(schema='spec-expert-union/1',
        sources=sources, input_sha256=manifest, constants=constants, curves=curves,
        baseline=baseline, predictions=all_predictions, comparison_D042=comparisons,
        recommendations=rules), indent=2)+'\n')

    text = ['# T-SPECM — Expert-union speculation model', '',
        'CPU-only analysis of local evidence. Writes are confined to `cc/spec_model/**` and '
        '`artifacts/codexSPECM/**`. No planner, engine, calibration or source trace was modified.', '',
        '## Evidence and window definition', '',
        'Qwen3-30B-A3B Q4_K_M: 48 layers, 128 experts/layer, top_k=8. Each domain '
        'has one 64-token decode continuation. Only records with phase `decode` enter '
        'the analysis; prefill is excluded. Every decode token must have all layers, '
        'exactly eight distinct in-range expert IDs and no duplicate token/layer. '
        'Records are ordered by sequence and position, and gaps/sequence boundaries split runs. '
        'For each layer and d=1..16, take every overlapping consecutive length-d window '
        '(65-d windows/domain), count the union of expert IDs, then average. Overlapping '
        'windows are correlated; these are descriptive curves, not independent confidence samples. '
        'Pooled curves weight all within-domain windows equally and never concatenate domains. '
        'Prose, chat and code are all descriptive here; none estimates a workload acceptance rate. '
        'Hashes and run lengths are recorded in `results.json`.', '',
        table(['d', 'naive d×8', 'prose', 'chat', 'code', 'pooled'],
              [[d, 8*d]+[f'{curves[x][d]["mean"]:.3f}' for x in curves] for d in range(1,17)]), '',
        'Full layer curves: `union-per-layer.csv`; aggregate curves: `union-curves.csv`.', '',
        '## Latency and accepted-token convention', '',
        'Let U_l(d) be the layer union mean, B_t the actual full expert tensor bytes, '
        'and E=128. Static K offloads the first K expert layers. For each tensor in layer l, '
        'active bytes per step = B_t × U_l(d)/E = PL4 single-token bytes × U_l(d)/8. '
        'Sum CPU bytes by quantization type for l<K and GPU expert bytes for l≥K. '
        'All nonexpert weights are on GPU and read once per step, as dense weight reuse '
        'permits; they have no expert union. The CSV also includes a conservative literal '
        'whole-GPU scaling sensitivity, multiplying nonexpert bytes by mean(U)/8.', '',
        '`T_us(K,d) = Σ_type CPU_bytes(type)/(BW_CPU(type)×1000) + '
        'GPU_bytes/(BW_GPU×1000) + K×handoff_us + draft_us + d×extra_us_per_token`', '',
        '`accepted_drafts_per_step = a×d; predicted_tps = 1e6×a×d/T_us; '
        'speedup = a×d×T_us(K,1)/T_us(K,d)`', '',
        '**a is the observed aggregate accepted/proposed draft fraction on engaged steps**, '
        'so the numerator is exactly expected accepted drafts when step length is fixed. '
        'Bonus/correction tokens and unengaged ordinary steps are excluded from the main grid. '
        'Thus d=1,a=1 is the ordinary PL4 baseline; d=1,a<1 counts only accepted drafts '
        'and is not an end-to-end decoder throughput prediction. If a instead means '
        'independent conditional prefix survival, expected accepted drafts are '
        '`Σ_(i=1..d) a^i`, not a×d. Substitute that numerator (or its implied aggregate '
        'fraction) rather than interpreting this grid as that different process. '
        'A bonus-token model needs explicit verification-token accounting; no free extra '
        'token is credited here.', '',
        'Tables assume draft_us=0 and extra_us_per_token=0, once-only expert weight reads '
        'within a batch and no new compute/KV bottleneck. All d proposed positions cost '
        'traffic even when rejected. Actual rejected-path routing may differ from the '
        'observed accepted continuation, especially at a=0.5; this grid holds U(d) fixed '
        'as an approximation. Repeated expert execution, output projection arithmetic, '
        'attention, activation traffic, KV allocation, drafting and sampling overhead '
        'can lower performance. This is a bandwidth forecast, not measured speculative throughput.', '',
        f'PL4 constants: GPU {constants["bw_gpu_gbs"]:.3f} GB/s; '
        f'CPU Q4_K/Q6_K {constants["bw_cpu_gbs"]["Q4_K"]:.3f} GB/s; '
        f'handoff {constants["handoff_us"]:.3f} µs/CPU layer. Q6_K shares Q4_K bandwidth '
        'as in PL4. The model retains each layer’s actual Q4_K/Q6_K tensor bytes. '
        'No additional 1.02 Q8-reuse multiplier is applied. PL4 has no other identifiable '
        'fixed per-step cost; inventing an intercept or adding a measured baseline residual '
        'would double-count or refit it.', '',
        table(['K', 'ordinary PL4 t/s', 'CPU ms', 'GPU ms', 'handoff ms'],
              [[k, f'{baseline[k]["tps"]:.2f}']+
               [f'{baseline[k][x]/1000:.3f}' for x in ('cpu_us','gpu_us','handoff_us')] for k in KS]), '',
        f'K=0 is omitted: weights alone are { (stats["expert_bytes"]+stats["nonexpert_bytes"])/2**20:.2f} MiB, '
        f'exceeding the recorded {host["gpu"]["memory_mib"]} MiB GPU before KV/runtime buffers. '
        'K=18,30,42,48 have measured static precedents (D039). This does not establish '
        'their speculative batch buffer fit; the planner must check that separately.', '',
        '## Predicted accepted-draft tokens/s', '',
        'Each table gives the full requested K×d×a grid. All step components, speedups, '
        'whole-GPU scaling sensitivity and break-even acceptance are in `predictions.csv`.', '']
    for domain in curves:
        text += [f'### {domain}', '', table(['K','a']+[f'd={d}' for d in DS],
            [[k, a]+[f'{all_predictions[domain][k][d]["tps"]*a:.1f}' for d in DS]
             for k in KS for a in ACCEPTANCE]), '']
    text += ['## D042 check: implied effective mean draft length', '',
        'D042 §1 reports rewrite 35.6→85.2 t/s (rounded 2.4×), repeat 35.2→106.6 '
        '(rounded 3.0×), all proposed drafts accepted, K=30. The raw PL4 baseline is '
        f'{baseline[30]["tps"]:.2f} t/s. Invert the a=1 curve. Between adjacent integer '
        'lengths, mix step times and token rewards to solve for a two-length mixture; '
        'do not linearly interpolate throughput. Relative inversion uses the rounded '
        '2.4×/3.0×; absolute inversion uses 85.2/106.6 t/s directly.', '',
        table(['trace proxy','workload','relative implied d','absolute implied d'],
              [[c['domain'],c['workload'],
                f'{c["relative"]["mean_d"]:.2f}' if c['relative'] else '>16 / outside curve',
                f'{c["absolute"]["mean_d"]:.2f}' if c['absolute'] else '>16 / outside curve']
               for c in comparisons]), '',
        'This is a consistency check, not independent validation: these three routing '
        'continuations are not the D042 rewrite/repeat prompts, and D042 does not report '
        'the length distribution or step latencies. Acceptance ≈100% does not mean '
        'speculation engages on every step. Ordinary fallback steps or positive draft '
        'overhead would generally require longer engaged batches than these idealized '
        'effective lengths. A mean alone cannot identify E[U(D)] for a broad distribution. '
        'Do not infer observed draft lengths from these inversions. Capture per-step '
        'proposed/accepted lengths, target verification time, draft time and engagement '
        'alongside D042 timing counters to test the mechanism.', '',
        '## Recommended planner `spec` lever', '',
        'Use `ngram-mod` as a workload-gated candidate for copying, rewrite/refactor and '
        'repeat tasks. Require fit for the model, KV and chosen verification batch. '
        'Log draft_n and draft_n_accepted; estimate aggregate a by draft length and the '
        'fraction rho of output tokens produced during engaged steps. Open chat in D042 '
        'had zero drafts: select no speculative speedup when rho=0. Measure drafting '
        'and incremental per-token compute costs rather than assuming they vanish.', '',
        'For each feasible d=2..16, compute the worst supplied domain speedup '
        '`s(d)=min_domain a(d)×d×T1/[T(d)+draft_us(d)+extra_us_per_token×d]`. '
        'Use `S(d)=1/[1-rho+rho/s(d)]` for a token-fraction engagement mixture. '
        'Enable only if the best S≥1.10 (an explicit engineering margin), and choose '
        'its d as draft_max. Recompute with observed a(d); a fixed acceptance fraction '
        'at every d is only the requested sensitivity grid. If the optimum reaches '
        '16, it is the largest evaluated length, not a demonstrated global optimum. '
        'Do not extrapolate past 16 from these 64-token traces.', '',
        table(['K','a','enable','draft_max','worst-domain gain (rho=1, zero extra costs)'],
              [[r['k'],r['a'],r['enable'],r['draft_max'],f'{r["predicted_speedup"]:.2f}×'] for r in rules]), '',
        'Treat this as an analytical candidate rule requiring an end-to-end check. '
        'Choose the smaller feasible length or disable if measured acceptance, '
        'engagement or added compute removes the margin. No existing planner files '
        'were changed; `cc/spec_model/model.py:recommend` provides the isolated rule.', '',
        '## Reproduction and tests', '',
        'From the repository root:', '', '```bash',
        'PYTHONDONTWRITEBYTECODE=1 python3 cc/spec_model/run.py',
        'PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider '
        '--basetemp=artifacts/codexSPECM/pytest-tmp cc/spec_model '
        '> artifacts/codexSPECM/pytest.txt', '```', '',
        'All analysis uses only Python’s standard library. Tests cross-check d=1 '
        'against PL4, brute-force real trace unions, sequence/gap handling, type-aware '
        'byte accounting, rejection numerator semantics, inversion and lever gates.']
    (OUT/'ANALYSIS.md').write_text('\n'.join(text)+'\n')
    print(f'Wrote {len(union_rows)} aggregate curve rows, {len(layer_rows)} layer rows, '
          f'{len(rows)} forecast rows to {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
