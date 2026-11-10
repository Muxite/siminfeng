#!/usr/bin/env python3
"""TRAIN-prefill fitted hot sets, held-out request coverage and conditional speed model."""
from collections import Counter
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
N, E, TOPK = 48, 128, 8
model = json.loads((OUT/'model.json').read_text())
cost = [model['per_layer_experts'][str(i)]['bytes_per_single_expert'] for i in range(N)]

def load(name):
    phases = {p: [[0]*E for _ in range(N)] for p in ('prefill','decode')}
    selections = {p: [] for p in phases}
    seen = set()
    positions = {p: set() for p in phases}
    rows = 0
    for line in (OUT/f'{name}.jsonl').open():
        row = json.loads(line)
        p, l, pos = row['phase'], row['layer'], row['pos']
        identity = (pos,l)
        assert identity not in seen and row['seq'] == 0
        seen.add(identity)
        assert len(row['experts']) == TOPK and len(set(row['experts'])) == TOPK
        assert all(0 <= e < E for e in row['experts'])
        positions[p].add(pos)
        for e in row['experts']:
            phases[p][l][e] += 1
        selections[p].append((pos,l,row['experts']))
        rows += 1
    assert len(positions['decode']) == 64
    allpos = sorted(set.union(*positions.values()))
    assert allpos == list(range(len(allpos)))
    assert rows == N*len(allpos)
    assert all((pos,l) in seen for pos in allpos for l in range(N))
    # Cross-check the unmodified official inspection adapter.
    official = json.loads((OUT/f'{name}-inspect.json').read_text())['routing_trace']
    for group in official['groups']:
        for layer, stats in group['layers'].items():
            assert stats['frequency'] == phases[group['phase']][int(layer)]
    return dict(hist=phases, selections=selections, rows=rows,
                tokens={p:len(s) for p,s in positions.items()})

names = ['prose','chat','code']
data = {name:load(name) for name in names if (OUT/f'{name}.jsonl').exists()}
train = data['prose']['hist']['prefill']
ranked = [sorted(range(E), key=lambda e:(-hist[e],e)) for hist in train]

def score(hist, sets):
    hits = sum(sum(hist[l][e] for e in sets[l]) for l in range(N))
    total = sum(map(sum,hist))
    return hits/total

def active(selections, sets):
    return sum(any(e not in sets[l] for e in es) for _,l,es in selections) / (len(selections)/N)

# Exact D039 endpoint fit, with seconds converted to ms.
c_ms = (1000/24.4-1000/93.5)/40
t_gpu_ms = 1000/93.5-8*c_ms
byte_layer_ms = TOPK*2.72*2**20/30e9*1000
static = {8:93.5,18:54.6,30:36.6,42:27.7}
results = []
for k, measured in static.items():
    budget = E*sum(cost[k:])
    h = int(budget//sum(cost))
    sizes = [h]*N
    used = h*sum(cost)
    # Spend remaining bytes on the next TRAIN-ranked expert with highest request/byte benefit.
    while True:
        eligible = [l for l in range(N) if sizes[l]<E and used+cost[l]<=budget]
        if not eligible:
            break
        l = min(eligible, key=lambda l:(-train[l][ranked[l][sizes[l]]]/cost[l], l))
        used += cost[l]
        sizes[l] += 1
    assert used <= budget and budget-used < min(cost)
    sets = [set(ranked[l][:sizes[l]]) for l in range(N)]
    coverage = {f'{name}_{phase}':score(d['hist'][phase],sets)
                for name,d in data.items() for phase in ('prefill','decode')}
    eval_hist = [[sum(data[name]['hist']['decode'][l][e] for name in data if name!='prose')
                  for e in range(E)] for l in range(N)]
    cov = score(eval_hist,sets)
    n_active = sum(active(d['selections']['decode'],sets)*d['tokens']['decode']
                   for name,d in data.items() if name!='prose') / sum(d['tokens']['decode'] for name,d in data.items() if name!='prose')
    byte_weighted_cov = sum(cost[l]*sum(eval_hist[l][e] for e in sets[l]) for l in range(N)) / sum(cost[l]*sum(eval_hist[l]) for l in range(N))
    cpu_ms = N*(1-cov)*byte_layer_ms
    pred = {str(us):1000/(t_gpu_ms+cpu_ms+N*us/1000) for us in (30,50,80)}
    threshold = 1-(1000/measured-t_gpu_ms-N*0.05)/(N*byte_layer_ms)
    critical_h_us = (1000/measured-t_gpu_ms-cpu_ms)/N*1000
    per_layer = {str(l): {'H':sizes[l], 'experts':ranked[l][:sizes[l]],
                  'train_coverage':sum(train[l][e] for e in sets[l])/sum(train[l]),
                  'eval_decode_coverage':sum(eval_hist[l][e] for e in sets[l])/sum(eval_hist[l])}
                 for l in range(N)}
    results.append(dict(K=k, static_tps=measured, expert_budget_MiB=budget/2**20,
        used_expert_MiB=used/2**20, unused_bytes=budget-used,
        H_min=min(sizes),H_max=max(sizes),H_sum=sum(sizes),coverage=coverage,
        eval_decode_coverage=cov,byte_weighted_eval_coverage=byte_weighted_cov,observed_cpu_active_layers=n_active,
        cpu_bytes_ms=cpu_ms, predicted_tps=pred,
        active_only_tps_at_50us=1000/(t_gpu_ms+cpu_ms+n_active*.05),
        coverage_crossover_at_50us=threshold,critical_handoff_us=critical_h_us,
        per_layer=per_layer))
report=dict(training='prose prefill only; tie expert ID ascending; no evaluation fitting',
    allocation='equal per-layer H floor, then next TRAIN frequency/byte until no expert fits',
    t_gpu_ms=t_gpu_ms,c_fit_ms=c_ms,byte_layer_ms=byte_layer_ms,
    handoff_us=[30,50,80],n_cpu_layers_assumed=48,
    validation={name:{'rows':d['rows'],'tokens':d['tokens']} for name,d in data.items()},results=results)
(OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| K | H range | TRAIN prefill | chat prefill | code prefill | chat decode | code decode | pooled decode | hybrid t/s (80-30 us) | static t/s |',
       '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in results:
    c=r['coverage']; p=r['predicted_tps']
    lines.append(f"| {r['K']} | {r['H_min']}-{r['H_max']} | {c['prose_prefill']:.2%} | {c['chat_prefill']:.2%} | {c['code_prefill']:.2%} | {c['chat_decode']:.2%} | {c['code_decode']:.2%} | {r['eval_decode_coverage']:.2%} | {p['80']:.1f}-{p['30']:.1f} | {r['static_tps']} |")
(OUT/'table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
