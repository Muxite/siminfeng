#!/usr/bin/env python3
"""Replay pre-registered planner requests and score against existing lab evidence."""
from __future__ import annotations
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import plan as planner
import decode_model as dm
from fit_decode import dump, read
ROOT = HERE.parents[1]
OUT = ROOT/'artifacts/codexPL4'


def score(stats, rows, constants, factor=1.):
    prediction = dm.predict(stats, rows[0]['family'], rows[0]['coord'], constants)['decode_tps']*factor
    measured = statistics.median(r['decode_tps'] for r in rows)
    return dict(coord=rows[0]['coord'], predicted_tps=prediction, measured_tps=measured,
                error_pct=100*(prediction/measured-1), samples=len(rows), component_factor=factor)


def main():
    cal = read(HERE/'calibration.json'); constants = cal['physical_decode']['constants']
    rows = read(OUT/'measured-points.json'); stats = read(OUT/'model-stats.json')
    host = read(ROOT/'cc/engine_handoff/hosts/x86-main.json')
    summary = dict(qwen_scenarios=[], granite_900={}, qwen_all_measured_coordinates=[])
    qrows = [r for r in rows if r['model']=='Qwen' and '_fa_' not in r['test'] and 'kv' not in r['test']]
    # The plain points have names qw_static_nK; FA/KV is a separate ninth configuration.
    qrows = [r for r in qrows if r['test'].split('#')[0] == f"qw_static_n{r['coord']}"]
    for k in sorted({r['coord'] for r in qrows}):
        group = [r for r in qrows if r['coord']==k]
        summary['qwen_all_measured_coordinates'].append(score(stats['qwen'],group,constants,1.02))
    for budget in (12000,8000,4000):
        old = read(ROOT/f'artifacts/x9-qwen/prereg-plan-{budget}.json')
        components = {k:{'enabled':True} for k in old['plans'][0]['components']}
        result = planner.plan(read(ROOT/'artifacts/x9-qwen/stats.json'),host,old['budget'],
                              allow_extrapolation=True, components=components,
                              all_representations=False,top=len(old['plans']))
        dump(OUT/f'plan-qwen-{budget}.json',result)
        original = old['plans'][0]; new = result['plans'][0]
        def coord(p): return int(p['flags'][p['flags'].index('--n-cpu-moe')+1])
        old_k, new_k = coord(original),coord(new)
        nearest = min(summary['qwen_all_measured_coordinates'],key=lambda r:abs(r['coord']-old_k))
        exact_old = [r for r in qrows if r['coord']==old_k]
        exact_new = [r for r in qrows if r['coord']==new_k]
        entry = dict(budget_mib=budget, original_id=original['id'], original_coord=old_k,
                     original_tps=original['decode_tps'],
                     original_coordinate_new_model_tps=dm.predict(stats['qwen'],'static',old_k,constants)['decode_tps']*1.02,
                     new_id=new['id'],new_coord=new_k,
                     new_tps=new['decode_tps'],new_gpu_mib=new['gpu_mib'],
                     same_original_coordinate=score(stats['qwen'],exact_old,constants,1.02) if exact_old else None,
                     new_coordinate=score(stats['qwen'],exact_new,constants,1.02) if exact_new else None,
                     nearest_original_measured=nearest,
                     caveat='D039 plain f16/default-batch measurements differ from selected quantized KV/FA/ubatch; same-K comparisons are placement proxies, not exact configuration validations. K41 has no measured exact match. D039 device deltas can exceed nominal budget despite transferred Granite memory guard.')
        summary['qwen_scenarios'].append(entry)
    result = planner.plan(read(ROOT/'artifacts/codexPL/granite-inspected.json'),host,
                          dict(gpu_mib=900,ctx=4096),components={},all_representations=False)
    dump(OUT/'plan-granite-900.json',result)
    measured = [r for r in rows if r['decision']=='D034']
    summary['granite_900'] = dict(selected=result['plans'][0]['id'],
        comparisons=[score(stats['mixE4K'],[r for r in measured if r['coord']==k],constants)
                     for k in sorted({r['coord'] for r in measured})])
    dump(OUT/'scenario-comparison.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
