"""Reproduce PL2 predictions from read-only T-PL inputs; no engine execution."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/codexPL2'
spec = importlib.util.spec_from_file_location('planner', ROOT/'cc/moe_planner/plan.py')
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
stats = planner.read_json(ROOT/'artifacts/codexPL/granite-inspected.json')
host = planner.read_json(ROOT/'cc/engine_handoff/hosts/x86-main.json')


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


baseline = {'components':{}}
fixed = {'components':{'kv_quant_fix':{'status':'pending','enabled':True}}}
stacked = {'components':dict(fixed['components'], q8_reuse={'status':'accepted'})}
save('components-baseline.json', baseline)
save('components-kvfix-enabled.json', fixed)
save('components-stacked.json', stacked)
summary = []
for name, gpu, ctx, extra, components in [
    ('plan-900-calibrated',900,4096,False,baseline),
    ('plan-900-diverse',900,4096,True,baseline),
    ('plan-1200-32k-baseline',1200,32768,True,baseline),
    ('plan-1200-32k-kvfix',1200,32768,True,fixed),
    ('plan-1200-32k-stacked',1200,32768,True,stacked),
]:
    result = planner.plan(stats,host,dict(gpu_mib=gpu,ctx=ctx),
                          allow_extrapolation=extra,components=components)
    save(name+'.json',result)
    save(name+'-lab-tests.json',planner.emit_lab_tests(result))
    summary.append(dict(name=name,gpu_mib=gpu,ctx=ctx,
                        families=result['family_count'],pareto_count=len(result['pareto_set']),
                        plans=[{k:p[k] for k in ('id','representation','kv_type','gpu_mib',
                            'gpu_guard_mib','decode_tps','prefill_tps','ttft_ms','confidence','kld')}
                               for p in result['plans']]))
save('examples-summary.json',summary)
for example in summary:
    print(example['name'], 'families=',example['families'],'pareto=',example['pareto_count'])
    for p in example['plans']:
        print(' ',p['id'],p['gpu_mib'],'MiB +',p['gpu_guard_mib'],'guard;',p['decode_tps'],'t/s;',p['ttft_ms'],'ms TTFT')
