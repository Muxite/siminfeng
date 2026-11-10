#!/usr/bin/env python3
"""Rebuild the physical calibration from immutable lab results (CPU, no engine runs)."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import decode_model as dm


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def collect(out):
    sys.path.insert(0, str(ROOT/'cc/model_inspect'))
    from inspect_model import inspect
    cal = read(HERE/'calibration.json')
    stats = {}
    for rep, rec in cal['representations'].items():
        # Only header pages are read; no GPU or tensor compute.
        if rep in ('mixE4K', 'Q8_0'):
            s = inspect(rec['model_path'])
            s.update(representation=rep, expert_bytes=rec['expert_bytes'], nonexpert_bytes=rec['nonexpert_bytes'])
            stats[rep] = s
    for name in ('mix4_imat', 'pap12_q3', 'pap18_q3', 'pap24_q3', 'pap24_q3_noimat',
                 'pap12_q2', 'pap18_q2', 'pap24_q2'):
        s = inspect(ROOT/f'artifacts/codexPAP/models/{name}.gguf')
        s.update(representation=name, expert_bytes=s['totals']['real_expert_bytes'],
                 nonexpert_bytes=s['totals']['non_expert_bytes_including_shared'])
        stats[name] = s
    s = read(ROOT/'artifacts/x9-qwen/stats.json')
    s.update(representation='Q4_K_M', expert_bytes=s['totals']['real_expert_bytes'],
             nonexpert_bytes=s['totals']['non_expert_bytes_including_shared'])
    stats['qwen'] = s
    dump(out/'model-stats.json', stats)
    definitions = read(ROOT/'lab/tests/plan1.json')
    jobs = [('D022','ad883810b3'), ('D029','34c09bec98'), ('D034','56115e2bf1'),
            ('D035','3d2fbbd061'), ('D039','9f50eae42a'), ('D040','d2e342e743')]
    rows = []
    for decision, job in jobs:
        path = ROOT/f'lab/jobs/job-{job}/att-1/result.json'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for test, result in read(path)['quality']['tests'].items():
            name = test.split('#')[0]
            if not result.get('ok'):
                continue
            tps = result.get('decode_tps')
            if decision == 'D029' and '_tg_' in name:
                tps = result['samples'][0]['avg_ts']
            if not tps:
                continue
            family, factor, coverage = 'static', 1., None
            if decision == 'D022':
                if 'static_n' in name:
                    k = int(re.search(r'_n(\d+)', name)[1])
                else:
                    k = int(re.search(r'_c(\d+)', name)[1]); family = 'lru'
                key = 'mixE4K'
            elif decision == 'D029':
                key, k = ('Q8_0' if '_q8_' in name else 'mixE4K'), 0
                factor = 1.02 if name.endswith('r1') else 1.
            elif decision == 'D034':
                args = definitions[name]['args']; k = int(args[args.index('--n-cpu-moe')+1]); key = 'mixE4K'
            elif decision in ('D035','D040'):
                key, k = re.fullmatch(r'pap_decode_(.+)_n(\d+)', name).groups(); k = int(k)
            else:
                key, k = 'qwen', int(re.search(r'_n(\d+)', name)[1])
                # D039 uses release-1, whose reuse component is enabled.
                factor = 1.02
            s = stats[key]
            rows.append(dict(decision=decision, job_id='job-'+job, attempt='att-1', test=test,
                             result_path=str(path.relative_to(ROOT)), result_sha256=sha,
                             model='Qwen' if key == 'qwen' else 'Granite', representation=key,
                             family=family, coord=k, decode_tps=tps, component_factor=factor,
                             gpu_mib=result.get('mem',{}).get('gpu_delta_mib'),
                             noisy=decision == 'D035', workload='bench-tg128' if '_tg_' in name else 'completion',
                             traffic=dm.traffic(s, family, k, coverage)))
    dump(out/'measured-points.json', rows)
    return rows, stats


def fit(rows):
    import numpy as np
    from scipy.optimize import lsq_linear
    # Qwen's Q6_K projections share Q4_K's effective path until an independent
    # Q6_K CPU screen exists. Fitting both is unstable in a one-model fold.
    kinds = sorted({('Q4_K' if k == 'Q6_K' else k) for r in rows
                    for k,b in r['traffic']['cpu_active_bytes'].items() if b})
    names = ['gpu'] + kinds + ['handoff', 'h2d']
    counts = defaultdict(int)
    for r in rows:
        counts[(r['job_id'], r['test'].split('#')[0])] += 1
    matrix, target, weights = [], [], []
    for r in rows:
        p = r['traffic']; cpu = defaultdict(float)
        for k,b in p['cpu_active_bytes'].items(): cpu['Q4_K' if k == 'Q6_K' else k] += b/1e6
        matrix.append([p['gpu_active_bytes']/1e6] + [cpu[k] for k in kinds] +
                      [p['cpu_touching_layers'], p['h2d_bytes']/1e6])
        target.append(1e6/r['decode_tps']*r['component_factor'])
        weights.append(1/counts[(r['job_id'], r['test'].split('#')[0])])
    a, y, w = np.array(matrix), np.array(target), np.sqrt(weights)
    # Anchor GPU bandwidth ONLY on training all-GPU points if present.
    anchors = [i for i,r in enumerate(rows) if r['family']=='static' and r['coord']==0]
    fixed = {}
    if anchors:
        ag, yg = a[anchors,0], y[anchors]
        aw = w[anchors]**2
        fixed[0] = float(np.dot(ag*aw,yg)/np.dot(ag*aw,ag))
    free = [i for i in range(len(names)) if i not in fixed and np.any(a[:,i])]
    rank = int(np.linalg.matrix_rank(a[:,free]*w[:,None]))
    prior = None
    if rank < len(free):
        h = names.index('handoff'); fixed[h] = 2030/12
        free.remove(h)
        prior = 'D031 extra split cost / 12 = 169.167 us/layer; rank-deficient training fold; not a universal measured handoff'
    residual = y.copy()
    for i,v in fixed.items(): residual -= a[:,i]*v
    lower = [0.5 if names[i] not in ('handoff',) else 0 for i in free]
    solution = lsq_linear(a[:,free]*w[:,None], residual*w, bounds=(lower, np.inf), tol=1e-12)
    coefficients = {names[i]:v for i,v in fixed.items()}
    coefficients.update({names[i]:float(v) for i,v in zip(free,solution.x)})
    cpu = {k:1000/coefficients[k] for k in kinds}
    if 'Q4_K' not in cpu: raise ValueError('need Q4_K training traffic')
    cpu['Q6_K'] = cpu['Q4_K']
    constants = dict(bw_gpu_gbs=1000/coefficients['gpu'], bw_cpu_gbs=cpu,
                     handoff_us=coefficients.get('handoff',0.),
                     bw_h2d_gbs=1000/coefficients['h2d'] if 'h2d' in coefficients else 12., cache_miss_us=0.)
    return dict(constants=constants, coefficients=coefficients, rank=rank, free_columns=len(free),
                training_models=sorted({r['model'] for r in rows}),
                training_jobs=sorted({r['job_id'] for r in rows}), training_point_count=len(rows),
                boundary_coefficients=[names[i] for i,lo in zip(free,lower) if abs(coefficients[names[i]]-lo)<1e-6],
                handoff_prior=prior, gpu_anchor_count=len(anchors),
                assumptions=['Q6_K shares Q4_K effective bandwidth (mixed Qwen experts)',
                             'all nonexpert weight bytes active each token (logical estimate)',
                             'uniform LRU coverage; fitted H2D bandwidth includes cache dispatch; miss latency not separately identifiable',
                             'equal weight per job/configuration; all repeats retained; D035 noisy points included',
                             'fit microseconds; D029/D039 reuse normalized by 1.02; workload/context effects not separately fitted'])


def evaluate(rows, stats, fitted):
    residuals = []
    for r in rows:
        p = dm.predict(stats[r['representation']], r['family'], r['coord'], fitted['constants'])
        tps = p['decode_tps']*r['component_factor']
        residuals.append(dict(model=r['model'], decision=r['decision'], test=r['test'],
                              family=r['family'], measured_tps=r['decode_tps'], predicted_tps=tps,
                              error_pct=100*(tps/r['decode_tps']-1), fallbacks=p['fallbacks']))
    groups = {}
    for label in sorted({r['model'] for r in rows} | {r['decision'] for r in rows}):
        vals = [r['error_pct'] for r in residuals if label in (r['model'],r['decision'])]
        groups[label] = dict(count=len(vals), mean_error_pct=statistics.mean(vals),
                             mape_pct=statistics.mean(map(abs,vals)), max_abs_pct=max(map(abs,vals)))
    by_mechanism = {}
    for model in sorted({r['model'] for r in rows}):
        for family in ('static','lru'):
            vals = [r['error_pct'] for r in residuals if r['model']==model and r['family']==family]
            if vals:
                by_mechanism[f'{model}/{family}'] = dict(count=len(vals), mape_pct=statistics.mean(map(abs,vals)),
                    mean_error_pct=statistics.mean(vals),max_abs_pct=max(map(abs,vals)))
    return dict(summary=groups, by_mechanism=by_mechanism, residuals=residuals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT/'artifacts/codexPL4'))
    args = p.parse_args(); out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/codexPL4'): p.error('out must stay under artifacts/codexPL4')
    out.mkdir(parents=True, exist_ok=True)
    rows, stats = collect(out)
    fitted = fit(rows)
    fitted['evaluation'] = evaluate(rows,stats,fitted)
    folds = {}
    for held_out in ('Granite','Qwen'):
        training = [r for r in rows if r['model'] != held_out]
        f = fit(training)
        f['training_models'] = sorted({r['model'] for r in training})
        f['evaluation'] = evaluate([r for r in rows if r['model']==held_out], stats, f)
        folds[held_out] = f
    fitted['leave_one_model_out'] = folds
    dump(out/'fit-report.json', fitted)
    cal = read(HERE/'calibration.json')
    old = cal.get('physical_decode',{})
    def compact(report):
        return {k:({a:b for a,b in v.items() if a != 'residuals'} if k == 'evaluation' else v)
                for k,v in report.items() if k != 'leave_one_model_out'}
    calibration = compact(fitted)
    calibration['leave_one_model_out'] = {k:compact(v) for k,v in folds.items()}
    cal['physical_decode'] = dict(schema='physical-decode/1', host_id=cal['host_id'], **calibration,
                                 report='artifacts/codexPL4/fit-report.json',
                                 points='artifacts/codexPL4/measured-points.json')
    for key in ('cpu_type_screen','screen_bw_cpu_gbs'):
        if key in old: cal['physical_decode'][key] = old[key]
    if 'screen_bw_cpu_gbs' in old:
        cal['physical_decode']['constants']['screen_bw_cpu_gbs'] = old['screen_bw_cpu_gbs']
    dump(HERE/'calibration.json',cal)
    print(json.dumps(dict(constants=fitted['constants'],residuals=fitted['evaluation']['summary'],
                          folds={k:v['evaluation']['summary'] for k,v in folds.items()}),indent=2))


if __name__ == '__main__': main()
