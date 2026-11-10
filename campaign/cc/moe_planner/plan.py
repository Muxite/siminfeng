#!/usr/bin/env python3
"""Offline hypotheses for pinned llama.cpp; never starts an engine or a GPU job."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MIB = 1048576


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return value


def line(xs, ys):
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if not denom:
        raise ValueError('fit needs distinct coordinates')
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return mean_y - slope * mean_x, slope


def grouped(cal, family):
    rows = [p for p in cal['points'] if p['family'] == family]
    return [dict(x=x, **{key: statistics.median(p[key] for p in rows if p['x'] == x)
                        for key in ('gpu_mib', 'decode_tps', 'prompt_ms', 'host_cgroup_peak_mib')})
            for x in sorted({p['x'] for p in rows})]


def fit(rows, family):
    """OLS GPU/time for static; log-time for LRU (always positive)."""
    result = {'log_time': family == 'lru'}
    for key in ('gpu_mib', 'decode_us', 'prompt_ms'):
        ys = [1e6 / r['decode_tps'] if key == 'decode_us' else r[key] for r in rows]
        if result['log_time'] and key != 'gpu_mib':
            ys = [math.log(y) for y in ys]
        result[key] = line([r['x'] for r in rows], ys)
    return result


def predict_fit(model, x):
    out = {}
    for key in ('gpu_mib', 'decode_us', 'prompt_ms'):
        a, b = model[key]
        value = a + b * x
        if model['log_time'] and key != 'gpu_mib':
            value = math.exp(value)
        out[key] = max(value, 1e-6)
    out['decode_tps'] = 1e6 / out['decode_us']
    return out


def residual_report(cal):
    report = {}
    for family in ('static', 'lru'):
        rows = grouped(cal, family)
        model = fit(rows, family)
        fitted, loo = [], []
        for i, r in enumerate(rows):
            for target, prediction in ((fitted, predict_fit(model, r['x'])),
                                       (loo, predict_fit(fit(rows[:i] + rows[i+1:], family), r['x']))):
                target.append(dict(x=r['x'], gpu_mib_error=prediction['gpu_mib']-r['gpu_mib'],
                                   decode_pct_error=100*(prediction['decode_tps']/r['decode_tps']-1),
                                   prompt_ms_error=prediction['prompt_ms']-r['prompt_ms']))
        report[family] = dict(coefficients=model, fitted=fitted, leave_one_coordinate_out=loo,
                             max_fit_gpu_mib=max(abs(r['gpu_mib_error']) for r in fitted),
                             max_loo_gpu_mib=max(abs(r['gpu_mib_error']) for r in loo),
                             max_fit_decode_pct=max(abs(r['decode_pct_error']) for r in fitted),
                             max_loo_decode_pct=max(abs(r['decode_pct_error']) for r in loo))
    return report


def adapt_stats(raw):
    """Small explicit adapter: canonical fields, tensors, or inspector totals."""
    if raw.get('schema', raw.get('format')) != 'model-stats/0':
        raise ValueError('expected model-stats/0')
    meta = raw.get('model', {})
    s = dict(raw)
    for name in ('arch', 'n_layer', 'n_expert', 'n_expert_used', 'context_length'):
        s[name] = raw.get(name, meta.get(name))
    if s['arch'] is None:
        s['arch'] = raw.get('architecture')
    for name in ('n_layer', 'n_expert', 'n_expert_used', 'context_length'):
        finite(s[name], name, positive=True)
        if int(s[name]) != s[name]:
            raise ValueError(f'{name} must be integral')
        s[name] = int(s[name])
    if s['n_expert_used'] > s['n_expert']:
        raise ValueError('n_expert_used exceeds n_expert')
    if 'expert_bytes' not in s or 'nonexpert_bytes' not in s:
        tensors = raw.get('tensors')
        if tensors is not None:
            s['expert_bytes'] = sum(t['bytes'] for t in tensors if t['role'] == 'expert_ffn')
            s['nonexpert_bytes'] = sum(t['bytes'] for t in tensors if t['role'] != 'expert_ffn')
        else:
            roles = raw.get('totals', {}).get('by_role', raw.get('totals_by_role', {}))
            sizes = {k: v['bytes'] if isinstance(v, dict) else v for k, v in roles.items()}
            if not sizes or 'expert_ffn' not in sizes:
                raise ValueError('need expert_bytes/nonexpert_bytes, tensors, or totals.by_role bytes')
            s['expert_bytes'] = sizes['expert_ffn']
            s['nonexpert_bytes'] = sum(v for k, v in sizes.items() if k != 'expert_ffn')
    for name in ('expert_bytes', 'nonexpert_bytes'):
        finite(s[name], name, positive=True)
    kv = raw.get('kv_bytes_per_token', raw.get('kv', {}).get('bytes_per_token'))
    if kv is None and 'kv_cache' in raw:
        kv = {k: v['bytes_per_token'] for k, v in raw['kv_cache']['cache_types'].items()
              if v.get('bytes_per_token') is not None}
    if not isinstance(kv, dict) or 'f16' not in kv:
        raise ValueError('need kv_bytes_per_token with total K+V f16 bytes')
    s['kv_bytes_per_token'] = {k: finite(v, f'KV {k}', positive=True) for k, v in kv.items()}
    s['model_path'] = raw.get('model_path', raw.get('gguf_path', raw.get('path', raw.get('source', {}).get('path'))))
    if not s['model_path']:
        raise ValueError('need model_path (or path/source.path) to emit exact model flags')
    s['representation'] = raw.get('representation') or next(
        (r for r in ('mixE4K', 'mixE3K', 'mixE2K', 'Q4_K_M', 'Q8_0', 'bf16')
         if r in Path(s['model_path']).name), 'unknown')
    return s


def variants(s, cal):
    """Only enumerate sibling GGUFs that exist, never imaginary quantizations."""
    out = [s]
    ref = cal['representations'].get(s['representation'])
    known = (s['arch'] == 'granitemoe' and s['n_layer'] == 24 and s['n_expert'] == 32
             and s['n_expert_used'] == 8 and ref
             and Path(s['model_path']).name == ref['model_file']
             and abs(s['expert_bytes'] / ref['expert_bytes'] - 1) < .001
             and abs(s['nonexpert_bytes'] / ref['nonexpert_bytes'] - 1) < .001)
    if known:
        for name, rec in cal['representations'].items():
            path = Path(s['model_path']).parent / rec['model_file']
            if name != s['representation'] and path.is_file():
                out.append(dict(s, representation=name, model_path=str(path),
                                expert_bytes=rec['expert_bytes'], nonexpert_bytes=rec['nonexpert_bytes']))
    return out


def resource_limits(sample, gpu_budget, host_budget):
    warnings = []
    if sample is None:
        return gpu_budget, host_budget, warnings
    if sample.get('schema') != 'resource-sample/0':
        raise ValueError('expected resource-sample/0')
    fields = sample.get('fields', {})
    def get(key, unit):
        f = fields.get(key, {})
        if f.get('status') != 'ok' or f.get('value') is None:
            warnings.append(f'{key} unavailable; no zero or free-memory assumption')
            return None
        if f.get('unit') != unit:
            raise ValueError(f'{key}: expected unit {unit}')
        return finite(f['value'], key)
    total, used = get('gpu.memory.total', 'MiB'), get('gpu.memory.used', 'MiB')
    if total is not None and used is not None:
        gpu_budget = min(gpu_budget, max(0, total-used))
    available = get('host.mem.available', 'B')
    if available is not None:
        host_budget = min(host_budget, available/MIB)
    warnings.append('Resource sample is a non-atomic availability hint; no reservation or process attribution')
    return gpu_budget, host_budget, warnings


def hybrid_hit_rate(stats, spec):
    """Apply the existing TRAIN-selected IDs to an explicitly supplied decode trace."""
    groups = [g for g in stats.get('routing_trace', {}).get('groups', []) if g['phase'] == 'decode']
    if len(groups) != 1 or len(groups[0]['layers']) != stats['n_layer']:
        return sum(spec['hot_counts'])/(stats['n_layer']*stats['n_expert']), 'uniform routing fallback; no complete unambiguous decode trace'
    total, hits = 0, 0
    for layer, hot_ids in enumerate(spec['hot_ids']):
        hist = groups[0]['layers'][str(layer)]['frequency']
        if len(hist) != stats['n_expert'] or any(type(n) is not int or n < 0 for n in hist):
            raise ValueError('invalid routing frequency')
        total += sum(hist)
        hits += sum(hist[e] for e in hot_ids)
    if not total:
        raise ValueError('decode routing trace has no requests')
    return hits/total, f"decode trace split={groups[0]['split']}; hot IDs fixed from prose prefill TRAIN; train/eval disjointness must be verified"


def load_components(source=None):
    """Registry may be absent. Accepted entries default on; pending needs enabled=true."""
    if source is None:
        source = ROOT / 'cc/factory/components.json'
        if not source.is_file():
            return {}
    if isinstance(source, (str, Path)) and Path(source) == ROOT/'cc/factory/components.json' and not Path(source).is_file():
        return {}
    raw = read_json(source) if isinstance(source, (str, Path)) else source
    entries = raw.get('components', raw) if isinstance(raw, dict) else raw
    if isinstance(entries, list):
        entries = {r if isinstance(r, str) else r.get('id', r.get('name')):
                   {'enabled': True} if isinstance(r, str) else r for r in entries}
    if not isinstance(entries, dict):
        raise ValueError('components must be a mapping or list')
    out = {}
    for name, rec in entries.items():
        if isinstance(rec, bool):
            rec = {'enabled': rec}
        if not isinstance(rec, dict):
            continue  # registry metadata
        enabled = rec.get('enabled', rec.get('status') == 'accepted')
        if type(enabled) is not bool:
            raise ValueError(f'component {name}: enabled must be boolean')
        if enabled:
            out[name] = rec
    return out


def crossover_model(stats, cal):
    rec = cal['planner_v2']['hybrid_crossover']
    per_byte = rec['cpu_expert_us'] / rec['reference_expert_bytes']
    fixed = rec['extra_token_us'] / rec['extra_handoff_layers']
    size = stats['expert_bytes'] / (stats['n_layer'] * stats['n_expert'])
    threshold = fixed / per_byte
    return dict(source=rec['source'], expert_bytes=size, cpu_us_per_byte=per_byte,
                per_layer_fixed_us=fixed, crossover_expert_bytes=threshold,
                crossover_expert_mib=threshold/MIB,
                required_margin=rec['required_margin'],
                predicted_beneficial=size*per_byte > rec['required_margin']*fixed,
                note='Necessary amortization condition only; routing savings and GPU qualification still required')


def prefill_prediction(cal, family, coord, prompt_tokens, ctx, flash):
    """Fit prefill us/token independently of decode, using measured prompt counts."""
    source_family = 'lru' if family == 'lru' else 'static'
    points = [r for r in cal['points'] if r['family'] == source_family]
    xs = sorted({r['x'] for r in points})
    ys = [statistics.median(1e6/r['prefill_tps'] if r.get('prefill_tps') else
                           r['prompt_ms']*1000/r['prompt_tokens']
                           for r in points if r['x'] == x) for x in xs]
    a, b = line(xs, ys)
    us = max(1e-6, a+b*(coord if family != 'hybrid' else 24))
    source = 'D022 measured prefill_tps (62-token chat); linear prompt-length hypothesis'
    d028 = cal['planner_v2']['flash_attention']
    if ctx >= 8192:
        # Transfer the independently measured CPU-placement penalty to D028's
        # long-prompt GPU endpoint. Context/prompt/placement transfer is unvalidated.
        baseline = 1e6/d028['prefill_tps_on' if flash else 'prefill_tps_off']
        us = baseline + max(0, us-a)
        source = 'D028 1560-token prefill endpoint + D022 placement us/token penalty; extrapolated'
    return dict(prefill_tps=1e6/us, prefill_ms=prompt_tokens*us/1000, source=source)


def pareto_front(candidates):
    """Minimize GPU and risk, maximize decode; unknown risk is +infinity."""
    def values(p):
        return p['gpu_mib'], -p['decode_tps'], p['quality_risk'] if p['quality_risk'] is not None else math.inf
    frontier = []
    for p in sorted(candidates, key=lambda p: (*values(p), p['id'])):
        v = values(p)
        if any(all(a <= b for a, b in zip(values(q), v)) and
               any(a < b for a, b in zip(values(q), v)) for q in frontier):
            continue
        frontier = [q for q in frontier if not (
            all(a <= b for a, b in zip(v, values(q))) and any(a < b for a, b in zip(v, values(q))))]
        frontier.append(p)
    return frontier


def load_quality_ledger(source=None):
    """Load explicit evidence; an empty ledger gives no quality permissions."""
    if isinstance(source, dict):
        ledger = source
    else:
        ledger = read_json(HERE/'quality_ledger.json' if source is None else source)
    if ledger.get('schema') != 'quality-ledger/1':
        raise ValueError('expected quality-ledger/1')
    for row in ledger['entries']:
        if row['metric'] not in ('kld', 'top1', 'ppl'):
            raise ValueError('unknown quality metric')
        finite(row['value'], 'ledger metric')
        if not row['source'].get('result_sha256') or not row['source'].get('job_id'):
            raise ValueError('quality entry needs result hash and job')
    return ledger


def resolve_quality(ledger, rep, family, coord, kv, flash, components, *, matched=True,
                    ctx=4096, max_ppl_change=.01):
    """Evidence-scoped screening inheritance, never an additive KLD estimate."""
    finite(max_ppl_change, 'max_ppl_change')
    out = dict(kld=None, status='unknown', rules=[], evidence=[], caveats=[],
               ppl_relative_change=None, ppl_status='unknown')
    def evidence(row):
        src = row['source']
        tag = f"{src['decision']} {src['job_id']} {src['test']}"
        if tag not in out['evidence']:
            out['evidence'].append(tag)
    def rule(name):
        if name not in out['rules']:
            out['rules'].append(name)
    if not matched or family != 'static' or type(coord) is not int or not 0 <= coord <= 24:
        return out
    rows = [r for r in ledger['entries'] if r['representation'] == rep]
    bases = [r for r in rows if r['metric'] == 'kld' and r['kv_type'] == 'f16'
             and not r['components'] and r['placement']['family'] == 'static']
    exact = [r for r in bases if r['placement']['params']['n_cpu_moe'] == coord]
    if exact:
        base = max(exact, key=lambda r: r['value'])
        evidence(base)
    else:
        # Require agreement across every measured placement, not just a pair
        # cherry-picked from conflicting evidence for this representation.
        placements = {r['placement']['params']['n_cpu_moe'] for r in bases}
        values = {r['value'] for r in bases}
        if len(placements) < 2 or len(values) != 1:
            return out
        base = bases[0]
        rule('b')
        for row in bases:
            evidence(row)
    out['caveats'].append(f"KLD is a wikitext screening measurement at ctx{base['ctx']}, {base['n_chunks']} chunks; task/context transfer is unvalidated")
    for component in components:
        if component == 'kv_quant_fix':
            # D033 measures both the f16 control and quantized variants with this
            # scheduler capacity correction. It does not certify arbitrary fixes.
            controls = [r for r in rows if r['metric'] == 'ppl' and r['kv_type'] == 'f16'
                        and r['components'] == ['kv_quant_fix'] and r['source']['decision'] == 'D033']
            if not controls:
                return out
            evidence(controls[0])
            rule('c')
        else:
            proofs = [p for p in ledger.get('computation_preserving', [])
                      if p['component'] == component and p['representation'] == rep
                      and p.get('text_identity') and p.get('ppl_identity')]
            if not proofs:
                return out
            proof = proofs[0]
            rule('a')
            src = proof['source']
            out['evidence'].append(f"{src['decision']} {src['job_id']} paired text SHA256 + PPL identity")
    if kv != 'f16':
        controls = [r for r in rows if r['metric'] == 'ppl' and r['kv_type'] == 'f16'
                    and r['source']['decision'] == 'D033']
        variants = [r for r in rows if r['metric'] == 'ppl' and r['kv_type'] == kv
                    and r['source']['decision'] == 'D033']
        if not controls or not variants or 'kv_quant_fix' not in components:
            return out
        control, variant = controls[0], variants[0]
        delta = variant['value']/control['value']-1
        out['ppl_relative_change'] = delta
        evidence(control)
        evidence(variant)
        rule('c')
        if abs(delta) > max_ppl_change:
            out.update(status='quality-failing', ppl_status='failing')
            return out
        # Only q8 has the D033 inheritance permission. A loose PPL budget does
        # not create a KLD measurement for q4 or another KV representation.
        if kv != 'q8_0':
            return out
        out['ppl_status'] = 'within-budget'
        out['caveats'].append('KV KLD inherited by PPL policy; combined KLD was not measured')
    if flash:
        fa_ppl = [r for r in rows if r['metric'] == 'ppl' and r['source']['decision'] == 'D033' and r['fa']]
        if not fa_ppl:
            return out
        rule('d')
        evidence(fa_ppl[0])
        out['caveats'].append('PPL measured with FA only; no non-FA PPL comparison (D033); caveat, not a block')
    out['kld'] = base['value']
    out['rules'].sort()
    out['status'] = 'inherited' if out['rules'] else 'measured'
    out['description'] = ('inherited (rule ' + ','.join(out['rules']) + '; evidence ' +
                          '; '.join(out['evidence']) + ')') if out['rules'] else 'measured'
    return out


def plan(raw, host, budget, *, allow_extrapolation=False, top=5, sample=None,
         all_representations=True, cal=None, prompt_tokens=62, components=None, quality_ledger=None):
    ledger = load_quality_ledger(quality_ledger)
    cal = cal or read_json(HERE / 'calibration.json')
    s = adapt_stats(raw)
    enabled = load_components(components)
    crossover = crossover_model(s, cal)
    ctx = budget['ctx']
    finite(ctx, 'ctx', positive=True)
    if int(ctx) != ctx or ctx > s['context_length']:
        raise ValueError('ctx must be integral and no greater than model context_length')
    ctx = int(ctx)
    if type(top) is not int or top < 1:
        raise ValueError('top must be a positive integer')
    finite(prompt_tokens, 'prompt_tokens', positive=True)
    if int(prompt_tokens) != prompt_tokens or prompt_tokens >= ctx:
        raise ValueError('prompt_tokens must be integral and less than ctx')
    prompt_tokens = int(prompt_tokens)
    objective = budget.get('objective', 'decode')
    if objective not in ('decode', 'ttft', 'balanced'):
        raise ValueError('objective must be decode, ttft, or balanced')
    finite(budget.get('max_ppl_change', .01), 'max_ppl_change')
    max_kld = budget.get('max_quality_loss')
    if max_kld is not None:
        finite(max_kld, 'max_quality_loss')
    gb = finite(budget['gpu_mib'], 'gpu_mib', positive=True)
    hb = finite(budget.get('host_mib', host['ram_bytes']/MIB), 'host_mib', positive=True)
    gb = min(gb, finite(host['gpu']['memory_mib'], 'host GPU capacity', positive=True))
    hb = min(hb, finite(host['ram_bytes'], 'host RAM', positive=True)/MIB)
    gb, hb, warnings = resource_limits(sample, gb, hb)
    report = residual_report(cal)
    fits = {f: report[f]['coefficients'] for f in ('static', 'lru')}
    same_host = (host.get('host_id') == cal['host_id'] and
                 host.get('memory_model') == cal['conditions']['memory_model'] and
                 host.get('gpu', {}).get('model') == cal['conditions']['gpu'] and
                 host.get('cpu', {}).get('model') == cal['conditions']['cpu'])
    candidates, rejected = [], []
    for v in variants(s, cal) if all_representations else [s]:
        rep = v['representation']
        ref = cal['representations'].get(rep)
        matched = (ref and v['arch'] == cal['arch'] and v['n_layer'] == 24 and v['n_expert'] == 32
                   and v['n_expert_used'] == 8 and Path(v['model_path']).name == ref['model_file'] and
                   abs(v['expert_bytes']/ref['expert_bytes']-1) < .001 and
                   abs(v['nonexpert_bytes']/ref['nonexpert_bytes']-1) < .001 and
                   abs(v['kv_bytes_per_token']['f16']-49152) < 1)
        quality_kld = next((r['value'] for r in ledger['entries']
                            if matched and r['representation'] == rep and r['metric'] == 'kld'
                            and r['placement'] == {'family': 'static', 'params': {'n_cpu_moe': 0}}
                            and not r['components'] and r['kv_type'] == 'f16'), None)
        # Every layer count is interpolation; LRU only measured cache sizes.
        placements = [('static', k, None) for k in range(v['n_layer']+1)]
        placements += [('lru', c, None) for c in (128, 256, 384, 640)]
        if rep == 'mixE4K' and matched:
            for spec in cal.get('hybrid_specs', []):
                path = ROOT / spec['model_path']
                if path.is_file():
                    placements.append(('hybrid', sum(spec['hot_counts']), str(path)))
        caches = [(k, z) for k, z in v['kv_bytes_per_token'].items() if k in ('f16', 'q8_0', 'q4_0')]
        for family, coord, split_path in placements:
            for kv, kv_bpt in caches:
                # Batch/ubatch alternatives are proposed tests, never calibrated improvements.
                for batch, ubatch in ((2048, 512), (512, 128), (512, 512)):
                    reasons, notes = [], []
                    flash = ctx >= 8192 or kv != 'f16'
                    component_ids = sorted(k for k in enabled if k in ('q8_reuse', 'kv_quant_fix'))
                    ident = f'{rep}-{family}{coord}-{kv}-c{ctx}-b{batch}-u{ubatch}'
                    if flash:
                        ident += '-fa'
                    if component_ids:
                        ident += '-cmp-' + '-'.join(component_ids)
                    if kv != 'f16' and 'kv_quant_fix' not in enabled:
                        rejected.append(dict(id=ident, reasons=['D028: quantized KV aborted; requires enabled component kv_quant_fix']))
                        continue
                    if family == 'hybrid' and not crossover_model(v, cal)['predicted_beneficial']:
                        rejected.append(dict(id=ident, reasons=['D030/D031: hybrid rejected in small-expert regime; below crossover margin']))
                        continue
                    if not same_host:
                        reasons.append('host differs from discrete x86-main calibration')
                    if not matched:
                        reasons.append('model stats differ from calibrated Granite representation')
                    if rep != 'mixE4K':
                        reasons.append('representation timing/placement transferred from mixE4K')
                    if ctx != 4096:
                        reasons.append('context outside measured singleton 4096')
                    if kv != 'f16':
                        reasons.append('quantized KV: speed extrapolated; quality resolved separately from ledger')
                    if (batch, ubatch) != (2048, 512):
                        reasons.append('batch/ubatch memory scaling and prefill unmeasured')
                    if prompt_tokens != 62:
                        reasons.append('prompt length outside measured singleton 62')
                    if family == 'lru' and coord / (v['expert_bytes']/MIB) < .60:
                        rejected.append(dict(id=ident, reasons=['D022/D019 policy: LRU resident-byte coverage <60% on discrete PCIe']))
                        continue
                    if family == 'hybrid':
                        reasons.append('hybrid end-to-end GPU speed/memory/quality unmeasured')
                    if reasons and not allow_extrapolation:
                        rejected.append(dict(id=ident, reasons=['requires --allow-extrapolation'] + reasons))
                        continue
                    base = predict_fit(fits['lru' if family == 'lru' else 'static'],
                                       coord if family != 'hybrid' else 0)
                    base_exp = cal['representations']['mixE4K']['expert_bytes']/MIB
                    exp, non = v['expert_bytes']/MIB, v['nonexpert_bytes']/MIB
                    kv_mib = ctx*kv_bpt/MIB
                    compute_mib = 108*ubatch/512
                    if family == 'static':
                        resident = exp*(v['n_layer']-coord)/v['n_layer']
                        # Preserve D022 intercept and slope, plus analytical byte deltas.
                        gpu = base['gpu_mib'] + (resident-base_exp*(24-coord)/24) + non-cal['representations']['mixE4K']['nonexpert_bytes']/MIB
                        decode_us = base['decode_us']
                        if rep != 'mixE4K' or not matched:
                            screen = cal['representation_screen'].get(rep, {}).get('decode_tps', 684)
                            decode_us = 1e6/screen + coord*(base['decode_us']-predict_fit(fits['static'], 0)['decode_us'])/max(coord, 1)*exp/base_exp
                        flags = ['--n-cpu-moe', str(coord)] if coord else []
                    elif family == 'lru':
                        resident = min(exp, coord)
                        gpu = base['gpu_mib'] + non-cal['representations']['mixE4K']['nonexpert_bytes']/MIB
                        decode_us = base['decode_us'] * exp/base_exp
                        flags = ['--n-cpu-moe', str(v['n_layer']), '--moe-cache-mib', str(coord)]
                        notes.append('LRU log-time fit is weak; prefer static at matched memory (D022)')
                    else:
                        # Null bank per layer is mandatory in hybrid_split; includes its bytes.
                        slot = exp/(v['n_expert']*v['n_layer'])
                        resident = (coord+v['n_layer'])*slot
                        gpu = predict_fit(fits['static'], 24)['gpu_mib'] + resident
                        spec = next(z for z in cal['hybrid_specs'] if sum(z['hot_counts']) == coord)
                        hit, trace_note = hybrid_hit_rate(v, spec)
                        hybrid_cost = crossover_model(v, cal)
                        decode_us = predict_fit(fits['static'], 0)['decode_us'] + hybrid_cost['cpu_us_per_byte']*slot*MIB*v['n_layer']*v['n_expert_used']*(1-hit) + hybrid_cost['per_layer_fixed_us']*v['n_layer']
                        flags = ['-ot', r'\.cold$=CPU', '--no-op-offload', '--no-repack']
                        notes += [trace_note, 'D030/D031 effective per-layer dispatch/handoff cost included; larger-expert transfer remains unmeasured.',
                                  'D027 amendment: focused CUDA correctness test failed; exploratory until repaired and requalified.']
                    gpu += kv_mib-192 + compute_mib-108
                    attention_delta = 0.0
                    d028 = cal['planner_v2']['flash_attention']
                    if ctx >= 8192:
                        scale = (ctx-4096)/(d028['ctx']-4096)
                        off_delta = d028['gpu_mib_off'] - (predict_fit(fits['static'], 0)['gpu_mib'] + (d028['ctx']-4096)*v['kv_bytes_per_token']['f16']/MIB)
                        attention_delta = (off_delta - (d028['gpu_mib_off']-d028['gpu_mib_on']))*scale
                        gpu += attention_delta
                        decode_us /= d028['decode_tps_on']/d028['decode_tps_off']
                        notes.append('D028: -fa on default at ctx >=8192; f16 memory/decode transfer is extrapolated; FA PPL evidence/caveat resolved separately from ledger')
                    if 'q8_reuse' in enabled:
                        decode_us /= cal['planner_v2']['q8_reuse']['decode_factor']
                        notes.append('D029: enabled q8_reuse multiplies decode by 1.02; prefill unchanged; placement/context transfer unmeasured')
                    # Conservative file-backed working set including load; not warm cgroup RSS.
                    host_mib = exp+non+400+(v['n_layer']*exp/(v['n_expert']*v['n_layer']) if family == 'hybrid' else 0)
                    guard = report['lru' if family == 'lru' else 'static']['max_loo_gpu_mib']
                    prefill = prefill_prediction(cal, family, coord, prompt_tokens, ctx, flash)
                    prompt_ms = prefill['prefill_ms']
                    ttft_ms = prompt_ms + decode_us/1000
                    quality = resolve_quality(ledger, rep, family, coord, kv, flash, component_ids,
                                              matched=bool(matched), ctx=ctx,
                                              max_ppl_change=budget.get('max_ppl_change', .01))
                    kld = quality['kld']
                    notes.append('Host estimate = complete mapped weights +400 MiB D015 anonymous allowance; load peak/working-set hypothesis, not measured RSS')
                    notes += quality['caveats']
                    if max_kld is not None and (kld is None or kld > max_kld):
                        rejected.append(dict(id=ident, quality=quality, reasons=['strict KLD gate: combined quality unknown, quality-failing, or exceeds max_quality_loss']))
                        continue
                    if gpu+guard > gb or host_mib > hb:
                        rejected.append(dict(id=ident, reasons=['GPU estimate + LOO guard exceeds budget' if gpu+guard > gb else 'host working set exceeds budget (D015 page-cache cliff)']))
                        continue
                    if reasons:
                        notes += reasons
                    model_path = split_path or v['model_path']
                    flags = ['-m', model_path, '-ngl', '99', '--fit', 'off', '-c', str(ctx),
                             '-b', str(batch), '-ub', str(ubatch), '-ctk', kv, '-ctv', kv,
                             '-t', '8', '-tb', '8'] + flags
                    if flash:
                        flags += ['-fa', 'on']
                    confidence = 'extrapolated' if reasons else 'calibrated'
                    p = dict(id=ident, representation=rep, placement=family, gpu_mib=round(gpu,3),
                             gpu_guard_mib=round(guard,3), host_mib=round(host_mib,3),
                             decode_tps=round(1e6/decode_us,3), ttft_ms=round(ttft_ms,3),
                             ttft_definition='warm prompt-eval + one decode proxy; excludes load, tokenizer, queue and sampling',
                             quality_class=quality['status'], quality=quality,
                             kld=kld, representation_kld=quality_kld,
                             quality_risk=kld, quality_risk_unknown=kld is None,
                             kv_type=kv, kv_mib=round(kv_mib,3), flash_attention=flash,
                             attention_delta_mib=round(attention_delta,3), components=component_ids,
                             runtime_env={'GGML_CUDA_Q8_REUSE':'1'} if 'q8_reuse' in enabled else {},
                             technique_family='|'.join([rep, family, kv, 'fa-on' if flash else 'fa-off', ','.join(component_ids) or 'base']),
                             prefill_tps=round(prefill['prefill_tps'],3), prefill_ms=round(prompt_ms,3),
                             prefill_source=prefill['source'],
                             techniques=[rep, family, f'kv:{kv}', f'ctx:{ctx}', f'batch:{batch}', f'ubatch:{ubatch}'],
                             confidence=confidence, metric_confidence=dict(gpu_mib=confidence, decode_tps=confidence, host_mib='extrapolated', prefill_tps='proxy', ttft_ms='proxy'),
                             flags=flags, model_path=model_path, notes=notes,
                             required_engine='hybrid_split patch + pinned base; CUDA qualification required' if family == 'hybrid' else 'llama.cpp 31288c9c6',
                             prerequisites=['Check mechanism engagement (D024)', 'Measure shim peak AND baseline-subtracted device peak (D023)', 'Paired quality check before accepting drift; 3 alternated timing rounds'])
                    if family == 'hybrid':
                        p['hot_counts_per_layer'] = spec['hot_counts']
                        p['routing_hit_estimate'] = hit
                        p['prerequisites'].insert(0, 'Resolve D027 CUDA correctness failure; pass test-hybrid-split and GET_ROWS i32 on CUDA')
                    if 'q8_reuse' in enabled:
                        p['metric_confidence']['decode_tps'] = 'extrapolated-component-transfer'
                        p['required_engine'] += ' + q8_reuse patch (D029)'
                    if 'kv_quant_fix' in enabled:
                        p['required_engine'] += ' + kv_quant_fix patch (D033)'
                    candidates.append(p)
    def rank(p):
        if objective == 'ttft':
            score = -p['ttft_ms']
        elif objective == 'balanced':
            score = math.log(p['decode_tps'])-math.log(p['ttft_ms'])
        else:
            score = p['decode_tps']
        # Measured regime precedes unvalidated hypotheses regardless of predicted speed.
        return (p['confidence'] != 'calibrated', -score, p['gpu_mib'], p['id'])
    candidates.sort(key=rank)
    best = {}
    for p in candidates:
        best.setdefault(p['technique_family'], p)
    family_best = list(best.values())
    pareto = sorted(pareto_front(candidates), key=rank)
    if len(family_best) < top:
        warnings.append(f'Only {len(family_best)} feasible technique families; top is a maximum, not a quota. Use --allow-extrapolation to consider unmeasured alternatives.')
    return dict(schema='moe-plan/0', prediction_only=True, prompt_tokens=prompt_tokens, budget=dict(gpu_mib=gb, host_mib=hb, ctx=ctx, max_quality_loss=max_kld, objective=objective),
                plans=family_best[:top], family_best=family_best, family_count=len(family_best),
                pareto_set=pareto, pareto_axes=['gpu_mib:min', 'decode_tps:max', 'quality_risk:min'],
                quality_risk_policy='measured or evidence-inherited screening KLD; null is unknown/failing and treated as +infinity, never zero',
                hybrid_crossover=crossover, enabled_components=sorted(enabled),
                technique_catalog=cal['planner_v2']['techniques'],
                feasible_count=len(candidates), rejected=rejected, calibration_residuals=report,
                warnings=warnings+['No hard VRAM guarantee; predictions require measurement.',
                                   'Calibrated confidence describes GPU/decode regime; host is analytical and TTFT is a warm proxy.',
                                   'Inherited KLD is an evidence-scoped screening prior; PPL acceptance is not a measured combined KLD bound.',
                                   'Missing TTFT/host calibration cannot be resolved by --allow-extrapolation.'])


def emit_lab_tests(result):
    """Exact bin/lab definition mapping; model path resides in each job's model directory."""
    if result['prompt_tokens'] != 62:
        raise ValueError('lab tests use the D022 62-token chat fixture; cannot emit another prompt length')
    tests = {}
    for p in result['plans']:
        tag = 'pl_'+hashlib.sha256(p['id'].encode()).hexdigest()[:12]
        flags = p['flags']
        assert flags[:1] == ['-m']
        common = flags[2:]
        remaining = result['budget']['ctx'] - result['prompt_tokens']
        for phase, n in (('engagement', min(8, remaining)), ('timing', min(160, remaining))):
            tests[f'{tag}_{phase}'] = dict(tool='llama-completion', parser='completion',
                model_file=Path(p['model_path']).name,
                docker_args=[item for k,v in p.get('runtime_env', {}).items() for item in ('-e', f'{k}={v}')]+['-v', str(ROOT / 'artifacts/traces/in')+':/tin:ro'],
                args=common+['-f', '/tin/chat.txt', '-n', str(n), '--temp', '0', '-s', '1', '-no-cnv', '--no-warmup'])
        if p['representation_kld'] is not None and (ROOT/'artifacts/kl/bf16.kl').is_file():
            # Existing reference logits are 20x512. Do not claim this tests ctx=4096.
            quality_flags = list(common)
            quality_flags[quality_flags.index('-c')+1] = '512'
            tests[f'{tag}_quality512'] = dict(tool='llama-perplexity', parser='kl', needs_data=True,
                model_file=Path(p['model_path']).name,
                docker_args=[item for k,v in p.get('runtime_env', {}).items() for item in ('-e', f'{k}={v}')]+['-v', str(ROOT/'artifacts/kl')+':/planner-kl:ro'],
                args=quality_flags+['--chunks', '20', '--kl-divergence', '--kl-divergence-base', '/planner-kl/bf16.kl'])
    return tests


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stats', required=True)
    a.add_argument('--host', required=True)
    a.add_argument('--budget-gpu-mib', type=float, required=True)
    a.add_argument('--budget-host-mib', type=float)
    a.add_argument('--ctx', type=int, required=True)
    a.add_argument('--max-kld', type=float)
    a.add_argument('--quality-ledger', help='Quality evidence JSON (default: bundled quality_ledger.json)')
    a.add_argument('--objective', choices=('decode', 'ttft', 'balanced'), default='decode')
    a.add_argument('--top', type=int, default=5)
    a.add_argument('--allow-extrapolation', action='store_true')
    a.add_argument('--only-input-representation', action='store_true')
    a.add_argument('--resource-sample')
    a.add_argument('--components', nargs='?', const=str(ROOT/'cc/factory/components.json'), help='Component registry JSON (default: cc/factory/components.json if present); pending entries require enabled=true')
    a.add_argument('--prompt-tokens', type=int, default=62)
    a.add_argument('--emit-lab-tests')
    args = a.parse_args(argv)
    try:
        budget = dict(gpu_mib=args.budget_gpu_mib, ctx=args.ctx, max_quality_loss=args.max_kld, objective=args.objective)
        if args.budget_host_mib is not None:
            budget['host_mib'] = args.budget_host_mib
        result = plan(read_json(args.stats), read_json(args.host), budget,
                      allow_extrapolation=args.allow_extrapolation, top=args.top,
                      sample=read_json(args.resource_sample) if args.resource_sample else None,
                      all_representations=not args.only_input_representation, prompt_tokens=args.prompt_tokens,
                      components=args.components, quality_ledger=args.quality_ledger)
        if args.emit_lab_tests:
            dest = Path(args.emit_lab_tests).resolve()
            # Assignment scope also constrains paths requested through this CLI.
            if not any(dest.is_relative_to(base) for base in (HERE, ROOT/'artifacts/codexPL3')):
                raise ValueError('--emit-lab-tests must stay under cc/moe_planner or artifacts/codexPL3')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(emit_lab_tests(result), indent=2)+'\n')
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        a.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
