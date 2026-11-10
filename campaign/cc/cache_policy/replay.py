#!/usr/bin/env python3
"""Offline layout-group cache replay; see README for source citations and limits."""
import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
POLICIES = ('LRU', 'SLRU*', 'ClassLRU', 'Static')
MIB = 2**20


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


class Cache:
    """One layout bank, batch-one plans; slot order matches pinned ClassLRU."""
    def __init__(self, capacity, policy, static=(), share=0):
        if capacity < 8 or policy not in POLICIES or not 0 <= share <= 1:
            raise ValueError('invalid capacity, policy, or share')
        self.capacity, self.policy = capacity, policy
        self.share_q = int(share * 65536 + .5)
        self.carry = defaultdict(int)
        self.clock = defaultdict(int)
        self.crf, self.last = {}, {}
        self.maps = {}
        self.keys = [None] * capacity
        self.lists = [OrderedDict() for _ in range(8)]
        self.classes = [0] * capacity
        self.lists[0].update((s, None) for s in range(capacity))
        self.filled = 0
        self.static = set(static)

    def move(self, slot, cls):
        del self.lists[self.classes[slot]][slot]
        self.lists[cls][slot] = None
        self.classes[slot] = cls

    def victim(self, selected):
        if self.policy == 'SLRU*':
            # Reconstruction: empty/probationary LRU, then protected LRU.
            order = (0, 1)
        else:
            order = range(8) if self.policy == 'ClassLRU' else (0,)
        best, best_score = None, -1
        for cls in order:
            slot = next((s for s in self.lists[cls] if self.keys[s] not in selected), None)
            if slot is None:
                continue
            key = self.keys[slot]
            if key is None:
                return slot
            if self.policy != 'ClassLRU':
                return slot
            score = (self.clock[key[0]] - self.last[key]) / 2**cls
            if score > best_score:
                best, best_score = slot, score
        if best is None:
            raise ValueError('no unselected victim')
        return best

    def plan(self, layer, experts):
        if len(set(experts)) != len(experts) or len(experts) > self.capacity:
            raise ValueError('plan requires distinct expert IDs fitting capacity')
        keys = [(layer, e) for e in experts]
        if self.policy == 'Static':
            hits = sum(k in self.static for k in keys)
            misses = len(keys) - hits
            return hits, 0 if self.share_q else misses, misses if self.share_q else 0
        selected = set(keys)
        misses, hits = [], 0
        for key in keys:
            old = self.crf.get(key, 0.)
            decay = f32(math.exp2(-(self.clock[layer] - self.last.get(key, -1)) / 128.))
            self.crf[key] = f32(f32(old * decay) + 1.)
            self.last[key] = self.clock[layer]
            if key in self.maps:
                hits += 1
                slot = self.maps[key]
                cls = min(7, int(math.log2(self.crf[key]))) if self.policy == 'ClassLRU' else 0
                if self.policy == 'SLRU*':
                    cls = 1
                self.move(slot, cls)
                if self.policy == 'SLRU*' and len(self.lists[1]) > self.capacity // 2:
                    self.move(next(iter(self.lists[1])), 0)
            else:
                misses.append(key)
        share = self.share_q if self.filled == self.capacity else 0
        acc = len(misses) * share + self.carry[layer]
        n_cpu, self.carry[layer] = acc >> 16, acc & 0xffff
        # Pinned #31853 chooses the lowest CRF misses; rank tie is expert ID.
        misses.sort(key=lambda k: (self.crf[k], k[1]))
        uploads = sorted(misses[n_cpu:])
        for key in uploads:
            slot = self.victim(selected)
            victim = self.keys[slot]
            if victim is not None:
                del self.maps[victim]
            else:
                self.filled += 1
            self.keys[slot] = key
            self.maps[key] = slot
            cls = min(7, int(math.log2(self.crf[key]))) if self.policy == 'ClassLRU' else 0
            self.move(slot, cls)
        self.clock[layer] += 1
        return hits, len(uploads), n_cpu


def read_trace(path, n_layer, n_expert, top_k):
    """Validate all records and complete contiguous single-sequence phases."""
    rows, seen = [], set()
    positions = defaultdict(lambda: defaultdict(set))
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                for k in ('seq', 'pos', 'layer'):
                    if type(row.get(k)) is not int or row[k] < 0:
                        raise ValueError('invalid identity')
                if row['seq'] != 0 or row['layer'] >= n_layer or row.get('phase') not in ('prefill', 'decode'):
                    raise ValueError('invalid layer, sequence or phase')
                es, ws = row['experts'], row['weights']
                if len(es) != top_k or len(set(es)) != top_k or any(type(e) is not int or not 0 <= e < n_expert for e in es):
                    raise ValueError('invalid top-k experts')
                if len(ws) != top_k or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in ws):
                    raise ValueError('invalid weights')
                identity = row['seq'], row['pos'], row['layer']
                if identity in seen:
                    raise ValueError('duplicate identity')
                seen.add(identity)
                positions[row['phase']][row['pos']].add(row['layer'])
                rows.append(row)
            except (ValueError, KeyError, TypeError) as err:
                raise ValueError(f'{path}:{line_no}: {err}') from err
    if set(positions) != {'prefill', 'decode'}:
        raise ValueError('both phases required')
    for phase, pp in positions.items():
        if sorted(pp) != list(range(min(pp), max(pp) + 1)) or any(ll != set(range(n_layer)) for ll in pp.values()):
            raise ValueError(f'incomplete {phase} phase')
    return sorted(rows, key=lambda r: (r['seq'], r['pos'], r['layer']))


def inspect_model(path):
    spec = importlib.util.spec_from_file_location('model_inspect', ROOT / 'cc/model_inspect/inspect_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect(path)


def groups_from_model(model):
    groups, by_signature = [], {}
    for layer in range(model['n_layer']):
        tensors = [t for t in model['tensors'] if t['role'] == 'expert_ffn' and t['layer'] == layer]
        tensors.sort(key=lambda t: t['name'].split('ffn_', 1)[1])
        signature = tuple((t['ggml_type'], tuple(t['shape']), t['bytes_per_expert']) for t in tensors)
        if signature not in by_signature:
            by_signature[signature] = len(groups)
            groups.append({'layers': [], 'projection_bytes': [t['bytes_per_expert'] for t in tensors],
                           'signature': signature})
        groups[by_signature[signature]]['layers'].append(layer)
    for g in groups:
        g['expert_bytes'] = sum(g['projection_bytes'])
        g['host_bytes'] = g['expert_bytes'] * len(g['layers']) * model['n_expert']
    return groups


def allocate(groups, budget, n_expert=128, share=0):
    host = sum(g['host_bytes'] for g in groups)
    result = []
    for g in groups:
        group_budget = int(budget * g['host_bytes'] / host)
        reserve = 1 + (8 if share else 0)
        def size(n):
            return sum(((b * (n + reserve) + 127) // 128) * 128 for b in g['projection_bytes'])
        slots = min(n_expert * len(g['layers']), group_budget // g['expert_bytes'])
        while slots > 0 and size(slots) > group_budget:
            slots -= 1
        if slots < 8:
            raise ValueError('budget too small')
        result.append({'slots': slots, 'allocated_bytes': size(slots), 'budget_bytes': group_budget})
    return result


def static_sets(groups, allocation, counts, n_expert):
    return [sorted(((l, e) for l in g['layers'] for e in range(n_expert)),
                   key=lambda k: (-counts[k], k))[:a['slots']] for g, a in zip(groups, allocation)]


def simulate(rows, groups, allocation, policy, share, hotsets):
    caches = [Cache(a['slots'], policy, hot, share) for a, hot in zip(allocation, hotsets)]
    group_of = {l: i for i, g in enumerate(groups) for l in g['layers']}
    hits = uploads = cpu = requests = upload_bytes = cpu_bytes = cpu_active = 0
    tokens = set()
    layer_stats = defaultdict(lambda: {'hits': 0, 'requests': 0})
    for row in rows:
        l = row['layer']
        ig = group_of[l]
        h, u, c = caches[ig].plan(l, row['experts'])
        hits += h
        uploads += u
        cpu += c
        requests += len(row['experts'])
        upload_bytes += u * groups[ig]['expert_bytes']
        cpu_bytes += c * groups[ig]['expert_bytes']
        cpu_active += c > 0
        tokens.add((row['seq'], row['pos']))
        layer_stats[l]['hits'] += h
        layer_stats[l]['requests'] += len(row['experts'])
    nt = len(tokens)
    return {'hits': hits, 'requests': requests, 'hit_rate': hits / requests,
            'uploads': uploads, 'cpu_misses': cpu, 'tokens': nt,
            'upload_bytes_per_token': upload_bytes / nt, 'cpu_bytes_per_token': cpu_bytes / nt,
            'cpu_active_layers_per_token': cpu_active / nt, 'layers': dict(layer_stats)}


def predict(stat, protocol, mean_expert_bytes, multiplier=1):
    transfer = stat['upload_bytes_per_token'] / (protocol['h2d_decimal_GB_s'] * 1e9)
    cpu = stat['cpu_bytes_per_token'] / (8 * mean_expert_bytes) * protocol['cpu_full_layer_seconds']
    boundary = stat['cpu_active_layers_per_token'] * protocol['cpu_boundary_seconds']
    seconds = protocol['gpu_floor_seconds'] + multiplier * transfer + cpu + boundary
    return {'tps': 1 / seconds, 'seconds': seconds, 'upload_seconds': transfer,
            'cpu_seconds': cpu, 'boundary_seconds': boundary}


def fit_multiplier(points, base):
    xs = [p['upload_seconds'] for p in points]
    ys = [1 / p['observed_tps'] - base for p in points]
    return max(0., sum(x*y for x, y in zip(xs, ys)) / sum(x*x for x in xs))


def crossings(grid):
    result = []
    for (b0, m0), (b1, m1) in zip(grid, grid[1:]):
        if m0 * m1 < 0:
            result.append({'GB': b0 + (b1 - b0) * (-m0) / (m1 - m0),
                           'direction': 'cache wins' if m1 > 0 else 'static wins',
                           'bracket_GB': [b0, b1]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/codexPOL')
    parser.add_argument('--model', type=Path, default=Path('/home/muk/models/campaign/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf'))
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'artifacts/codexPOL'):
        parser.error('output must be under artifacts/codexPOL')
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / 'artifacts/codexPOL/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    model = inspect_model(args.model)
    (out / 'model-header.json').write_text(json.dumps(model, indent=2) + '\n')
    groups = groups_from_model(model)
    n_layer, n_expert, top_k = (model[k] for k in ('n_layer', 'n_expert', 'n_expert_used'))
    traces, hashes = {}, {}
    for name in ('prose', 'chat', 'code'):
        path = ROOT / 'artifacts/codexQRT' / (name + '.jsonl')
        traces[name] = read_trace(path, n_layer, n_expert, top_k)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(sum(r['phase'] == 'decode' for r in rows) == 64 * n_layer for rows in traces.values())
    counts = Counter((r['layer'], e) for r in traces['prose'] if r['phase'] == 'prefill' for e in r['experts'])
    eval_rows = {n: [r for r in traces[n] if r['phase'] == 'decode'] for n in ('chat', 'code')}
    mean_bytes = sum(g['expert_bytes'] * len(g['layers']) for g in groups) / n_layer
    report = {'schema': 'cache-policy-replay/1', 'protocol': protocol,
              'simulator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'trace_sha256': hashes, 'groups': groups, 'mean_expert_bytes': mean_bytes,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'source').glob('*.cpp')},
              'results': [], 'calibration': [], 'crossovers': {}}
    memo = {}
    def run(budget, policy, mode, names=('chat', 'code')):
        key = budget, policy, mode, names
        if key in memo:
            return memo[key]
        share = protocol['miss_modes'][mode]
        allocation = allocate(groups, budget, n_expert, share)
        hotsets = static_sets(groups, allocation, counts, n_expert)
        domains = {n: simulate(eval_rows[n], groups, allocation, policy, share, hotsets) for n in names}
        stats = list(domains.values())
        pooled = {k: sum(s[k] for s in stats) / len(stats) for k in
                  ('upload_bytes_per_token', 'cpu_bytes_per_token', 'cpu_active_layers_per_token')}
        pooled['hits'] = sum(s['hits'] for s in stats)
        pooled['requests'] = sum(s['requests'] for s in stats)
        pooled['hit_rate'] = pooled['hits'] / pooled['requests']
        result = {'budget_bytes': budget, 'policy': policy, 'mode': mode,
                  'allocation': allocation, 'static_sets': hotsets if policy == 'Static' else None,
                  'domains': domains, 'pooled': pooled}
        memo[key] = result
        return result
    for mib, measured in ((2200, 18.1), (6000, 51.3)):
        s = run(mib * MIB, 'LRU', 'upload', ('chat',))['pooled']
        pred = predict(s, protocol, mean_bytes)
        report['calibration'].append({'MiB': mib, 'observed_tps': measured, 'hit_rate': s['hit_rate'], **pred})
    alpha = fit_multiplier(report['calibration'], protocol['gpu_floor_seconds'])
    for p in report['calibration']:
        fitted = 1 / (protocol['gpu_floor_seconds'] + alpha * p['upload_seconds'])
        other = [q for q in report['calibration'] if q is not p]
        loo = fit_multiplier(other, protocol['gpu_floor_seconds'])
        loo_tps = 1 / (protocol['gpu_floor_seconds'] + loo * p['upload_seconds'])
        p.update(fitted_tps=fitted, residual_tps=fitted - p['observed_tps'],
                 residual_ms=1000 * (1 / fitted - 1 / p['observed_tps']),
                 leave_one_out_tps=loo_tps, leave_one_out_error_tps=loo_tps - p['observed_tps'])
    report['upload_multiplier'] = alpha
    report['fit_rmse_tps'] = math.sqrt(sum(p['residual_tps']**2 for p in report['calibration']) / 2)
    report['fit_rmse_ms'] = math.sqrt(sum(p['residual_ms']**2 for p in report['calibration']) / 2)
    # Whole-layer static D041 fit; discrete K at each exact suffix-byte budget.
    layer_bytes = {l: g['expert_bytes'] * n_expert for g in groups for l in g['layers']}
    cpu_layer = (1 / 27.4 - 1 / 37.1) / 12
    static_base = 1 / 37.1 - 30 * cpu_layer
    def static_time(budget):
        k = next(k for k in range(n_layer + 1) if sum(layer_bytes[l] for l in range(k, n_layer)) <= budget)
        return {'K': k, 'tps': 1 / (static_base + k * cpu_layer), 'seconds': static_base + k * cpu_layer,
                'expert_bytes': sum(layer_bytes[l] for l in range(k, n_layer))}
    report['static_layer_fit'] = {'base_seconds': static_base, 'cpu_layer_seconds': cpu_layer,
                                  'training_rmse_tps': 0., 'degrees_of_freedom': 0}
    for gb in protocol['budgets_decimal_GB']:
        for policy in POLICIES:
            for mode in protocol['miss_modes']:
                item = dict(run(int(gb * 1e9), policy, mode))
                item['budget_GB'] = gb
                item['physics'] = predict(item['pooled'], protocol, mean_bytes)
                item['calibrated'] = predict(item['pooled'], protocol, mean_bytes, alpha)
                item['whole_layer_static'] = static_time(int(gb * 1e9))
                report['results'].append(item)
    grid = [i / 2 for i in range(1, 25)]
    for policy in POLICIES[:-1]:
        for mode in protocol['miss_modes']:
            for comparison in ('whole_layer_static', 'top_frequency_static'):
                margins = []
                for gb in grid:
                    budget = int(gb * 1e9)
                    s = run(budget, policy, mode)['pooled']
                    seconds = predict(s, protocol, mean_bytes, alpha)['seconds']
                    baseline = static_time(budget)['seconds'] if comparison == 'whole_layer_static' else predict(run(budget, 'Static', mode)['pooled'], protocol, mean_bytes, alpha)['seconds']
                    margins.append((gb, baseline - seconds))
                report['crossovers'][f'{policy}/{mode}/{comparison}'] = {'grid_margin_seconds': margins, 'transitions': crossings(margins)}
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_report(report, out)
    print(json.dumps({'fit_rmse_tps': report['fit_rmse_tps'], 'fit_rmse_ms': report['fit_rmse_ms'], 'upload_multiplier': alpha}))


def write_report(report, out):
    lines = ['# Frozen offline cache replay', '',
             'SLRU* is an unverified 50/50 reconstruction: upstream historical blob unavailable. Other policies are source-based. All throughput numbers are predictions.', '',
             'Budgets are decimal GB of cache banks, excluding other model/runtime weights. Each evaluation trace begins cold at decode. Static ranks use prose prefill only. Pooled chat/code decode has equal token weight.', '',
             '| GB | Policy | Upload hit % | CPU-share=1 hit % | Upload t/s physics | Upload t/s fitted | CPU t/s | Whole-layer static t/s |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for gb in report['protocol']['budgets_decimal_GB']:
        for policy in POLICIES:
            up, cpu = [next(r for r in report['results'] if r['budget_GB'] == gb and r['policy'] == policy and r['mode'] == mode) for mode in ('upload', 'CPU')]
            lines.append(f"| {gb} | {policy} | {100*up['pooled']['hit_rate']:.2f} | {100*cpu['pooled']['hit_rate']:.2f} | {up['physics']['tps']:.2f} | {up['calibrated']['tps']:.2f} | {cpu['calibrated']['tps']:.2f} | {up['whole_layer_static']['tps']:.2f} |")
    lines += ['', '## Calibration', '',
              'D041 LRU chat anchors, with fixed GPU floor and H2D=28.7 GB/s. One nonnegative upload multiplier fitted in latency space; CPU forecasts use the same multiplier only for initial uploads.', '',
              '| Cache MiB | Observed t/s | Physics t/s | Fitted t/s | Fit error t/s | Fit error ms | Leave-one-out t/s |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for p in report['calibration']:
        lines.append(f"| {p['MiB']} | {p['observed_tps']:.2f} | {p['tps']:.2f} | {p['fitted_tps']:.2f} | {p['residual_tps']:+.2f} | {p['residual_ms']:+.2f} | {p['leave_one_out_tps']:.2f} |")
    lines += ['', f"Upload multiplier: {report['upload_multiplier']:.5f}; fit RMSE {report['fit_rmse_tps']:.3f} t/s, {report['fit_rmse_ms']:.3f} ms/token.", '',
              '## Crossover predictions', '',
              'Piecewise linear interpolation of latency margins on the preregistered 0.5 GB grid. Whole-layer static K is discrete; crossings are bracketed model forecasts, not hardware measurements. Positive margin means cache wins. All transitions, including reversals, are saved in results.json.', '',
              '| Policy | Miss mode | Against whole-layer static | Against trained top-frequency static |',
              '|---|---|---|---|']
    def cell(key):
        c = report['crossovers'][key]
        if not c['transitions']:
            return 'wins throughout grid' if c['grid_margin_seconds'][0][1] > 0 else 'no crossing; loses throughout grid'
        initial = 'cache wins at 0.5 GB; ' if c['grid_margin_seconds'][0][1] > 0 else ''
        return initial + '; '.join(f"{t['GB']:.3f} GB ({t['direction']}, bracket {t['bracket_GB']})" for t in c['transitions'])
    for policy in POLICIES[:-1]:
        for mode in ('upload', 'CPU'):
            lines.append(f"| {policy} | {mode} | {cell(f'{policy}/{mode}/whole_layer_static')} | {cell(f'{policy}/{mode}/top_frequency_static')} |")
    lines += ['', '## Interpretation and limits', '',
              'The calibration residual and leave-one-out predictions test whether a fixed miss-price model explains both D041 points. Cold 64-token replay omits D041 prefill warming and follows different greedy continuations. The fitted multiplier absorbs exposed copy overhead, synchronization, and model mismatch; it is not a measured PCIe bandwidth. A fit with large residuals does not validate absolute t/s or a precise crossover.', '',
              'CPU-share=1 bypasses every miss once a group fills, so its resident set freezes. Its hit rates must not be replaced with upload-mode hit rates. Partial CPU-share can improve admission and may perform better; no share was tuned on evaluation. CPU cost is D039/T-QRT 0.757210 ms per fully CPU layer, scaled by missed bytes, plus 30 us per CPU-active layer. This is an aggregate CPU calibration, not a pure bandwidth measurement; double-counting some boundary cost is possible. GPU work, CPU computation and uploads are serialized in the forecast; overlap and scheduler overhead are unknown.', '',
              'SLRU* lacks source verification and must not be cited as faithful #30637. LRU and ClassLRU share the pinned grouping/guard allocation for a controlled comparison. LRU/SLRU with CPU-share and Static with upload-on-miss are counterfactual policy/miss-mode combinations, not claims of upstream options. Static top-frequency residents never change, so upload misses are transient staging operations with no resident eviction. Whole-layer static is separately anchored to D041 K30/K42; its two-point fit has zero degrees of freedom and is not independent validation.', '',
              'See cc/cache_policy/README.md for exact source lines, formulas, reproduction, and missing-source evidence. results.json includes per-domain and per-layer counts, allocations, selected static IDs, source/trace/protocol hashes and crossover margins.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
