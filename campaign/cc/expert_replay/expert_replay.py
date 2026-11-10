#!/usr/bin/env python3
"""Generic logical expert-cache replay; Python 3.10 standard library only."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


def integer(x, minimum=0):
    if type(x) is not int or x < minimum:
        raise ValueError(f'expected integer >= {minimum}, got {x!r}')
    return x


def read_trace(path, layers=None, experts=None):
    rows, metadata, errors, seen = [], {}, [], set()
    truncated = 0
    with Path(path).open(encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('expected JSON object')
                if obj.get('type') == 'metadata':
                    for name in ('layers', 'experts'):
                        if name in obj:
                            value = integer(obj[name], 1)
                            if name in metadata and metadata[name] != value:
                                raise ValueError('conflicting metadata')
                            metadata[name] = value
                    continue
                seq = integer(obj.get('seq', obj.get('request', 0)))
                tok = integer(obj.get('pos', obj.get('tok')))
                layer = integer(obj.get('layer'))
                ids = obj.get('experts')
                if not isinstance(ids, list) or not ids:
                    raise ValueError('experts must be a nonempty list')
                ids = [integer(e) for e in ids]
                if len(set(ids)) != len(ids):
                    raise ValueError('duplicate expert IDs')
                if obj.get('phase') not in ('prefill', 'decode'):
                    raise ValueError('phase must be prefill or decode')
                if 'weights' in obj:
                    ws = obj['weights']
                    if (not isinstance(ws, list) or len(ws) != len(ids) or
                        any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in ws)):
                        raise ValueError('invalid aligned weights')
                key = (seq, tok, layer)
                if key in seen:
                    raise ValueError('duplicate sequence/token/layer')
                seen.add(key)
                rows.append(dict(seq=seq, tok=tok, layer=layer, experts=ids, phase=obj['phase']))
            except (ValueError, TypeError) as exc:
                tail = not line.endswith('\n')
                truncated += int(tail)
                errors.append(dict(line=number, error=str(exc), possible_truncated_tail=tail))
    layers = layers if layers is not None else metadata.get('layers')
    experts = experts if experts is not None else metadata.get('experts')
    integer(layers, 1)
    integer(experts, 1)
    if any(metadata.get(k, v) != v for k, v in [('layers', layers), ('experts', experts)]):
        raise ValueError('flags conflict with trace metadata')
    valid = []
    for row in rows:
        if row['layer'] >= layers or any(e >= experts for e in row['experts']):
            # Bounds violations are fatal: never silently simulate the wrong model.
            raise ValueError(f'expert/layer ID outside declared dimensions: {row}')
        valid.append(row)
    if not valid:
        raise ValueError(f'no valid routing records; malformed_lines={len(errors)}, possible_truncated_tail={truncated}')
    valid.sort(key=lambda r: (r['seq'], r['tok'], r['layer']))
    phases = {}
    for r in valid:
        key = (r['seq'], r['tok'])
        if phases.setdefault(key, r['phase']) != r['phase']:
            raise ValueError('inconsistent phases within a token')
    coverage = Counter((r['seq'], r['tok']) for r in valid)
    incomplete = sum(n != layers for n in coverage.values())
    return valid, layers, experts, dict(incomplete_layer_tokens=incomplete,
                                       malformed_lines=len(errors), possible_truncated_tail=truncated,
                                       errors=errors, valid_records=len(valid))


def token_keys(rows):
    return list(dict.fromkeys((r['seq'], r['tok']) for r in rows))


def partition(rows, split):
    if split is None:
        return [], rows
    try:
        train, test = map(float, split.split(':'))
        if not all(math.isfinite(x) for x in (train, test)) or train < 0 or test <= 0:
            raise ValueError()
    except ValueError:
        raise ValueError('split must be nonnegative-train:positive-test weights') from None
    keys = token_keys(rows)
    cut = int(len(keys) * train / (train + test))
    training = set(keys[:cut])
    return ([r for r in rows if (r['seq'], r['tok']) in training],
            [r for r in rows if (r['seq'], r['tok']) not in training])


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def make_profile(rows, layers, experts, weighting):
    totals = Counter(r['seq'] for r in rows for _ in r['experts'])
    counts = defaultdict(float)
    for r in rows:
        for e in r['experts']:
            counts[r['layer'], e] += 1 if weighting == 'raw' else 1 / totals[r['seq']]
    ranked = sorted(((l, e) for l in range(layers) for e in range(experts)),
                    key=lambda p: (-counts[p], p))
    return dict(schema_version=1, layers=layers, experts=experts, weighting=weighting,
                ranked=[list(p) for p in ranked], scores=[counts[p] for p in ranked],
                training_records=[fingerprint(r) for r in rows])


def replay(rows, layers, experts, capacity, policy='lru', per_layer=False, warmup=0,
           expert_bytes=0, layer_bytes=None, bw_gbs=1.0, compute_ms=0.0, profile=None):
    integer(capacity)
    integer(warmup)
    if policy not in ('lru', 'lfu', 'belady', 'static-profile') and not policy.startswith('lfu-decay:'):
        raise ValueError('unknown policy')
    half = float(policy.split(':', 1)[1]) if policy.startswith('lfu-decay:') else None
    if half is not None and (not math.isfinite(half) or half <= 0):
        raise ValueError('half-life must be finite and positive')
    if not math.isfinite(bw_gbs) or bw_gbs <= 0 or not math.isfinite(compute_ms) or compute_ms < 0:
        raise ValueError('bandwidth must be positive and compute time nonnegative')
    integer(expert_bytes)
    sizes = [expert_bytes] * layers if layer_bytes is None else layer_bytes
    if len(sizes) != layers:
        raise ValueError('per-layer bytes must have exactly L entries')
    for size in sizes:
        integer(size)
    keys = token_keys(rows)
    unscored = set(keys[:warmup])
    events = [(r, (r['layer'], e)) for r in rows for e in r['experts']]
    future = defaultdict(deque)
    for i, (_, pair) in enumerate(events):
        future[pair].append(i)
    caches = defaultdict(dict)
    selected = set()
    if policy == 'static-profile':
        if profile is None or (profile.get('layers'), profile.get('experts')) != (layers, experts):
            raise ValueError('profile dimensions mismatch or profile missing')
        used = Counter()
        seen_pairs = set()
        for item in profile['ranked']:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError('invalid profile pair')
            l, e = item
            integer(l); integer(e)
            if l >= layers or e >= experts or (l, e) in seen_pairs:
                raise ValueError('invalid/duplicate profile pair')
            seen_pairs.add((l, e))
            scope = l if per_layer else 0
            if used[scope] < capacity:
                selected.add((l, e)); used[scope] += 1
    stats = defaultdict(lambda: dict(hits=0, requests=0, bytes_moved=0, tokens=set()))
    for i, (r, pair) in enumerate(events):
        future[pair].popleft()
        cache = caches[r['layer'] if per_layer else 0]
        hit = pair in cache
        if (r['seq'], r['tok']) not in unscored:
            for name in ('overall', 'decode' if r['phase'] == 'decode' else 'prefill', f"layer:{r['layer']}"):
                s = stats[name]
                s['requests'] += 1; s['hits'] += hit
                s['bytes_moved'] += 0 if hit else sizes[r['layer']]
                s['tokens'].add((r['seq'], r['tok']))
        if hit:
            freq, last = cache[pair]
            if half is not None:
                freq *= 2 ** (-(i - last) / half)
            cache[pair] = (freq + 1, i)
        elif capacity and (policy != 'static-profile' or pair in selected):
            if len(cache) >= capacity:
                if policy == 'belady':
                    # Include incoming pair: bypass permits the oracle to dominate pinned policies.
                    victim = max([*cache, pair], key=lambda p: future[p][0] if future[p] else math.inf)
                    if victim == pair:
                        continue
                elif policy == 'lru':
                    victim = min(cache, key=lambda p: cache[p][1])
                else:
                    victim = min(cache, key=lambda p: (cache[p][0] * (2 ** (-(i-cache[p][1])/half) if half else 1), cache[p][1]))
                del cache[victim]
            cache[pair] = (1.0, i)
    def finish(s):
        n, h, t = s['requests'], s['hits'], len(s['tokens'])
        misses = n - h
        return dict(hits=h, requests=n, misses=misses, tokens=t, hit_rate=h/n if n else None,
                    misses_per_token=misses/t if t else None, bytes_moved=s['bytes_moved'],
                    bytes_moved_per_token=s['bytes_moved']/t if t else None)
    overall = finish(stats['overall'])
    decode = finish(stats['decode'])
    prediction = (compute_ms / 1000 + decode['bytes_moved_per_token'] / (bw_gbs * 1e9)
                  if decode['tokens'] else None)
    return dict(policy=policy, oracle_label='reads-the-future upper bound' if policy == 'belady' else None,
                capacity_experts=capacity, per_layer_capacity=per_layer, warmup_tokens=len(unscored),
                overall=overall, decode_only=decode,
                per_layer={str(l): finish(stats[f'layer:{l}']) for l in range(layers)},
                prediction=dict(decode_seconds_per_token=prediction, bw_bytes_s=bw_gbs*1e9,
                                compute_s=compute_ms/1000),
                semantics='sequential expert lookups in recorded rank order; cold demand admission; persistent across phases/requests')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('simulate', 'profile', 'sweep'):
        p = commands.add_parser(name)
        p.add_argument('--trace', required=True)
        p.add_argument('--layers', type=int)
        p.add_argument('--experts', type=int)
        p.add_argument('--split', help='chronological token train:test weights; profile uses train, replay uses test')
        if name == 'profile':
            p.add_argument('--out', required=True)
            p.add_argument('--weighting', choices=['raw', 'per-request'], default='raw')
        else:
            p.add_argument('--capacity-experts', type=int, required=name == 'simulate')
            p.add_argument('--per-layer-capacity', action='store_true')
            p.add_argument('--policy', default='lru')
            p.add_argument('--warmup', type=int, default=0)
            p.add_argument('--expert-bytes', type=int, default=0)
            p.add_argument('--per-layer-bytes', help='comma-separated byte sizes, one per layer')
            p.add_argument('--bw-gbs', type=float, default=1.0)
            p.add_argument('--compute-ms', type=float, default=0.0)
            if name == 'sweep':
                p.add_argument('--capacities', required=True, help='comma-separated capacities')
                p.add_argument('--out', help='CSV path; default stdout')
    args = parser.parse_args(argv)
    try:
        rows, layers, experts, diagnostics = read_trace(args.trace, args.layers, args.experts)
        train, test = partition(rows, args.split)
        if diagnostics['malformed_lines']:
            print(json.dumps({'trace_diagnostics': diagnostics}), file=sys.stderr)
        if args.command == 'profile':
            chosen = train if args.split else rows
            if not chosen:
                raise ValueError('empty training partition')
            result = make_profile(chosen, layers, experts, args.weighting)
            result['trace_diagnostics'] = diagnostics
            Path(args.out).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            return 0
        if not test:
            raise ValueError('empty test partition')
        policy, profile = args.policy, None
        if policy.startswith('static-profile:'):
            profile = json.loads(Path(policy.split(':', 1)[1]).read_text(encoding='utf-8'))
            policy = 'static-profile'
            if args.split:
                if 'training_records' not in profile:
                    raise ValueError('split replay requires profile training provenance')
                if set(profile['training_records']) & {fingerprint(r) for r in test}:
                    raise ValueError('profile leakage: training includes test records')
        sizes = list(map(int, args.per_layer_bytes.split(','))) if args.per_layer_bytes else None
        capacities = list(map(int, args.capacities.split(','))) if args.command == 'sweep' else [args.capacity_experts]
        results = []
        for capacity in capacities:
            result = replay(test, layers, experts, capacity, policy, args.per_layer_capacity,
                            args.warmup, args.expert_bytes, sizes, args.bw_gbs, args.compute_ms, profile)
            result.update(layers=layers, experts=experts, trace_diagnostics=diagnostics,
                          training_tokens=len(token_keys(train)), test_tokens=len(token_keys(test)))
            results.append(result)
        if args.command == 'simulate':
            print(json.dumps(results[0], indent=2, allow_nan=False))
        else:
            output = Path(args.out).open('w', newline='', encoding='utf-8') if args.out else sys.stdout
            try:
                writer = csv.DictWriter(output, fieldnames=['capacity_experts', 'policy', 'oracle_label', 'hit_rate',
                    'decode_hit_rate', 'misses_per_token', 'bytes_moved_per_token', 'prediction_decode_seconds_per_token',
                    'malformed_lines', 'possible_truncated_tail'])
                writer.writeheader()
                for r in results:
                    writer.writerow(dict(capacity_experts=r['capacity_experts'], policy=r['policy'], oracle_label=r['oracle_label'],
                        hit_rate=r['overall']['hit_rate'], decode_hit_rate=r['decode_only']['hit_rate'],
                        misses_per_token=r['overall']['misses_per_token'], bytes_moved_per_token=r['overall']['bytes_moved_per_token'],
                        prediction_decode_seconds_per_token=r['prediction']['decode_seconds_per_token'],
                        malformed_lines=diagnostics['malformed_lines'], possible_truncated_tail=diagnostics['possible_truncated_tail']))
            finally:
                if args.out:
                    output.close()
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
