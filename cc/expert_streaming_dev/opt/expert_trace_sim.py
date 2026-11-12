#!/usr/bin/env python3
"""Offline expert lab. CSV IDs are space separated; appended runs stay independent.

Predictors train on prior prompt/decode observations only. Unions interleave source
rankings and truncate to K. Caches use equal-sized (layer, expert) objects, warm on
prompt, and count decode misses. No speculative IO is charged in cache results.
The default 1.83 MiB size estimates Satay; override for another model. Belady sees
future accesses, so it is an optimistic demand-cache bound, not a deployable policy.
"""
import argparse
import csv
import heapq
import json
import math
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    run: str
    token: int
    layer: int
    phase: str
    selected: tuple[int, ...]
    predicted: tuple[int, ...] = ()


def read_trace(path):
    records = []
    seen = set()
    with open(path, newline='', encoding='utf-8') as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            try:
                record = Record(row.get('run', '0'), int(row['token_idx']), int(row['layer']),
                                row['phase'], tuple(map(int, row['selected'].split())),
                                tuple(map(int, row['predicted'].split())))
                key = (record.run, record.token, record.layer)
                if (record.token < 0 or record.layer < 0 or record.phase not in ('prompt', 'decode')
                        or not record.selected or min(record.selected + record.predicted) < 0
                        or len(set(record.selected)) != len(record.selected) or key in seen):
                    raise ValueError('invalid IDs, phase, index, or duplicate record')
                seen.add(key)
                records.append(record)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{line}: {error}') from error
    return sorted(records, key=lambda r: (r.run, r.token, r.layer))


def union_ranked(sources, k):
    result = []
    for rank in range(max(map(len, sources), default=0)):
        for source in sources:
            if rank < len(source) and source[rank] not in result:
                result.append(source[rank])
                if len(result) == k:
                    return result
    return result


def predictor_report(records, k):
    counts = defaultdict(Counter)
    previous = {}
    hits = defaultdict(Counter)
    actual = Counter()
    coverage = Counter()
    names = ('router', 'previous_token', 'frequency', 'router_previous',
             'router_frequency', 'previous_frequency', 'router_previous_frequency')
    for r in records:
        key = (r.run, r.layer)
        frequency = sorted(counts[key], key=lambda e: (-counts[key][e], e))[:k]
        temporal = previous.get(key, ())[:k]
        router = r.predicted[:k]
        predictions = (router, temporal, frequency, union_ranked((router, temporal), k),
                       union_ranked((router, frequency), k), union_ranked((temporal, frequency), k),
                       union_ranked((router, temporal, frequency), k))
        if r.phase == 'decode':
            actual[r.layer] += len(r.selected)
            coverage[r.layer] += bool(router)
            for name, prediction in zip(names, predictions):
                hits[r.layer][name] += len(set(prediction).intersection(r.selected))
        counts[key].update(r.selected)
        previous[key] = r.selected

    def summary(layer_hits, denominator):
        return {name: layer_hits[name] / denominator if denominator else 0.0 for name in names}

    overall = Counter()
    for layer_hits in hits.values():
        overall.update(layer_hits)
    return {'overall': summary(overall, sum(actual.values())),
            'per_layer': {str(layer): summary(hits[layer], actual[layer]) for layer in sorted(actual)},
            'router_records': sum(coverage.values()), 'actual_experts': sum(actual.values())}


def accesses(records):
    return [(r, (r.layer, expert)) for r in records for expert in r.selected]


def cache_report(records, capacity, expert_bytes, policy):
    """Demand paging; LFU ties by LRU. 2Q protects repeated hits in half the cache."""
    sequence = accesses(records)
    future = defaultdict(deque)
    if policy == 'belady':
        for index, (r, key) in enumerate(sequence):
            future[(r.run, key)].append(index)
    cache = OrderedDict()
    protected = OrderedDict()
    frequency = Counter()
    heap = []
    hits = misses = 0
    run = None
    for index, (r, key) in enumerate(sequence):
        if run != r.run:
            cache.clear()
            protected.clear()
            frequency.clear()
            heap.clear()
            run = r.run
        if policy == 'belady':
            future[(r.run, key)].popleft()
        hit = key in cache or key in protected
        if r.phase == 'decode':
            hits += hit
            misses += not hit
        if capacity == 0:
            continue
        if policy == 'lru_frequency':
            if key in protected:
                protected.move_to_end(key)
            elif key in cache:
                del cache[key]
                protected[key] = None
            else:
                if len(cache) + len(protected) >= capacity:
                    (cache if cache else protected).popitem(last=False)
                cache[key] = None
            if len(protected) > max(1, capacity // 2):
                demoted, _ = protected.popitem(last=False)
                cache[demoted] = None
            continue
        if not hit:
            if len(cache) >= capacity:
                if policy == 'lru':
                    victim, _ = cache.popitem(last=False)
                else:
                    while heap:
                        priority, stamp, victim = heapq.heappop(heap)
                        if victim in cache and cache[victim] == (priority, stamp):
                            del cache[victim]
                            break
                frequency.pop(victim, None)
            cache[key] = None
        if policy == 'lru':
            cache.move_to_end(key)
        else:
            frequency[key] += 1
            next_use = future[(r.run, key)][0] if policy == 'belady' and future[(r.run, key)] else math.inf
            priority = -next_use if policy == 'belady' else frequency[key]
            cache[key] = (priority, index)
            heapq.heappush(heap, (priority, index, key))
            # Discard stale hit entries to keep memory proportional to residency.
            if len(heap) > 4 * capacity + 32:
                heap = [(score, stamp, item) for item, (score, stamp) in cache.items()]
                heapq.heapify(heap)
    tokens = len({(r.run, r.token) for r in records if r.phase == 'decode'})
    return {'capacity_experts': capacity, 'hits': hits, 'misses': misses,
            'hit_rate': hits / (hits + misses) if hits + misses else 0.0,
            'bytes_per_decode_token': misses * expert_bytes / tokens if tokens else 0.0}


def reuse_report(records):
    """Token gaps and distinct-object stack distances, power-of-two bins; cold separate."""
    sequence = accesses(records)
    tree = [0] * (len(sequence) + 1)
    last = {}
    token_gaps = Counter()
    stack_distances = Counter()
    active = 0

    def update(pos, delta):
        while pos < len(tree):
            tree[pos] += delta
            pos += pos & -pos

    def prefix(pos):
        total = 0
        while pos:
            total += tree[pos]
            pos -= pos & -pos
        return total

    def bucket(n):
        if n == 0:
            return '0'
        low = 1 << (n.bit_length() - 1)
        return str(low) if low == 1 else f'{low}-{2 * low - 1}'

    for index, (r, key) in enumerate(sequence, 1):
        identity = (r.run, key)
        prior = last.get(identity)
        if r.phase == 'decode':
            token_gaps[bucket(r.token - prior[1]) if prior else 'cold'] += 1
            stack_distances[bucket(active - prefix(prior[0])) if prior else 'cold'] += 1
        if prior:
            update(prior[0], -1)
        else:
            active += 1
        update(index, 1)
        last[identity] = (index, r.token)
    return {'token_gap': dict(token_gaps), 'distinct_experts_between_uses': dict(stack_distances)}


def analyze(records, k=8, budgets=(2, 3, 4, 5), expert_bytes=round(1.83 * 2**20)):
    if k < 1 or expert_bytes < 1 or any(b < 0 or not math.isfinite(b) for b in budgets):
        raise ValueError('K and expert size must be positive; budgets must be finite and nonnegative')
    return {'records': len(records),
            'decode_tokens': len({(r.run, r.token) for r in records if r.phase == 'decode'}),
            'k': k, 'expert_bytes': expert_bytes,
            'predictor_recall_at_k': predictor_report(records, k),
            'caches': {str(b): {p: cache_report(records, int(b * 2**30) // expert_bytes, expert_bytes, p)
                               for p in ('lru', 'lfu', 'lru_frequency', 'belady')} for b in budgets},
            'reuse_distance': reuse_report(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--budgets-gib', type=float, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--expert-mib', type=float, default=1.83)
    args = parser.parse_args()
    try:
        report = analyze(read_trace(args.trace), args.k, args.budgets_gib, round(args.expert_mib * 2**20))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
