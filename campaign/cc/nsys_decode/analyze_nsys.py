#!/usr/bin/env python3
"""Read-only, stdlib-only per-token decode accounting for nsys SQLite exports.

Graph-level exports expose graph envelopes, not constituent kernel timings.
Never interpret an empty steady-state KERNEL table as an empty GPU workload.
"""
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def duration(intervals):
    return sum(b - a for a, b in merge(intervals))


def intersection(left, right):
    left, right = merge(left), merge(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if b > a:
            result.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def complement(intervals, a, b):
    result = []
    cursor = a
    for x, y in merge(intervals):
        x, y = max(a, x), min(b, y)
        if y <= x:
            continue
        if x > cursor:
            result.append((cursor, x))
        cursor = max(cursor, y)
    if cursor < b:
        result.append((cursor, b))
    return result


def rows(conn, table):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return []
    return [dict(r) for r in conn.execute(f'SELECT * FROM "{table}" ORDER BY start')]


def top_kernels(events, strings, a, b):
    totals = defaultdict(lambda: [0, 0])
    for e in events:
        if a <= e['start'] < b:
            name = strings.get(e.get('demangledName', e.get('shortName')), 'unknown')
            totals[name][0] += 1
            totals[name][1] += max(0, min(b, e['end']) - e['start'])
    return [dict(name=k, count=v[0], total_us=v[1] / 1000)
            for k, v in sorted(totals.items(), key=lambda kv: -kv[1][1])]


def analyze(path, warmup=5, marker_bytes=196620, device=0, gap_us=10):
    if warmup < 0:
        raise ValueError('warmup must be nonnegative')
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        strings = dict(conn.execute('SELECT id,value FROM StringIds'))
        kinds = dict(conn.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        tables = {key: rows(conn, table) for key, table in {
            'kernel': 'CUPTI_ACTIVITY_KIND_KERNEL',
            'graph': 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE',
            'copy': 'CUPTI_ACTIVITY_KIND_MEMCPY',
            'memset': 'CUPTI_ACTIVITY_KIND_MEMSET',
            'api': 'CUPTI_ACTIVITY_KIND_RUNTIME'}.items()}
    finally:
        conn.close()
    for key in ('kernel', 'graph', 'copy', 'memset'):
        tables[key] = [e for e in tables[key] if e.get('deviceId', 0) == device]
    pids = {e.get('globalPid') for key in ('kernel', 'graph', 'copy') for e in tables[key]}
    if len(pids - {None}) > 1:
        raise ValueError('multiple CUDA processes: export/filter a single process first')
    markers = sorted(e['end'] for e in tables['copy']
                     if e['copyKind'] == 2 and e['bytes'] == marker_bytes)
    if len(markers) <= warmup + 1 or len(set(markers)) != len(markers):
        raise ValueError('insufficient or duplicate logits markers; adjust --marker-bytes/--warmup')
    # First marker terminates prefill. Each subsequent interval is one decode.
    # Include the outgoing logits DMA in its token; API tails cross this boundary.
    starts = {k: [e['start'] for e in events] for k, events in tables.items()}
    prefix_ends = {}
    for k, events in tables.items():
        high = -1
        prefix_ends[k] = []
        for e in events:
            high = max(high, e['end'])
            prefix_ends[k].append(high)

    def select(key, a, b):
        lo = bisect_left(prefix_ends[key], a + 1)
        hi = bisect_left(starts[key], b)
        return [e for e in tables[key][lo:hi] if e['end'] > a]

    tokens = []
    aggregate_apis = defaultdict(lambda: [0, 0])
    aggregate_copies = defaultdict(lambda: [0, 0, 0, 0, 0])
    graphs = defaultdict(lambda: [0, 0])
    # Correlation may repeat between processes, so retain process/thread IDs.
    api_by_corr = defaultdict(list)
    for e in tables['api']:
        if 'Memcpy' in strings.get(e['nameId'], ''):
            api_by_corr[e.get('correlationId')].append(e)
    for index in range(warmup, len(markers) - 1):
        a, b = markers[index:index + 2]
        ev = {k: select(k, a, b) for k in tables}
        spans = {k: [(max(a, e['start']), min(b, e['end'])) for e in es]
                 for k, es in ev.items()}
        compute = merge(spans['graph'] + spans['kernel'])
        busy = merge(compute + spans['copy'] + spans['memset'])
        idle = complement(busy, a, b)
        sync_copy = []
        counts = defaultdict(int)
        api_times = defaultdict(int)
        token_copies = defaultdict(lambda: [0, 0, 0, 0])
        for e in ev['api']:
            name = strings.get(e['nameId'], str(e['nameId']))
            ns = min(b, e['end']) - max(a, e['start'])
            owned = a <= e['start'] < b
            counts[name] += int(owned)
            api_times[name] += ns
            aggregate_apis[name][0] += int(owned)
            aggregate_apis[name][1] += ns
            if 'Synchronize' in name or 'Memcpy' in name:
                sync_copy.append((max(a, e['start']), min(b, e['end'])))
        for e in ev['copy']:
            if not a <= e['start'] < b:
                continue
            key = (kinds.get(e['copyKind'], str(e['copyKind'])), e['bytes'])
            v = aggregate_copies[key]
            v[0] += 1
            v[1] += e['bytes']
            v[2] += min(b, e['end']) - e['start']
            tv = token_copies[key]
            tv[0] += 1
            tv[1] += min(b, e['end']) - e['start']
            for api in api_by_corr[e.get('correlationId')]:
                if api['start'] <= e['end'] and api['end'] >= a:
                    v[3] += 1
                    v[4] += max(0, min(b, api['end']) - max(a, api['start']))
                    tv[2] += 1
                    tv[3] += max(0, min(b, api['end']) - max(a, api['start']))
        for e in ev['graph']:
            if a <= e['start'] < b:
                v = graphs[str(e.get('graphExecId', e.get('graphId')))]
                v[0] += 1
                v[1] += min(b, e['end']) - e['start']
        idle_api = duration(intersection(idle, sync_copy))
        token = dict(index=index + 1, start_ns=a, end_ns=b,
                     wall_us=(b-a)/1000,
                     graph_launches=sum(v for k, v in counts.items() if k.startswith('cudaGraphLaunch')),
                     kernel_launch_apis=sum(v for k, v in counts.items()
                                            if k.startswith(('cudaLaunchKernel', 'cuLaunchKernel'))),
                     explicit_kernel_records=sum(a <= e['start'] < b for e in ev['kernel']),
                     stream_sync_calls=sum(v for k,v in counts.items() if k.startswith('cudaStreamSynchronize')),
                     stream_sync_api_us=sum(v for k,v in api_times.items() if k.startswith('cudaStreamSynchronize'))/1000,
                     memcpy_api_calls=sum(v for k,v in counts.items() if 'Memcpy' in k),
                     memcpy_api_us=sum(v for k,v in api_times.items() if 'Memcpy' in k)/1000,
                     compute_envelope_us=duration(compute)/1000,
                     transfer_only_us=(duration(busy)-duration(compute))/1000,
                     gpu_busy_us=duration(busy)/1000,
                     gpu_idle_us=duration(idle)/1000,
                     idle_sync_copy_api_us=idle_api/1000,
                     idle_other_host_us=(duration(idle)-idle_api)/1000,
                     idle_gap_count=len(idle),
                     large_idle_gap_count=sum(y-x >= gap_us*1000 for x,y in idle),
                     large_idle_gap_us=sum(y-x for x,y in idle if y-x >= gap_us*1000)/1000,
                     d2h_bytes=sum(e['bytes'] for e in ev['copy'] if a <= e['start'] < b and e['copyKind']==2),
                     h2d_bytes=sum(e['bytes'] for e in ev['copy'] if a <= e['start'] < b and e['copyKind']==1))
        token['copies'] = [dict(kind=k[0], bytes=k[1], calls=v[0], device_us=v[1]/1000,
                                matched_api_calls=v[2], matched_api_us=v[3]/1000)
                           for k,v in sorted(token_copies.items())]
        tokens.append(token)
    n = len(tokens)
    metrics = {k: dict(mean=statistics.mean(t[k] for t in tokens),
                       median=statistics.median(t[k] for t in tokens),
                       min=min(t[k] for t in tokens), max=max(t[k] for t in tokens))
               for k in tokens[0] if k not in ('index','start_ns','end_ns','copies')}
    return dict(source=str(path), token_count=n, marker_count=len(markers), warmup=warmup,
                marker_bytes=marker_bytes, gap_threshold_us=gap_us,
                boundary='end of D2H logits DMA; first marker ends prefill; discard first warmup decode intervals',
                metrics=metrics,
                apis=[dict(name=k, calls_per_token=v[0]/n, us_per_token=v[1]/n/1000)
                      for k,v in sorted(aggregate_apis.items(), key=lambda kv:-kv[1][1])],
                copies=[dict(kind=k[0], bytes=k[1], calls_per_token=v[0]/n,
                             bytes_per_token=v[1]/n, device_us_per_token=v[2]/n/1000,
                             matched_api_calls_per_token=v[3]/n, matched_api_us_per_token=v[4]/n/1000)
                        for k,v in sorted(aggregate_copies.items())],
                graph_execs=[dict(id=k, calls_per_token=v[0]/n, us_per_token=v[1]/n/1000)
                             for k,v in sorted(graphs.items(), key=lambda kv:int(kv[0]))],
                steady_top_explicit_kernels=top_kernels(tables['kernel'],strings,markers[warmup],markers[-1]),
                first_decode_top_explicit_kernels=top_kernels(tables['kernel'],strings,markers[0],markers[1]),
                first_decode_wall_us=(markers[1]-markers[0])/1000,
                tokens=tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', nargs='+', type=Path)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--marker-bytes', type=int, default=196620,
                        help='size of one D2H logits copy per token; verify against the model')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gap-us', type=float, default=10)
    parser.add_argument('--output', type=Path, help='JSON output; defaults to stdout')
    args = parser.parse_args()
    result = [analyze(p,args.warmup,args.marker_bytes,args.device,args.gap_us) for p in args.sqlite]
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
