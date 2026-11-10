"""Summarize long-form Nsight Compute CSV without NVIDIA tooling."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import io
import json
import math
from pathlib import Path
import re

TIME = 'gpu__time_duration.sum'
READ = 'dram__bytes_read.sum'
WRITE = 'dram__bytes_write.sum'
DRAM = 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
SM = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
METRICS = (TIME, READ, DRAM, SM)


def quantity(value, unit, kind):
    factors = {'time': {'ns': 1e-9, 'us': 1e-6, 'µs': 1e-6, 'ms': 1e-3, 's': 1},
               'bytes': {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9},
               'percent': {'%': 1}}[kind]
    if unit not in factors:
        raise ValueError(f'unsupported {kind} unit: {unit!r}')
    try:
        result = float(Decimal(value.replace(',', '')) * Decimal(str(factors[unit])))
    except InvalidOperation as exc:
        raise ValueError(f'invalid metric value: {value!r}') from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'invalid metric value: {value!r}')
    return result


def kernel_class(name):
    """Group template instantiations by function, preserving distinct functions."""
    match = re.search(r'([\w:]+)\s*(?:<|\()', name)
    return match.group(1) if match else name.strip()


def summarize(source):
    lines = [line for line in source.splitlines() if line.strip() and not line.startswith('==')]
    reader = csv.DictReader(io.StringIO('\n'.join(lines)))
    required = {'ID', 'Process ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError('missing ncu CSV header columns')
    launches = {}
    for row in reader:
        if row['ID'] == 'ID':  # repeated export header
            continue
        metric = row['Metric Name']
        if metric not in (*METRICS, WRITE):
            continue
        key = (row['Process ID'], row['ID'])
        launch = launches.setdefault(key, {'name': row['Kernel Name'], 'metrics': {}})
        if launch['name'] != row['Kernel Name'] or metric in launch['metrics']:
            raise ValueError(f'ambiguous duplicate launch/metric: {key}, {metric}')
        kind = 'time' if metric == TIME else 'bytes' if metric in (READ, WRITE) else 'percent'
        launch['metrics'][metric] = quantity(row['Metric Value'], row['Metric Unit'], kind)
    if not launches:
        raise ValueError('no kernel metrics')
    if any(WRITE in x['metrics'] for x in launches.values()) and not all(
            WRITE in x['metrics'] for x in launches.values()):
        raise ValueError('incomplete write-byte coverage')
    groups = {}
    tiny_time = tiny_count = 0
    for launch in launches.values():
        metrics = launch['metrics']
        if not set(METRICS).issubset(metrics):
            raise ValueError(f'incomplete metrics for {launch["name"]}')
        duration = metrics[TIME]
        if duration <= 0:
            raise ValueError('kernel duration must be positive')
        tiny = duration < 5e-6 and metrics[DRAM] < 5
        tiny_count += int(tiny)
        tiny_time += duration if tiny else 0
        group = groups.setdefault(kernel_class(launch['name']), dict(count=0, time_s=0.,
            read_bytes=0., write_bytes=0., dram_weight=0., sm_weight=0., launch_bound_count=0,
            launch_bound_time_s=0.))
        group['count'] += 1
        group['time_s'] += duration
        group['read_bytes'] += metrics[READ]
        group['write_bytes'] += metrics.get(WRITE, 0)
        group['dram_weight'] += duration * metrics[DRAM]
        group['sm_weight'] += duration * metrics[SM]
        group['launch_bound_count'] += int(tiny)
        group['launch_bound_time_s'] += duration if tiny else 0
    total = sum(g['time_s'] for g in groups.values())
    for group in groups.values():
        group['time_share'] = group['time_s'] / total
        group['GB_s'] = (group['read_bytes'] + group['write_bytes']) / group['time_s'] / 1e9
        group['dram_pct'] = group.pop('dram_weight') / group['time_s']
        group['sm_pct'] = group.pop('sm_weight') / group['time_s']
        group['launch_bound'] = group['launch_bound_count'] == group['count']
    return dict(kernels=len(launches), time_s=total,
                effective_GB_s=sum(g['read_bytes'] + g['write_bytes'] for g in groups.values()) / total / 1e9,
                byte_scope='read+write' if any(WRITE in x['metrics'] for x in launches.values()) else 'read',
                launch_bound_count=tiny_count, launch_bound_time_share=tiny_time / total,
                classes=dict(sorted(groups.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv.read_text()), indent=2))


if __name__ == '__main__':
    main()
