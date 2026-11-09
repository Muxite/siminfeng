#!/usr/bin/env python3
"""Read-only, dependency-free resource observation. See CONTRACT.md."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping

GPU_QUERY = 'uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem'
APP_QUERY = 'gpu_uuid,pid,used_memory'
GPU_SPECS = {
    'gpu.memory.used': ('memory.used', 'MiB'),
    'gpu.memory.total': ('memory.total', 'MiB'),
    'gpu.util': ('utilization.gpu', '%'),
    'gpu.temp': ('temperature.gpu', 'degC'),
    'gpu.power': ('power.draw', 'W'),
    'gpu.clocks.sm': ('clocks.current.sm', 'MHz'),
    'gpu.clocks.memory': ('clocks.current.memory', 'MHz'),
}
HOST_SPECS = {
    'host.mem.total': 'MemTotal', 'host.mem.available': 'MemAvailable',
    'host.mem.swap.total': 'SwapTotal', 'host.mem.swap.free': 'SwapFree',
}
CGROUP_SPECS = {
    'cgroup.memory.current': 'memory.current',
    'cgroup.memory.peak': 'memory.peak',
    'cgroup.memory.swap.current': 'memory.swap.current',
    'cgroup.memory.swap.peak': 'memory.swap.peak',
    'cgroup.io.rbytes': 'io.stat',
}
MISSING = {'', '[n/a]', 'n/a', '[not supported]', 'not supported', '-'}


def field(value, unit, source, scope, status='ok', reason=None):
    return dict(value=value if status == 'ok' else None, unit=unit,
                source=source, scope=scope, status=status, reason=reason)


def number(raw, unit, source, scope, scale=1):
    if raw is None or raw.strip().lower() in MISSING:
        return field(None, unit, source, scope, 'unavailable',
                     'missing or unsupported field' if raw is None else f'unsupported value: {raw!r}')
    # Only accept a number and its expected source unit, not arbitrary trailing text.
    source_unit = 'kB' if scale == 1024 else unit
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(?:' + re.escape(source_unit) + r')?\s*', raw)
    if not match:
        return field(None, unit, source, scope, 'error', f'malformed numeric value: {raw!r}')
    value = Decimal(match[1]) * scale
    if not value.is_finite():
        return field(None, unit, source, scope, 'error', 'non-finite numeric value')
    return field(int(value) if value == value.to_integral_value() else float(value), unit, source, scope)


def csv_rows(text):
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return [], []
    # Real nvidia-smi headers carry units; normalize them away.
    headers = [re.sub(r'\s*\[.*?\]\s*$', '', c.strip()) for c in rows[0]]
    return headers, [[c.strip() for c in r] for r in rows[1:]]


def parse_gpu(text, gpu='0'):
    """Parse CSV with headers; choose one index or exact UUID, never sum devices."""
    headers, rows = csv_rows(text)
    selected = None
    if str(gpu).isdigit():
        index = int(gpu)
        selected = rows[index] if index < len(rows) else None
    else:
        for row in rows:
            if 'uuid' in headers and len(row) > headers.index('uuid') and row[headers.index('uuid')] == gpu:
                selected = row
                break
    values = dict(zip(headers, selected)) if selected is not None else {}
    malformed = selected is not None and len(selected) != len(headers)
    fields = {}
    for name, (column, unit) in GPU_SPECS.items():
        # Allow names used by older NVIDIA versions / synthetic fixtures.
        aliases = {'clocks.current.sm': 'clocks.sm', 'clocks.current.memory': 'clocks.mem'}
        raw = values.get(column, values.get(aliases.get(column)))
        fields[name] = (field(None, unit, f'nvidia-smi:{column}', 'device-total', 'error', 'malformed GPU CSV row')
                        if malformed else number(raw, unit, f'nvidia-smi:{column}', 'device-total'))
    uuid = values.get('uuid')
    if uuid is not None and uuid.lower() in MISSING:
        uuid = None
    return fields, uuid


def parse_apps(text, uuid):
    source = 'nvidia-smi:compute-apps'
    if not uuid:
        return field(None, 'records', source, 'per-process-on-device', 'unavailable', 'selected GPU UUID unavailable')
    headers, rows = csv_rows(text)
    if not headers:
        return field(None, 'records', source, 'per-process-on-device', 'unavailable', 'empty output without CSV header')
    if not {'gpu_uuid', 'pid', 'used_gpu_memory'}.issubset(headers) and not {'gpu_uuid', 'pid', 'used_memory'}.issubset(headers):
        return field(None, 'records', source, 'per-process-on-device', 'error', 'missing compute-app CSV columns')
    result = []
    for row in rows:
        if len(row) != len(headers):
            return field(None, 'records', source, 'per-process-on-device', 'error', 'malformed compute-app CSV row')
        values = dict(zip(headers, row))
        if values['gpu_uuid'] != uuid:
            continue
        if not values['pid'].isdigit() or int(values['pid']) <= 0:
            return field(None, 'records', source, 'per-process-on-device', 'error', 'malformed process PID')
        result.append({'pid': int(values['pid']), 'used_memory': number(
            values.get('used_gpu_memory', values.get('used_memory')), 'MiB', source, 'per-process-on-device')})
    return field(result, 'records', source, 'per-process-on-device')


def run_query(query, timeout):
    source = f'nvidia-smi:{query}'
    try:
        proc = subprocess.run(['nvidia-smi', f'--{query}', '--format=csv'],
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, 'unavailable', 'nvidia-smi absent'
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, 'error', f'{source}: {exc}'
    if proc.returncode:
        return None, 'error', f'exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:1000]}'
    return proc.stdout, 'ok', None


def nvidia_backend(gpu, timeout):
    text, status, reason = run_query('query-gpu=' + GPU_QUERY, timeout)
    if text is None:
        fields = {name: field(None, unit, f'nvidia-smi:{column}', 'device-total', status, reason)
                  for name, (column, unit) in GPU_SPECS.items()}
        uuid = None
    else:
        fields, uuid = parse_gpu(text, gpu)
    apps, app_status, app_reason = run_query('query-compute-apps=' + APP_QUERY, timeout)
    fields['gpu.compute_apps'] = (parse_apps(apps, uuid) if apps is not None else
        field(None, 'records', 'nvidia-smi:compute-apps', 'per-process-on-device', app_status, app_reason))
    return fields, uuid


def read_text(path):
    try:
        return Path(path).read_text(), 'ok', None
    except FileNotFoundError:
        return None, 'unavailable', f'not found: {path}'
    except OSError as exc:
        return None, 'error', str(exc)


def parse_meminfo(text, source='/proc/meminfo'):
    entries = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
    return {name: number(entries.get(key), 'B', f'{source}:{key}', 'host', scale=1024)
            for name, key in HOST_SPECS.items()}


def parse_io(text, source):
    # io.stat is per block device: sum the cumulative read bytes across devices.
    if not text.strip():
        return field(0, 'B', source, 'cgroup-subtree')
    values = []
    for line in text.splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        reads = [t[7:] for t in tokens[1:] if t.startswith('rbytes=')]
        if not re.fullmatch(r'\d+:\d+', tokens[0]) or len(reads) != 1 or not reads[0].isdigit():
            return field(None, 'B', source, 'cgroup-subtree', 'error', 'malformed io.stat row or missing rbytes')
        values.append(int(reads[0]))
    return field(sum(values), 'B', source, 'cgroup-subtree')


def linux_backend(proc_root, cgroup_path, pid):
    path = Path(proc_root) / 'meminfo'
    text, status, reason = read_text(path)
    fields = parse_meminfo(text, str(path)) if text is not None else {
        name: field(None, 'B', f'{path}:{key}', 'host', status, reason) for name, key in HOST_SPECS.items()}
    for name, filename in CGROUP_SPECS.items():
        source = str(Path(cgroup_path) / filename) if cgroup_path is not None else f'cgroup-v2:{filename}'
        text, status, reason = read_text(source) if cgroup_path is not None else (None, 'unavailable', 'cgroup_path not supplied')
        fields[name] = ((parse_io(text, source) if filename == 'io.stat' else
                         number(text.strip(), 'B', source, 'cgroup-subtree')) if text is not None else
                        field(None, 'B', source, 'cgroup-subtree', status, reason))
    source = str(Path(proc_root) / str(pid) / 'status') if pid is not None else '/proc/<pid>/status'
    text, status, reason = read_text(source) if pid is not None else (None, 'unavailable', 'pid not supplied')
    if text is not None:
        entries = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
        fields['process.rss'] = number(entries.get('VmRSS'), 'B', source + ':VmRSS', 'process', scale=1024)
    else:
        fields['process.rss'] = field(None, 'B', source + ':VmRSS', 'process', status, reason)
    return fields


def sample(*, backends=('nvidia-smi', 'linux'), gpu='0', cgroup_path=None,
           pid=None, proc_root='/proc', timeout=2.0, extra_backends=None):
    """Sample explicitly selected backends. Extensions return platform.<name>.* fields."""
    if pid is not None and (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0):
        raise ValueError('pid must be a positive integer')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    extensions = extra_backends or {}
    names = list(backends)
    if len(names) != len(set(names)) or any(n not in {'nvidia-smi', 'linux'} | extensions.keys() for n in names):
        raise ValueError('unknown or duplicate backend')
    started = time.monotonic()
    result = {'schema': 'resource-sample/0',
              'ts': {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'monotonic': started},
              'host_id': socket.gethostname(), 'backends': names, 'gpu_id': None, 'fields': {}}
    # Keep common fields present even if their backend is disabled.
    fields = {name: field(None, unit, f'nvidia-smi:{column}', 'device-total', 'unavailable', 'backend disabled')
              for name, (column, unit) in GPU_SPECS.items()}
    fields['gpu.compute_apps'] = field(None, 'records', 'nvidia-smi:compute-apps', 'per-process-on-device', 'unavailable', 'backend disabled')
    for name in HOST_SPECS:
        fields[name] = field(None, 'B', 'linux:procfs', 'host', 'unavailable', 'backend disabled')
    for name in CGROUP_SPECS:
        fields[name] = field(None, 'B', 'linux:cgroup-v2', 'cgroup-subtree', 'unavailable', 'backend disabled')
    fields['process.rss'] = field(None, 'B', 'linux:procfs', 'process', 'unavailable', 'backend disabled')
    for name in names:
        if name == 'nvidia-smi':
            observed, result['gpu_id'] = nvidia_backend(gpu, timeout)
        elif name == 'linux':
            observed = linux_backend(proc_root, cgroup_path, pid)
        else:
            observed = extensions[name]()
            for key, metric in observed.items():
                if not key.startswith(f'platform.{name}.') or not isinstance(metric, Mapping) or set(metric) != {'value', 'unit', 'source', 'scope', 'status', 'reason'}:
                    raise ValueError('extension fields must be namespaced and use the field envelope')
                if metric['status'] not in {'ok', 'unavailable', 'error'} or (metric['status'] != 'ok' and (metric['value'] is not None or not metric['reason'])):
                    raise ValueError('invalid extension status/value/reason')
        fields.update(observed)
    result['fields'] = fields
    result['collection_seconds'] = time.monotonic() - started
    return result


def sample_series(interval, duration=None, *, until=None, sampler=sample,
                  clock=time.monotonic, sleep=time.sleep, **sample_kwargs) -> Iterator[dict]:
    """Yield samples and a final resource-series/0 accounting record.

    `until` is an absolute monotonic deadline. Scheduled ticks in [start, end)
    count as expected; late ticks are skipped rather than collected in bursts.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('interval must be finite and positive')
    if (duration is None) == (until is None):
        raise ValueError('supply exactly one of duration or until')
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise ValueError('duration must be finite and nonnegative')
    start = clock()
    end = start + duration if duration is not None else until
    if not math.isfinite(end):
        raise ValueError('until must be a finite monotonic timestamp')
    span = duration if duration is not None else max(0.0, end - start)
    # Use the original duration: adding it to a large monotonic clock can
    # introduce enough cancellation error to invent an endpoint tick.
    ratio = span / interval
    if not math.isfinite(ratio):
        raise ValueError("requested sample count is too large")
    expected = max(0, math.ceil(math.nextafter(ratio, -math.inf)))
    obtained = 0
    slot = 0
    first = last = None
    max_gap = 0.0
    failures = []
    while slot < expected:
        target = start + slot * interval
        now = clock()
        if now < target:
            sleep(target - now)
            continue
        if now >= end:
            break
        # Take at most one sample, assigned to the most recent due tick.
        slot = max(slot, int((now - start) / interval))
        if slot >= expected:
            break
        actual = clock()
        try:
            record = sampler(**sample_kwargs)
        except Exception as exc:
            failures.append({'slot': slot, 'reason': f'{type(exc).__name__}: {exc}'})
        else:
            obtained += 1
            if first is None:
                first = actual
            if last is not None:
                max_gap = max(max_gap, actual - last)
            last = actual
            yield record
        slot += 1
    finish = clock()
    yield {'schema': 'resource-series/0', 'interval_seconds': interval,
           'start_monotonic': start, 'end_monotonic': end, 'finished_monotonic': finish,
           'expected_samples': expected, 'obtained_samples': obtained,
           'lost_samples': expected - obtained, 'max_gap_seconds': max_gap if obtained >= 2 else None,
           'leading_gap_seconds': (first - start) if first is not None else span,
           'trailing_gap_seconds': max(0.0, end - last) if last is not None else span,
           'collection_errors': failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('once', 'watch'))
    parser.add_argument('--interval', type=float, default=1.0)
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument('--duration', type=float, default=None, help='watch seconds (default: 10)')
    stop.add_argument('--until', type=float, help='absolute time.monotonic() deadline')
    parser.add_argument('--out', choices=('jsonl',), default='jsonl', help='format; stdout is the destination')
    parser.add_argument('--backend', action='append', choices=('linux', 'nvidia-smi'), dest='backends')
    parser.add_argument('--gpu', default='0', help='GPU row index or exact UUID')
    parser.add_argument('--cgroup-path', type=Path)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--timeout', type=float, default=2.0)
    args = parser.parse_args(argv)
    options = dict(backends=args.backends if args.backends is not None else ('nvidia-smi', 'linux'),
                   gpu=args.gpu, cgroup_path=args.cgroup_path, pid=args.pid, timeout=args.timeout)
    try:
        if args.command == 'once':
            print(json.dumps(sample(**options), allow_nan=False), flush=True)
        else:
            duration = args.duration if args.duration is not None else (None if args.until is not None else 10.0)
            for record in sample_series(args.interval, duration, until=args.until, **options):
                print(json.dumps(record, allow_nan=False), flush=True)
    except ValueError as exc:
        parser.error(str(exc))
    except (KeyboardInterrupt, BrokenPipeError):
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
