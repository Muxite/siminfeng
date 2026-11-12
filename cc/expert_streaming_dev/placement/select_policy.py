#!/usr/bin/env python3
"""One opt-in policy, chosen before process startup using existing engine knobs."""
import argparse
import csv
import json
import math
from pathlib import Path


def select(baseline, cpu, curve, mem_available_mib, enabled):
    decision = {'enabled': enabled, 'threshold': 32, 'flag': '--op-offload', 'reason': 'disabled; engine default',
                'mem_available_mib': mem_available_mib, 'decision_level': 'before a run', 'curve_threshold': None}
    if not enabled:
        return decision
    for run in (baseline, cpu):
        if run['status'] != 'ok' or run['timing'] != 'untraced':
            raise ValueError('calibration runs must be valid and untraced')
    for key in ('prompt_tokens', 'prompt_sha256', 'decode_runs', 'stdout_sha256', 'host_id', 'image'):
        if baseline[key] != cpu[key]:
            raise ValueError(f'calibration differs in {key}; inspect before enabling policy')
    if abs(cpu['decode_ms_per_token'] / baseline['decode_ms_per_token'] - 1) > 0.05:
        raise ValueError('decode drift exceeds 5%; calibrations do not isolate prompt cost')
    expected = {1, 2, 4, 8, 16, 32, 64, 128, 512}
    times = {}
    for shape in ('up_gate', 'down'):
        rows = [r for r in curve if r['shape'] == shape]
        if len(rows) != 9 or {int(r['n_tokens']) for r in rows} != expected:
            raise ValueError('incomplete crossover curve')
        for r in rows:
            for key in ('cpu_boundary_ms', 'gpu_staged_ms'):
                if not math.isfinite(float(r[key])) or float(r[key]) <= 0:
                    raise ValueError('invalid crossover timing')
            if not 0 <= float(r['nmse']) < 5e-4:
                raise ValueError('crossover numerical check failed')
            n = int(r['n_tokens'])
            c, g = times.get(n, (0, 0))
            factor = 2 if shape == 'up_gate' else 1
            times[n] = c + factor * float(r['cpu_boundary_ms']), g + factor * float(r['gpu_staged_ms'])
    cross = next((n for n in sorted(expected) if n >= 32 and all(g <= 0.9*c for t, (c, g) in times.items() if t >= n)), None)
    decision['curve_threshold'] = cross
    # Budget: supplied 2.5 GiB non-experts + 537 MiB scratch + 143 MiB KV/recurrent + 1 GiB reserve.
    if mem_available_mib < 4264:
        decision.update(flag='--no-op-offload', reason='less than 4264 MiB available before model load')
    elif cpu['prompt_ms'] <= 0.9 * baseline['prompt_ms']:
        decision.update(flag='--no-op-offload', reason='CPU prompt at least 10% faster on this calibrated workload')
    elif baseline['prompt_ms'] <= 0.9 * cpu['prompt_ms'] and cross is not None and cross <= min(512, baseline['prompt_tokens']):
        decision.update(threshold=cross, reason='GPU prompt wins; use first sustained staged crossover, floor 32')
    else:
        decision.update(reason='insufficient evidence for a change; keep threshold 32')
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--curve', type=Path, required=True)
    parser.add_argument('--meminfo', type=Path, default=Path('/proc/meminfo'))
    parser.add_argument('--prompt-sha256', help='Hash of the prompt about to run')
    parser.add_argument('--enable', action='store_true')
    args = parser.parse_args()
    # Default off also works before any calibration artifacts exist.
    if not args.enable:
        decision = select(None, None, None, None, False)
    else:
        mem = dict(line.split(':', 1) for line in args.meminfo.read_text().splitlines())
        available = int(mem['MemAvailable'].split()[0]) / 1024
        baseline = json.loads(args.baseline.read_text())
        if args.prompt_sha256 != baseline['prompt_sha256']:
            raise ValueError('prompt differs from calibration')
        decision = select(baseline, json.loads(args.cpu.read_text()), list(csv.DictReader(args.curve.open())), available, True)
        decision['calibration_files'] = [str(args.baseline), str(args.cpu), str(args.curve)]
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f'select_policy: {exc}')
