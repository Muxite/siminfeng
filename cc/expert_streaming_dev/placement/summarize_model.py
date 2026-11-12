#!/usr/bin/env python3
"""Extract separate prompt/decode timings; reject incomplete or guarded-out runs."""
import hashlib
import json
from pathlib import Path
import re
import sys


def summarize(out):
    result = json.loads((out / 'result.json').read_text())
    text = (out / 'stderr.txt').read_text(errors='replace')
    prompt = re.findall(r'prompt eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens', text)
    decode = re.findall(r'(?<!prompt )\beval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs', text)
    if result['status'] != 'ok' or result['exit_code'] != 0 or result['watchdog_trips'] != 0:
        raise ValueError(f"invalid run: {result['status']}")
    if not prompt or not decode:
        raise ValueError('missing prompt/decode timing lines')
    prompt_ms, prompt_n = float(prompt[-1][0]), int(prompt[-1][1])
    decode_ms, decode_n = float(decode[-1][0]), int(decode[-1][1])
    if prompt_n <= 0 or decode_n != 63:
        raise ValueError(f'expected 64-token completion (63 decode runs); got prompt={prompt_n} decode={decode_n}')
    command = (out / 'cmd.txt').read_text()
    if 'expert_fill: inactive' in text or 'expert_pregate: inactive' in text:
        raise ValueError('fill or pregate fell back; inspect stderr.txt')
    if 'expert_fill: active' not in text or 'expert_pregate: active' not in text:
        raise ValueError('expected fill and pregate activation logs are missing')
    fill = re.findall(r'decode_bytes_read=(\d+)', text)
    demand = re.findall(r'demand_wait_ns_per_token=([\d.]+)', text)
    record = {
        'status': 'ok', 'host_id': result['host_id'], 'image': result['image'],
        'argv': result['argv'], 'env': result['env'],
        'prompt_tokens': prompt_n, 'prompt_ms': prompt_ms,
        'decode_runs': decode_n, 'decode_ms': decode_ms, 'decode_ms_per_token': decode_ms / decode_n,
        'prompt_plus_decode_ms': prompt_ms + decode_ms, 'wall_s': result['wall_s'],
        'stdout_sha256': hashlib.sha256((out / 'stdout.txt').read_bytes()).hexdigest(),
        'prompt_sha256': (out / 'prompt.sha256').read_text().split()[0],
        'cmd_sha256': hashlib.sha256(command.encode()).hexdigest(),
        'decode_bytes_read': int(fill[-1]) if fill else None,
        'demand_wait_ms_per_token': float(demand[-1]) / 1e6 if demand else None,
        'timing': 'untraced',
    }
    for phase in ('start', 'end'):
        record[f'page_cache_{phase}'] = json.loads((out / f'page_cache.{phase}.json').read_text())
    start = record['page_cache_start']
    if start.get('protocol') != 'cold' or start.get('error') or start.get('resident_pct', 100) > 1:
        raise ValueError('expected successful per-file cold page-cache preparation')
    return record


if __name__ == '__main__':
    try:
        root = Path(sys.argv[1])
        summary = summarize(root)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({k: summary[k] for k in ('prompt_ms', 'decode_ms_per_token', 'stdout_sha256')}))
    except (ValueError, KeyError, OSError, IndexError) as exc:
        sys.exit(f'summarize_model: {exc}')
