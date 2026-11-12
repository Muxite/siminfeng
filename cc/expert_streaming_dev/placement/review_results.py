#!/usr/bin/env python3
"""Read the six-run campaign without running any engine or GPU work."""
import json
from pathlib import Path
import sys


def review(root):
    names = ['01-baseline', '02-no-offload', '05-adaptive', '06-baseline-replay']
    runs = {name: json.loads((root/name/'summary.json').read_text()) for name in names}
    a, cpu, candidate, b = (runs[name] for name in names)
    reasons = []
    for field in ('prompt_tokens', 'prompt_sha256', 'decode_runs', 'stdout_sha256', 'host_id', 'image'):
        if len({r[field] for r in runs.values()}) != 1:
            reasons.append(f'{field} differs')
    if abs(b['decode_ms_per_token'] / a['decode_ms_per_token'] - 1) > .05:
        reasons.append('baseline decode drift >5%')
    if abs(b['prompt_ms'] / a['prompt_ms'] - 1) > .10:
        reasons.append('baseline prompt drift >10%')
    prompt_reference = (a['prompt_ms'] + b['prompt_ms']) / 2
    decode_reference = (a['decode_ms_per_token'] + b['decode_ms_per_token']) / 2
    improvement = 1 - candidate['prompt_ms'] / prompt_reference
    decode_change = candidate['decode_ms_per_token'] / decode_reference - 1
    if improvement < .10:
        reasons.append('candidate prompt improvement <10%')
    if abs(decode_change) > .05:
        reasons.append('candidate decode change exceeds 5%')
    return {'status': 'reject_or_inconclusive' if reasons else 'passed_this_workload_screen',
            'reasons': reasons, 'prompt_improvement_fraction': improvement, 'decode_change_fraction': decode_change,
            'no_offload_prompt_ms': cpu['prompt_ms'],
            'limitation': 'One calibration pair and one replay; this does not establish transfer to other workloads.'}


if __name__ == '__main__':
    try:
        root = Path(sys.argv[1])
        record = review(root)
        (root/'review.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))
    except (KeyError, ValueError, OSError, IndexError) as exc:
        sys.exit(f'review_results: {exc}')
