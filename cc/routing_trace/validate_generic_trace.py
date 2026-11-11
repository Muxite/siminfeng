#!/usr/bin/env python3
"""Validate a generic capture, including its clean-exit footer (streaming)."""
import json
import math
import sys
from pathlib import Path


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('invalid nonnegative integer')
    return value


def validate(path):
    header = None
    footer = None
    records = tokens = 0
    current = None
    observed = []
    previous_phase = None
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            if not line.endswith('\n'):
                raise ValueError('unterminated line')
            row = json.loads(line)
            if not isinstance(row, dict) or footer is not None:
                raise ValueError('invalid object or data after footer')
            if header is None:
                header = row
                n = integer(header.get('n_layer'), 1)
                if (header.get('type') != 'metadata' or header.get('layers') != n or
                    not isinstance(header.get('arch'), str) or
                    not isinstance(header.get('model_path'), str) or
                    header.get('shared_experts_routed') is not False):
                    raise ValueError('invalid header')
                for key in ('n_expert', 'k', 'shared_experts_present'):
                    if not isinstance(header.get(key), list) or len(header[key]) != n:
                        raise ValueError('invalid per-layer dimensions')
                for e, k, shared in zip(header['n_expert'], header['k'], header['shared_experts_present']):
                    integer(e)
                    integer(k)
                    if (not 0 <= k <= e or bool(e) != bool(k) or type(shared) is not bool):
                        raise ValueError('invalid layer dimensions')
                routed = [i for i, k in enumerate(header['k']) if k]
                if not routed or header.get('experts') != max(header['n_expert']):
                    raise ValueError('no routing or invalid replay dimensions')
                continue
            if 'trace_complete' in row:
                if row.get('trace_complete') is not True or row.get('type') != 'metadata':
                    raise ValueError('invalid footer')
                footer = row
                continue
            tok, layer = integer(row.get('tok')), integer(row.get('layer'))
            if type(row.get('seq')) is not int or row['seq'] != 0:
                raise ValueError('capture requires seq=0')
            phase = row.get('phase')
            if phase not in ('prefill', 'decode') or (previous_phase == 'decode' and phase != 'decode'):
                raise ValueError('invalid phase')
            if tok != current:
                if (current is not None and observed != routed) or tok != tokens:
                    raise ValueError('missing token or layer coverage')
                current, observed = tok, []
                tokens += 1
                previous_phase = phase
            if phase != previous_phase or layer not in routed or layer in observed:
                raise ValueError('invalid layer/phase coverage')
            observed.append(layer)
            ids, weights = row.get('experts'), row.get('weights')
            k, e = header['k'][layer], header['n_expert'][layer]
            if (not isinstance(ids, list) or len(ids) != k or
                any(type(i) is not int or not 0 <= i < e for i in ids) or len(set(ids)) != k):
                raise ValueError('invalid expert IDs')
            if (not isinstance(weights, list) or len(weights) != k or
                any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights)):
                raise ValueError('invalid weights')
            records += 1
    if (footer is None or not records or observed != routed or
        integer(footer.get('tokens')) != tokens or integer(footer.get('records')) != records):
        raise ValueError('missing/mismatched completion footer or coverage')
    return dict(tokens=tokens, records=records)


if __name__ == '__main__':
    print(json.dumps(validate(sys.argv[1])))
