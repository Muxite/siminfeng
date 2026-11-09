#!/usr/bin/env python3
"""Select static experts for ONE GPU / ONE homogeneous layout group.

Budget is persistent bank bytes, including alignment and the MMQ guard slot.
Requires raw routing_trace JSONL, not aggregated experiments/routing profiles.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

# Reuse the pinned trace component's strict Granite top-8 record validation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'routing_trace'))
from trace_replay import read_trace


def bank_bytes(slots, strides, alignment):
    return sum(((stride * (slots + 1) + alignment - 1) // alignment) * alignment
               for stride in strides)


def select_hot(records, *, budget, strides, alignment, layers, train_seqs,
               train_phase, train_start, train_end):
    if (budget < 0 or not strides or any(s <= 0 for s in strides)
            or alignment <= 0 or layers <= 0 or not train_seqs
            or any(type(s) is not int or s < 0 for s in train_seqs)
            or train_phase not in ('prefill', 'decode')
            or not 0 <= train_start < train_end):
        raise ValueError('invalid budget/layout/TRAIN split')
    if any(r['layer'] >= layers for r in records):
        raise ValueError('trace layer outside declared model')
    # Match the engine, which currently requires >= top-8 slots per group,
    # even though static mode itself does not require eight resident experts.
    if bank_bytes(8, strides, alignment) > budget:
        raise ValueError('budget cannot allocate the engine minimum of eight slots')
    capacity = 8
    while capacity < layers * 32 and bank_bytes(capacity + 1, strides, alignment) <= budget:
        capacity += 1
    train = [r for r in records if r['seq'] in train_seqs
             and r['phase'] == train_phase and train_start <= r['pos'] < train_end]
    if not train:
        raise ValueError('TRAIN split contains no records')
    counts = Counter((r['layer'], e) for r in train for e in r['experts'])
    # Equal-size items: global frequency maximizes TRAIN hits at this capacity.
    ranked = sorted(counts, key=lambda key: (-counts[key], *key))
    selected = sorted(ranked[:capacity])
    return selected, {
        'schema_version': 1,
        'scope': 'one GPU, one homogeneous expert-layout group; 32 experts, top-8',
        'train': {'seqs': sorted(train_seqs), 'phase': train_phase,
                  'pos_start_inclusive': train_start, 'pos_end_exclusive': train_end},
        'train_records': len(train), 'train_requests': 8 * len(train),
        'train_hits': sum(counts[key] for key in selected),
        'budget_bytes': budget, 'expert_bank_strides': list(strides),
        'alignment': alignment, 'layers': layers, 'capacity': capacity,
        'engine_bank_bytes': bank_bytes(capacity, strides, alignment),
        'selected_experts': len(selected), 'selected_payload_bytes': len(selected) * sum(strides),
        'tie_break': 'frequency descending, layer ascending, expert ascending',
        'unobserved_experts': 'not selected; allocated spare slots remain empty',
        'evaluation': 'excluded from ranking; use disjoint sequences or positions outside TRAIN',
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path)
    p.add_argument('--budget-bytes', type=int, required=True)
    p.add_argument('--expert-bytes', type=int, nargs='+', required=True,
                   help='actual nb[2] per bank: gate/up/down or gate_up/down')
    p.add_argument('--alignment', type=int, required=True, help='GPU buffer type alignment')
    p.add_argument('--layers', type=int, default=24)
    p.add_argument('--train-seq', type=int, nargs='+', required=True)
    p.add_argument('--train-phase', choices=['prefill', 'decode'], required=True)
    p.add_argument('--train-pos-start', type=int, default=0)
    p.add_argument('--train-pos-end', type=int, required=True, help='exclusive absolute position')
    args = p.parse_args(argv)
    try:
        selected, report = select_hot(
            read_trace(args.trace), budget=args.budget_bytes, strides=args.expert_bytes,
            alignment=args.alignment, layers=args.layers, train_seqs=set(args.train_seq),
            train_phase=args.train_phase, train_start=args.train_pos_start,
            train_end=args.train_pos_end)
        report['trace_sha256'] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
        metadata = json.dumps(report, sort_keys=True)
        args.output.write_text('# LLAMA_MOE_CACHE_STATIC v1: layer expert (zero based)\n'
                               + '# ' + metadata + '\n'
                               + ''.join(f'{il} {e}\n' for il, e in selected), encoding='utf-8')
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(metadata)
    except (OSError, ValueError) as error:
        p.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
