#!/usr/bin/env python3
"""Recompute a gate from saved summaries without changing original evidence."""
import argparse
import json
from pathlib import Path
import sys

if __package__:
    from . import abrun, gate, run_summary
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cc.abrun import abrun, gate, run_summary


def recompute(outdir, exclude=(), metric=None, reason=''):
    outdir = Path(outdir)
    spec = json.loads((outdir / 'spec.json').read_text())
    grouped = {arm: [] for arm in spec['arms']}
    paths = sorted(outdir.glob('*/summary.json'))
    known = {p.parent.name for p in paths}
    excluded = list(dict.fromkeys(exclude))
    unknown = set(excluded) - known
    if unknown:
        raise ValueError('unknown excluded run IDs: ' + ', '.join(sorted(unknown)))
    included, invalid = [], []
    for path in paths:
        run_id = path.parent.name
        if run_id in excluded:
            continue
        arm = run_id.split('-', 1)[1]
        if arm not in grouped:
            raise ValueError(f'{run_id}: unknown arm {arm!r}')
        summary = json.loads(path.read_text())
        # Recompute available raw evidence, including precision and phase metrics.
        # Summary-only archives remain readable; never overwrite their files.
        if (path.parent / 'stderr.txt').exists():
            observed = run_summary.summarize(path.parent, abrun.switch_patterns(spec), write=False)
            summary.update(observed)
        elif 'switch_matches' not in summary:
            observed = run_summary.summarize(path.parent, abrun.switch_patterns(spec), write=False)
            summary.update({key: observed[key] for key in ('switch_matches', 'switch_lines')})
        invalid.extend(f'{run_id}: {error}' for error in abrun.switch_errors(spec, arm, summary))
        grouped[arm].append(summary)
        included.append(run_id)
    verdict = gate.evaluate(grouped, metric or spec.get('metric', 'eval_ms_per_token'),
                            direction=spec.get('direction') if metric is None else None)
    invalid.extend(verdict['reasons'])
    verdict.update(status='INVALID' if invalid else 'completed', invalid_reasons=invalid,
                   excluded_runs=excluded, included_runs=included, reason=reason)
    if invalid:
        for comparison in verdict['comparisons'].values():
            comparison['verdict'] = 'null'
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('--exclude', nargs='+', action='extend', default=[], metavar='RUN_ID')
    parser.add_argument('--metric')
    parser.add_argument('--reason', default='')
    parser.add_argument('--no-write', action='store_true')
    parser.add_argument('--dry', action='store_true', help='print secondary table without writing any files')
    args = parser.parse_args()
    try:
        verdict = recompute(args.outdir, args.exclude, args.metric, args.reason)
        text = json.dumps(verdict, indent=2) + '\n'
        if not args.no_write and not args.dry:
            (args.outdir / 'verdict-regate.json').write_text(text)
        if args.dry:
            print(f'{args.outdir.name}: metric={verdict["metric"]}, direction={verdict["direction"]}')
            print(gate.secondary_table(verdict))
        else:
            print(text, end='')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'regate: {exc}\n')


if __name__ == '__main__':
    main()
