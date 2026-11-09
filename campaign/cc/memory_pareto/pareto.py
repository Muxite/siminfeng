#!/usr/bin/env python3
"""Describe measured memory/speed/quality trade-offs; Python standard library only."""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELDS = ('config', 'desc', 'class', 'gpu_mib', 'host_mib', 'tg_ts', 'pp_ts',
          'kld', 'top1_pct', 'ppl_ratio', 'notes')
NUMERIC = ('gpu_mib', 'host_mib', 'tg_ts', 'pp_ts', 'kld', 'top1_pct', 'ppl_ratio')
OBJECTIVES = ('gpu_mib', 'tg_ts', 'kld')
QUALITY = ('kld', 'top1_pct', 'ppl_ratio')


def number(value):
    """Missing/nonfinite values are unavailable, never zero-filled."""
    if value is None or str(value).strip().lower() in ('', 'na', 'n/a', 'null', 'none', 'nan'):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid numeric value: %r' % value) from exc
    return parsed if math.isfinite(parsed) else None


def read_csv(path):
    rows = []
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError('missing CSV columns: ' + ', '.join(sorted(missing)))
        for line, raw in enumerate(reader, 2):
            row = {key: (raw.get(key) or '').strip() for key in FIELDS}
            if not row['config']:
                raise ValueError('line %d: config is required' % line)
            if row['class'] not in ('transferable', 'discrete-only', 'mixed'):
                raise ValueError('line %d: invalid class %r' % (line, row['class']))
            for key in NUMERIC:
                row[key] = number(row[key])
                if row[key] is not None and row[key] < 0:
                    raise ValueError('line %d: %s must be nonnegative' % (line, key))
            if row['top1_pct'] is not None and row['top1_pct'] > 100:
                raise ValueError('line %d: top1_pct must be <= 100' % line)
            row['source'] = '%s:%d' % (path, line)
            rows.append(row)
    return rows


def read_lab(root):
    """Keep each successful measurement separate; never infer a quality join."""
    definitions = {}
    for path in sorted((root / 'tests').glob('*.json')):
        definitions.update(json.loads(path.read_text()))
    rows = []
    for path in sorted((root / 'jobs').glob('*/att-*/result.json')):
        result = json.loads(path.read_text())
        job_path = path.parent.parent / 'job.json'
        spec = json.loads(job_path.read_text()).get('spec', {}) if job_path.exists() else {}
        if result.get('execution', {}).get('status') != 'completed':
            continue
        for name, test in result.get('quality', {}).get('tests', {}).items():
            if test.get('exit_code') != 0 or test.get('partial') or test.get('ok') is False:
                continue
            definition = definitions.get(name.split('#')[0], {})
            memory = test.get('mem', {})
            samples = test.get('samples') or [{}]
            for index, sample in enumerate(samples):
                row = dict.fromkeys(FIELDS, None)
                row.update(config='%s/%s/%s/sample%d' % (path.parent.parent.name, path.parent.name, name, index),
                           desc=definition.get('model_file') or spec.get('model') or 'unknown model',
                           **{'class': 'unclassified'},
                           gpu_mib=number(memory.get('gpu_peak_mib')),
                           host_mib=number(memory.get('host_cgroup_peak_mib')),
                           tg_ts=number(sample.get('avg_ts')) if sample.get('n_prompt') == 0 and sample.get('n_gen', 0) > 0 else None,
                           pp_ts=number(sample.get('avg_ts')) if sample.get('n_gen') == 0 and sample.get('n_prompt', 0) > 0 else None,
                           notes='Lab fallback; class unassigned; no cross-test joins. ppl=%s; ppl_stderr=%s; args=%s; env=%s' % (
                               test.get('ppl'), test.get('ppl_stderr'), definition.get('args', []), definition.get('env', {})),
                           source=str(path) + ':' + name)
                rows.append(row)
    return rows


def frontier(rows):
    """Return indices not strictly dominated in all three specified objectives."""
    eligible = [i for i, row in enumerate(rows) if all(row.get(k) is not None for k in OBJECTIVES)]
    def dominates(a, b):
        av = (a['gpu_mib'], -a['tg_ts'], a['kld'])
        bv = (b['gpu_mib'], -b['tg_ts'], b['kld'])
        return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))
    return {i for i in eligible if not any(dominates(rows[j], rows[i]) for j in eligible if j != i)}


def analyze(rows, budget):
    rows = [dict(row) for row in rows]
    optimal = frontier(rows)
    fitting = [i for i, row in enumerate(rows) if row.get('gpu_mib') is not None and row['gpu_mib'] <= budget]
    budget_optimal = {fitting[i] for i in frontier([rows[i] for i in fitting])}
    class_optimal = set()
    for category in {row['class'] for row in rows}:
        indices = [i for i, row in enumerate(rows) if row['class'] == category]
        class_optimal.update(indices[i] for i in frontier([rows[i] for i in indices]))
    for i, row in enumerate(rows):
        row.update(missing_quality=[k for k in QUALITY if row.get(k) is None],
                   missing_objectives=[k for k in OBJECTIVES if row.get(k) is None],
                   fits_budget=None if row.get('gpu_mib') is None else i in fitting,
                   pareto_optimal=i in optimal, budget_pareto_optimal=i in budget_optimal,
                   class_pareto_optimal=i in class_optimal)
    return sorted(rows, key=lambda row: (row.get('gpu_mib') is None, row.get('gpu_mib') or 0, row['config']))


def fmt(value):
    if value is None:
        return 'NA'
    if isinstance(value, float):
        return '%.6g' % value
    return str(value).replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def scatter(rows, budget, width=58, height=14):
    points = [(i + 1, r) for i, r in enumerate(rows) if r['gpu_mib'] is not None and r['kld'] is not None]
    if not points:
        return 'No scatter points: GPU MiB and KLD are both required.'
    xmax = max([budget] + [r['gpu_mib'] for _, r in points]) or 1
    ymax = max(r['kld'] for _, r in points) or 1
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    bx = round(budget / xmax * (width - 1))
    for line in grid:
        line[bx] = ':'
    for _, row in points:
        x = round(row['gpu_mib'] / xmax * (width - 1))
        y = height - 1 - round(row['kld'] / ymax * (height - 1))
        symbol = '*' if row['pareto_optimal'] else 'o'
        grid[y][x] = '+' if grid[y][x] in ('o', '*', '+') else symbol
    lines = ['KLD (lower is better)', '%g +%s' % (ymax, '-' * width)]
    lines += ['     |' + ''.join(line) for line in grid]
    lines += ['   0 +' + '-' * width, '      0 to %g GPU MiB (lower is better)' % xmax,
              '      : budget %g MiB; * frontier; o other; + overlapping points' % budget,
              'Points (table row: GPU MiB, KLD): ' + '; '.join('%d: %g, %g' % (i, r['gpu_mib'], r['kld']) for i, r in points)]
    return '\n'.join(lines)


CAVEATS = ('Pareto objectives: minimize GPU MiB, maximize TG tokens/s, minimize KLD. '
           'Only rows with all three objectives are eligible; exact ties all remain optimal. '
           'Budget fit is GPU MiB <= the budget, independent of quality availability. '
           'Global and budget frontiers are descriptive pooled comparisons, not transferable claims. '
           'Class frontiers compare rows only within their labelled class. Discrete-only, mixed and '
           'unclassified measurements do not establish transferable savings. No statistical significance '
           'is established; workloads, hardware, memory accounting and quality references must be comparable.')


def markdown(rows, budget, source):
    columns = [('config', 'Config'), ('desc', 'Description'), ('class', 'Class'),
               ('gpu_mib', 'GPU MiB'), ('host_mib', 'Host MiB'), ('tg_ts', 'TG t/s'),
               ('pp_ts', 'PP t/s'), ('kld', 'KLD'), ('top1_pct', 'Top1 %'), ('ppl_ratio', 'PPL ratio')]
    lines = ['# Memory / quality / speed Pareto report', '', 'Input: ' + fmt(source), '',
             'GPU budget: **%g MiB** (default 700: rounded scaled analogue of 500 GB -> 128 GB on a 2.67 GB model; not a physical capacity prediction).' % budget,
             '', CAVEATS, '', '| ' + ' | '.join(label for _, label in columns) + ' | Fits budget | Pareto | Budget Pareto | Class Pareto | Missing quality | Missing objectives | Notes | Source |',
             '| ' + ' | '.join(['---'] * (len(columns) + 8)) + ' |']
    for row in rows:
        cells = [fmt(row[key]) for key, _ in columns]
        cells += ['unknown' if row['fits_budget'] is None else ('yes' if row['fits_budget'] else 'no')]
        cells += ['yes' if row[k] else ('ineligible' if row['missing_objectives'] else 'no') for k in ('pareto_optimal', 'budget_pareto_optimal', 'class_pareto_optimal')]
        cells += [', '.join(row['missing_quality']) or 'none', ', '.join(row['missing_objectives']) or 'none', fmt(row['notes']), fmt(row['source'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    if not rows:
        lines += ['', 'No measurements available.']
    lines += ['', '```text', scatter(rows, budget), '```', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', type=Path, help='primary CSV; explicit missing paths are errors')
    parser.add_argument('--budget-mib', type=float, default=700)
    parser.add_argument('--lab-root', type=Path, default=ROOT / 'lab', help='fallback if the default CSV is absent')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_mib) or args.budget_mib < 0:
        parser.error('--budget-mib must be finite and nonnegative')
    source = args.csv or ROOT / 'experiments/memory/configs.csv'
    try:
        if args.csv is not None or source.exists():
            rows = read_csv(source)
            source_kind = 'csv'
        else:
            rows = read_lab(args.lab_root)
            source = args.lab_root
            source_kind = 'lab-fallback'
        rows = analyze(rows, args.budget_mib)
        report = markdown(rows, args.budget_mib, str(source))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'pareto.md').write_text(report, encoding='utf-8')
        payload = dict(source=str(source), source_kind=source_kind, budget_mib=args.budget_mib,
                       objectives={'gpu_mib': 'min', 'tg_ts': 'max', 'kld': 'min'},
                       caveats=CAVEATS, rows=rows)
        (args.output_dir / 'pareto.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(report, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
