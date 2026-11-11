#!/usr/bin/env python3
"""Lead-only network entry point: equal-per-category frozen MMLU-Pro sets."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cc.acc import common, datasets


def balanced_subset(rows, per_category, seed=1):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row['meta']['category'])].append(row)
    if len(groups) != 14:
        raise ValueError(f'expected 14 MMLU-Pro categories, found {len(groups)}')
    selected = []
    # Use datasets.py's deterministic sampling independently within each category.
    for category in sorted(groups):
        if len(groups[category]) < per_category:
            raise ValueError(f'category {category} has fewer than {per_category} items')
        selected.extend(datasets.subset(groups[category], per_category, seed))
    random.Random(seed).shuffle(selected)
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir', type=Path, default=Path('~/models/campaign/data/acc').expanduser())
    args = p.parse_args()
    args.outdir = args.outdir.expanduser()
    paths = [args.outdir / f'mmlu_pro_{n}.jsonl' for n in (140, 70)]
    for path in paths:
        if path.exists() or path.with_suffix('.meta.json').exists():
            raise FileExistsError(f'refusing overwrite: {path}; inspect existing frozen sets')
    dataset, config, split = datasets.SOURCES['mmlu-pro']
    raw, pages = datasets.fetch_rows(dataset, config, split)
    source = datasets.rows_url(dataset, config, split)
    normalized = [datasets.normalize('mmlu-pro', i, row, source, seed=1) for i, row in raw]
    # Compute both before writing so malformed source data leaves neither set behind.
    selected_sets = [balanced_subset(normalized, count, seed=1) for count in (10, 5)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path, selected in zip(paths, selected_sets):
        with path.open('x') as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        common.write_json(path.with_suffix('.meta.json'), dict(dataset=dataset, config=config,
            split=split, source_url=source, seed=1, n=len(selected), total_rows=len(raw), pages=pages,
            row_offsets=[r['meta']['row_offset'] for r in selected], sha256=common.sha256(path),
            selection='equal per category using cc.acc.datasets.subset independently; seed 1',
            categories=dict(Counter(r['meta']['category'] for r in selected))))
        print(path)


if __name__ == '__main__':
    main()
