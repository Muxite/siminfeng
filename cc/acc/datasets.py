"""Fetch reproducible subsets through the public HF datasets-server rows API."""
import argparse
from collections import Counter, defaultdict
import hashlib
from email.utils import parsedate_to_datetime
import math
import json
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from .common import sha256, write_json
except ImportError:
    from common import sha256, write_json

API = "https://datasets-server.huggingface.co/rows"
SOURCES = {
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "default", "test"),
    "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
}
# Probe at fetch time; no claim that an unverified mirror is available or canonical.
GPQA_MIRRORS = [("fingertap/GPQA-Diamond", "default", "test"),
                ("fingertap/GPQA-Diamond", "default", "train")]


def rows_url(dataset, config, split, offset=0, length=100):
    if not 1 <= length <= 100:
        raise ValueError("rows length must be 1..100")
    return API + "?" + urllib.parse.urlencode(dict(dataset=dataset, config=config,
                                                 split=split, offset=offset, length=length))


def parse_page(payload):
    """The documented envelope is {rows: [{row_idx, row, ...}], num_rows_total}."""
    rows = [(int(entry["row_idx"]), entry["row"]) for entry in payload["rows"]]
    return rows, int(payload["num_rows_total"])


def retry_after(headers):
    value = headers.get("Retry-After") if headers else None
    if value is None:
        return 0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 0
    return max(0, delay) if math.isfinite(delay) else 0


def fetch_page(url, opener):
    for attempt in range(8):
        try:
            with opener(url, timeout=60) as response:
                return response.read()
        except urllib.error.URLError as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code != 429 and not 500 <= exc.code < 600:
                raise
            if attempt == 7:
                raise
            delay = max(2 ** attempt + random.uniform(0, 1),
                        retry_after(getattr(exc, "headers", None)))
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            time.sleep(delay)


def fetch_rows(dataset, config, split, opener=None, page_delay=0.5):
    opener = opener or urllib.request.urlopen
    if not math.isfinite(page_delay) or page_delay < 0:
        raise ValueError("page delay must be finite and nonnegative")
    offset, total = 0, None
    result, pages = [], []
    while total is None or offset < total:
        url = rows_url(dataset, config, split, offset)
        if pages:
            time.sleep(page_delay)
        raw = fetch_page(url, opener)
        rows, count = parse_page(json.loads(raw))
        if total is not None and count != total:
            raise ValueError("dataset size changed while fetching")
        total = count
        if not rows or [i for i, _ in rows] != list(range(offset, offset + len(rows))):
            raise ValueError("empty or noncontiguous datasets-server page")
        result.extend(rows)
        pages.append(dict(url=url, offset=offset, length=len(rows), sha256=hashlib.sha256(raw).hexdigest()))
        offset += len(rows)
    if len(result) != total:
        raise ValueError("incomplete dataset")
    return result, pages


def normalize(name, index, row, source, seed=42):
    meta = {"row_offset": index, "category": row.get("category", row.get("domain", "all"))}
    if name == "mmlu-pro":
        question, choices = row["question"], row["options"]
        answer = row.get("answer")
        if answer is None:
            answer = chr(65 + int(row["answer_index"]))
        meta["question_id"] = row.get("question_id")
    elif name == "arc-challenge":
        question = row["question"]
        choices = row["choices"]["text"]
        labels = [str(label) for label in row["choices"]["label"]]
        answer = chr(65 + labels.index(str(row["answerKey"])))
        meta["original_id"] = row.get("id")
    elif name == "gsm8k":
        question, choices, answer = row["question"], None, row["answer"]
    elif name == "gpqa-diamond":
        question = row["Question"]
        original = [row["Correct Answer"], *[row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]]
        order = list(range(4))
        random.Random(f"{seed}:{source}:{index}").shuffle(order)
        choices = [original[i] for i in order]
        answer = chr(65 + order.index(0))
        meta["choice_permutation"] = order
        meta["category"] = row.get("Subdomain", row.get("High-level domain", "all"))
    else:
        raise ValueError(name)
    if choices is not None:
        if not 2 <= len(choices) <= 26 or answer not in [chr(65 + i) for i in range(len(choices))]:
            raise ValueError(f"invalid MCQ at row {index}")
    return dict(id=f"{name}:{index}", source=source, question=question,
                choices=choices, answer=answer, meta=meta)


def subset(rows, n, seed, mode="proportional"):
    """Seeded sampling with equal category counts or proportional largest remainders."""
    if mode not in ("proportional", "per-category"):
        raise ValueError("unknown selection mode: " + mode)
    if not 1 <= n <= len(rows):
        raise ValueError(f"n must be 1..{len(rows)}")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["meta"].get("category", "all"))].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    if mode == "per-category":
        for key in keys:
            if len(groups[key]) < n:
                raise ValueError(f"category {key!r} has {len(groups[key])} rows; need {n}")
        quotas = {key: n for key in keys}
    else:
        quotas = {key: n * len(groups[key]) // len(rows) for key in keys}
        ranked = sorted(keys, key=lambda key: n * len(groups[key]) % len(rows), reverse=True)
        for key in ranked[:n - sum(quotas.values())]:
            quotas[key] += 1
    selected = []
    for key in sorted(keys):
        group = sorted(groups[key], key=lambda row: row["id"])
        rng.shuffle(group)
        selected.extend(group[:quotas[key]])
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[*SOURCES, "gpqa-diamond"])
    parser.add_argument("-n", type=int, help="Sample size (per category for MMLU-Pro by default)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--per-category", type=int, metavar="N", help="Select exactly N rows per category")
    modes.add_argument("--proportional", action="store_true", help="Allocate -n rows proportionally")
    parser.add_argument("--page-delay", type=float, default=0.5, help="Delay between pages in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("~/models/campaign/data/acc").expanduser())
    parser.add_argument("--gpqa-dataset", help="Public Diamond mirror with original GPQA columns")
    parser.add_argument("--config", help="Override source config")
    parser.add_argument("--split", help="Override split (GPQA mirrors may use train)")
    args = parser.parse_args()
    if args.per_category is not None and args.n is not None:
        parser.error("use --per-category N or -n, not both")
    n = args.per_category if args.per_category is not None else args.n
    if n is None or n < 1:
        parser.error("provide a positive -n or --per-category N")
    equal_categories = args.per_category is not None or (args.name == "mmlu-pro" and not args.proportional)
    mode = "per-category" if equal_categories else "proportional"
    if args.name == "gpqa-diamond":
        sources = [(args.gpqa_dataset, args.config or "default", args.split or "train")] if args.gpqa_dataset else [
            (dataset, args.config or config, args.split or split) for dataset, config, split in GPQA_MIRRORS]
    else:
        dataset, config, split = SOURCES[args.name]
        sources = [(dataset, args.config or config, args.split or split)]
    failures = []
    for dataset, config, split in sources:
        try:
            raw, pages = fetch_rows(dataset, config, split, page_delay=args.page_delay)
            break
        except urllib.error.HTTPError as exc:
            if args.name != "gpqa-diamond" or exc.code not in (401, 403, 404, 422):
                raise
            failures.append(f"{dataset}/{config}/{split}: HTTP {exc.code}")
    else:
        print("SKIP GPQA-Diamond: no accessible public mirror; " + "; ".join(failures), file=sys.stderr)
        return
    source = rows_url(dataset, config, split)
    page_sources = {i: page["url"] for page in pages
                    for i in range(page["offset"], page["offset"] + page["length"])}
    normalized = [normalize(args.name, i, row, page_sources[i], args.seed) for i, row in raw]
    selected = subset(normalized, n, args.seed, mode)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f"{args.name}-{mode}-{n}-s{args.seed}.jsonl"
    metadata = path.with_suffix(".meta.json")
    if path.exists() or metadata.exists():
        raise FileExistsError(f"refusing to overwrite {path} or {metadata}")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    write_json(metadata, dict(dataset=dataset, config=config, split=split, source_url=source,
               seed=args.seed, n=len(selected), requested_n=n, mode=mode, total_rows=len(raw), pages=pages,
               row_offsets=[r["meta"]["row_offset"] for r in selected], sha256=sha256(path),
               selection=mode,
               categories=dict(Counter(str(r["meta"]["category"]) for r in selected)),
               gpqa_mirror_unverified=args.name == "gpqa-diamond"))
    print(path)


if __name__ == "__main__":
    main()
