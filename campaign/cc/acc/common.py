"""Shared scoring, HTTP and durable result helpers (no third-party imports)."""
import hashlib
import json
import math
from pathlib import Path
import urllib.request


def wilson(correct, n, z=1.959963984540054):
    if not 0 <= correct <= n:
        raise ValueError("require 0 <= correct <= n")
    if not n:
        return None, None
    p = correct / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., center - radius), min(1., center + radius)


def request_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON; repair interrupted line before resuming") from exc
    return rows


def indexed(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"duplicate id: {row['id']}")
        result[row["id"]] = row
    return result


def pending_items(items, output):
    done = indexed(read_jsonl(output))
    ids = indexed(items)
    if not done.keys() <= ids.keys():
        raise ValueError("result contains IDs outside this dataset")
    return [item for item in items if item["id"] not in done]


def summary(rows):
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    lo, hi = wilson(correct, n)
    rates = [r["decode_tps"] for r in rows if r.get("decode_tps") is not None]
    return dict(n=n, correct=correct, acc=correct / n if n else None,
                wilson95_lo=lo, wilson95_hi=hi,
                no_letter=sum(bool(r.get("no_letter", False)) for r in rows),
                mean_ms_per_item=sum(r["elapsed_ms"] for r in rows) / n if n else None,
                prompt_tokens_total=sum(r["prompt_tokens"] for r in rows),
                mean_decode_tps=sum(rates) / len(rates) if rates else None)


def response_metrics(response):
    timing = response.get("timings", {})
    rate = timing.get("predicted_per_second")
    if rate is None and timing.get("predicted_ms", 0) > 0:
        rate = timing["predicted_n"] * 1000 / timing["predicted_ms"]
    return dict(prompt_tokens=response.get("usage", {}).get("prompt_tokens",
                response.get("tokens_evaluated", timing.get("prompt_n", 0))), decode_tps=rate)


def thinking_settings(props):
    # Templates without this variable cannot promise to disable thinking.
    supported = "enable_thinking" in (props.get("chat_template") or "")
    return {"chat_template_kwargs": {"enable_thinking": False}} if supported else {}


def messages(prompt, system, props):
    result = []
    if props.get("chat_template_caps", {}).get("supports_system_role") is not False:
        result.append({"role": "system", "content": system})
    result.append({"role": "user", "content": prompt})
    return result


def prepare_run(args, mode, protocol):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    model = Path(args.model).expanduser().resolve()
    stat = model.stat()
    identity = dict(mode=mode, dataset_sha256=sha256(args.data), image=args.image,
                    model=str(model), model_size=stat.st_size, model_mtime_ns=stat.st_mtime_ns,
                    model_sha256=sha256(model),
                    engine_argv=args.engine_arg, env=args.env, protocol=protocol)
    manifest = out / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("resume settings differ from run.json; use a new outdir")
    elif (out / "items.jsonl").exists():
        raise ValueError("cannot resume without run.json")
    else:
        write_json(manifest, identity)
    return out
