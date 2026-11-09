#!/usr/bin/env python3
"""Header-only GGUF weight/KV accounting. No tensor payloads are accessed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

# Keep the reference tree read-only, including Python's implicit cache writes.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref/llama.cpp/gguf-py"))
import gguf  # noqa: E402


# Ordered, explicit llama.cpp GGUF names; suffixes include weight/bias/scales and
# hybrid .weight.hot/.weight.cold. Architecture labels document coverage rather
# than restricting matches (converters reuse these names across architectures).
ROLE_PATTERNS = [
    ("router", r"blk\.\d+\.ffn_(?:gate_inp(?:_shexp)?|exp_probs_b|expert_ids)(?:\..*)?",
     "granitemoe, llama/mixtral, qwen2moe/qwen3moe, deepseek2, gpt-oss, hybrid ID maps"),
    ("shared_ffn", r"blk\.\d+\.ffn_(?:gate|up|down|gate_up)_(?:shexp|shared_exps)(?:\..*)?",
     "qwen2moe/qwen3moe and deepseek2 shared experts"),
    ("expert_ffn", r"blk\.\d+\.ffn_(?:gate|up|down|gate_up|norm)_exps(?:\..*)?",
     "granitemoe, llama/mixtral, qwen2moe/qwen3moe, deepseek2, gpt-oss fused gate_up; hot/cold"),
    ("norm", r"(?:output_norm|blk\.\d+\.(?:attn|ffn|layer|post_attention|post_ffn)[\w]*norm[\w]*)(?:\..*)?",
     "all architectures, including deepseek2 attn_q_a_norm/attn_kv_a_norm"),
    ("attn_qkv", r"blk\.\d+\.attn_(?:q|k|v|qkv|q_a|q_b|kv_a_mqa|kv_b|k_b|v_b|kv)(?:\..*)?",
     "GQA/MHA and deepseek2 MLA projections"),
    ("attn_out", r"blk\.\d+\.attn_(?:output|output_a|output_b|sinks|gate)(?:\..*)?",
     "attention output and gpt-oss attention sinks"),
    ("dense_ffn", r"blk\.\d+\.ffn_(?:gate|up|down|gate_up|act)(?:\..*)?",
     "dense FFNs, including deepseek2 initial dense layers"),
    ("embed", r"(?:token_embd|position_embd|token_types)(?:\..*)?", "token/position embeddings"),
    ("output", r"output(?:\..*)?", "untied output head; tied embeddings counted only once"),
]
ROLES = tuple(dict.fromkeys(role for role, _, _ in ROLE_PATTERNS)) + ("other",)
COMPILED_ROLES = [(role, re.compile(pattern)) for role, pattern, _ in ROLE_PATTERNS]
CACHE_TYPES = ("F16", "BF16", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0")


def classify(name):
    return next((role for role, pattern in COMPILED_ROLES if pattern.fullmatch(name)), "other")


class HeaderReader(gguf.GGUFReader):
    """Reuse pinned metadata parsing, but never create or read payload views.

    GGUFReader maps the file read-only; only header pages are touched. Overriding
    this hook also allows inspection of metadata for tensor types that have no
    numpy dequantizer. This is the only use of a private gguf-py reader hook.
    """

    def _build_tensors(self, start_offs, fields):
        self.tensor_headers = []
        seen = set()
        for field in fields:
            _, _, _, dims, raw_type, offset = field.parts
            name = field.name
            if name in seen:
                raise ValueError(f"duplicate tensor: {name}")
            seen.add(name)
            shape = [int(x) for x in dims]
            dtype = gguf.GGMLQuantizationType(int(raw_type[0]))
            block, size = gguf.GGML_QUANT_SIZES[dtype]
            if not shape or any(d <= 0 for d in shape) or shape[0] % block:
                raise ValueError(f"invalid shape/block alignment for {name}: {shape}, {dtype.name}")
            nbytes = math.prod(shape) // block * size
            if start_offs + int(offset[0]) + nbytes > self.data.nbytes:
                raise ValueError(f"tensor extends beyond file: {name}")
            self.tensor_headers.append({"name": name, "ggml_type": dtype.name,
                                        "shape": shape, "bytes": nbytes})


def _layer_values(value, n_layer):
    if isinstance(value, list):
        if len(value) != n_layer:
            raise ValueError("per-layer metadata length differs from block_count")
        return value
    return [value] * n_layer


def kv_stats(model, warnings):
    """Logical full-context cache, one sequence, K and V of the same type.

    Quantization blocks apply to each flattened GQA row, not to total elements
    across layers. No fictitious fractional blocks: invalid rows yield null.
    """
    layers = model["n_layer"]
    contexts = list(dict.fromkeys([2048, 4096, 8192, 32768] +
                                 ([model["context_length"]] if model["context_length"] else [])))
    is_mla = bool(model["mla_head_dim"] and model["mla_value_head_dim"])
    result = {"basis": ("pinned llama.cpp compressed MLA: K stores latent+rope, V aliases K" if is_mla else
                        "standard/legacy expanded K/V") +
                       "; full context, one sequence; no allocator/context padding or sliding-window cap",
              "contexts": contexts, "cache_types": {}}
    if not layers:
        warnings.append("KV unavailable: missing block_count")
        return result
    heads = _layer_values(model["n_head_kv"], layers)
    keys = _layer_values(model["head_dim"], layers)
    values = _layer_values(model["value_head_dim"], layers)
    for name in CACHE_TYPES:
        block, size = gguf.GGML_QUANT_SIZES[getattr(gguf.GGMLQuantizationType, name)]
        rows = [(h * k, 0 if is_mla else h * v) for h, k, v in zip(heads, keys, values)
                if h is not None and k is not None and v is not None]
        reason = None
        if len(rows) != layers:
            reason = "missing KV head count or key/value head dimensions"
        elif any(k % block or v % block for k, v in rows):
            reason = f"K or V row not divisible by GGML block size {block}"
        per_token = None if reason else sum((k + v) // block * size for k, v in rows)
        result["cache_types"][name.lower()] = {
            "block_elements": block, "block_bytes": size, "bytes_per_token": per_token,
            "MiB_by_context": {str(c): per_token * c / 2**20 if per_token is not None else None
                               for c in contexts}, "unavailable_reason": reason}
    if model["kv_lora_rank"]:
        if not is_mla:
            warnings.append("Legacy MLA metadata: no key_length_mla/value_length_mla; table uses stored K/V dimensions, as pinned llama.cpp does")
        result["mla"] = {"kv_lora_rank": model["kv_lora_rank"],
                         "compressed_cache": is_mla,
                         "note": "key_length_mla/value_length_mla trigger K-only allocation in pinned src/llama-kv-cache.cpp; engine support for each cache type is not implied."}
    return result


def inspect(gguf_path, trace=None):
    """Return a JSON-serializable model-stats/0 object (GGUF-order shapes)."""
    reader = HeaderReader(gguf_path, "r")
    try:
        def field(key, default=None):
            value = reader.get_field(key)
            return value.contents() if value is not None else default

        arch = field("general.architecture")
        if not arch:
            raise ValueError("missing general.architecture")

        def meta(suffix, default=None):
            return field(f"{arch}.{suffix}", default)

        warnings = []
        tensors = reader.tensor_headers
        inferred_layers = max((int(m[1]) + 1 for t in tensors
                               if (m := re.match(r"blk\.(\d+)\.", t["name"]))), default=0)
        n_layer = meta("block_count", inferred_layers)
        embed = meta("embedding_length")
        heads = meta("attention.head_count")
        if isinstance(heads, list):
            inferred_dim = [embed // h if embed and h else None for h in heads]
        else:
            inferred_dim = embed // heads if embed and heads else None
        model = {
            "arch": arch, "n_layer": n_layer,
            "n_expert": meta("expert_count", 0), "n_expert_used": meta("expert_used_count"),
            "n_shared_expert": meta("expert_shared_count"),
            "embedding_dim": embed, "ffn_dim": meta("feed_forward_length"),
            "expert_ffn_dim": meta("expert_feed_forward_length"),
            "shared_expert_ffn_dim": meta("expert_shared_feed_forward_length"),
            "n_head": heads, "n_head_kv": meta("attention.head_count_kv", heads),
            "head_dim": meta("attention.key_length", inferred_dim),
            "value_head_dim": meta("attention.value_length", inferred_dim),
            "context_length": meta("context_length"), "kv_lora_rank": meta("attention.kv_lora_rank"),
            "mla_head_dim": meta("attention.key_length_mla"),
            "mla_value_head_dim": meta("attention.value_length_mla"),
        }
        by_role = {r: 0 for r in ROLES}
        by_type = Counter()
        per_layer = {}
        hot_counts = field("hybrid_split.hot_counts")
        null_declared = field("hybrid_split.null_expert")
        unknown = []
        for t in tensors:
            name, nbytes = t["name"], t["bytes"]
            role = t["role"] = classify(name)
            match = re.match(r"blk\.(\d+)\.", name)
            layer = t["layer"] = int(match[1]) if match else None
            by_role[role] += nbytes
            by_type[t["ggml_type"]] += nbytes
            if role == "other":
                unknown.append(name)
            if role != "expert_ffn":
                continue
            if len(t["shape"]) < 2:
                warnings.append(f"cannot identify expert axis: {name}")
                continue
            count = t["shape"][-1]
            part = name.rsplit(".", 1)[-1]
            bank = part if part in ("hot", "cold") else "unsplit"
            nulls = 1 if bank == "hot" and null_declared else 0
            if hot_counts is not None and bank in ("hot", "cold"):
                expected = hot_counts[layer] + nulls if bank == "hot" else model["n_expert"] - hot_counts[layer]
                if count != expected:
                    raise ValueError(f"hybrid bank count mismatch: {name}")
            per_expert = nbytes // count
            t.update(bank=bank, stored_expert_count=count, null_expert_count=nulls,
                     bytes_per_expert=per_expert, null_bytes=nulls * per_expert)
            item = per_layer.setdefault(str(layer), {
                "expert_bytes": 0, "real_expert_bytes": 0, "null_bytes": 0,
                "bytes_per_single_expert": 0, "per_projection": {}, "banks": {}})
            projection = re.search(r"ffn_(.*?)_exps", name)[1]
            # Weight and bias are separate projections in the accounting table.
            suffix = re.sub(r"\.(?:hot|cold)$", "", name).split("_exps", 1)[1]
            projection += suffix
            proj = item["per_projection"].setdefault(projection, {
                "bytes": 0, "real_bytes": 0, "null_bytes": 0, "banks": {}})
            for dest in (item, proj):
                dest["null_bytes"] += t["null_bytes"]
            item["expert_bytes"] += nbytes
            item["real_expert_bytes"] += nbytes - t["null_bytes"]
            proj["bytes"] += nbytes
            proj["real_bytes"] += nbytes - t["null_bytes"]
            proj["banks"][bank] = {"stored_experts": count, "real_experts": count - nulls,
                                   "bytes_per_expert": per_expert, "ggml_type": t["ggml_type"]}
            item["banks"][bank] = item["banks"].get(bank, 0) + nbytes
        for item in per_layer.values():
            for proj in item["per_projection"].values():
                rates = {b["bytes_per_expert"] for b in proj["banks"].values()}
                proj["bytes_per_single_expert"] = next(iter(rates)) if len(rates) == 1 else None
            rates = [p["bytes_per_single_expert"] for p in item["per_projection"].values()]
            item["bytes_per_single_expert"] = sum(rates) if all(r is not None for r in rates) else None
            item["real_expert_count"] = model["n_expert"]

        expert_weights = [t for t in tensors if t["role"] == "expert_ffn" and ".weight" in t["name"]]
        if model["expert_ffn_dim"] is None and expert_weights:
            up = next((t for t in expert_weights if "ffn_up_exps" in t["name"]), None)
            down = next((t for t in expert_weights if "ffn_down_exps" in t["name"]), None)
            if up:
                model["expert_ffn_dim"] = up["shape"][1]
            elif down:
                model["expert_ffn_dim"] = down["shape"][0]
            model["expert_ffn_dim_source"] = "tensor shape"
        shared = [t for t in tensors if t["role"] == "shared_ffn"]
        model["has_shared_experts"] = bool(shared)
        if shared and model["n_shared_expert"] is None:
            warnings.append("shared FFN present but shared expert count absent; count cannot be inferred reliably from a fused shared FFN")

        total = sum(by_role.values())
        expert = by_role["expert_ffn"]
        null_bytes = sum(i["null_bytes"] for i in per_layer.values())
        real_expert = expert - null_bytes
        non_expert = total - expert  # Includes shared FFNs exactly once.
        e, k = model["n_expert"], model["n_expert_used"]
        if e and (k is None or not 0 <= k <= e):
            active = None
            real_active = None
            warnings.append("active byte estimate unavailable: missing/invalid expert_used_count")
        else:
            active = non_expert + (k / e * expert if e else 0)
            real_active = non_expert + (k / e * real_expert if e else 0)
        floors = [{"strategy": "all_experts_on_host", "hot_fraction": 0, "gpu_weight_bytes": non_expert},
                  {"strategy": "all_experts_on_gpu", "hot_fraction": 1, "gpu_weight_bytes": total}]
        floors += [{"strategy": "hot_fraction", "hot_fraction": f,
                    "gpu_weight_bytes": non_expert + f * expert} for f in (0.25, 0.5, 0.75)]
        report = {
            "schema": "model-stats/0", "gguf_path": str(Path(gguf_path).resolve()),
            **model,
            "shape_order": "GGML ne[0..] (reverse of numpy/PyTorch)",
            "tensors": tensors, "totals": {"weight_bytes": total, "by_role": by_role,
                "by_type": dict(by_type), "expert_fraction": expert / total if total else 0,
                "non_expert_bytes_including_shared": non_expert, "real_expert_bytes": real_expert,
                "null_expert_bytes": null_bytes},
            "per_layer_experts": per_layer,
            "active_bytes_per_token": {"bytes": active,
                "formula": "non_expert_including_shared + n_expert_used/n_expert * stored_expert_bytes",
                "real_experts_only_bytes": real_active,
                "note": "requested proportional storage estimate includes physical null overhead; real_experts_only_bytes excludes nulls; weights only, not measured bandwidth"},
            "strategy_floors": {"basis": "physical stored weight bytes; fractional experts, no KV, compute buffers, runtime or transfer staging",
                                "scenarios": floors},
            "unknown_tensors": unknown, "warnings": warnings,
            "role_patterns": [{"role": r, "pattern": p, "coverage": c} for r, p, c in ROLE_PATTERNS],
        }
        if null_bytes:
            report["hybrid_split"] = {
                "hot_counts": hot_counts, "null_expert": null_declared,
                "null_expert_bytes": null_bytes,
                "stored_expert_slots_per_layer": {
                    layer: {bank: max(p["banks"][bank]["stored_experts"]
                                      for p in data["per_projection"].values() if bank in p["banks"])
                            for bank in data["banks"]} for layer, data in per_layer.items()},
                "actual_hot_gpu_weight_bytes": non_expert + sum(d["banks"].get("hot", 0) for d in per_layer.values())}
        if unknown:
            warnings.append(f"{len(unknown)} unknown tensor(s) retained as other")
        report["kv_cache"] = kv_stats(model, warnings)
        if trace is not None:
            report["routing_trace"] = inspect_trace(trace, e, n_layer, k)
        return report
    finally:
        # No header array escapes this function; close the mapping promptly.
        reader.data._mmap.close()


def frequency_stats(hist):
    n, total = len(hist), sum(hist)
    ranked = sorted(range(n), key=lambda e: (-hist[e], e))
    probs = [h / total for h in hist] if total else [0] * n
    entropy = -sum(p * math.log2(p) for p in probs if p)
    ordered = sorted(hist)
    gini = sum((2 * i - n - 1) * x for i, x in enumerate(ordered, 1)) / (n * total) if total else None
    cumulative = 0
    curve = []
    for expert in ranked:
        cumulative += hist[expert]
        curve.append(cumulative / total if total else None)
    return {"frequency": hist, "fraction": probs, "requests": total,
            "entropy_bits": entropy if total else None, "gini": gini,
            "ranked_experts": ranked, "coverage_curve": curve,
            "recommended_hot_set_size": {str(t): next((i + 1 for i, c in enumerate(curve)
                if c is not None and c >= t), None) for t in (0.5, 0.75, 0.9)}}


def inspect_trace(path, n_expert, n_layer, top_k=None):
    """Read routing_trace JSONL or its replay JSON; never merge phases/splits.

    Raw trace rows have no train/eval declaration in the official format. Such
    input is explicitly marked unspecified, rather than assuming it is TRAIN.
    """
    if not n_expert:
        raise ValueError("trace requires positive model expert_count")
    groups = defaultdict(lambda: defaultdict(lambda: [0] * n_expert))
    provenance = {}
    with Path(path).open(encoding="utf-8") as src:
        first = src.readline()
        if not first:
            raise ValueError("trace is empty")
        try:
            first_row = json.loads(first)
        except json.JSONDecodeError:
            first_row = None
        if first_row is not None and not isinstance(first_row, dict):
            raise ValueError("trace must contain JSON objects")
        if first_row is None or "experts" not in first_row:
            src.seek(0)
            data = json.load(src)
            if not isinstance(data, dict):
                raise ValueError("replay must be a JSON object")
            if data.get("assumptions", {}).get("experts") != n_expert:
                raise ValueError("replay expert count differs from model")
            provenance = {key: data[key] for key in ("split", "train", "evaluation", "source") if key in data}
            split = data.get("split", "unspecified")
            if not isinstance(split, str):
                split = json.dumps(split, sort_keys=True)
            for phase, info in data["phases"].items():
                if phase not in ("prefill", "decode"):
                    raise ValueError("invalid replay phase")
                for layer, item in info["layers"].items():
                    hist = item["histogram"]
                    if (not 0 <= int(layer) < n_layer or len(hist) != n_expert or
                            any(type(x) is not int or x < 0 for x in hist)):
                        raise ValueError("invalid replay histogram/layer")
                    groups[(split, phase)][str(int(layer))] = hist
            input_format = "routing_trace replay JSON"
        else:
            seen = set()
            sequences = set()
            positions = {}

            def consume(row, number):
                try:
                    if not isinstance(row, dict):
                        raise ValueError("record must be a JSON object")
                    for key in ("seq", "pos", "layer"):
                        if type(row.get(key)) is not int or row[key] < 0:
                            raise ValueError(f"invalid {key}")
                    if row["layer"] >= n_layer or row.get("phase") not in ("prefill", "decode"):
                        raise ValueError("invalid layer/phase")
                    experts, weights = row["experts"], row["weights"]
                    if (not isinstance(experts, list) or not experts or
                            (top_k is not None and len(experts) != top_k) or
                            any(type(e) is not int or not 0 <= e < n_expert for e in experts) or
                            len(set(experts)) != len(experts)):
                        raise ValueError("invalid expert IDs/top-k")
                    if (not isinstance(weights, list) or len(weights) != len(experts) or
                            any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights)):
                        raise ValueError("invalid routing weights")
                    split = row.get("split", "unspecified")
                    if not isinstance(split, str):
                        raise ValueError("split must be a string")
                    identity = (split, row["seq"], row["pos"], row["layer"])
                    if identity in seen:
                        raise ValueError("duplicate routing record")
                    seen.add(identity)
                    sequences.add(row["seq"])
                    range_key = f"{split}/{row['phase']}/seq{row['seq']}"
                    lo, hi = positions.get(range_key, (row["pos"], row["pos"]))
                    positions[range_key] = (min(lo, row["pos"]), max(hi, row["pos"]))
                    hist = groups[(split, row["phase"])][str(row["layer"])]
                    for expert in experts:
                        hist[expert] += 1
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error

            consume(first_row, 1)
            for number, line in enumerate(src, 2):
                consume(json.loads(line), number)
            provenance = {"sequences": sorted(sequences), "position_ranges_inclusive": positions}
            input_format = "routing_trace JSONL"
    if not groups:
        raise ValueError("trace contains no layer histograms")
    return {"source": str(Path(path).resolve()), "format": input_format,
            "warning": "train\u2260eval: hot sets fitted to this trace are descriptive; validate on a disjoint evaluation split. Partial traces are accepted; completeness is not proven.",
            "metric": "unweighted routed expert requests; coverage_curve[H-1] for H=1..E; Gini includes zero-frequency experts",
            "provenance": provenance,
            "groups": [{"split": split, "phase": phase,
                        "layers": {layer: frequency_stats(hist) for layer, hist in sorted(layers.items(), key=lambda x: int(x[0]))}}
                       for (split, phase), layers in sorted(groups.items())]}


def summary(stats):
    mib = lambda b: "n/a" if b is None else f"{b / 2**20:,.2f}"
    rows = [(role, mib(value)) for role, value in stats["totals"]["by_role"].items() if value]
    lines = [f"{stats['arch']}: {stats['n_layer']} layers, {stats['n_expert']} experts, top-{stats['n_expert_used']}",
             f"embedding={stats['embedding_dim']} FFN={stats['ffn_dim']} expert-FFN={stats['expert_ffn_dim']} "
             f"heads={stats['n_head']}/{stats['n_head_kv']} head_dim={stats['head_dim']} context={stats['context_length']}",
             "| Weight role | MiB |", "| --- | ---: |"]
    lines += [f"| {role} | {value} |" for role, value in rows]
    lines += [f"| TOTAL | {mib(stats['totals']['weight_bytes'])} |",
              f"Expert share: {stats['totals']['expert_fraction']:.2%}; active weight MiB/token: {mib(stats['active_bytes_per_token']['bytes'])}",
              "", "| Weight placement floor | GPU MiB |", "| --- | ---: |"]
    for row in stats["strategy_floors"]["scenarios"]:
        lines.append(f"| {row['strategy']} (f={row['hot_fraction']}) | {mib(row['gpu_weight_bytes'])} |")
    lines += ["", "| KV cache type | Bytes/token | MiB at model context |", "| --- | ---: | ---: |"]
    for name, row in stats["kv_cache"]["cache_types"].items():
        at_max = row["MiB_by_context"].get(str(stats["context_length"]))
        lines.append(f"| {name} | {row['bytes_per_token']} | {'n/a' if at_max is None else f'{at_max:,.2f}'} |")
    lines += ["", "Floors exclude KV, runtime, compute buffers and transfer staging."]
    lines += [f"Warning: {w}" for w in stats["warnings"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        stats = inspect(args.gguf, args.trace)
        encoded = json.dumps(stats, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(encoded, encoding="utf-8")
        if args.summary:
            sys.stdout.write(summary(stats))
        elif not args.out:
            sys.stdout.write(encoded)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"inspect_model: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
