#!/usr/bin/env python3
"""Replay complete top-8 routing records using only the Python standard library."""

import argparse
import json
import math
from pathlib import Path
import sys

HEADER = ("SIMULATION on a logical trace: no residency/transfer semantics; "
          "Granite fully fits GPU memory.")
CAPACITIES = (4, 8, 12, 16, 24)
POLICIES = ("LRU", "LFU")


class SlotCache:
    """Score at token entry, then admit/touch in trace rank order."""

    def __init__(self, capacity, policy):
        if capacity <= 0 or policy not in POLICIES:
            raise ValueError("invalid capacity or policy")
        self.capacity = capacity
        self.policy = policy
        self.slots = {}  # expert -> (resident frequency, last access clock)
        self.clock = 0

    def access_token(self, experts):
        hits = sum(expert in self.slots for expert in experts)
        for expert in experts:
            self.clock += 1
            if expert in self.slots:
                frequency, _ = self.slots[expert]
                self.slots[expert] = (frequency + 1, self.clock)
                continue
            if len(self.slots) == self.capacity:
                if self.policy == "LRU":
                    victim = min(self.slots, key=lambda e: self.slots[e][1])
                else:
                    victim = min(self.slots, key=lambda e: self.slots[e])
                del self.slots[victim]
            self.slots[expert] = (1, self.clock)
        return hits


def read_trace(path):
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
                for key in ("seq", "pos", "layer"):
                    if type(row.get(key)) is not int or row[key] < 0:
                        raise ValueError(f"{key} must be a nonnegative integer")
                if row.get("phase") not in ("prefill", "decode"):
                    raise ValueError("phase must be prefill or decode")
                experts, weights = row.get("experts"), row.get("weights")
                if not isinstance(experts, list) or len(experts) != 8:
                    raise ValueError("exactly eight expert IDs required")
                if any(type(e) is not int or not 0 <= e < 32 for e in experts):
                    raise ValueError("expert IDs must be integers in [0,32)")
                if len(set(experts)) != 8:
                    raise ValueError("expert IDs must be distinct")
                if not isinstance(weights, list) or len(weights) != 8:
                    raise ValueError("exactly eight weights required")
                if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0
                       for w in weights):
                    raise ValueError("weights must be finite nonnegative numbers")
                identity = (row["seq"], row["pos"], row["layer"])
                if identity in seen:
                    raise ValueError("duplicate (seq,pos,layer)")
                seen.add(identity)
                records.append(row)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise ValueError("trace is empty")
    return records


def rate(hits, requests):
    return {"hits": hits, "requests": requests,
            "hit_rate": hits / requests if requests else None}


def replay(records):
    """Each phase, layer and sequence has an independent cold cache."""
    report = {
        "header": HEADER,
        "schema_version": 1,
        "assumptions": {
            "experts": 32, "top_k": 8, "capacities_per_layer": list(CAPACITIES),
            "initial_state": "cold per phase, layer and sequence; no prefill warming of decode",
            "hit_accounting": "all eight lookups use token-entry slots; then update in array order",
            "LFU": "resident counts start at 1 on admission, reset on eviction; LRU breaks ties",
            "overlap": "intersection size / 8, adjacent positions only, within sequence and phase",
            "entropy": "Shannon bits of unweighted expert selection counts (maximum 5 bits)",
            "overall": "request-weighted hit rate; weights do not affect hit counting",
        },
        "phases": {},
    }
    for phase in ("prefill", "decode"):
        rows = sorted((r for r in records if r["phase"] == phase),
                      key=lambda r: (r["seq"], r["pos"], r["layer"]))
        layers = {}
        previous = {}
        caches = {}
        for row in rows:
            layer = str(row["layer"])
            item = layers.setdefault(layer, {
                "tokens": 0, "histogram": [0] * 32, "overlap_pairs": 0,
                "overlap_experts_sum": 0,
                "cache": {p: {str(c): rate(0, 0) for c in CAPACITIES} for p in POLICIES},
            })
            item["tokens"] += 1
            for expert in row["experts"]:
                item["histogram"][expert] += 1
            key = (row["seq"], row["layer"])
            if key in previous:
                old = previous[key]
                if row["pos"] == old["pos"] + 1:
                    item["overlap_pairs"] += 1
                    item["overlap_experts_sum"] += len(set(old["experts"]) & set(row["experts"]))
            previous[key] = row
            for policy in POLICIES:
                for capacity in CAPACITIES:
                    cache_key = (*key, policy, capacity)
                    if cache_key not in caches:
                        caches[cache_key] = SlotCache(capacity, policy)
                    hits = caches[cache_key].access_token(row["experts"])
                    stat = item["cache"][policy][str(capacity)]
                    stat["hits"] += hits
                    stat["requests"] += 8
        for item in layers.values():
            counts = item["histogram"]
            total = sum(counts)
            item["entropy_bits"] = -sum((n / total) * math.log2(n / total) for n in counts if n)
            pairs = item["overlap_pairs"]
            item["mean_overlap_experts"] = item["overlap_experts_sum"] / pairs if pairs else None
            item["mean_overlap_fraction"] = item["overlap_experts_sum"] / (8 * pairs) if pairs else None
            for policy in POLICIES:
                for capacity in CAPACITIES:
                    stat = item["cache"][policy][str(capacity)]
                    stat.update(rate(stat["hits"], stat["requests"]))
        overall = {p: {} for p in POLICIES}
        for policy in POLICIES:
            for capacity in CAPACITIES:
                stats = [v["cache"][policy][str(capacity)] for v in layers.values()]
                overall[policy][str(capacity)] = rate(sum(s["hits"] for s in stats),
                                                     sum(s["requests"] for s in stats))
        report["phases"][phase] = {"records": len(rows), "layers": layers, "overall": overall}
    return report


def markdown(report):
    lines = [HEADER, "", "Cold caches; token-entry lookups; overall rates weighted by expert requests.", "",
             "| Phase | Policy | 4 slots | 8 slots | 12 slots | 16 slots | 24 slots |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for phase, data in report["phases"].items():
        for policy in POLICIES:
            cells = []
            for capacity in CAPACITIES:
                value = data["overall"][policy][str(capacity)]["hit_rate"]
                cells.append("n/a" if value is None else f"{value:.2%}")
            lines.append("| " + " | ".join([phase, policy, *cells]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json-out", type=Path, help="default: JSON on stdout")
    parser.add_argument("--markdown-out", type=Path, help="default: Markdown on stderr")
    args = parser.parse_args(argv)
    try:
        report = replay(read_trace(args.trace))
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        table = markdown(report)
        if args.json_out:
            args.json_out.write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
        if args.markdown_out:
            args.markdown_out.write_text(table, encoding="utf-8")
        else:
            sys.stderr.write(table)
    except (OSError, ValueError) as error:
        parser.exit(1, f"trace_replay: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
