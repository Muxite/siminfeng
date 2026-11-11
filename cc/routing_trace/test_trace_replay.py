import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from trace_replay import CAPACITIES, HEADER, POLICIES, SlotCache, read_trace, replay


def row(pos, experts=None, phase="decode", layer=0, seq=0):
    return {"seq": seq, "pos": pos, "layer": layer,
            "experts": list(range(8)) if experts is None else experts,
            "weights": [0.125] * 8, "phase": phase}


def write_trace(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_repeat_top8_cold_misses_entropy_overlap(tmp_path):
    rows = read_trace(write_trace(tmp_path / "repeat.jsonl", [row(p) for p in range(1000)]))
    report = replay(rows)
    layer = report["phases"]["decode"]["layers"]["0"]
    assert layer["histogram"] == [1000] * 8 + [0] * 24
    assert layer["entropy_bits"] == 3
    assert layer["overlap_pairs"] == 999
    assert layer["mean_overlap_experts"] == 8
    assert layer["mean_overlap_fraction"] == 1
    for policy in POLICIES:
        for capacity in CAPACITIES:
            stat = layer["cache"][policy][str(capacity)]
            assert stat["requests"] == 8000
            assert stat["hits"] == (4 if capacity == 4 else 8) * 999
            assert stat["hit_rate"] == (0.4995 if capacity == 4 else 0.999)
    assert report["phases"]["prefill"]["overall"]["LRU"]["8"]["hit_rate"] is None


def test_uniform_random_independent_top8():
    rng = random.Random(20261109)
    report = replay([row(p, rng.sample(range(32), 8)) for p in range(12000)])
    layer = report["phases"]["decode"]["layers"]["0"]
    assert math.isclose(layer["entropy_bits"], 5, abs_tol=0.003)
    assert math.isclose(layer["mean_overlap_fraction"], 0.25, abs_tol=0.01)
    for policy in POLICIES:
        for capacity in CAPACITIES:
            assert math.isclose(layer["cache"][policy][str(capacity)]["hit_rate"],
                                capacity / 32, abs_tol=0.015)


def test_lfu_retains_hot_expert_lru_tracks_recency():
    lru, lfu = SlotCache(2, "LRU"), SlotCache(2, "LFU")
    for cache in (lru, lfu):
        for _ in range(4):
            cache.access_token([0])
        cache.access_token([1])
        cache.access_token([2])
    assert set(lru.slots) == {1, 2}
    assert set(lfu.slots) == {0, 2}
    assert lru.access_token([0]) == 0
    assert lfu.access_token([0]) == 1


def test_lfu_ties_use_lru_and_counts_reset():
    cache = SlotCache(2, "LFU")
    for expert in (0, 1, 2):
        cache.access_token([expert])
    assert set(cache.slots) == {1, 2}
    cache.access_token([1])
    cache.access_token([0])
    assert set(cache.slots) == {0, 1}
    assert cache.slots[0][0] == 1


def test_layer_sequence_phase_isolation_and_weighted_overall():
    rows = [row(0, phase="prefill"), row(1), row(2), row(3, layer=1), row(4, seq=1)]
    report = replay(list(reversed(rows)))
    decode = report["phases"]["decode"]
    assert decode["overall"]["LRU"]["8"] == {"hits": 8, "requests": 32, "hit_rate": 0.25}
    assert decode["layers"]["0"]["cache"]["LRU"]["8"]["hit_rate"] == 1 / 3
    assert decode["layers"]["0"]["overlap_pairs"] == 1
    assert decode["layers"]["1"]["mean_overlap_fraction"] is None


def test_overlap_ignores_gaps_and_respects_sequence():
    report = replay([row(0), row(2), row(3, list(range(4, 12))), row(4, seq=1)])
    layer = report["phases"]["decode"]["layers"]["0"]
    assert layer["overlap_pairs"] == 1
    assert layer["mean_overlap_experts"] == 4
    assert layer["mean_overlap_fraction"] == 0.5


@pytest.mark.parametrize("field,value", [
    ("experts", list(range(6))), ("experts", [0] * 8),
    ("experts", list(range(7)) + [32]), ("experts", list(range(7)) + [True]),
    ("weights", [0.1] * 7), ("weights", [float("nan")] * 8),
    ("weights", [float("inf")] * 8), ("weights", [-0.1] * 8),
    ("weights", [10 ** 400] * 8),
    ("weights", [True] * 8), ("phase", "unknown"),
    ("pos", -1), ("pos", 1.5), ("seq", True), ("layer", "0"),
])
def test_reject_invalid_trace(tmp_path, field, value):
    record = row(0)
    record[field] = value
    with pytest.raises(ValueError, match=":1:"):
        read_trace(write_trace(tmp_path / "bad.jsonl", [record]))


def test_reject_duplicates_including_cross_phase(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        read_trace(write_trace(tmp_path / "duplicate.jsonl", [row(0), row(0, phase="prefill")]))


@pytest.mark.parametrize("data", ["", "not json\n", "[]\n", "null\n", "\n"])
def test_reject_empty_or_malformed(tmp_path, data):
    source = tmp_path / "bad.jsonl"
    source.write_text(data)
    with pytest.raises(ValueError):
        read_trace(source)


def test_cli_outputs_machine_json_and_markdown(tmp_path):
    script = Path(__file__).with_name("trace_replay.py")
    source = write_trace(tmp_path / "trace.jsonl", [row(0), row(1)])
    result = subprocess.run([sys.executable, str(script), str(source)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["header"] == HEADER
    assert result.stderr.startswith(HEADER)
    assert "| decode | LRU |" in result.stderr
    json_out, md_out = tmp_path / "report.json", tmp_path / "report.md"
    subprocess.run([sys.executable, str(script), str(source), "--json-out", str(json_out),
                    "--markdown-out", str(md_out)], check=True, capture_output=True)
    assert json.loads(json_out.read_text()) == json.loads(result.stdout)
    assert md_out.read_text() == result.stderr


def test_cli_missing_input_is_nonzero(tmp_path):
    script = Path(__file__).with_name("trace_replay.py")
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "missing")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "trace_replay:" in result.stderr
