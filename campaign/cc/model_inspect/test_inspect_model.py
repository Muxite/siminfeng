"""Real-file acceptance checks plus independent tiny-header expectations."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_model import (ROOT, classify, frequency_stats, gguf, inspect,
                           inspect_trace, main, summary)
import numpy as np
import pytest

MODELS = Path.home() / "models/campaign/gguf"
STEM = "granite-3.1-1b-a400m-instruct-"


def real_path(variant):
    path = MODELS / f"{STEM}{variant}.gguf"
    if not path.is_file():
        pytest.skip(f"real model unavailable: {path}")
    return path


@pytest.fixture(scope="module")
def granite():
    return inspect(real_path("bf16"))


@pytest.mark.parametrize("variant", ["bf16", "Q8_0", "Q4_K_M", "mixE4K"])
def test_real_variants(variant):
    stats = inspect(real_path(variant))
    assert stats["schema"] == "model-stats/0"
    assert (stats["arch"], stats["n_layer"], stats["n_expert"], stats["n_expert_used"]) == ("granitemoe", 24, 32, 8)
    assert stats["embedding_dim"] == 1024
    assert stats["expert_ffn_dim"] == 512
    assert stats["unknown_tensors"] == []
    assert len(stats["per_layer_experts"]) == 24
    totals = stats["totals"]
    assert sum(t["bytes"] for t in stats["tensors"]) == totals["weight_bytes"]
    assert sum(totals["by_role"].values()) == sum(totals["by_type"].values()) == totals["weight_bytes"]
    assert totals["weight_bytes"] < real_path(variant).stat().st_size
    for layer in stats["per_layer_experts"].values():
        assert layer["expert_bytes"] == 32 * layer["bytes_per_single_expert"]
        assert len(layer["per_projection"]) == 3
    if variant == "bf16":
        assert totals["expert_fraction"] == pytest.approx(0.9045, abs=0.001)
        # 24 layers, 32 experts, 3 projections, [1024,512], two bytes each.
        assert totals["by_role"]["expert_ffn"] == 24 * 32 * 3 * 1024 * 512 * 2
    if variant == "mixE4K":
        assert {t["ggml_type"] for t in stats["tensors"] if t["role"] == "expert_ffn"} == {"Q4_K"}
        # Norms/router stay F32 in the source Q8_0 model; the remaining matrices are Q8_0.
        matrices = [t for t in stats["tensors"] if t["role"] in ("attn_qkv", "attn_out", "embed", "output")]
        assert {t["ggml_type"] for t in matrices} == {"Q8_0"}
        q8 = inspect(real_path("Q8_0"))
        assert [(t["name"], t["ggml_type"], t["bytes"]) for t in stats["tensors"] if t["role"] != "expert_ffn"] == [
            (t["name"], t["ggml_type"], t["bytes"]) for t in q8["tensors"] if t["role"] != "expert_ffn"]


def test_real_hybrid():
    path = ROOT / "artifacts/codexR/models/granite-split-n12.gguf"
    if not path.is_file():
        pytest.skip(f"real split unavailable: {path}")
    stats = inspect(path)
    assert stats["unknown_tensors"] == []
    hot_counts = stats["hybrid_split"]["hot_counts"]
    assert len(hot_counts) == 24
    assert all(0 <= h < 32 for h in hot_counts)
    assert len(set(hot_counts)) > 1  # Read actual header counts, not the filename.
    for index, layer in stats["per_layer_experts"].items():
        assert layer["real_expert_count"] == 32
        assert layer["expert_bytes"] == 33 * layer["bytes_per_single_expert"]
        assert layer["null_bytes"] == layer["bytes_per_single_expert"]
        assert layer["real_expert_bytes"] == 32 * layer["bytes_per_single_expert"]
        for proj in layer["per_projection"].values():
            assert proj["banks"]["hot"]["stored_experts"] == hot_counts[int(index)] + 1
            assert proj["banks"]["cold"]["stored_experts"] == 32 - hot_counts[int(index)]
    assert stats["totals"]["null_expert_bytes"] == sum(l["bytes_per_single_expert"] for l in stats["per_layer_experts"].values())
    assert {t["bank"] for t in stats["tensors"] if t["role"] == "expert_ffn"} == {"hot", "cold"}
    assert sum(t["null_expert_count"] for t in stats["tensors"] if t["role"] == "expert_ffn") == 24 * 3
    active = stats["active_bytes_per_token"]
    totals = stats["totals"]
    assert active["bytes"] == totals["non_expert_bytes_including_shared"] + totals["by_role"]["expert_ffn"] / 4
    assert active["bytes"] - active["real_experts_only_bytes"] == totals["null_expert_bytes"] / 4


def test_active_and_floors(granite):
    total = granite["totals"]["weight_bytes"]
    expert = granite["totals"]["by_role"]["expert_ffn"]
    assert granite["active_bytes_per_token"]["bytes"] == total - expert + expert / 4
    for row in granite["strategy_floors"]["scenarios"]:
        assert row["gpu_weight_bytes"] == total - expert + row["hot_fraction"] * expert


def test_kv_independent_block_sizes(granite):
    # Independently known GGML block encodings; not copied from implementation.
    expected = {"f16": (1, 2), "bf16": (1, 2), "q8_0": (32, 34),
                "q5_1": (32, 24), "q5_0": (32, 22), "q4_1": (32, 20), "q4_0": (32, 18)}
    for kind, (block, size) in expected.items():
        row = granite["kv_cache"]["cache_types"][kind]
        assert row["block_elements"] == block
        assert row["block_bytes"] == size
        per_token = 24 * 8 * 64 * 2 // block * size
        assert row["bytes_per_token"] == per_token
        assert row["MiB_by_context"] == {str(c): per_token * c / 2**20 for c in (2048, 4096, 8192, 32768, 131072)}


def write_tiny(path, arch="qwen3moe", key_dim=32):
    writer = gguf.GGUFWriter(path, arch)
    for key, value in {"block_count": 1, "expert_count": 4, "expert_used_count": 2,
                       "expert_shared_count": 1, "embedding_length": 64,
                       "feed_forward_length": 128, "expert_feed_forward_length": 32,
                       "expert_shared_feed_forward_length": 48, "attention.head_count": 2,
                       "attention.head_count_kv": 1, "attention.key_length": key_dim,
                       "attention.value_length": 16, "context_length": 16384}.items():
        writer.add_uint32(f"{arch}.{key}", value)
    if arch == "deepseek2":
        writer.add_uint32(f"{arch}.attention.kv_lora_rank", 16)
    names = {
        "token_embd.weight": (8, 64), "output.weight": (8, 64),
        "output_norm.weight": (64,), "blk.0.ffn_gate_inp.weight": (4, 64),
        "blk.0.ffn_gate_inp_shexp.weight": (1, 64),
        "blk.0.ffn_up_exps.weight": (4, 32, 64),
        "blk.0.ffn_gate_exps.weight": (4, 32, 64),
        "blk.0.ffn_down_exps.weight": (4, 64, 32),
        "blk.0.ffn_up_shexp.weight": (48, 64),
        "blk.0.ffn_gate_shexp.weight": (48, 64),
        "blk.0.ffn_down_shexp.weight": (64, 48),
        "blk.0.attn_output.weight": (64, 64),
        "blk.0.ffn_norm.weight": (64,), "unknown_payload.weight": (2, 3),
    }
    if arch == "deepseek2":
        names.update({f"blk.0.attn_{name}.weight": (16, 64)
                      for name in ("q_a", "q_b", "kv_a_mqa", "kv_b", "k_b", "v_b")})
        names.update({"blk.0.attn_q_a_norm.weight": (16,), "blk.0.attn_kv_a_norm.weight": (16,),
                      "blk.0.ffn_up.weight": (32, 64)})
    else:
        names.update({f"blk.0.attn_{name}.weight": (32, 64) for name in ("q", "k", "v")})
    for name, shape in names.items():
        writer.add_tensor(name, np.zeros(shape, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize("arch", ["qwen3moe", "deepseek2"])
def test_synthetic_architectures(tmp_path, arch):
    path = tmp_path / f"{arch}.gguf"
    write_tiny(path, arch)
    stats = inspect(path)
    assert stats["arch"] == arch
    assert stats["n_shared_expert"] == 1 and stats["has_shared_experts"]
    assert stats["shared_expert_ffn_dim"] == 48
    assert stats["totals"]["by_role"]["shared_ffn"] == 3 * 48 * 64 * 4
    assert stats["totals"]["by_role"]["expert_ffn"] == 3 * 4 * 32 * 64 * 4
    assert stats["unknown_tensors"] == ["unknown_payload.weight"]
    assert stats["totals"]["by_role"]["other"] == 24
    assert stats["per_layer_experts"]["0"]["bytes_per_single_expert"] == 3 * 32 * 64 * 4
    non_expert = stats["totals"]["weight_bytes"] - 3 * 4 * 32 * 64 * 4
    assert stats["active_bytes_per_token"]["bytes"] == non_expert + 3 * 2 * 32 * 64 * 4
    assert stats["kv_cache"]["cache_types"]["f16"]["bytes_per_token"] == (32 + 16) * 2
    # V row has 16 elements: do not silently divide into a half Q4 block.
    assert stats["kv_cache"]["cache_types"]["q4_0"]["bytes_per_token"] is None
    if arch == "deepseek2":
        assert stats["totals"]["by_role"]["attn_qkv"] == 6 * 16 * 64 * 4
        assert stats["totals"]["by_role"]["dense_ffn"] == 32 * 64 * 4
        assert any("MLA" in w for w in stats["warnings"])


def test_modern_compressed_mla(tmp_path):
    path = tmp_path / "modern-mla.gguf"
    writer = gguf.GGUFWriter(path, "deepseek2")
    for key, value in {"block_count": 2, "embedding_length": 128,
                       "attention.head_count": 4, "attention.head_count_kv": 1,
                       "attention.key_length": 96, "attention.value_length": 64,
                       "attention.key_length_mla": 48, "attention.value_length_mla": 32,
                       "attention.kv_lora_rank": 64, "context_length": 4096}.items():
        writer.add_uint32(f"deepseek2.{key}", value)
    writer.add_tensor("token_embd.weight", np.zeros((2, 128), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    stats = inspect(path)
    cache = stats["kv_cache"]
    assert cache["mla"]["compressed_cache"] is True
    assert cache["cache_types"]["f16"]["bytes_per_token"] == 2 * 96 * 2
    assert cache["cache_types"]["q8_0"]["bytes_per_token"] == 2 * 3 * 34
    assert cache["contexts"] == [2048, 4096, 8192, 32768]
    assert stats["active_bytes_per_token"]["bytes"] == stats["totals"]["weight_bytes"]


@pytest.mark.parametrize("name,role", [
    ("blk.2.ffn_gate_up_exps.weight", "expert_ffn"),
    ("blk.2.ffn_gate_up_exps.bias", "expert_ffn"),
    ("blk.2.ffn_down_exps.weight.hot", "expert_ffn"),
    ("blk.2.ffn_down_exps.weight.cold", "expert_ffn"),
    ("blk.2.ffn_expert_ids.ids_hot", "router"),
    ("blk.2.ffn_exp_probs_b.bias", "router"),
    ("blk.2.ffn_gate_inp_shexp.weight", "router"),
    ("blk.2.ffn_gate_shexp.weight", "shared_ffn"),
    ("blk.2.attn_q_a_norm.weight", "norm"),
    ("blk.2.attn_kv_a_mqa.weight", "attn_qkv"),
    ("blk.2.attn_sinks.weight", "attn_out"),
    ("made_up_tensor", "other"),
])
def test_pattern_table(name, role):
    assert classify(name) == role


def test_never_reads_tensor_payload(monkeypatch):
    original = gguf.GGUFReader._get

    def guarded(reader, offset, *args, **kwargs):
        if hasattr(reader, "data_offset"):
            assert offset < reader.data_offset, "attempt to read payload"
        return original(reader, offset, *args, **kwargs)

    monkeypatch.setattr(gguf.GGUFReader, "_get", guarded)
    stats = inspect(real_path("bf16"))
    assert stats["totals"]["weight_bytes"] > 2 * 2**30


def test_frequency_statistics():
    uniform = frequency_stats([10] * 4)
    assert uniform["entropy_bits"] == 2
    assert uniform["gini"] == 0
    assert uniform["coverage_curve"] == [.25, .5, .75, 1]
    assert uniform["recommended_hot_set_size"] == {"0.5": 2, "0.75": 3, "0.9": 4}
    skew = frequency_stats([10, 0, 0, 0])
    assert skew["entropy_bits"] == 0
    assert skew["gini"] == .75
    assert skew["ranked_experts"] == [0, 1, 2, 3]
    assert frequency_stats([0] * 4)["recommended_hot_set_size"]["0.5"] is None


def trace_row(**kwargs):
    return {"seq": 0, "pos": 0, "layer": 0, "experts": [0, 1], "weights": [.7, .3],
            "phase": "prefill", **kwargs}


def test_trace_splits_and_phases(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [trace_row(split="train"), trace_row(split="eval", experts=[2, 3]),
            trace_row(pos=1, phase="decode", experts=[0, 3])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = inspect_trace(path, 4, 1, 2)
    groups = {(g["split"], g["phase"]): g["layers"]["0"] for g in stats["groups"]}
    assert groups[("train", "prefill")]["frequency"] == [1, 1, 0, 0]
    assert groups[("eval", "prefill")]["frequency"] == [0, 0, 1, 1]
    assert groups[("unspecified", "decode")]["frequency"] == [1, 0, 0, 1]
    assert "train\u2260eval" in stats["warning"]


@pytest.mark.parametrize("rows", [[], [trace_row(), trace_row()],
    [trace_row(experts=[0, 4])], [trace_row(experts=[0, 0])],
    [trace_row(weights=[float("nan"), 0])], [trace_row(layer=1)],
    [trace_row(phase="train")], [trace_row(experts=[0], weights=[1])]])
def test_bad_trace(tmp_path, rows):
    path = tmp_path / "bad.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        inspect_trace(path, 4, 1, 2)


def test_real_replay_and_raw_agree():
    replay_path = ROOT / "experiments/routing/prose.json"
    raw_path = ROOT / "artifacts/traces/out/prose.jsonl"
    if not replay_path.is_file() or not raw_path.is_file():
        pytest.skip("real routing evidence unavailable")
    replay = inspect_trace(replay_path, 32, 24, 8)
    raw = inspect_trace(raw_path, 32, 24, 8)
    assert replay["groups"] == raw["groups"]
    for group in raw["groups"]:
        assert group["split"] == "unspecified"
        assert len(group["layers"]) == 24
        assert group["layers"]["0"]["coverage_curve"][-1] == 1


def test_cli(tmp_path, granite, capsys):
    model = tmp_path / "tiny.gguf"
    output = tmp_path / "stats.json"
    write_tiny(model)
    assert main([str(model), "--out", str(output), "--summary"]) == 0
    assert "| Weight role | MiB |" in capsys.readouterr().out
    assert json.loads(output.read_text())["schema"] == "model-stats/0"
    assert main([str(model)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "model-stats/0"
    assert main([str(model), "--out", str(output)]) == 0
    assert capsys.readouterr().out == ""
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "absent.gguf")])
    assert error.value.code == 1
    assert "2,547.20" in summary(granite)
