"""Reproduce retained model-stats reports with one process and header reads."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cc/model_inspect"))
from inspect_model import inspect, summary

destination = Path(__file__).resolve().parent
model_dir = Path.home() / "models/campaign/gguf"
rows = []
for variant in ("bf16", "Q8_0", "Q4_K_M", "mixE4K", "split-n12"):
    path = (ROOT / "artifacts/codexR/models/granite-split-n12.gguf" if variant == "split-n12" else
            model_dir / f"granite-3.1-1b-a400m-instruct-{variant}.gguf")
    trace = ROOT / "experiments/routing/prose.json" if variant == "mixE4K" else None
    stats = inspect(path, trace)
    (destination / f"granite-{variant}.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    (destination / f"granite-{variant}-summary.md").write_text(summary(stats))
    totals = stats["totals"]
    rows.append({"variant": variant, "weight_MiB": totals["weight_bytes"] / 2**20,
                 "expert_MiB": totals["by_role"]["expert_ffn"] / 2**20,
                 "expert_fraction": totals["expert_fraction"],
                 "active_MiB_per_token": stats["active_bytes_per_token"]["bytes"] / 2**20,
                 "host_experts_gpu_floor_MiB": totals["non_expert_bytes_including_shared"] / 2**20,
                 "null_expert_MiB": totals["null_expert_bytes"] / 2**20})
(destination / "granite-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
table = ["| Granite variant | Weights MiB | Experts MiB | Expert share | Active MiB/token | Host-expert GPU weight floor MiB |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    table.append(f"| {r['variant']} | {r['weight_MiB']:,.2f} | {r['expert_MiB']:,.2f} | "
                 f"{r['expert_fraction']:.2%} | {r['active_MiB_per_token']:,.2f} | {r['host_experts_gpu_floor_MiB']:,.2f} |")
table += ["", "24 layers, 32 experts, top-8; embedding 1024; expert FFN 512; attention heads 16/8; head dimension 64; max context 131072.",
          "", "MiB = 2^20 bytes. All values are weights only. Physical split storage and its proportional active estimate include 20.25 MiB of null experts; JSON additionally reports an active estimate excluding nulls.",
          "", "Real validations: Granite variants and hybrid split. Qwen3 shared experts and legacy/current DeepSeek2 MLA are synthetic fixtures. Other named architectures have explicit pattern checks but no real GGUF in this assignment."]
(destination / "granite-comparison.md").write_text("\n".join(table) + "\n")
print("\n".join(table))
