# hybrid_expert / `hybrid_moe_bench` (agent-facing)

A focused harness, not an engine change. It executes the **batch-1 Granite MoE expert FFN** with **expert-granular placement**.
Hits run on the GPU and misses run on the CPU, concurrently. It uses real weights and real routing, and is numerically checked.
It exists to confirm or refute the D014 prediction (expert-granular GPU-hit/CPU-miss dispatch with a linear 33.3 us/miss cost) on real kernels.
Results: `RESULTS.md`. Raw samples: `results.json` (24 layers) and `results_L12_single_layer.json`.

## What it does
- **Weights.** Real `blk.N.ffn_{gate,up,down}_exps` (Q4_K, [1024,512,32] / [512,1024,32], 0.844 MiB per expert) are read via the gguf API from
  `granite-3.1-1b-a400m-instruct-mixE4K.gguf`. Each layer has three copies:
  - a pinned host copy, used as the H2D source;
  - a CPU compute copy in `CPU_REPACK`, which is what llama.cpp uses for `--n-cpu-moe` experts (see below; `--cpu-repack 0` uses the plain copy);
  - a full GPU copy, used by the reference path.
- **Routing.** Top-8 ids and weights are replayed from `artifacts/traces/out/{prose,code,chat}.jsonl`. The timed stream is the decode tokens of all three traces
  concatenated: 352 positions per layer. The static slot profile is taken from the **prefill** tokens only, so the decode tokens are never used to choose slots.
- **Graph.** This is the llama.cpp `build_moe_ffn` op sequence: `mul_mat_id(gate)`, `mul_mat_id(up)`, `swiglu_split`, `mul_mat_id(down)`, `mul(weights)`, then a sum of views.
  Graphs are prebuilt per (backend, n experts = 1..8) with `ggml_gallocr`.
- **Paths, per layer.**
  - `gpu_ref`: all 32 experts on the GPU.
  - `cpu_all`: all experts on the CPU.
  - `hybrid` (overlap or serial): K GPU slots plus a slot table. The hit graph over slots is launched async on ggml-cuda, the miss graph runs on the ggml CPU backend
    (8-thread threadpool) concurrently, and then y = y_gpu + y_cpu. Each side multiplies only its own experts by their routing weights, so every weight is applied exactly once.
  - `upload`: llama.cpp `--moe-cache-mib` semantics. Misses are copied H2D into LRU slots first, then all 8 are computed on the GPU.
- **Policies.**
  - `static`: top-K by prefill frequency.
  - `lru`: cold start; after each token every miss is admitted, never evicting a slot used by that token. The admission H2D is timed **separately**, between tokens.
- **Synthetic sweep.** Exactly m = 0..8 misses per layer against the static top-16, using real routing weights. It also measures components: the CPU graph alone for m
  experts and the GPU graph alone for 8-m experts.
- **Micro-benchmarks.** H2D of one expert (pinned and pageable) and the bare GPU round trip (4 KB H2D + 4 KB D2H + sync).
- **Multi-layer mode.** `--layers 0-23` dispatches each decode position through all 24 real layers, each with its own routing, slots and weights. This is the
  **cache-realistic** mode: 24 × 27 MiB of expert weights cycle through the CPU's L3 and the GPU's L2, as in a real decode. A single repeated layer (`--layers 12`)
  is L3-hot and makes CPU misses about 3× too cheap (see RESULTS.md).

## Correctness rule and why
The CPU and GPU kernels do **not** compute bit-identical results for the same Q4_K weights. The CPU `vec_dot_q4_K_q8_K` and repack kernels quantize activations to
**Q8_K** (blocks of 256), while CUDA `mmvq` quantizes to **Q8_1** (blocks of 32). So "match to float rounding order" is not achievable between backends.
Against an fp64 reference (dequantized weights, unquantized activations), the endpoints have about 1.2e-2 (GPU) and 1.8e-2 (CPU) median relative L2 error.

The rule, per (layer, token): rel-L2(hybrid, fp64) ≤ 1.5 × max(rel-L2(gpu_ref), rel-L2(cpu_all)) + 1e-6. The endpoints are the same token through the all-GPU and
all-CPU graphs. The factor 1.5 allows the hybrid to be a mix of the two error sources; a dropped or double-applied routing weight gives errors of order 10–100%, far above it.

In addition, two **bitwise** checks: all-hit hybrid == `gpu_ref`, and zero-hit hybrid == `cpu_all`. These hold because the kernels and accumulation order are the same,
and they show the slot remapping and the weight split are exact. The process exits with code 2 if any check fails.

## Build and run (container `campaign/cuda-tools:c1`, label `campaign=c1`, `--cpus 8 --memory 16g --pids-limit 512`, `--network none`)
```
./build.sh ggml-cuda      # first time: ggml CPU+CUDA (sm_120) from READ-ONLY ref/llama.cpp mounted at /llama, about 6 min
./build.sh                # the harness; build dir = artifacts/opus-hybrid/build (sources mounted :ro)
./run.sh <out.json> --layers 0-23 --threads 8 --passes 1 --synth-n 200 --warmup 10 --check-n 16   # about 62 s GPU (cache-realistic)
./run.sh <out.json> --layers 12 --threads 8                                                       # about 9 s GPU (single layer, cache-hot)
python3 analyze.py <out.json>                                                                     # tables used in RESULTS.md
```
`run.sh` waits up to 10 min for `nvidia-smi --query-compute-apps` to be empty. The CPU ISA is `-march=native` (Zen 2 AVX2). The engine image `campaign/llamacpp:b0`
also runs AVX2 for Q4_K: `ggml_vec_dot_q4_K_q8_K` contains ymm `vpmaddubsw` (checked by disassembly; lib copy in `artifacts/opus-hybrid/engine-b0-libs/`).
In llama.cpp, CPU-overridden experts go to the first CPU extra buffer type, `CPU_REPACK` (`src/llama-model-loader.cpp`, override branch), so this harness defaults to the same.

## What this establishes, and what it does NOT
Establishes, on this machine (RTX 5070 Ti over PCIe 4.0 x16 on Ryzen 7 3700X, Zen 2 AVX2, dual-channel DDR4) for Granite-1B dims and Q4_K experts:
- that expert-granular hit/miss dispatch is numerically exact up to the inherent backend difference;
- the measured cost of a CPU-computed expert;
- how much CPU/GPU overlap a straightforward single-host-thread ggml implementation gets;
- the H2D admission cost, and LRU versus static versus upload-on-miss on real decode routing.

Does **NOT** establish:
- **Full-model decode t/s.** There is no attention, norms, router or other ops between layers, and each hybrid layer is host-synchronized.
- **Prefill or batch > 1.**
- **Quality.** Routing is replayed, not recomputed. The traces are from BF16, the weights are Q4_K, and activations are synthetic: `rms_norm(N(0,1)) * ffn_norm` of
  the real layer.
- **Other CPUs or thread counts.** Only 8 threads were measured.
- **Behaviour of models with larger experts or more experts.**
- **Anything about the SSD tier.**

All timings are **UNSCORED / exploratory** unless the orchestrator re-runs them.

**Mapping to the unified-memory target (128 GB UMA + SSD).** There is no PCIe on the target. The "GPU slot" and the "CPU copy" are the same LPDDR pages, so the
H2D admission cost measured here (38 us/expert) is replaced by nothing (UMA) or by an SSD read (a different tier). What transfers:
- the dispatch structure (slot table, split, combine, weight-once rule, bitwise endpoint checks);
- the finding that on a bandwidth-bound CPU a miss costs about expert_bytes / CPU_stream_BW plus a per-layer fixed cost;
- the finding that overlap must not share the launching host thread.

On UMA, "CPU miss versus GPU hit" becomes a question of which compute unit streams the expert, and both share one memory bus, so overlap gains will be smaller than
on a discrete GPU. The per-miss cost must be re-measured on the target.
