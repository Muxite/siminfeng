# D041 — Upstream `--moe-cache-mib` beats static offload on Qwen at 8 GB (INDICATIVE); ncu decode roofline

Status: **INDICATIVE**. The screen was unfrozen, with 2 alternating rounds and no pre-registered protocol. A frozen follow-up is required before the factory adopts this.
Date: 2026-11-10. Lane: x86-main.
Engine: release-1 (`run_ref sha256:5b8a6a22…`, pin 31288c9c6). That pin contains upstream PR #29887, `--moe-cache-mib`.
Script: `artifacts/mc-qwen/screen.sh`, op T-MCQ. Raw output is in `artifacts/mc-qwen/`.

## Why
D022 found that static `--n-cpu-moe` beats the engine LRU cache at every footprint. That test used **Granite**, whose experts are 0.84 MiB each.
External reports for the upstream cache show decode gains on 16–32 GB cards with large MoEs (R9700 11→27.6 t/s; 5060 Ti 16 GB 15→19 t/s). They also say a hit rate of about 75% is needed.
D031 had already flagged a proxy-scale bias, so the D022 conclusion needed retesting on Qwen3-30B-A3B Q4_K_M.

## Result
Settings: chat prompt, 128 tokens, greedy decoding, FA on, ctx 4096. Peak VRAM is from nvidia-smi polled every 0.5 s.

| Config | Peak MiB | Decode t/s (r1, r2) | Prompt eval ms | Text sha |
|---|---:|---|---:|---|
| static `--n-cpu-moe 30` | 8193 | 36.88, 37.33 | 757, 783 | 1189257b |
| all-CPU experts + `--moe-cache-mib 6000` | 7831 | **51.13, 51.44** | 1082, 1089 | bf80f280 |
| static `--n-cpu-moe 42` | 4105 | 27.93, 26.77 | 1000, 996 | 77a3045a |
| all-CPU experts + `--moe-cache-mib 2200` | 4029 | 18.00, 18.15 | 1088, 1088 | bf80f280 |

- **At about 8 GB, the cache gives +38% decode with 362 MiB less VRAM.**
  - TTFT is about 40% worse, 1085 ms against 770 ms. This matches the reported prompt-processing loss.
- At about 4 GB, the cache is −34%. The crossover lies between the two budgets, and the hit rate is the likely driver.
  - That fits T-QRT's coverage curves. Held-out decode coverage is 77% at the K=8-sized hot set, falling to 15% at K=42.
- Every placement produced different greedy text, including static K30 against K42. This is the usual numeric divergence between CPU and GPU, not evidence of a bug. Even so, it needs a **KLD/PPL check**, not text identity.
  - The two cache configs produce identical text, which is consistent with all expert math running on the GPU.
- The host was quiet: no compiles, load average 2.4.

## Consequences
1. The D022 scope is narrowed to "small experts / low coverage". The planner must not hard-reject the expert cache.
2. Planner: add the cache as a lever. Model the hit rate from routing-trace coverage curves (T-QRT), and price misses at PCIe upload cost. The decision rule becomes: cache when predicted coverage ≥ about 70%, otherwise static.
3. The report must show the TTFT cost. The factory should expose a "decode-optimised vs TTFT-optimised" choice.
4. Frozen follow-up: ABBA ×3 at budgets 4/6/8/10/12 GB, with the cache size swept to find the crossover. Add KLD against a reference and the hit rate from engine logs.

## ncu decode roofline (Granite Q8_0, release-1, all-GPU), op T-NCU1
Settings: `ncu --launch-skip 2000 --launch-count 600` on `llama-bench tg32`. Raw data is in `artifacts/ncu1/ncu.csv`.
Caveat: ncu serialises kernels and flushes caches, so small-kernel times are inflated against graph replay.

- The effective DRAM bandwidth over the profiled window is 222 GB/s. The 5070 Ti's peak is about 896 GB/s.
- The large expert `mul_mat_vec_q` reaches 578 GB/s (66% of peak DRAM).
- About 40% of kernel time goes to tiny 2–4 µs kernels at under 1% DRAM: `quantize_q8_1`, `rms_norm`, `add`, `scale`, `topk_moe`, `moe_weighted_reduction`.
  - On a model this small, decode is launch- and latency-bound, not bandwidth-bound.
  - **Fusion candidates:** rms_norm with quantize_q8_1, and scale with add.
- An unprofiled nsys timeline is needed to size the real gain. ncu's serialisation exaggerates small kernels.
