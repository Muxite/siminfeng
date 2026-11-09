# hybrid_moe_bench results: UNSCORED / exploratory (agent opus-hybrid, task T-J, parent W7)

**Setup.** RTX 5070 Ti (PCIe gen4 x16, max as reported by nvidia-smi), Ryzen 7 3700X with 8 CPU threads (container `--cpus 8`), mixE4K Q4_K experts,
CPU-miss weights in CPU_REPACK (as llama.cpp `--n-cpu-moe`).

**Runs** (recorder ops in `evidence/opus-hybrid.jsonl`):

| file | op | description |
|---|---|---|
| `results.json` | op-2ea9dea3e2e5 | 24 layers, 352 decode positions, 1 timed pass; synthetic 200 per m |
| `results_L12_single_layer.json` | op-6da36c612c6b | layer 12 only, 2 passes = 704 samples; synthetic 600 per m |
| (first run, superseded) | op-cc0ff424b52f | same build family, `artifacts/opus-hybrid/runs/r1_L12_t8_repack.json` |

**Builds:** op-eb0f9a7be446 (ggml), op-6ea3308d1c34, op-33d0baf30db1, op-3fbac14b1c98.

**Timing.** Host `steady_clock` around each layer dispatch. The interval includes input staging, CUDA launch, CPU compute, sync and the combine. The step value is
the sum over layers. Medians and p95 are over at least 200 samples per config. CUDA events were **not** used: ggml-cuda runs on its own internal stream with
timing-disabled events, so GPU parts are host-measured launch-to-sync intervals. Full tables: `artifacts/opus-hybrid/analysis.md` (from `analyze.py`).

## 1. Correctness: PASS (all configs, both runs)
- **Bitwise.** All-hit hybrid (K=32, and synthetic m=0) is identical to `gpu_ref`; zero-hit hybrid (m=8) is identical to `cpu_all`. This holds in all 24 layers.
  It shows the slot remap, the split and the weight-once combine are exact.
- **Mixed hit/miss.** Every checked (layer, token) satisfies rel-L2(hybrid, fp64) ≤ 1.5 × max(endpoint errors), 0 failures:
  - 24-layer run: 16 positions × 24 layers per config;
  - single-layer run: all 352 positions.
- **Endpoint errors vs fp64.** GPU median 1.16e-2 (max 1.65e-2); CPU median 1.83e-2 (max 3.48e-2). These are inherent: Q8_1 activations on the GPU versus Q8_K
  on the CPU (README). Hybrid max is 3.5e-2 on trace tokens. One synthetic token reached 7.6e-2, and the endpoints were equally large on that token.

## 2. Miss cost per CPU-computed expert (the key number)

| regime | marginal per CPU expert | fixed per layer that has ≥1 miss |
|---|---|---|
| **24 layers, cache-realistic** (`cpu_only_n1..8`, least squares on medians) | **30.6 us** | 15.2 us |
| 1 layer, L3-hot (27 MiB layer stays resident; not representative of decode) | 10.2 us | 13.4 us |

30.6 us × 0.844 MiB works out to about 29 GB/s, i.e. **DRAM-bandwidth-bound**, as expected for DDR4.

**The D014 / predict_slots calibration of 33.3 us/miss holds as a marginal cost, within about 8%.** That calibration came from whole-layer static offload, and this
is an independent per-expert measurement. The single-layer number would have refuted it wrongly; that run is a benchmark artefact.

## 3. Overlap efficiency: poor in this implementation, so the perfect-overlap bound is NOT reached
In the synthetic sweep (24 layers, per step), the GPU part is about 27–50 us/layer (gpu-only with 8-m experts: 641–1203 us per step) and the CPU part is
42–256 us/layer.

| m | overlap saving ÷ min(GPU part, CPU part) |
|---|---|
| 1 | 0.32 |
| 2 | 0.25 |
| 3 | 0.33 |
| 4 | 0.06 |
| 5 | 0.18 |
| 6 | 0.32 |
| 7 | about 0 (noise) |

The GPU always finishes first: wait-after-CPU is about 1 us/layer. Why the saving is small anyway: total − cpu − wait = **25–35 us per layer of host time spent
launching the ggml-cuda graph** (staging, `graph_compute_async`, `get_async`) **before** the CPU work starts. In this design the thread that launches is also CPU
worker 0, so that overhead is serialized. The bare GPU round trip is only 6.1 us. Launch overhead, not GPU execution, is what does not overlap.

Fitted per step: overlap = 51 us/layer + 30.1 us/miss; serial = 73 us/layer + 26 us/miss.

## 4. H2D admission cost (reported separately)
- One expert (884,736 B), pinned source: **37.9 us median** (p95 38.3), which is 23.3 GB/s. Pageable source: 90 us. Bare 4 KB round trip: 6.1 us.
- **On this PCIe gen4 box, uploading a missed expert (38 us) costs about the same as computing it on the CPU (31 us).** Consistent with that, `upload_lru`
  (llama.cpp `--moe-cache-mib` semantics) roughly ties the CPU-miss hybrid at equal hit rate:

  | K | upload_lru | hybrid static overlap | hybrid LRU overlap |
  |---|---|---|---|
  | 16 | 2685 us | 2808 us | 2895 us |
  | 24 | 1562 us | 1597 us | 1621 us |

- **LRU admission traffic, between steps (mean per step):**

  | K | admission time | hit rate, LRU | hit rate, static (prefill profile) |
  |---|---|---|---|
  | 8 | 3886 us | | |
  | 12 | 2786 us | | |
  | 16 | 1853 us | 74.4% | 73.3% |
  | 24 | 590 us | | |

  Admitting every miss costs **more than the CPU compute it saves**. At K=16 LRU barely beats the prefill-profile static set, so **static (or a
  frequency-gated admission) is the better policy here**; admit-all LRU is not. Hit rates match D014 (about 70–78% at 16 slots).

## 5. Trace-driven per-step times vs the predict_slots formula (24 layers; 352 real decode positions)
t0 = `gpu_ref` = 1198 us per step for the expert FFN of all 24 layers. The prediction is computed per token from that token's actual miss count, then the median
is taken.

| config | hit % | misses/step | measured median (p95) us | pred serial t0 + 33.3·miss | pred perfect overlap | measured ÷ pred serial |
|---|---|---|---|---|---|---|
| hybrid static overlap K8 | 46.1 | 103.6 | 4265 (4908) | 4694 | 3496 | 0.91 |
| hybrid static overlap K12 | 60.8 | 75.2 | 3501 (4489) | 3795 | 2597 | 0.92 |
| hybrid static overlap K16 | 73.3 | 51.3 | 2808 (3277) | 2996 | 1798 | 0.94 |
| hybrid static overlap K24 | 90.8 | 17.6 | 1597 (1973) | 1797 | 1198 | 0.89 |
| hybrid static serial K16 | 73.3 | 51.3 | 3172 (3694) | 2996 | 1798 | 1.06 |
| hybrid LRU overlap K16 (excl. admission) | 74.4 | 49.2 | 2895 (4371) | 2729 | 1531 | 1.06 |
| cpu_all | 0 | 192 | 6336 (8153) | 7590 | 6392 | 0.83 |

**Verdict on the linear 33 us/miss model.** The **serial** form (t0 + 33.3 us × misses) predicts measured hybrid step time to within −11% / +6% for every K in
both modes. It holds.

It breaks in three places:
- (a) The **perfect-overlap** form is too optimistic by 1.5–1.6× at K = 12–16: 2808 measured vs 1798 predicted at K16. Only about 6–33% of the hideable time is
  hidden, because host launch overhead serializes with the CPU work. D014's "517–704 t/s if CPU misses overlap GPU hits" upper range is **not supported** by this
  implementation. The serial-bound range (298–358 t/s) is the one the data backs.
- (b) There is a **per-layer fixed cost** of about 15 us whenever a layer has any miss. This matters when misses are spread thinly (synthetic m=1: 77 us/layer
  measured vs 50 + 33 predicted). With all-CPU work batched per layer (`cpu_all`), the model over-predicts by 17%: the CPU amortizes better when given all 8 experts.
- (c) In cache-hot conditions (small models whose experts fit in L3) the per-miss cost is about 3× lower than the model.

**Derived estimate, not measured end-to-end.** Adding the K16-static-overlap expert-FFN delta (+1.61 ms/step) to the engine's measured 1.42 ms/token (n0) gives
about 3.0 ms/token, i.e. **about 330 t/s at about 897 MiB**, against static `--n-cpu-moe 12`, which measured 212 t/s at 894 MiB. This assumes the harness's
per-layer host sync and launch costs are representative of an engine integration. That assumption is unverified, and the engine's own GPU path is pipelined
without per-layer syncs.

## Honest limits
- Synthetic activations; routing replayed, not recomputed (BF16 trace on Q4_K weights).
- 352 decode positions from 3 prompts, concatenated, with LRU cold-start each pass.
- One thread count (8); one machine; an 8-CPU cgroup quota.
- No attention or other ops interleaved between layers.
- Each hybrid layer is host-synchronized, which an engine needs too, but its cost may differ.
- p95 in the 24-layer run is noticeably wider than the median (e.g. LRU K16: 2895 / 4371), likely CPU and cgroup scheduling jitter. Not investigated.
- The single-layer numbers are cache-hot and must not be used for cost modelling.

## Most valuable next step
Take CUDA launch and sync off the CPU-miss critical path and re-measure overlap in this harness: a persistent CPU-miss worker pool that does not include the
launching thread, or launching layer l's GPU hits before waiting on layer l−1. The gap between measured 2808 us and the 1798 us perfect-overlap bound at K16 is
the whole upside of overlap. If it closes, D014's upper range becomes reachable; if not, the serial formula is the planning model. On the UMA target, re-measure the
per-miss cost as CPU stream bandwidth on that box. Admission and upload costs are PCIe-specific and do not transfer.
