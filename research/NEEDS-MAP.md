# Needs map — what the inference factory needs, learned from c1 work (Jetson lane view, 2026-11-10)
Status key: **HAVE** (exercised, evidence) · **NOW** (blocking the over-memory goal) · **NEXT** (valuable, not blocking) · **NO** (rejected, with reason).
Evidence: decisions D0xx, experiments/jetson/*, research/strata-*.md (Strata = MIT reference, own engine; no Jetson/sm_87 build).

## 0. Honest position
- Two full-engine gains exist, both on a model that FITS (Granite-1B): D025 bf16 +19%, D027 Q4_K experts +23.6% (identical outputs) — local to Orin.
- **Nothing has yet run a model larger than memory end to end.** x86 only emulated it (D014 static offload, D015 page-cache cliff, D019 single-layer harness). Jetson (unified DRAM + NVMe) is the closest real proxy of the 128 GB unified + SSD target → over-memory track started: Qwen3.6-35B-A3B UD-Q4_K_XL, 20.8 GiB on 5.9 GiB usable (~3.5x).
- Designs ARE model×hardware specific (D027: unfuse helps bf16/q4_K, hurts q8_0 on Orin). So the factory's product is: components with declared applicability + cheap per-platform qualification screens + a reproducible measurement method — not one universal engine config.

## 1. Tools (operations agents invoke)
| Need | Status | Notes / evidence |
|---|---|---|
| Build CC (identified source+patch → native image + record; amd64 & Jetson) | HAVE | D016, J-T1/J-T1b; ccache 22x (D026). Gap: binaries not bit-reproducible (D026) → identity by record+functional check; add `-ffile-prefix-map` NEXT |
| Model identity (`gguf_manifest`: tensor-data hash vs metadata) | HAVE | J-E001 file sha differs from x86; tensor compare pending x86 |
| Quantize/representation recipes as recorded ops | HAVE | J-E006 via Build CC `--extra-target llama-quantize` |
| Kernel microbench screen (`test-backend-ops perf -o MUL_MAT_ID` per type/shape/fusion) | **NOW** | would have found D027's per-type rule in minutes; required before porting dispatch rules to x86/other boards |
| Expert routing trace capture (per token/layer, prompt vs generated) | **NOW** | x86 cc/routing_trace patch exists (Granite); must qualify on qwen35moe + record completeness (Strata trace reader silently dropped truncated tails) |
| Expert-cache replay simulator (LRU/LFU/decay/static/Belady, generic L×E) | NOW (J-OM2 running) | Strata pattern, hardcoded 48×512 there |
| Static hot-set placement generator (profile → `-ot` tensor-override args / cache profile) | **NOW** | turns a routing profile into an engine placement; needs per-expert tensor split support check in llama.cpp (experts are fused 3-D tensors → may need a component) |
| Result acceptance / perf_summary with A/A-aware gate | NEXT | adopt Strata autotune rule: median per-round ratio > max(3%, 2×A/A gap, 2×drift); byte-identical outputs auto-accept, else KL gate |
| CUDA register-spill / static checks in Build gate | NEXT | Strata tools/check_cuda_spills.py pattern |
| Launch identity (effective argv/env/config hashed per attempt) | NEXT | Strata serve/launch_identity.py pattern; our run_op records argv, not engine-effective config |
| nvpmodel/clock pinning helper | NEXT (needs owner sudo authorization) | GPU devfreq 306–1020 MHz unpinned; we rely on warm-up discard + A/B interleave |

## 2. Profiling / tracing / observability
| Need | Status | Notes |
|---|---|---|
| nsys timeline + kernel inventory (sm_87, sm_120) | HAVE | J-P001..P003; gotcha: stale .sqlite → 0-byte CSV, use `--force-export=true` |
| ncu counters on Jetson | NO (for now) | RmProfilingAdminOnly=1; needs root — use derived bytes/time instead (done) |
| Resource telemetry with scopes (tegrastats/proc/cgroup/sysfs/nvml) | HAVE | cc/resource_observe; never sum scopes; unavailable-with-reason |
| I/O + fault telemetry (vmstat majfault/refault, NVMe MiB/s, per-pid/cgroup faults, io.stat) | **NOW** (J-OM1 running) | over-memory runs are I/O bound; current telemetry blind to it |
| Safety watchdog (MemAvailable/swap trip → kill only the owned run) | **NOW** (J-OM1) | required before any oversubscription run on an 8 GB host |
| Quality: ppl smoke + paired KL + **decode-path (ub=1) KL** | HAVE | methodology counterexample J-E005: batch-512 ppl blind to decode kernels |
| Quality reference when the full model cannot be resident | NOW | same GGUF, placement-invariant reference run (slow but exact) + a resident smaller-quant reference; KL top-k file for 20 GiB model must be bounded (c128 × few chunks) |
| Engine-internal counters (cache hit/miss, bytes uploaded, stall time) | NOW | llama.cpp `--moe-cache-mib` exists (LRU upload-on-miss) — need its counters exposed or an nsys NVTX range |

## 3. Methodologies (agent procedures / decision rules)
| Rule | Status |
|---|---|
| Counterbalanced ABBA/ABAB, ≥2 runs/arm, discard first sample, steady mean | HAVE (J-E004/5/7) |
| A/A before trusting a delta; same image, env-switch arms (isolates code path) | HAVE |
| Classify every result (code gain / placement / representation / memory-fit / screen / prediction) | HAVE |
| Decode-only changes need ub=1 KL; prefill unaffected must be shown | HAVE |
| Shared-memory hosts: pin `-c`; never trust free-memory-derived engine defaults (D023); `-lm auto` disables mmap on Orin iGPU (llama-model.cpp:1590, ggml-cuda.cu:5473) → over-memory baseline must say `-lm mmap` | HAVE (new) |
| Fair "default" baseline for over-memory = what a competent user runs (`-lm mmap`, `-cmoe`/`-ncmoe N`, `--moe-cache-mib`), not the broken auto default; report the auto default's failure separately | NOW |
| Benchmark protocol IDs + corpus sha in every result; `nonconforming` status; cache-hit prompt check | NEXT (Strata BENCHMARK_PROTOCOL.md) |
| "Measured on:" line before every number; contract doc per component | NEXT (Strata docs style) |
| Per-platform applicability table per component (type × cc × batch) | HAVE (cc/moe_dispatch_sm87) |

## 4. Build tools / dependency management
| Need | Status | Decision |
|---|---|---|
| Docker + Buildx (amd64), host-native JetPack builder + l4t runtime image (Jetson) | HAVE | one CC, two native targets |
| ccache with stable basedir | HAVE | D026 |
| Python deps: stdlib-first; venv for heavy converters (torch/transformers) | HAVE | ~/venvs/convert; system numpy broken on Jetson |
| Pinned upstream sources (llama.cpp 31288c9c6, Strata 504b1433 ref-only) | HAVE | record pins in decisions |
| Nix | NO | Strata's flake is x86_64-only; JetPack is apt/L4T-bound |
| Multi-platform registry index | NEXT | only after both native outputs + a registry exist |

## 5. Code fragments / functional modules (component CCs)
| Component | Status |
|---|---|
| moe-gateup-unfuse by type (patch 03) | HAVE (Orin qualified; x86 untested) |
| Static hot-expert placement (resident hot set + streamed cold experts) | **NOW** — main candidate for "better than default" over-memory |
| Router-aware next-layer prefetch / async read of cold experts (madvise WILLNEED / readahead on mmap ranges) | NEXT — needs trace + I/O telemetry first |
| Expert tiering policy (LFU-decay swap) à la Strata exchange rotation | NEXT — simulator first (J-OM2) |
| gpu_budget shim (cudaMemGetInfo/cudaMalloc cap) | NEXT on Jetson — natural cap for D023; x86 X3 owns qualification |
| x86 hybrid hit/miss dispatch (D019) | x86-owned; Jetson relevance low (no PCIe; CPU weak: 6×A78AE) |

## 6. Isolation / lab / Manager
| Need | Status |
|---|---|
| Admission by declared host capabilities; rejected/needs_information; hashed admission | HAVE (Jetson live) |
| Sandbox kill / runner death / reconcile / new attempt | HAVE (Jetson live) |
| Memory-cap isolation for over-memory tests (cgroup memory.max on container: charges page cache; does it charge nvmap/CUDA? unknown) | **NOW** — qualify with a bounded test before the 20 GiB runs |
| External evidence endpoint off-host | NEXT (git-only today; protects against sandbox loss only) |
| Storage admission (model files 20+ GiB, page cache effects, free-disk check) | NOW (62 GB free; 22 GB model) |
| GPU slot per host; no concurrent builds during timing windows | HAVE (rule), not enforced by code |
