# MoE Inference-Engine Factory: Toolkit Catalog

This catalog lists the tools, modules, skills and methods the factory needs.
- **HAVE** means it is in the repo and demonstrated.
- **GAP** means the work is done by hand today, or the tool doesn't exist yet.
- **TRY** means it's an external tool or method worth trying.

Last updated 2026-11-10, after D041. Statuses come from the decision notes.

## 1. Model intake and inspection
| Item | State | Where / what |
|---|---|---|
| GGUF header inspector: expert count/size, KV bytes per ctx, tensor-class bytes | HAVE | `cc/model_inspect/inspect_model.py` (pinned gguf-py) |
| Per-tensor type verifier (`role@layers=TYPE`) | HAVE | `.claude/skills/model-variant/scripts/verify_types.py` |
| Lossless GGUF expert splitter | HAVE | `cc/hybrid_split/split_gguf.py` |
| Model/dataset fetch with hash pinning (HF etag → sha256 → manifest), plus a quiet-host-aware download | **GAP** | curl/hf by hand. Downloads polluted timings in D035/D040 |
| HF safetensors to GGUF conversion as a CC | GAP | `convert_hf_to_gguf.py`, not yet wrapped |

## 2. System probing and calibration
| Item | State | Where / what |
|---|---|---|
| Resource observer (nvidia-smi, cgroup; "unavailable" ≠ 0) | HAVE (x86) | `cc/resource_observe/`. Jetson/tegrastats backend is a stub |
| Host-load monitor (compiles/downloads/codex) | HAVE | `frozen-experiment/scripts/loadmon.sh` |
| **Host calibration microbench**: GPU BW, CPU expert BW per quant, PCIe H2D, per-layer handoff | **GAP (top priority)** | PL4 showed constants don't transfer between models. Candidates: `test-backend-ops perf -o MUL_MAT_ID`, nvbandwidth, STREAM |
| PCIe / NVLink bandwidth | TRY | `nvbandwidth`, CUDA `bandwidthTest`. Neither is installed, both can be built in cuda-tools |
| CPU DRAM bandwidth / Zen 2 counters | TRY | STREAM/mbw, `likwid-perfctr`, AMD uProf. `perf` needs `perf_event_paranoid` lowered (currently 4, a host change needing approval) |
| Jetson telemetry | TRY | `tegrastats`, jtop (jetson-stats) |

## 3. Planning and modeling
| Item | State | Where / what |
|---|---|---|
| Placement planner (budget → K, KV type, FA, ctx) | HAVE | `cc/moe_planner/plan.py` |
| Bandwidth decode model plus leave-one-model-out validation | HAVE (review pending) | T-PL4: `decode_model.py`, `fit_decode.py`. Granite→Qwen 7.9% MAPE; Qwen→Granite 53% |
| Quality ledger (fail-closed inheritance of measured quality) | HAVE | T-PL3 `build_quality_ledger.py` (64 entries) |
| Pareto frontier/budget fit | HAVE | `cc/memory_pareto/pareto.py` |
| Cache hit-rate predictor from routing traces | HAVE (Granite), extended | `experiments/routing/predict_slots.py`, `cc/routing_trace/trace_replay.py`, T-QRT coverage curves |
| **Expert-cache lever in the planner** (coverage → hit rate → PCIe miss cost) | **GAP (new, D041)** | Cache gave +38% at 8 GB on Qwen and −34% at 4 GB |
| TTFT/load-time model | GAP | The planner predicts decode only. D041 shows the cache trades TTFT for decode |
| Unified-memory (Jetson) host profile | GAP | Offloading saves nothing there. Levers are quant, KV and mmap |

## 4. Build and dependency management
| Item | State | Where / what |
|---|---|---|
| Build CC: pinned source + patch stack → digest-pinned runtime image + record | HAVE | `cc/build_engine/build_engine.py` (v2, `run_ref`, cmake defines, extra targets) |
| Host-side ELF linkage check | HAVE | `cc/build_engine/host_linkage.py` |
| GPU-less nvcc compile check of a patch | HAVE | `bin/cuda_compile_check` |
| Component registry, patch stacking with order-independence test, ABI diff | HAVE | `cc/components/` (`components.json`, `patch_stack.py`, `abi_diff.py`) |
| CUDA tools image (nvcc, nsys, ncu, compute-sanitizer, cuobjdump) | HAVE | `campaign/cuda-tools:c1` |
| **Patch packager**: sha, apply-check, compile-check, register component | **GAP** | Repeated by hand per patch (`artifacts/codex*/package_patch.py`) |
| Build cache (ccache via `CMAKE_CUDA_COMPILER_LAUNCHER`, BuildKit cache mounts) | TRY | Cuts the 4–5 min rebuilds |
| aarch64/Jetson build path | TRY | dusty-nv `jetson-containers`, `ghcr.io/nvidia-ai-iot/llama_cpp`. Build on the device itself, not by cross-compiling |
| Engine forks as alternative "engines" | TRY | ik_llama.cpp (fused MoE ops, IQK quants), KTransformers (AMX/AVX512; little benefit on AVX2) |

## 5. Lab and execution
| Item | State | Where / what |
|---|---|---|
| Lab: image add, idempotent submit (embedded hashed inputs and tests), run, reconcile, export | HAVE | `bin/lab`, `lab/tests/*.json` |
| Alternating-order generator, per-test median aggregator | HAVE | `frozen-experiment/scripts/make_order.py`, `agg.py` |
| Engine-handoff envelopes (proposal/admit/result, host templates) | HAVE | `cc/engine_handoff/` |
| GPU budget shim (cooperative LD_PRELOAD cap) | HAVE | `cc/gpu_budget/` |
| **Quiet-host gate** (lab refuses to time, or auto-labels "untimed", under load) | **GAP** | Several reruns were caused by contamination |
| **Protocol → tests-JSON generator** | GAP | Each experiment's JSON is hand-written |
| Peak-VRAM sampler for ad-hoc runs | GAP (ad hoc) | `artifacts/mc-qwen/screen.sh` polling loop. Should go into resource_observe |

## 6. Profiling and tracing
| Item | State | Where / what |
|---|---|---|
| nsys per-token decode breakdown | HAVE | `cc/nsys_decode/analyze_nsys.py` (D031) |
| **ncu kernel roofline per kernel class** | HAVE (first use, D041) | `artifacts/ncu1`. Needs `--cap-add SYS_ADMIN`. Found about 40% of kernel time in tiny launch-bound kernels |
| Routing tracer (top-k IDs per layer/token; Granite + Qwen3) | HAVE | `cc/routing_trace/` patch + replay |
| **Profiler capture wrapper** (nsys/ncu flags, mounts, extracting the engine `/app` into cuda-tools) | **GAP** | The recipes live only in D004/D011/D041 |
| ncu summarizer (by kernel: time share, GB/s, DRAM %) | GAP (inline script) | Promote from the D041 inline Python |
| compute-sanitizer memcheck/racecheck for engine patches | TRY | Already in cuda-tools, never run |
| eBPF uprobe tracer on `ggml_compute_forward_mul_mat_id` (ProfInfer method) | TRY | bpftrace installed; needs root |
| `cb_eval` per-node timing by `ffn_moe_*` tag | TRY | Pitfall: `ffn_moe_topk` is a strided view |
| `GGML_SCHED_DEBUG=2` split/assignment dump | TRY | Explains backend splits and handoffs |

## 7. Correctness and quality gates
| Item | State | Where / what |
|---|---|---|
| Greedy-text fingerprint grouped by test | HAVE | `engine-candidate/scripts/text_fp.py` |
| Engagement check (prove a patch path is actually taken before timing) | HAVE (method) | D024 |
| Paired PPL + KLD against bf16 logits | HAVE | `llama-perplexity --kl-divergence-base`, `artifacts/kl/` |
| test-backend-ops for patched ops | HAVE | Build CC `--extra-target` |
| **Verdict tool**: frozen criteria → `verdict.json` (speed Δ, text, PPL, KLD, backend-ops) | **GAP** | Checked by hand in each D0xx note |
| Task-level eval (lm-eval-harness against llama-server, logprob tasks) | TRY | Catches regressions that KLD misses on long generation |
| Qwen KLD reference logits | GAP | Needed to gate D041 cache configs and Qwen variants |

## 8. Model variants and compression
| Item | State | Where / what |
|---|---|---|
| imatrix calibration plus `--tensor-type` mixed recipes | HAVE | D035: imatrix is the default (−33% KLD at the same bytes) |
| **Generic `make_variant`**: imatrix → quantize → verify_types → KLD gate → provenance | **GAP** | `cc/pap/make_pap.sh` is PAP-specific |
| PAP (CPU-side experts at Q3) | REJECTED | D040 |
| IQ / IQK quants (fewer bytes per unit of quality) | TRY | Decode is byte-bound, so bytes per quality is the metric |
| Expert pruning (REAP) / merging (REAM, MC-SMoE) | TRY | A smaller model fits more on the GPU. Needs a task-level eval |
| Dynamic expert skipping (MoBiLE, AdapMoE) | TRY (research) | Cuts bytes per token; needs kernels and a quality check |

## 9. Engine components (patches)
| Item | State |
|---|---|
| q8_reuse (+2.1%) | ACCEPTED (D029) |
| kv_quant_fix (q8_0 KV at 32k: −720 MiB, PPL ≈) | ACCEPTED (D033) |
| release-stack-1 (both) | SHIPPED (D037/D040) |
| moe_static_hot, hybrid_split, own LRU | REJECTED (D024, D030/31, D022) |
| upstream `--moe-cache-mib` | **RE-OPENED (D041)**: wins on Qwen at 8 GB |
| Kernel fusion: rms_norm+quantize_q8_1, scale+add, topk+reduce | TRY (from ncu, D041) |
| Speculative decoding: n-gram self-spec (no model needed); MTP heads (`--spec-type draft-mtp`) | TRY. Verifying several tokens at once reads each expert once; watch the growth in the union of experts touched |
| Shared-expert-on-GPU pinning, async CPU/GPU overlap (KTransformers ideas) | TRY (research) |

## 10. Provenance, orchestration, reporting
| Item | State | Where / what |
|---|---|---|
| Recorded ops (intent/outcome JSONL, stdout/stderr artifacts) | HAVE | `bin/run_op`, `bin/index_evidence` |
| Cost report (wall time, agent tokens) | HAVE (unused) | `bin/cost_report` |
| Codex delegation (assignment template, launch, wait, review on real records) | HAVE | `.claude/skills/codex-assignment` |
| Decision notes, frozen-protocol template | HAVE | `.claude/skills/decision-note` |
| Workflow skills | HAVE | frozen-experiment, engine-candidate, model-variant |
| Factory pipeline (inspect → system → plan → recipe → build → measure → report) | HAVE (measure/report not yet run for real) | `cc/factory/factory.py` |
| Demo script | HAVE | `cc/demo/demo.sh` |
| Brief linter (catches wording that triggers provider refusals) | GAP | gaps.md: T-KVB refusal |
| Real-fixture capture (snapshot real tool output to use as test fixtures) | GAP | Codex mocks hid real failures twice |
| cc-inventory.csv refresh | GAP | Stale rows |

## Methodologies (the rules the tools enforce)
- **Pre-register, then measure.** Freeze the protocol before results exist (D026+).
- **A/A, then matched A/B.** Use alternating ABBA order and medians, on a quiet host.
- **Engagement before timing.** Prove the code path ran: fingerprint against references (D024).
- **Text identity for exact rewrites; KLD/PPL for numerical changes.** Placement changes the text (D041), so they need KLD.
- **Real-record fixtures** for any parser of another tool's output.
- **Transferability labels.** Proxy results (Granite) don't carry over to big MoEs. D022 was overturned on Qwen in D041.
- **Budget-guarantee taxonomy:** none, cooperative, cgroup, hard (D021/D023).
- **Image identity:** run by `run_ref`, never by `config_digest`.
- **Profiler perturbation check:** compare profiled and unprofiled timing (D008). ncu inflates small kernels.

## Build next (ranked)
1. Host calibration microbench (§2). It feeds the PL4 model per host.
2. Expert-cache lever plus a TTFT model in the planner (§3), followed by the frozen D041 sweep with Qwen KLD reference logits (§7).
3. Verdict tool and quiet-host gate (§5, §7). These remove most of the manual review.
4. Generic `make_variant` and the patch packager (§4, §8).
5. Profiler capture wrapper and ncu summarizer (§6). After that, prototype kernel fusion.
6. Speculative-decoding screen (n-gram first) on Qwen.
7. Unified-memory host profile for the Jetson.
