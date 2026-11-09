# Candidate source claims for local investigation

Input: `/home/muk/projects/Strata`, commit `ffa80d73133ea5c548aefe57889a16e8085a938f` (HEAD verified). All citations below are relative to that input at that commit. This is static source analysis: no network, builds, GPU execution, model loading, or reproduced measurements. The stated target is a **16 GB RTX 5070 Ti**; CPU, RAM, OS, storage, driver, CUDA toolchain and model files are unspecified. VRAM capacity alone does not establish model fit.

“Granite” is not identified by model name, revision, quantization or runtime. No Granite model integration was found in the inspected Strata source/docs (the `granite` word in `tools/dgx_bench.py:42` is prompt vocabulary). `docs/AI_REFERENCE.md:30-38` names DeepSeek V4.1, Qwen3.8 and Qwen3.6; `tests/core/gguf_geometry_test.cpp:263-283` exercises refusal of other architectures. Consequently, Granite can supply a separate implementation or small input fixture only after a compatible interface is established. Running a different model does not reproduce a Qwen performance or quality result.

Classes: **reproduction** = repeat the claimed behavior in the same relevant implementation and conditions (a numerical historical reproduction also needs the original hardware/workload); **related mechanism** = test a narrower analogous property with fixtures or a different model; **new hypothesis** = a proposed extrapolation absent upstream evidence. The class in each entry applies to the proposed local question. All historical numbers below are **upstream claims, not established facts or local results**.

## C01 — GPU, RAM and SSD are separate model-dependent tiers

- **Exact source:** `docs/AI_REFERENCE.md:10-12`, with limits at `docs/AI_REFERENCE.md:76-90`.
- **Claim:** the engine uses GPU, system RAM and, for applicable models/modes, SSD; GPU support does not establish support for every model/feature. RAM, VRAM, download size and conversion space must be counted separately.
- **Conditions:** a supported family/backend and its actual memory mode, context and quantization. No universal throughput or fit guarantee is stated.
- **Granite / this machine:** a 16 GB card makes a tier experiment plausible, but neither Granite support nor host-memory sufficiency follows. Use generated expert blobs for a local tier test without full weights.
- **Falsifiable local question:** for a tiny file-backed expert fixture, do RAM-resident blobs equal file bytes, and does eviction restore file fallback without substituting another expert? `tests/core/file_expert_source_test.cpp:144`, `:192`, `:259` and its opt-in GPU rotation cases cover parts of this.
- **Class:** related mechanism.

## C02 — More VRAM holds more experts and reduces CPU work

- **Exact source:** `docs/DETAILS.md:85-87`.
- **Claim:** every additional GB holds about 700 more experts, and an expert on the GPU is one the CPU does not compute; the document argues VRAM matters more than GPU speed.
- **Conditions:** the page's Qwen packs and expert geometry, memory reserved for dense weights, context and draft state, and a workload that routes to the additional cached experts. The approximate expert count is not a universal bytes-per-expert constant.
- **Granite / this machine:** 16 GB rather than the page's measured 12 GB configuration changes the cache budget; Granite's expert size and routing may be different or it may be dense. A fixture can establish accounting and miss behavior, not the speed ranking.
- **Falsifiable local question:** with fixed synthetic expert sizes and routing, does increasing cache capacity produce the predicted extra slots and fewer CPU misses, without changing expert identity? Report actual blob sizes and hits rather than assuming 700/GB.
- **Class:** related mechanism.

## C03 — Adaptive swaps can run between windows, with compatibility restrictions

- **Exact source:** `docs/DETAILS.md:257-259`.
- **Claim:** `--adapt-async 1` moves adaptive swaps onto a helper between decode windows; works with batch slots, excludes batch groups and peer-device mode, and is not bit-exact run to run.
- **Conditions:** opt-in serve path with adaptive/resident tiers; safe publication of incoming bytes and ownership changes before consumers use them.
- **Granite / this machine:** small staged blobs and host concurrency fixtures suffice for ownership/order checks. Actual Strata async inference requires a supported family and runtime; arbitrary Granite runtime concurrency is a different implementation.
- **Falsifiable local question:** while a swap is staged, is conflicting RAM replacement refused, and after commit do every expert's host/device aliases still expose its own bytes? Check `tests/core/file_expert_source_test.cpp:761`, `:851`, `:940` (GPU cases), plus `tests/core/exchange_storage_test.cpp:109-194` (CPU ownership model). This does not alone validate the complete async serve loop.
- **Class:** reproduction of the narrow ownership contract; complete concurrent serve validation remains open.

## C04 — IQ grouping can change greedy answers; deterministic controls narrow that difference

- **Exact source:** `docs/DETAILS.md:346-350`.
- **Claim:** IQ one-token and multi-token CPU experts round differently; `STRATA_IQ_MT_MIN=1` removes drafting-dependent grouping, and byte-identical A/B runs additionally disable prompt cache, adaptive swaps and PCIe experts. A Ryzen 7600 IQ3_S decode cost of 1–3% is reported.
- **Conditions:** native IQ packs, actual CPU kernels, identical inputs and numerical path; the extra controls are part of the stated repeatability recipe. The cost is historical hardware-specific evidence.
- **Granite / this machine:** a compatible native IQ expert fixture can compare grouping without loading the model. Granite quantization/runtime may not use these kernels. Host ISA and worker topology are unknown; no 1–3% prediction is justified.
- **Falsifiable local question:** do the same expert/input rows agree across group sizes when the stated exact path is forced, and can the unforced path differ? No dedicated end-to-end drafting/cache-invariance test was found under the assigned inventory roots; native numerical parity programs also live outside them in `src/kernels/`.
- **Class:** related mechanism.

## C05 — Native Q2_0 fused prefill may gain mostly through a larger chunk

- **Exact source:** `docs/DETAILS.md:1147-1163`, especially `:1154-1160`.
- **Claim:** on the author's RTX 5080 16 GB, native Q2_0 fused prefill gained 2–3.5% at fixed chunk 4,864, versus 32–33% on a 16K prompt with auto chunk growing to 5,632; generated text can change. A one-layer test and first-token KL figures are reported, with Ampere explicitly unverified.
- **Conditions:** Linux, CUDA 13.0, i9-14900K, 2,100 resident experts, CPU prompt share off, identical build, interleaved rounds; switch `STRATA_PF_FUSED_Q2N=1`. Q2_0/Q2_0 native experts on sm_80+.
- **Granite / this machine:** a 16 GB 5070 Ti can be a new hardware check of the kernel fixture if the relevant code is built. Its capacity matching the 5080 does not replicate the throughput. Full Qwen/host fit is not established; Granite does not reproduce the reported model result.
- **Falsifiable local question:** on this card, does `prefill_fused_iq_test_q2n` satisfy its numerical oracle, and, later with a fitting supported model, do fixed-chunk and auto-chunk A/B runs separate kernel savings from changed work? Record chunk size, memory, prompt and decode separately.
- **Class:** related mechanism (kernel fixture); transferring the speedup to this card/model is a new hypothesis.

## C06 — A pointer-keyed activation cache is invalid across layers

- **Exact source:** `include/strata/core/expert_source.hpp:10-14`; implementation explanation `src/core/expert_source.cpp:4745-4747`.
- **Claim:** every layer overwrites the same pinned `x_f` address. Caching activation quantization by pointer would reuse layer 0's values for subsequent layers and yield finite but incorrect output; conversion must happen every call.
- **Conditions:** host-loop adapter, same buffer address, changed contents, legacy CPU expert activation quantization.
- **Granite / this machine:** no large model or GPU is conceptually required to falsify pointer-only caching. A small fixture with two distinct activations at one address can expose it; it tests the actual adapter only if wired through that adapter. A Granite equivalent is an analogous test.
- **Falsifiable local question:** after overwriting the same buffer with a second activation, does the second expert call equal a freshly quantized reference and differ from the first? Include zero/nonzero and sign-changing inputs. **No direct two-call adapter regression test found under `tests/` or `tools/test_*`.**
- **Class:** related mechanism until the production callback is exercised.

## C07 — Router weights must be applied once at combination

- **Exact source:** `include/strata/core/expert_source.hpp:15-17`.
- **Claim:** the pool produces unweighted expert parts; device `moe_combine` multiplies them by routing weights. Applying the weight in the pool too would apply it twice. The embedded `layer.cpp:479` reference is a historical comment, not a verified current line locator.
- **Conditions:** this adapter/pool/combine contract. Other engines can place the one multiplication elsewhere and remain correct.
- **Granite / this machine:** a tiny independent scalar/reference example is enough to distinguish `sum(w_i*y_i)` from `sum(w_i*w_i*y_i)`; use non-unit, unequal weights and nonzero distinguishable expert outputs. Testing an unrelated Granite MoE only checks its own contract.
- **Falsifiable local question:** does pool-to-combine output match exactly one weight application for weights such as 0.2 and 0.8? Include a deliberately double-weighted negative control. **No direct pool-to-combine once-only weighting regression test found in the assigned roots.** Fused Qwen tests compare expert products; that is not proof of this integration contract. A related DeepSeek synthetic token/window-combination oracle exists at `tests/dsv41/ds4_gguf_test.cu:341-455`, using non-unit weights and a once-weighted reference; it exercises a different implementation.
- **Class:** related mechanism until wired through pool and combine.

## C08 — The legacy CPU expert kernel requires compatible geometry

- **Exact source:** `include/strata/core/expert_source.hpp:18-21`; `src/core/expert_source.cpp:4728-4737`.
- **Claim:** legacy artifact geometry is H=2560, FF=640, blob=1,382,400; incompatible dimensions must be refused instead of walking into adjacent expert bytes. Native IQ has a separate window path; do not generalize this legacy blob size to all packs.
- **Conditions:** the fixed-format artifact/kernel and its adapter, distinct from metadata parsing or native variable layouts.
- **Granite / this machine:** this is a first gate for any Granite attempt. A different architecture or dimensions do not become compatible because the weights fit in 16 GB. Synthetic GGUF/layout fixtures can test rejection with no GPU; arbitrary Granite files should not be forced through this path.
- **Falsifiable local question:** do a valid fixture and one-key dimension/architecture mutations produce acceptance/refusal with a named diagnostic before data interpretation? `tests/core/gguf_geometry_test.cpp:242-283`, `model_family_test.cpp:1-11`, and `expert_layout_test.cpp:8-17` cover related guards; no direct bad-width host-adapter invocation test found in these roots.
- **Class:** reproduction of metadata/layout guards; related mechanism for the legacy callback refusal.

## C09 — Choosing hits from the previous layer's IDs severely changed logits

- **Exact source:** `src/core/expert_source.cpp:4749-4756`.
- **Claim:** an earlier implementation made its hit decision too late and computed previous-layer experts into current rows; the comment reports C1 mean KL worsening from 9.69e-02 to 1.03e+00. This is a historical upstream measurement comment, not a reproduced quality result.
- **Conditions:** Launch-before-pool callback sequencing; current IDs must determine current hits; job-array index and output-row index diverge when some experts hit. The comment does not fully specify C1 model, hardware or corpus.
- **Granite / this machine:** alternating disjoint layer IDs with per-expert byte/output sentinels is a small local test; the old C1 KL figure cannot be reproduced from this comment alone. File/cache identity tests protect pieces but not the entire callback sequence.
- **Falsifiable local question:** after switching IDs and a mixed hit/miss mask, does each output row belong to the current `(layer, expert)` and its original routing position? Include a deliberately one-layer-delayed ID negative control. See the identity coverage and gaps in `tests.md`.
- **Class:** related mechanism.

## C10 — Hit activation quantization needs current data and CPU-compatible scales

- **Exact source:** `src/core/expert_source.cpp:5373-5400`, especially `:5385-5393`.
- **Claim:** the hit output must be zeroed to avoid adding a previous layer's stale hit row; the current activation must be quantized rather than reusing a previous-layer buffer. A historical `bench/micro/act_quant_parity.cu` measurement reportedly found all 80 chunks different, up to 4.761e-04 relative, for fp16 versus fp32 scale rules, motivating the scaled quantizer.
- **Conditions:** legacy grouped hit path, current `mixed` activation and fp32 per-block scales matching CPU `act_quant_q8_1`. A finite output is not a correctness oracle.
- **Granite / this machine:** generated 2560-wide activations and synthetic expert blobs can probe the mechanism on a small memory budget. No Granite integration is established; the historical bench path is outside the inventory roots and its availability must be checked independently.
- **Falsifiable local question:** when a row changes hit→miss across successive calls, is its obsolete hit contribution zero, and does scaled GPU quantization match the CPU reference on small/large/sign-changing blocks? **No dedicated successive-layer hit-buffer/current-activation test found under the assigned roots.**
- **Class:** related mechanism; reproducing the historical 80-chunk number needs the original fixture and scale/error definition.

## C11 — Async checkpoint/park paths claim unchanged snapshot and answer bytes

- **Exact source:** `docs/DETAILS.md:1575-1589`.
- **Claim:** opt-in CUDA checkpoint/park copies move through VRAM/background threads while keeping answers and parked images byte-identical. Contributor CMP 170HX timing claims include checkpoint 104–105→34–35 ms (fresh vectors), with separate reused-vector figures; the project explicitly says it did not measure that card.
- **Conditions:** supported CUDA path, reader waits, allocation-failure fallback; slot features require their parent async switch; several paths require 1.5 GB free; gather was not tested across a layer split. Some reported park timings combined switches.
- **Granite / this machine:** state-size and architecture differ; a tiny snapshot fixture can test lifetime and byte parity. A 16 GB budget may decline the staging/slot path, so prove the requested arm actually ran. PCIe link and OS are unknown; CMP timings cannot transfer.
- **Falsifiable local question:** while an async copy is outstanding, do restore, eviction, clear and shutdown wait for ownership safely, and do restored bytes equal the synchronous reference? Offline `tools/test_conversation_cache_*` validate evidence-checking logic, **not execution of these GPU paths**. Relevant actual snapshot/transfer tests also live outside the requested roots in `src/core/`.
- **Class:** related mechanism; performance on the 5070 Ti is a new hypothesis.

## C12 — Architecture-specific measured defaults must not silently generalize

- **Exact source:** `src/core/nv_defaults.cpp:53-80` (GB10 sm_121 branch), especially `:58-66` and `:73-79`.
- **Claim:** the GB10 defaults were selected from reported measurements: fused IQ prefill ratios 1.139/1.149/1.121 at 4K/16K/32K and five identical greedy answers; NVFP4 prompt ratios 1.106/1.229 and 24 identical answers; cost-aware MTP decode 1.047x. These are upstream comments about one part, not local facts or universal bitwise guarantees.
- **Conditions:** sm_121 DGX Spark, stated packs/workloads, default-version gate and explicit user overrides. The code branches on major=12, minor=1; the target 5070 Ti is the sm_120 family in this source's naming, not this branch.
- **Granite / this machine:** pure rule checks can confirm sm_120 does not inherit the sm_121 table; a kernel fixture can investigate an analogous improvement. Neither a small Granite run nor shared “Blackwell” branding reproduces GB10's unified-memory results.
- **Falsifiable local question:** does the defaults function return only the intended architecture's settings and preserve explicit user choices? If fused IQ is explicitly tested on the 5070 Ti, does the numerical oracle pass before any speed comparison? Defaults tests exist outside the requested inventory in `src/core/arch_defaults_test.cpp`; the assigned roots include fused-kernel fixtures, not the sm_121 defaults unit test.
- **Class:** related mechanism; extrapolating the reported ratios is a new hypothesis.

## Local priorities and open inputs

1. Start with the existing synthetic geometry/layout, source-byte, ownership and route/order fixtures listed in `tests.md`; they can address correctness without the full Qwen weights. This assignment only inventories them; none was built or run.
2. Add direct adapter tests for C06/C07/C09/C10 in a separately authorized implementation task. Require changed contents at an unchanged address, changing `(layer, expert)` IDs and hit masks, non-unit weights, and explicit negative controls. A passing independent toy implementation is insufficient evidence about the adapter.
3. Later GPU fixtures should record exact GPU/toolchain, memory use, actual dispatch and skip reasons. sm_80/sm_90-only and HIP-only paths are not verified by a successful or skipped test on an RTX 5070 Ti.
4. Resolve Granite model/revision/quant/runtime, available CPU ISA/RAM/OS/storage, and existing model/binary availability before choosing any end-to-end run. Historical measurements require their original workloads/builds and hardware for literal reproduction; otherwise report a new-machine related experiment.
