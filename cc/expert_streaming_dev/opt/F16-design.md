# F16 implementation notes

All components are independent diffs against 6e9d96cf4, enabled only by their new environment switches. The existing fill gate remains required for pregate. No graph selection, tensor contents, sampling, or stdout code changes.

## P12

CSV columns: `run,token_idx,layer,phase,selected,predicted`. IDs are space separated; predictions are the issued distance-1 router list for that layer. Each appended context has its own run ID. Prompt rows pruned before the final MoE layer are absent; surviving rows are indexed through the ubatch output mask, including noncontiguous output positions. Token indices count processed ubatch tokens, including CLI warmup. Phase follows the existing streaming convention: ubatch size 1 is decode; larger ubatches are prompt. Thus a single-token prompt is classified as decode. This is an observation of evaluation calls, not a sequence-position identifier.

The callback copies each token's host IDs before demand/prediction arrival handling. Records accumulate in memory and flush at 128-token boundaries between graphs or at teardown. The largest prompt ubatch is an additional buffer bound. No trace file calls occur per layer. The trace observer also works without fill or router prefetch. Tracing adds graph observation when these were otherwise off.

The simulator sorts layer-major prompt records into token/layer order and keeps appended runs independent. Predictors use past data only; prompt history warms frequency and previous-token predictors. Unions interleave source rankings before truncation to K. Router coverage is reported separately, since Granite cannot enable the base router pregate implementation. Cache policies are demand-only, equal-sized (layer, expert) objects with prompt warmup, independent caches per run, global residency across layers, and cold-start misses included. `lru_frequency` is a two-queue frequency/LRU approximation (not full ARC). Belady uses full future knowledge and standard demand admission. Reported bytes are model-size estimates, not measured storage bytes, and exclude speculation, kernel readahead, page rounding, dense weights, KV, and GPU allocation. Reuse histograms include token gaps and distinct-object stack distance; cold references are separate. All default sizes and budgets describe Satay, not the Granite test model.

## P13

Populate workers call MADV_POPULATE_READ on page-aligned, bounded mmap chunks. Both demand and explicit async jobs cover cached chunks as well: mincore cannot prove that PTEs are installed. Successful populate jobs allocate no scratch and perform no pread. EINVAL/ENOSYS fall back to the existing pread loop for each chunk, allocate scratch only when needed, and log once per process. Other errors propagate through existing demand/async error paths. The old opt-in FILL_POPULATE second pass is suppressed when the populate backend itself is selected.

`populate_calls` includes old diagnostic calls and new demand/async backend calls; `populate_bytes` counts successfully populated virtual bytes, including already cached/mapped pages, not disk traffic; `populate_fallbacks` counts chunks using the unsupported-kernel fallback. `bytes_read` and async bytes/read counters retain their pread meaning. Thus zero pread bytes on the populate backend does not prove zero disk IO.

## P14

Each selection increments a running per-(layer, expert) count on decode calls. The first ranking occurs before the next decode after the configured warmup has completed, then every PIN_EVERY decode calls. Prompt calls do not train counts. Complete expert bundles are ranked by frequency, then stable layer/expert order. Budget accounting conservatively charges all page-aligned tensor slices before merging shared boundary pages, so the pinned union cannot exceed the configured budget. Never pin a partial expert.

Repinning unlocks the old page union and locks the new union between graphs. This also handles shared boundary pages correctly. The first mlock failure releases successful locks and switches permanently to MADV_WILLNEED on each subsequent ranking. In this mode pinned_mib and pinned_experts are zero: advice is not residency. Small container RLIMIT_MEMLOCK values therefore cause a logged `pin_mode=willneed`, not an inference failure. Plain mmap mode is required, avoiding ownership conflict with model-wide mlock. This experiment assumes one pinning context per model mapping; Linux mlock is not reference-counted across contexts.

Pinning consumes the same shared RAM pool needed by GPU, KV, dense weights, and pageable experts. The 8192 MiB parser maximum is not a safe recommendation for a 7.4 GiB machine. Start small and measure phase changes; running frequency does not decay.

## P15

Temporal mode reuses the previous decode token's selection at the target layer, issuing it at the current token's preceding layer (and at the existing distance-2 hook if configured). Each layer's history updates only when its actual IDs arrive, after all earlier-layer predictions have already used its old history. Prompt ubatches clear temporal history. No router weights, scalar dot products, or architecture-specific input assumptions are needed in temporal mode, so it also runs on Granite. Router and hybrid modes retain the existing Qwen35MoE restrictions.

Hybrid forms the union of router top-K and the previous-token set, ranks by router score, then membership in the previous token at score ties, then expert ID; it truncates to K. A strict router-score-first rule normally reproduces router top-K. Meaningful temporal substitution without ties would require changing the requested policy, e.g. a source quota or temporal score bonus.

`recall_router` and `recall_temporal` compare the independent source sets against actual IDs at eligible prediction arrivals, with cold history counted as misses. Router mode tracks temporal history for comparison too. Temporal mode does not evaluate router scores, so its empty router source reports recall_router=0. The existing total recall/timeliness counters still describe the actual issued list. Distance 1 and 2 contribute separately when both are enabled. Layer 0 has no preceding-layer prediction.

## CPU fixture caveat

The prescribed Granite command permits CPU_REPACK on this ARM host: the existing CPU override chooses extra CPU buffer types, so its expert weights are not mmap-backed and the baseline fill path is inactive. Exact-command switch comparisons are retained. Supplemental runs add the existing LLAMA_ARG_REPACK=0 setting to exercise the actual mmap fill/pin/temporal paths. LLAMA_ARG_LOG_VERBOSITY=4 exposes baseline library INFO messages that completion normally hides. New required summaries are emitted directly on stderr. Granite still cannot exercise the architecture-specific router or hybrid predictor; temporal mode is architecture independent.

## Satay expectations

40 layers x 8 experts x 1.83 MiB = 585.6 MiB of expert bundles selected per decode token. Expert bytes total 18.3 GiB; 2-5 GiB of cache holds about 11-27% of that set before overhead. Dense weights, GPU memory, KV, OS, and scratch further restrict the 7.4 GiB shared pool. Cold or uniform routing still needs hundreds of MiB of storage reads per token, regardless of predictor quality.

P12 can identify whether frequency, temporal reuse, or better eviction policy can actually reduce this traffic. P13 removes the private scratch copy (up to roughly 1.14 GiB of read+write memory traffic per all-cold token) and later mmap faults; it wins when memory bandwidth or page faults matter, but does not remove storage traffic. P14 wins only when pinned hot experts cover more requests than the displaced pageable set; stale pinning or oversized budgets can hurt GPU and demand residency. P15 wins when previous-token recall and layer-to-layer lead time let async reads finish before demand. Wrong predictions add storage/memory bandwidth, queued predictions are canceled, and running reads may duplicate demand. K=8 has no traffic headroom for low precision; the trace and timeliness counters should determine whether it is useful.
