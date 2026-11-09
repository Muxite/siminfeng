# Logging llama.cpp router expert IDs: options and limits

Assignment T-C, 2026-11-09. Offline source analysis at llama.cpp `31288c9c66e59377f7788d920678a565bb89dd37`; no runtime image, GGUF, profiler run, harness build, or timing measurement was supplied/performed. Source root is `/home/muk/projects/siminfeng/campaign/ref/llama.cpp`; citations below are `path:line` from that root. Strata comparison revision is `ffa80d73133ea5c548aefe57889a16e8085a938f`.

**Recommended first tool:** a small out-of-tree program using public `llama_context_params.cb_eval` against an existing, ABI-matched libllama. Filter at `ask=true`, collect the full I32 `ffn_moe_topk-L` tensor with byte strides, and explicitly map every row to the input token. This can leave the runtime image unchanged if the image exposes usable libraries/headers and allows a bind-mounted executable. It cannot attach a callback to an already-running `llama-server` through its HTTP API. If the immutable image contains only a standalone executable, the no-image-change harness option is conditional, not assured.

## Options ranked by cost

Costs are qualitative engineering/runtime costs inferred from source, not measured estimates.

| Option | Changes required | Cost | Can prove | Cannot prove / main limitation |
| --- | --- | --- | --- | --- |
| Existing logs + `GGML_SCHED_DEBUG=1/2` | Environment/log-level settings only, if binary supports them. | Lowest setup; metadata output overhead. | Model allocation, graph splits, assigned backends and copied input names. `ggml/src/ggml-backend.cpp:1146-1182,1551-1553,1962-1963`. | No runtime router values, complete route trace, per-expert byte correctness or CPU/GPU overlap. Enable application's debug logging or GGML debug messages may be suppressed. |
| Existing `llama-debug --tensor-filter 'ffn_moe_topk-.*'` | None if this binary is already present. | Low setup, **very high perturbation**: callback requests every node even with filter. | Sample IDs and tensor shape/name; useful discovery and single-token smoke checks. `examples/debug/debug.cpp:231-234`; `common/arg.cpp:4595-4601`. | Default printer truncates **every dimension** to 3 head/tail elements; K>6 loses middle experts, T>6 loses tokens. `common/debug.cpp:79-122,186`. Not a complete trace. Filter is checked after `ask` and GPU copies occur even for filtered-out tensors (`143-186`). |
| Existing `llama-eval-callback` | None if shipped. | Low setup, same generic logger overhead, potentially enormous output. | Demonstrates callback use; tokenizes prompt, decodes it and logs tensors. `examples/eval-callback/eval-callback.cpp:11-35,54-58`. | No selective full structured trace by default; it uses the same truncating printer, disables warmup, and is a prompt evaluation example rather than a full generation tracer. |
| **Out-of-tree selective cb_eval harness** | Build/mount a separate executable against existing matching library; no libllama patch and no image rebuild. | Small implementation; synchronization per observed layer/microbatch, tensor copy and record I/O. | Complete routed expert IDs, optionally probabilities/weights, per input token/layer under controlled execution. Public fields `include/llama.h:393-394`; scheduler contract `ggml/include/ggml-backend.h:311-318`. | Runtime timings are perturbed; no guarantee about unobserved fusion/capture, actual cache hits, weight bytes, transfer/computation overlap. Cannot change an existing server context via HTTP. |
| Application-only patched build | Replace generic logger or add trace option to an executable, leaving core math unchanged. | Small patch/build; same cb_eval runtime cost, but easier deployment in a dedicated trace binary. | Same data as harness, integrated token/batch metadata and complete output. | Requires replacement binary or image/deployment change; selective cb_eval still alters execution boundaries. |
| Cache copy-hook trace | Patch `llama_moe_cache::impl::copy` / selective copy path to record IDs and fills already read by the engine. | Moderate patch; low additional device synchronization on paths already reading IDs, plus CPU logging cost. | Cache planning decisions, logical expert -> slot, hits/evictions, bytes submitted, selected uploads. `src/llama-moe-cache.cpp:548-580`; `src/llama-context.cpp:2789-2837`. | Only layers/microbatches taking these paths. Cache-disabled/GPU-resident/CPU execution can be missing. Cache dedup loses per-token multiplicity if only `uniq`/fills is logged. Queued uploads are not proof of completion or correct bytes. |
| GPU trace-buffer kernel / backend hook | Patch graph/backend, preallocate trace buffer; device-side record and defer readback. | Highest implementation/testing; lower possible per-layer host overhead, device writes and final readback remain. | Complete IDs on intended GPU path with less host synchronization; can join device timestamps/slot metadata. | Must cover fused and unfused routes, CPU fallback and graph replay; capture cannot freeze positions/write indices. Still needs correctness and overhead calibration. |

`--tensor-filter` is restricted to the debug executable in the common argument table (`common/arg.cpp:4595-4601`); it is not a general-purpose routing flag on every tool. `examples/eval-callback/eval-callback.cpp:56` installs the generic logger directly, rather than a specialized routing collector.

## What to log

The logical Granite router output is named **`ffn_moe_topk-L`**, I32 dimensions `[K,T,1,1]` (`src/llama-graph.cpp:2267-2273`). `graph_get_cb` appends the layer number (`src/llama-context.cpp:2846-2852`).

Crucially, `ggml_argsort_top_k` returns a **strided view** of a full E-wide descending argsort (`ggml/src/ggml.c:5463-5476`). Its row stride can be `E*sizeof(int32_t)`, not `K*sizeof(int32_t)`. For rank r, row t, read:

```text
id(r,t) = I32_at(buffer + r*nb[0] + t*nb[1])
```

Do not copy K*T integers and then assume packed rows. Either retrieve the view's full byte span with `ggml_nbytes` and respect `nb[]`, or copy each row/rank with correct offsets. Use `ggml_backend_tensor_get`, not direct dereference of device data. The existing logger demonstrates backend-safe readback (`common/debug.cpp:176-185`). `ffn_moe_argsort-L` is the full-sort source; if collecting it as a fallback, retain only the first K IDs per row and record that larger transfer. The named logical output is preferable to filtering on `GGML_OP_TOP_K`, because that is a different builder here (`ggml/src/ggml.c:5481-5492`).

Optional companion tensors:

- `ffn_moe_logits-L`: F32 `[E,T]` for router-score diagnostics (`src/llama-graph.cpp:2185-2190`). Expensive compared with K IDs.
- `ffn_moe_weights_norm-L`: selected normalized weights; this callback is on a 2D `[K,T]` tensor before it is reshaped (`src/llama-graph.cpp:2295-2308`). If `expert_weights_scale` differs from 0 and 1, final weights have further scaling and `ffn_moe_weights_scaled-L` is `[1,K,T]` (`2310-2312`). Observe the appropriate final tensor or record the metadata scale explicitly.
- `ffn_moe_slots-L`: cache gather tensor `[1,K*T]`, later reshaped `[K,T]` (`src/llama-graph.cpp:2553-2571`). These values are **cache slots, not logical expert IDs**. They exist only on eligible cache graphs. Pair with logical IDs; a slot number alone cannot identify an expert after eviction.

All selected projection matmuls share the same logical selection or mapped slots. Logging IDs at every up/gate/down matmul would overcount routing by roughly three records per layer and, with cache enabled, may log slot IDs instead (`src/llama-graph.cpp:1675-1679,2353-2368,2468`).

## Out-of-tree harness design

A complete program can use only the public llama/GGML headers and libllama/GGML libraries. It need not compile `common` or edit sources. Existing initialization and decode examples establish the pattern; `examples/eval-callback/eval-callback.cpp:54-63` shows callback installation before context initialization, while `common/common.cpp:1736-1737` transfers common parameters into public context parameters.

The callback below is a design sketch, not a built or executed deliverable. It illustrates the minimum required semantics. `Trace` holds caller-owned metadata and an output sink; `emit` must copy values immediately and must not retain tensor pointers.

```cpp
// Include llama.h, ggml.h, ggml-backend.h and standard C++ headers.
bool route_cb(ggml_tensor * t, bool ask, void * user) {
    auto & trace = *static_cast<Trace *>(user);
    const char * name = ggml_get_name(t);
    const bool wanted = exact_layer_name(name, "ffn_moe_topk-");
    if (ask) {
        return wanted; // Selection happens BEFORE scheduler synchronization.
    }
    if (!wanted) {
        return true;
    }
    require(t->type == GGML_TYPE_I32 && t->ne[2] == 1 && t->ne[3] == 1);
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    for (int64_t row = 0; row < t->ne[1]; ++row) {
        for (int64_t rank = 0; rank < t->ne[0]; ++rank) {
            int32_t expert;
            std::memcpy(&expert,
                bytes.data() + row*t->nb[1] + rank*t->nb[0], sizeof expert);
            trace.emit(name, row, rank, expert); // Join caller's token metadata.
        }
    }
    return true; // Continue; false is not the way to decline an observation.
}

// Before llama_init_from_model:
auto cp = llama_context_default_params();
cp.cb_eval = route_cb;
cp.cb_eval_user_data = &trace; // Must outlive context and pending work.
```

Implementation details that make this reliable:

1. Match `ffn_moe_topk-` plus an entire numeric layer suffix; reject copied-name prefixes/suffixes and duplicate records rather than silently double-counting. Require observed K to match model metadata and IDs in `[0,E)`. Require one record per observed `(decode-call, microbatch, layer, input-row)`.
2. Initially use **one sequence and one token per decode call**, explicit position/sequence ID, and no speculative decoding. This avoids hidden microbatch/token reorder mapping. Feed a fixed token stream (teacher forcing) so numerical or sampling differences cannot change the routing workload.
3. Build the expected row table before each decode. The row is the input token whose forward pass produced the router values, not the token sampled from resulting logits. Include token ID, absolute position and sequence ID. Use all output rows when batching a corpus so final-layer FFN is not reduced to requested-logit rows (`src/models/granite.cpp:175-180`).
4. Only after one-token tracing is verified, support microbatches. A decode call can split/reorder into internal ubatches; do not assume a callback provides the public batch index. Either constrain each decode call to one known ubatch with validated ordering, or patch the graph/context path to attach its ubatch mapping. Different layers can have different row counts in the last layer. `cb_eval` receives only tensor, ask and user pointer (`ggml/include/ggml-backend.h:318`), not positions or ubatch metadata.
5. Explicitly tokenize with the model's tokenizer/BOS/chat-template choices. Store a manifest with GGUF hash, tokenizer/model metadata, prompt/token hash, revision, backend, quantization, n_ctx, batch/ubatch settings, placement overrides, cache budget, output-row policy and trace schema version.
6. Avoid initialization/warmup routes in the dataset. The public API harness can omit warmup entirely or tag an `init` phase; the eval-callback example sets `params.warmup=false` (`examples/eval-callback/eval-callback.cpp:58`). A common-init-based harness must account for its warmup behavior.
7. Copy raw data immediately, keep a bounded in-memory record buffer, and flush outside decode where possible. Report overflow/loss explicitly. Do not hold graph tensor pointers across callbacks, evaluation or context reset.
8. Return `true` after successful data collection. The scheduler's false-return handling is for stopping computation (`ggml/src/ggml-backend.cpp:1930-1931`), not selective filtering. Keep exceptions/errors within the C callback boundary and surface a harness error after evaluation.

Suggested row schema:

```text
run_id, phase, request_id, decode_call, ubatch_id,
input_token_id, position, seq_id, layer, K,
expert_ids[K], optional_weights[K], tensor_ne[4], tensor_nb[4]
```

For one-token calls, `ubatch_id=0` and input-row mapping is explicit. For batched routes do not fill an invented `ubatch_id` or position when it is unknown; treat missing mapping as a failed trace, not a usable complete dataset.

### Keeping the image immutable

First inspect the actual executable set and shared-library exports/read-only headers, without installing packages or fetching sources. Use exactly matching headers: this revision's context-parameter struct includes new fields, so matching symbol names alone is not enough to guarantee ABI compatibility. A host-built binary also needs a compatible loader, libc/C++ runtime, backend plugins and CUDA libraries.

If compatible dependencies are present, compile externally and place the executable/output directory on a bind mount, invoking it with a read-only model mount. Use the existing library/backend search paths. That does **not** modify the image but runs a separate process and needs its own model/context allocations. An existing server's occupied VRAM may prevent this; run serially or provision separate capacity. If only a static server binary is shipped, there is no general supported way to inject `cb_eval` through HTTP; use a separate source-built harness/runtime outside the image, or a patched image for integrated deployment. No image inspection or command execution is claimed here.

## Perturbation and what a route trace can prove

The scheduler asks nodes until one is requested, computes that graph view, **synchronizes the backend**, then invokes the observation (`ggml/src/ggml-backend.cpp:1905-1934`). A selective callback therefore adds observation boundaries, typically one per routed layer per microbatch; even a callback that requests no values still enters the callback branch and synchronizes its submitted graph views. Keeping a non-null callback is not a zero-cost baseline.

Per observation the useful ID payload is `4*K*T` bytes, but a strided view's full-span readback can approach `4*E*T` bytes. This is separate from the synchronization cost, which can dominate. Collecting logits adds `4*E*T` bytes and more graph boundaries. The built-in generic logger requests **every node**, copies every device tensor before filtering, and only then decides what to print (`common/debug.cpp:150-186`); it is much more invasive than a selective collector.

CUDA fuses router operations (`ggml/src/ggml-cuda/ggml-cuda.cu:3819-3891`). Splitting a submitted graph at top-k may prevent fusion across weights normalization. Validate observed names/counts and compare route IDs and logits with callback-off execution at the same fixed input; do not assume observation is numerically neutral near top-k ties. Tie order can depend on backend/reduction arithmetic; record rank and compare set membership separately where appropriate.

A complete logical route trace supports frequency, temporal locality, distinct experts per microbatch, all-K-hit potential, profile transfer, and offline policy simulation. It **cannot** establish actual slot hits, effective DRAM/PCIe bytes, CPU-miss/GPU-hit overlap, slot byte correctness, or real end-to-end speed. An offline LRU replay must model llama.cpp's grouping, deduplication, batch eligibility and bypass rules to predict its counters (`src/llama-moe-cache.cpp:74-118,167-175,462-512,574-577`). Strata's per-occurrence hook counts require conversion to the same denominator.

## Patched-build choices

### A. Smallest application patch

Use the existing eval-callback example as a template but replace `common_debug_cb_eval` with the selective structured collector. Add explicit input token/position and output schema handling in the application; no graph math change is necessary. If modifying the generic printer instead, both early `ask` filtering and removal of truncation are needed. This proves routing cheaply in engineering terms, while retaining cb_eval synchronization overhead.

### B. Cache/copy trace with existing synchronization

At `src/llama-moe-cache.cpp:548-580`, emit the **original ID buffer before dedup**, layer, slot map before/after, victim key and fill list, byte ranges, and a monotonic evaluation/ubatch ID. This path already synchronizes for ID retrieval. Record requested fills separately from completion; use ordered events or verification after synchronization for completion claims. Add corresponding transient-copy records at `src/llama-context.cpp:2789-2837`.

This can explain real residency and transfer submissions but is not a universal router trace: ineligible cache layers, bypass batches, CPU-resident execution and fully GPU-resident weights may never enter either hook. Correlate with logical cb_eval traces during validation, then disable cb_eval for timing. Do not count gate/up/down uploads as three separate expert routing events.

### C. Low-overhead device trace

Insert a device-to-device copy/record immediately after selected IDs become valid, or extend the fused topk-MoE output path to write a bounded preallocated trace buffer. Preserve both IDs and weights as needed. Defer host readback to the end of a token/window/request; add separate CPU records for CPU router paths. For a CUDA graph capture, the record index and request metadata must be runtime inputs or graph-updated parameters, not values frozen at capture.

Test buffer overflow, concurrent sequences, capture replay, all fused/unfused implementations and backend fallback. Do not use device printf in throughput measurements. If using backend `MUL_MAT_ID` hooks, deduplicate repeated projection reads and distinguish logical IDs from remapped slot IDs. A single graph-level logical-ID trace point is easier to validate.

## Validation and experiment recipe

1. **Trace correctness:** use a fixed short token corpus, K>6 and multiple layers. Confirm K values per input token/layer, valid/distinct expert IDs, expected layer count, stride correctness and weights summing to the configured scale within tolerance. Test batched T>6 separately; the built-in printer must demonstrably not be used as the full trace oracle.
2. **Row alignment:** compare single-token teacher forcing with batched teacher forcing, requesting every output row; track expected final-layer counts. If near-tie routes differ, inspect logits and precision rather than assuming token mapping is wrong. Check prompt/BOS, first decode, second decode and multi-request boundaries.
3. **Numerical control:** compare callback-on/off logits, next-token distribution/KL and, on a representative corpus, perplexity. `tools/perplexity/perplexity.cpp:175-294` supplies KL machinery, but aggregate perplexity alone cannot locate a stale expert or bad slot. Add per-layer residual/expert-output comparison for dispatch changes.
4. **Residency controls:** run cache disabled and enabled at several budgets. Use microbatches 1/8/32/33 plus a capacity-guard case. Compare logged fills/hits to an implementation-faithful simulator; separately read back sampled slots and check all projection slices against GGUF bytes.
5. **State controls:** recreate context for cold expert-cache trials. Use `llama_memory_clear(llama_get_memory(ctx), true)` for a KV reset when needed, but label expert residency warm. State save/load serializes memory, not MoE cache (`src/llama-context.cpp:3608-3667,4385-4390`). New process does not guarantee cold OS page cache. Record cache state and initialization/warmup in every run manifest.
6. **Timing:** remove cb_eval entirely for performance baseline. `llama-bench` covers prompt/generation and static `n_cpu_moe` sweeps (`tools/llama-bench/llama-bench.cpp:474,2168-2227,2415-2445`); its separate parser has no cache-budget option in this revision. Use the harness or a small benchmark extension for cache sweeps. Report repeated uninstrumented prefill/decode wall times, CPU worker times, GPU events, transfer submissions/completions and peak memory; collect a timeline for overlap claims.

## Open questions

- Does the immutable image include the debug example or an ABI-compatible shared libllama/backend set? Without this, zero-image-change complete tracing is not established.
- Which exact Granite GGUF, backend/device and quantization are the target? Plain MoE, Shared, Hybrid and SWA must not share unverified layout assumptions.
- Is one-token tracing sufficient for the first policy study, or must traces preserve production batching/speculation? The latter needs richer input/ubatch mapping.
- What trace perturbation is acceptable, and what tolerance defines CPU/GPU numerical parity? Those require actual measured controls.
