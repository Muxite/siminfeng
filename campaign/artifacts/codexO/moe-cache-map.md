# Pinned engine cache map

Inspected source: `ref/llama.cpp` commit **31288c9c66e59377f7788d920678a565bb89dd37**. References below use original source line numbers. No reference-tree edits, network, Docker or GPU execution were performed.

## Compute and misses

`common/arg.cpp:2781` parses `--moe-cache-mib`; `llama-context.cpp:551` constructs one context-owned `llama_moe_cache` only for a positive budget. `llama-context.cpp:711` registers `sched_copy_experts` as the scheduler copy callback. Setting the proposed environment variable alone does not enable a cache.

`llama-graph.cpp:2532` builds the slot-map lookup when eligible. `build_lora_mm_id` at line 1669 replaces the original expert tensor with a bank view and the router IDs with slot IDs. **Original small-batch misses are uploaded into persistent slots and computed on the GPU; they are not computed on the CPU.** No hit/miss matmul split or CPU-miss worker exists here. The D019 harness has different execution semantics.

When no slot map is returned, the graph uses its original `MUL_MAT_ID(w, activation, expert_ids)`. Scheduler placement decides its backend. If the op runs on the GPU and needs a copy of a host weight, `llama-context.cpp:2762–2857` identifies used expert IDs from the first matmul in the split, downloads and synchronizes IDs, deduplicates them, groups adjacent expert ranges, calls `copy_experts` for available cached ranges (D2D), and uploads remaining ranges into the scheduler's temporary copy of the original weight. A small padding upload, up to 512 bytes per range, protects MMQ reads. This is GPU computation with selected-weight transfer, not automatic CPU computation. If placement stays on the CPU or the first-node test fails, normal scheduler behavior applies; cache acceleration is not promised.

## Identity, layout and allocation

`llama-moe-cache.cpp:123–149`: participating projections are gate_up, gate, up, down in that order when present. Only layers with **all** those tensors in host WEIGHTS buffers, assigned to an available GPU/IGPU, qualify. Tensor-parallel split mode is refused. Layers share a device/layout group only when projection count, ggml types, full shapes and `nb[2]` match. CPU_REPACK has a null `is_host` predicate (`ggml-cpu/repack.cpp:5248`), so it is excluded rather than copied as ordinary quantized bytes. This assumes the original engine's packed expert layout; it is not a general strided/repacked tensor conversion.

Each group has one bank per projection: `[ne0, ne1, n_slots + 1]`. The last slot is an MMQ guard; views expose only `n_slots`. Each expert occupies `nb[2]` bytes in each bank; a slot stores all projections for one logical `(layer, expert)`. Groups can share slots across layers. Layer slot maps are host I32 `[1,n_expert]`, initialized to -1. LRU keys are `layer*n_expert + expert`; pointer-keyed `bindings` map original `ggml_tensor *` weights to `(layer, bank index, bank view)`, and `layer_of` maps slot-map tensor pointers to layers. Tensor names are diagnostic, not identity keys.

The budget is split over devices by explicit tensor split or free memory, then over groups proportional to host bytes. Estimated bank allocation per group is `sum_projection align(nb[2]*(slots+1), device_alignment)`, capped at all group experts. At least top-k slots are required. Actual allocator bytes are logged and can differ from the estimator on another backend. Slot maps, scheduler staging copies, context/KV buffers and host copies are additional memory.

## Per-step admission and synchronization

Original `get_slot_map` permits nonzero batches up to 32 tokens, provided the worst-case distinct selection fits the pool. Large batches use the fallback above without changing LRU residency.

In original `copy()` (line 515), the callback locates the split's GET_ROWS reading the copied slot map, requires routing to be in an earlier split, downloads contiguous selected IDs and synchronizes the backend. `moe_cache_lru::plan` deduplicates IDs, touches hits first, evicts LRU slots for misses and updates host maps. Consecutive source experts assigned consecutive slots are merged into asynchronous bank uploads. The callback then enqueues the updated map copy to the same backend, so bank writes precede matmul reads. The next callback's backend synchronization prevents reuse before prior consumers finish. Stats count unique selected experts per layer/callback, not all router requests across a multi-token batch.

`ggml-backend.cpp:1823–1865` waits on the scheduler's previous-use event (or synchronizes the backend) before invoking a host-weight copy callback. `copy_experts` creates ephemeral source/destination views and enqueues a backend copy; context-owned banks remain alive. Scheduler split ordering and these waits retain responsibility for temporary-buffer lifetimes. Pipeline parallelism is already disabled for a positive cache budget (`llama-context.cpp:516–523`). Cache state belongs to the context; concurrent evaluation of the same context is not supported by this patch.

## CUDA graphs

Admission and ID readback happen in scheduler callbacks outside captured compute graphs; LRU's bank pointers/shapes can remain stable even while contents change. CUDA graph compatibility is separately checked per split (`ggml-cuda.cu:2699`, `ggml_cuda_mul_mat_id_needs_sync`); a fallback matmul that synchronizes disables capture. Cache presence therefore never guarantees graph capture or replay of a whole token. Host split boundaries and transfers still cost time.

The static patch returns no slot map, so there is no slot-map callback, no `plan()` call, and no persistent-bank mutation after init. It deliberately uses the existing fallback for every batch. Cached projections travel D2D into temporary tensors, misses and range padding still travel H2D, and routing ID readback/synchronization remain. Stable residency is useful but does not establish lower total transfer/latency or improve CUDA graph eligibility by itself. The untouched fallback's scheduling/graph policy must be measured.
