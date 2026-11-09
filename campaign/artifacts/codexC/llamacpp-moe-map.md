# llama.cpp MoE implementation map

Assignment T-C; inspected offline on 2026-11-09. **Static source analysis, not a performance measurement.** Both input worktrees were clean when checked. Verified revisions:

- llama.cpp: `31288c9c66e59377f7788d920678a565bb89dd37`, root `/home/muk/projects/siminfeng/campaign/ref/llama.cpp`.
- Strata: `ffa80d73133ea5c548aefe57889a16e8085a938f`, root `/home/muk/projects/Strata`.

Citations below use `path:line` relative to the llama.cpp root unless prefixed `Strata:`. Line ranges identify the complete supporting implementation. No network, model execution, source modification, or image modification was used. The companion documents are [comparison](compare-vs-strata.md) and [instrumentation](instrumentation-options.md).

## Principal finding

**This revision has an experimental, optional expert-slot GPU cache.** It is incorrect to describe it as only whole-layer CPU/GPU placement. `--moe-cache-mib` adds persistent GPU banks, `(layer, expert) -> slot` maps, LRU replacement, selected-expert uploads, and slot-indexed expert matmuls. The default remains disabled (`include/llama.h:399`; `src/llama-context.cpp:4004`; `common/arg.cpp:2780-2789`).

The important gap relative to Strata's hit-hook model is the executor: llama.cpp fills cache misses for GPU execution; Strata can compute GPU hits concurrently with CPU misses from the same routed set. Cache existence alone does not establish that overlap. See `src/llama-moe-cache.cpp:515-580` versus `Strata:src/core/session.cpp:726-748`.

## Granite MoE architecture and graph

| Stage | File:line | Behavior |
| --- | --- | --- |
| Architecture-specific hyperparameters | `src/models/granite-moe.cpp:3-19` | Loads RMS epsilon, logit scale, optional residual/embedding/attention scales; optional shared-expert FFN length. |
| Expert counts | `src/llama-model.cpp:1398-1400` | Loads expert count and scalar/per-layer expert-used counts from GGUF metadata. Do not hard-code a Granite model's E or K from a different checkpoint. |
| Expert tensors | `src/models/granite-moe.cpp:56-76` | Dense fallback if E=0; otherwise router plus expert gate/up/down tensors. Shared experts use separate dense matrices. |
| Graph implementation reuse | `src/models/models.h:1771-1778`; `src/models/granite-moe.cpp:81-83` | Granite MoE aliases `llama_model_granite::graph`; searching only the MoE file misses the forward graph. |
| Attention and final-layer pruning | `src/models/granite.cpp:136-180,203-238` | Embedding, positions, KV attention, then FFN. Last layer can select only requested output rows before its FFN. Thus a prompt batch can have fewer router rows in the final layer than earlier layers. |
| MoE invocation | `src/models/granite.cpp:241-287` | Attention residual, FFN RMS norm, router and expert projections; `LLM_FFN_SILU`, `norm_w=true`, `hparams.expert_weights_scale`, explicit SOFTMAX gating. |
| Shared experts | `src/models/granite.cpp:290-304` | Dense shared FFN evaluates the same normalized input and is added to routed output. It is not another top-k choice. |
| Residual and output | `src/models/granite.cpp:307-317,187-200` | Residual scaling/add, control vector, final norm and LM head; logits divided by Granite logit scale. |

For Granite's normal ungrouped router, with input x, E experts and K selected experts:

1. `logits = W_router * x`, shape `[E,T]` (`src/llama-graph.cpp:2183-2193`).
2. Softmax over experts, shape `[E,T]` (`src/llama-graph.cpp:2200-2221`).
3. Descending selection gives IDs `[K,T]` (`src/llama-graph.cpp:2267-2273`). The generic helper also supports biased/grouped selection for other architectures (`2223-2265`); those branches must not be assumed active for a particular Granite GGUF without inspecting its metadata.
4. Gather selected probabilities, normalize by their sum clamped below at `6.103515625e-5`, optionally scale (`src/llama-graph.cpp:2281-2313`). This is softmax over all experts followed by selected-weight normalization, not the SOFTMAX_WEIGHT branch used by some other models (`2288-2293`).
5. Up and gate indexed matmuls, split SwiGLU, down indexed matmul; multiply outputs by router weights and sum K expert rows (`src/llama-graph.cpp:2353-2380,2409-2415,2468-2527`).

### `ggml_top_k` versus the operation actually built

The graph here calls **`ggml_argsort_top_k`**, which builds descending `ARGSORT` and a K-wide **view** into its result (`ggml/src/ggml.c:5463-5476`). A separate `ggml_top_k` constructs an I32 `GGML_OP_TOP_K` result (`ggml/src/ggml.c:5481-5492`). Do not assume Granite's graph contains the latter op merely because the logical operation is top-k.

CUDA can fuse softmax, argsort/view, gather and normalization into its specialized topk-MoE kernel, retaining IDs and weights as outputs (`ggml/src/ggml-cuda/ggml-cuda.cu:3819-3891`). Observing an intermediate tensor can break this fusion. Names assigned by graph callbacks include `ffn_moe_logits-L`, `ffn_moe_argsort-L`, `ffn_moe_topk-L`, `ffn_moe_weights_norm-L` (`src/llama-graph.cpp:2190,2270-2273,2306`; suffixing at `src/llama-context.cpp:2846-2852`).

## GGUF tensor layout and conversion

Let H = embedding width, F = per-expert intermediate width, E = expert count. GGML dimensions put the contiguous/input dimension first:

| GGUF tensor name (with `.weight`) | GGML dimensions | Meaning |
| --- | --- | --- |
| `blk.L.ffn_gate_inp.weight` | `[H,E]` | Router; not expert gate projection. |
| `blk.L.ffn_gate_exps.weight` | `[H,F,E]` | One gate matrix per expert; loader marks optional. |
| `blk.L.ffn_up_exps.weight` | `[H,F,E]` | One up matrix per expert. |
| `blk.L.ffn_down_exps.weight` | `[F,H,E]` | One down matrix per expert. |

Source: `src/models/granite-moe.cpp:66-75`; names in `gguf-py/gguf/constants.py:1596-1598` and router mapping in `gguf-py/gguf/tensor_mapping.py:518-523`. These are **three stacked tensors per layer**, not a GGUF tensor name per individual expert. Expert e is a matrix slice at byte offset `e * nb[2]`. Quantized row bytes depend on GGML block type, so derive bytes from tensor strides/type rather than `H*F*sizeof(float)` (`src/llama-context.cpp:2786-2799,2832-2837`; cache layout check at `src/llama-moe-cache.cpp:133-148`). GGUF writer reverses the Python shape when serializing dimensions (`gguf-py/gguf/gguf_writer.py:261-269`). The common PyTorch shape `[E,F,H]` therefore becomes GGML `[H,F,E]`.

`convert_hf_to_gguf.py` is now an entry point into modular conversion code (`convert_hf_to_gguf.py:18-27,250,285`):

- Architecture registry maps `GraniteMoeForCausalLM` and `GraniteMoeSharedForCausalLM` to the Granite module (`conversion/__init__.py:120-128`).
- Their converter is `GraniteMoeModel`, GGUF architecture `GRANITE_MOE` (`conversion/granite.py:179-192`). Granite scale metadata is inherited from `GraniteModel` (`23-49`).
- Merged HF `block_sparse_moe.input_linear.weight`, shape with `2*F` on axis -2, is split into **gate then up**, preserving expert order (`conversion/granite.py:194-207`). Shared input projection is split similarly (`211-225`).
- `block_sparse_moe.router.layer` maps to gate input; `block_sparse_moe.output_linear` maps to down experts (`gguf-py/gguf/tensor_mapping.py:522,746-751`). Generic metadata conversion writes `num_local_experts`/`num_experts` and `num_experts_per_tok` to expert count/used count (`conversion/base.py:1690-1695`).
- Granite MoE SWA and Hybrid are separate architectures/converters; e.g. SWA can retain fused gate/up (`conversion/granite.py:124-176`). Do not generalize that layout to the plain Granite MoE path above.

## Static placement and selective offload

| Control | Implementation and scope |
| --- | --- |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | `common/arg.cpp:2801-2819`: integer, `auto`, or `all`. Layer device assignment uses the trailing offloaded layers plus output layer; input stays CPU (`src/llama-model.cpp:1667-1692`). This is allocation policy, not per-token routing. |
| `--cpu-moe` (`-cmoe`) | Adds CPU buffer override for expert tensors (`common/arg.cpp:2763-2769`). Regex covers `up/down/gate/gate_up` expert tensors, not router or shared dense FFNs (`common/common.h:1243-1252`). |
| `--n-cpu-moe N` (`-ncmoe`) | CPU overrides for the **first N layers** (`common/arg.cpp:2770-2779`; `common/common.h:1255-1261`). Whole stacked expert tensors are placed, not selected hot experts within each tensor. |
| `-ot`, `--override-tensor` | Regex-to-buffer-type overrides (`common/arg.cpp:2757-2762`). Loader takes **first matching override**, selects a compatible CPU buffer type when requested, and stops (`src/llama-model-loader.cpp:1233-1266`). It cannot match an individual expert index inside `ffn_up_exps.weight`. |
| `--moe-cache-mib N` | Additional context-owned GPU cache for eligible host expert tensors (`common/arg.cpp:2780-2789`; `src/llama-context.cpp:551-553`). Can be combined conceptually with offloaded layers whose experts were overridden to CPU. |

CPU *weight storage* does not guarantee CPU *execution*. Scheduler prefers weight backend but can offload a host-weight op when a higher-priority backend supports and requests it (`ggml/src/ggml-backend.cpp:1099-1127`). CUDA's decision uses op batch size, and for `MUL_MAT_ID` that is `ne[2]` (`ggml/src/ggml-cuda/ggml-cuda.cu:5996-6014`). CPU implementation exists at `ggml/src/ggml-cpu/ggml-cpu.c:1554-1727`; tiled expert dispatch at `ggml/src/ggml-cpu/tiled/tiled.cpp:1227`. Check actual backend assignment at the intended microbatch size. `--no-op-offload` disables the scheduler's opportunistic host-weight op offload (`common/arg.cpp:2960-2968`); disable the explicit MoE cache as well for a CPU-expert control. It does not change already GPU-resident weights into CPU weights.

Even without the persistent cache, llama.cpp has a **selected-expert weight-copy callback**: when the split starts with the relevant `MUL_MAT_ID`, it reads IDs, marks unique experts, and uploads consecutive used expert slices plus necessary padding instead of copying the entire stacked weight (`src/llama-context.cpp:2762-2843`). If this callback's conditions do not match, normal scheduler copy remains possible. Selected uploads are not evidence of persistent residency.

## Optional MoE cache: slots and residency already exist

| Mechanism | File:line and behavior |
| --- | --- |
| Eligibility | `src/llama-moe-cache.cpp:221-267`: GPU required, tensor parallel split rejected; MoE required. All expert projection tensors for a layer must be host weights, and its assigned layer device must be a cache GPU. CPU-assigned layers are not automatically cached. |
| Layout groups | `src/llama-moe-cache.cpp:133-148,167-188,252-263`: same device, shape, type and expert stride share a bank group/LRU. |
| Budget | `src/llama-moe-cache.cpp:270-331`: split by tensor split or free GPU memory, then expert bytes within a GPU; enough slots for at least K; extra guard expert for MMQ over-read. |
| Storage | `src/llama-moe-cache.cpp:357-384,418-436`: one bank per projection, views per layer; host I32 `[1,E]` slot map starts at -1; weights usage makes copies go through scheduler callback. |
| LRU | `src/llama-moe-cache.cpp:15-118`: key is `layer*E+expert`; deduplicate routed IDs, protect/touch hits first, sort misses, evict oldest and clear old owner's mapping. Hits/misses count unique experts per layer/microbatch, not every token-expert occurrence. |
| Eligibility per microbatch | `src/llama-moe-cache.cpp:196,462-472`: bypass zero tokens, >32 tokens, or worst-case `min(T*K,E)` greater than group slots. This deliberately avoids prompt-driven cache churn. |
| Graph seam | `src/llama-graph.cpp:2532-2571`: get slot map; ensure each expert tensor has bank binding; gather slots for selected IDs on target GPU; name `ffn_moe_slots-L`. |
| Populate | `src/llama-moe-cache.cpp:515-580`: copy hook locates lookup, verifies selection was in an earlier split, reads current IDs to host and synchronizes GPU, plans LRU, uploads missing projection slices, then uploads slot map. |
| Execution | `src/llama-graph.cpp:1675-1679,2318-2319`: cached expert matmuls use bank views with **slot IDs**; original logical IDs still gather router weights/scales. All selected experts on this path run on GPU, including newly filled misses. |
| Larger microbatches | `src/llama-moe-cache.cpp:480-512`; `src/llama-context.cpp:2817-2837`: reuse cached slices by device copy to transient expert buffer, upload other selected slices; do not update LRU through this path. |
| Statistics/lifetime | `src/llama-moe-cache.cpp:458-460,574-598`: destructor prints small/large microbatch hit/miss/upload totals and large-batch cache-copy bytes. Context owns cache (`src/llama-context.h:310`). |
| Limits | `src/llama-context.cpp:515-523`: scheduler pipeline parallelism disabled with cache. `src/llama-moe-cache.cpp:234-235`: tensor parallelism unsupported. |

These mechanisms support reusing llama.cpp as a baseline for a slot/residency experiment. They do not supply Strata's mixed CPU/GPU expert dispatch, profile-selected fixed residency, adaptive policy controls, or KV/expert memory lending in the inspected cache implementation.

## Expert matmul and CUDA execution

`ggml_mul_mat_id(as,b,ids)` expects stacked 3D expert matrices and I32 `[K,T]` indices; output is F32 `[as.ne[1],K,T,1]`. IDs are `src[2]` (`ggml/src/ggml.c:3411-3436`). It selects matrix slices, not a full dense computation over all E experts.

- CUDA dispatch: `ggml/src/ggml-cuda/ggml-cuda.cu:2042-2079`. Small quantized batches use indexed MMVQ; AMD float path may use MMVF; larger supported shapes use MMQ or MMF. Exact choice depends on tensor type, shape and device.
- MMVQ accepts IDs and derives expert/token strides (`ggml/src/ggml-cuda/mmvq.cu:1455-1485,1572-1586`).
- MMQ builds expert-grouped compact input/output row maps and expert bounds (`ggml/src/ggml-cuda/mmid.cu:27-38,104-127,150-178`). It quantizes/scatters activations and dispatches quantized matmul (`ggml/src/ggml-cuda/mmq.cu:280-352`). Expert grouping here is GPU work organization, not a host residency manager.
- Fallback reads IDs to host, synchronizes, sorts tokens by expert, executes active expert slices, and maps results back (`ggml/src/ggml-cuda/ggml-cuda.cu:2083-2156` and following slice loop). CUDA graph eligibility rejects this synchronization-requiring path (`2011-2039,2711-2715`). Therefore neither 'all expert matmuls synchronize' nor 'all are captured' is valid.

## Scheduler, copies, and synchronization

The GGML scheduler assigns operations based on preallocated buffers, op support, weight location, offload preference, and propagation (`ggml/src/ggml-backend.cpp:1070-1127,1319-1441`). It partitions the graph when backends change or a new incompatible weight input needs a split, then creates copied inputs (`1445-1549`).

At compute time, user inputs are copied immediately; reuse is protected by backend events or synchronization; callback-controlled host weights are copied after non-weight inputs (`ggml/src/ggml-backend.cpp:1823-1897`). Copy callback registration is at `src/llama-context.cpp:709-711`. Without eval callback the split is submitted asynchronously; with it, requested node boundaries create graph views and synchronization (`ggml/src/ggml-backend.cpp:1899-1935`). Split completion events protect reuse (`1938-1941`).

This is tensor/op scheduling. It does not partition one layer's selected experts into concurrent CPU misses and GPU hits. Strata's explicit Launch -> CPU pool -> miss H2D -> Combine seam is a separate component (`Strata:include/strata/core/hit_hook.hpp:13-39`; `Strata:src/core/session.cpp:726-748`).

## KV cache, session state, and cold/warm experiments

- Granite uses attention/KV graph input (`src/models/granite.cpp:143,234-236`). KV allocation/state is independent of expert bank storage.
- `llama_memory_clear(mem,data)` clears the memory module; sequence removal has a separate API (`src/llama-context.cpp:4385-4402`; `include/llama.h:755` and following sequence APIs). KV clear resets cells/heads and zeroes backing buffers only if `data=true` (`src/llama-kv-cache.cpp:391-401`). It does not unload model weights or clear the MoE LRU.
- Context state write/read serializes architecture and memory module; per-sequence variants serialize memory for that sequence (`src/llama-context.cpp:3608-3667`). Session-file APIs are declared at `include/llama.h:858-924`. There is no expert-cache snapshot in these state methods.
- To make the **expert cache** cold, recreate the context/cache. A fresh process is stronger for allocator state but still does not guarantee cold OS page cache. A KV reset alone leaves expert cache and its cumulative counters warm. A restored session skips computation for restored tokens, so those router events are absent.
- Do not assume generated token index equals router input token index: routing processes the token fed to decode, which produces logits for the next token.

## Memory fit and built-in tools

`llama-fit-params` calls the current `common_fit_params` implementation and prints context, GPU-layer, split and override arguments (`tools/fit-params/fit-params.cpp:30-75`). The README uses the older internal `llama_params_fit` name (`tools/fit-params/README.md:3-6`); source is authoritative here.

Fit creates a metadata/no-weight-allocation model/context to estimate model/context/compute memory (`common/fit.cpp:30-76`), tries context reduction where allowed (`373-457`), then layer/partial-MoE placement with tensor overrides (`565-627`). It honors constraints such as explicitly set GPU layer counts (`463-478`). Cache no-allocation accounting exists (`src/llama-moe-cache.cpp:387-401,628-638`; included by `src/llama-context.cpp:3710-3713`). This estimates **capacity**, not trace hit rate, CPU/GPU balance, transfer overlap, or optimal cache policy.

| Tool/hook | What it supplies | What it does not supply |
| --- | --- | --- |
| `llama-bench` | Prompt/generation microbenchmarks (`tools/llama-bench/llama-bench.cpp:2168-2227`), repetitions, placement sweeps including `n_cpu_moe` (`354,434,474,1332`), warmup and KV clears (`2387,2415-2445`). | Representative workload routing or residency correctness. Its parser is separate from common CLI; this revision has no `moe_cache_size`/`--moe-cache-mib` handling in that file. Use a harness or extend it for cache sweeps. |
| `llama-perplexity` | Teacher-forced corpus NLL/perplexity and logits/KL comparison machinery (`tools/perplexity/perplexity.cpp:109-175,175-294,296-430`). | Direct slot correctness or overlap evidence; large prompt batches may follow a different path than decode. |
| `llama-debug`, eval example | Tensor values via cb_eval (`examples/debug/debug.cpp:231-234`; `examples/eval-callback/eval-callback.cpp:54-58`). | Complete traces by default: printer truncates each dimension at 3 head/tail values (`common/debug.cpp:79-122,186`); filter does not avoid synchronization/copies (`143-186`). |
| Public `cb_eval` | Selective tensor observation (`include/llama.h:393-394`; `ggml/include/ggml-backend.h:311-318`; attached at `src/llama-context.cpp:1537`). | A public CPU/GPU hit dispatch override, or attaching to an already-running server. |
| `GGML_SCHED_DEBUG=1/2` | Split/backend/input and node metadata logging (`ggml/src/ggml-backend.cpp:1146-1182,1551-1553,1962-1963`). | Actual routed ID values, per-expert hits, slot bytes or GPU/CPU overlap. Debug log output must be enabled in the application. |
| MoE cache destructor log | Unique-expert cache hits/misses and transfer totals (`src/llama-moe-cache.cpp:574-598`). | Per-token routing, resident byte correctness, OS page faults, or proof of mixed CPU/GPU compute. |

## Open questions for execution

1. Which immutable runtime image, binaries, shared libraries, headers and model GGUF are actually available? The inspected source revision alone cannot answer ABI/feature availability in an image.
2. Which Granite variant, quantization, device and microbatch sizes will be tested? These select the graph/kernel/offload path.
3. How large are callback perturbations and LRU thrashing on the target workload? Measure separately from uninstrumented throughput.
4. How closely can CPU/GPU expert outputs match with the chosen quantizers? Compare weights, intermediate expert outputs and logits, not just fluent text.

## Verification performed

Verified both source commit identities and clean worktrees, checked all cited file paths and start-line bounds across the three reports, and reviewed the graph/cache/dispatch control flow directly. Runtime validation, image inspection, model conversion and hardware benchmarks were not run. All writes for this assignment are the three Markdown deliverables in `campaign/artifacts/codexC/`.
