# Q8_0 activation quantization reuse

Target: llama.cpp commit `31288c9c66e59377f7788d920678a565bb89dd37`.
This is a private campaign patch, with AI assistance, for orchestrator review and GPU validation. The reference checkout is unchanged.

## Safe reuse rule

A cache entry holds one activation quantization. Only the unfused `ggml_cuda_mul_mat_vec_q` path for `GGML_OP_MUL_MAT` with Q8_0 weights and a contiguous F32 input of shape `[N, 1, 1, 1]` may populate or read it. The output must be F32 and its byte range must not overlap its input. Weight buffers with COMPUTE usage are excluded: their padding-clearing writes would need additional alias analysis. Other weight types, batching and matmul implementations retain their original quantization and dispatch.

A hit requires the same `ggml_tensor *`, data pointer, type, all four dimensions, all four byte strides, and CUDA stream. The backend context owns the entry, so contexts cannot share it. Quantization is always the existing `quantize_row_q8_1_cuda`, with null ids, Q8_0 source-weight type, the same dimensions/strides, and `GGML_PAD(ne[0], MATRIX_ROW_PADDING)`. These fixed parameters and the saved layout determine the complete quantization call; a hit cannot cross a change in its parameters.

The entry survives only the following window in graph execution:

- Metadata/view/no-op nodes and nodes the evaluator does not execute perform no writes.
- An eligible Q8_0 matmul may preserve it, subject to checking the matmul output against the saved activation byte range before dispatch. A different input replaces the entry if it takes the direct MMVQ path.
- A standalone RoPE may preserve it only when its output byte range is disjoint from the saved activation. In-place RoPE on a different tensor is safe if that range is disjoint.
- Every other executed operation clears the entry before dispatch. Every successful fusion clears it after dispatch. Cache access is disabled throughout fusion dispatch, including fusions that launch multiple kernels.

`ggml_nbytes` covers a tensor's stride span, so the overlap check rejects both equal pointers and partial overlap through views. It subtracts ordered addresses instead of forming potentially overflowing end addresses. This is deliberately stricter than tensor SSA identity: an unrelated node that reuses the allocator's activation storage cannot leave a valid entry. Even an eligible matmul cannot seed an entry when its own output overlaps its input.

Reuse is disabled for the entire evaluation when the evaluator plans concurrent events; it also requires stream number zero. There is no cross-stream handoff or cache synchronization protocol. If an initially nonempty concurrent plan is discarded, preparation may not have allocated a buffer: MMVQ then falls back to the original pool allocation and quantization.

## Storage and lifetime

The context owns an exact-size map of separately `cudaMalloc`-allocated q8_1 buffers. Preparation scans eligible nodes only when the host will evaluate or capture the graph, and allocates missing sizes before `cudaStreamBeginCapture`. Pure executable-graph replay does not scan or allocate.

Each padded vector size has one permanent address for the context lifetime. Buffers never shrink, move, return to the pool, or get freed when another graph needs a larger size. Old graph executables can therefore retain their recorded addresses across shape changes, graph-key switches and recapture. During context teardown, graph handles are destroyed and each buffer is freed on the owning device before streams are destroyed. `cudaFree` completes outstanding use before returning.

The ordinary scoped `ggml_cuda_pool_alloc<char>` is still used for every ineligible or disabled call. No cache entry refers to a scoped pool allocation. The host entry stores a layout snapshot and uses its tensor pointer only for identity comparisons; it does not dereference an earlier evaluation's tensor pointer.

Memory cost is the sum of distinct padded vector sizes seen over the context lifetime, at `padded_N / QK8_1 * sizeof(block_q8_1)` bytes per size. Typical models use few sizes. Workloads with many distinct widths retain more memory until context destruction. This avoids freeing storage still referenced by an older captured executable.

## Evaluation reset and CUDA graphs

`src` and the cache-access gate are reset at the beginning of every `ggml_backend_cuda_graph_compute`, before graph inspection or any early replay decision. No entry survives between host evaluations.

During capture, the first eligible use in every window records the original quantization kernel; later hits record matvecs reading that same persistent buffer. Quantizations for subsequent windows are also recorded, and same-stream graph dependencies order every write after earlier matvec reads. Replay executes these producer kernels again on current input data on every launch. Resetting the host entry on replay does not remove or skip those recorded producers.

The existing property comparison snapshots the whole destination tensor (including source tensor identities), source data pointers, source dimensions and source strides. Changes cause warmup/direct evaluation and later recapture; reuse decisions are recomputed from an empty entry. As in the baseline, a nonzero unchanged graph UID asserts an immutable graph description; changing data *contents* is allowed, but mutating tensor metadata/allocation or the stream plan under the same immutable UID is outside that existing contract. Updated executable kernel parameters reference the retained scratch addresses, and each rerecorded window starts with a producer. If removal of quantization nodes changes topology, the existing `cudaGraphExecUpdate` failure path destroys and reinstantiates the executable.

All accesses to the shared scratch addresses occur on context stream zero. Launching multiple captured graphs or direct evaluations sequentially on that stream orders them, including when the CPU queues another evaluation before the previous launch has completed. Separate contexts have separate scratch maps. The baseline backend's serialization of host use of one context is required; simultaneous graph_compute calls on the same context are not introduced or supported.

The environment variable is sampled once per process: unset enables reuse; exactly `GGML_CUDA_Q8_REUSE=0` disables it. Use separate processes for A/B runs so an existing executable never needs to switch cache policy.

## Coverage and open risks

- MoE `GGML_OP_MUL_MAT_ID` is not covered. Existing gate/up/GLU and other MMVQ fusions keep their original quantization behavior. Supporting their multiple outputs, temporary weights and alternate layouts would expand the write analysis.
- Successful fusions, including RoPE/set-rows and matmul/bias fusions, terminate the window. Standalone intervening additions, copies or normalization also terminate it. Consequently this patch may save fewer launches than the hypothesis estimates, or none on a heavily fused graph. A changed input also evicts the single entry; this is not a general activation cache.
- Other quantized weight types, MMQ, cuBLAS, floating-point matvecs, multi-token inputs, split-device helper `ggml_cuda_op_mul_mat_vec_q`, concurrent streams and FWHT-specific paths are unchanged.
- GPU bit identity, nvcc compilation, capture/replay behavior and performance have not been tested here. No Docker or CUDA execution was attempted. The original quantization and matvec kernels are unchanged; identical operands are the basis for the bit-identity expectation, not a substitute for validation.
- Host smoke checks compile extracted cache, eligibility, overlap, allocation and quantization-selection code using real ggml headers and mocked CUDA/quantization. They check stale-cache hazards but do not execute CUDA kernels, the full graph evaluator or the CUDA graph driver.
- The requested KLD-zero gate needs interpretation by the orchestrator if baseline self-comparison is nonzero: this revision's perplexity file format clips and quantizes log probabilities to uint16. Exact raw F32 logits must be compared separately. See README.md; no numerical gate has been declared passed.
