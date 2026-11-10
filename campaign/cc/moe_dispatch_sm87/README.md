# J-T3: BF16 MoE decode dispatch on sm_87

Pinned source verified: `ref/llama.cpp` HEAD is `31288c9c66e59377f7788d920678a565bb89dd37`. All source references below are relative to that directory and refer to the original, unpatched line numbers. Source was read only. No builds, GPU work, or network access were performed. These are experimental candidate patches, not measured improvements.

## 1. What selects the kernels

**The asymmetry comes from graph fusion, not a special slow standalone gate/up dispatcher.** The fused gate/up+GLU path asks the MMVF heuristic; the ordinary NVIDIA `MUL_MAT_ID` dispatcher instead asks the MMF heuristic. Disabling only the eligible BF16 ID fusion is sufficient to send both projections through the existing MMF implementation.

### Shape and architecture interpretation

For gate/up, weight dimensions are `[1024,512,32,1]`; for down, `[512,1024,32,1]`. The first dimension is the reduction length, the second the output-row count, and the third the number of experts. Eight selected experts are output channels, not eight tokens. ID kernels use `dst->ne[2]` / `src1->ne[2]` for token count: `ggml/src/ggml-cuda/mmvf.cu:695-704`, `ggml/src/ggml-cuda/mmf.cu:55-69`. Thus decode has `ncols_dst=1` for both.

CC is encoded as 870 for sm_87 (Ampere threshold 800): `ggml/src/ggml-cuda/common.cuh:54-66`. `ampere_mma_available` checks NVIDIA plus highest compiled architecture >=800, not just the physical CC: `ggml/src/ggml-cuda/common.cuh:369-371`. The interpretation here assumes the Orin build includes Ampere MMA device code, consistent with the observed BF16 MMF kernel.

### Fused gate/up

- `ggml_cuda_should_fuse_mul_mat_vec_f` requires weights F32/F16/BF16, activations/output F32, and calls `ggml_cuda_should_use_mmvf` using `src1->ne[2]` for IDs. Fusion requires exactly one token for IDs (`dst->ne[2]==1`), or one output column for ordinary matmul: `ggml/src/ggml-cuda/ggml-cuda.cu:1847-1872`.
- MMVF requires an even reduction length, contiguous dimension zero, and higher strides divisible by twice the weight element size: `ggml/src/ggml-cuda/mmvf.cu:791-807`.
- For BF16 NVIDIA, `src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1)`. If small and MMF is ineligible with `mul_mat_id=false`, MMVF accepts up to `MMVF_MAX_BATCH_SIZE=8`; otherwise Ampere MMA accepts only `src0_small && ne11==1`. The later Ada/non-Ampere branches allow <=4 / <=3 respectively; the Ampere branch takes precedence on sm_87: `ggml/src/ggml-cuda/mmvf.cu:858-875`; maximum definition: `ggml/src/ggml-cuda/mmvf.cuh:3`.
- Gate/up have 512 rows and one token, so pass. Down has 1024 rows and 32 experts, so is not small (and does not participate in this gate/up pattern).
- The graph matches `{MUL_MAT_ID, MUL_MAT_ID, GLU}` or the bias form with two `ADD_ID`s, verifies subgraph eligibility and memory overlap safety: `ggml/src/ggml-cuda/ggml-cuda.cu:3478-3506`, with overlap implementation at `:3255-3306`. The semantic predicate requires matching weight type/shape/strides, identical activation pointer and ID tensor pointer, correct gate/up GLU inputs, both biases/scales when present, supported non-swapped GLU (SwiGLU, GEGLU, SwiGLU-OAI, SwiGLU-clamp): `:1753-1844`. Bias IDs must match their matmul IDs. Optional scale checks exist in this predicate, while the straightforward float ID fusion here is the no-scale/bias pattern.
- Compute calls fused MMVF and skips the intermediate nodes, passing gate weights and GLU operation/limit, plus biases where present: `ggml/src/ggml-cuda/ggml-cuda.cu:4228-4275` and `:4291-4316`. Fusion computes the two reductions then `up * activation(gate)` before writing F32: `ggml/src/ggml-cuda/mmvf.cu:348-380`.

### Standalone MUL_MAT_ID (down, or unfused gate/up)

`ggml/src/ggml-cuda/ggml-cuda.cu:2043-2081` requires F32 activations/output. Small batches enter MMVQ for supported quantized types, or MMVF **only on AMD**. NVIDIA BF16 bypasses that vector branch. MMQ rejects BF16 (supported-type switch): `ggml/src/ggml-cuda/mmq.cu:355-398`. Next, an eligible MMF runs directly with IDs. Only if these paths fail does the synchronized sorting/cuBLAS-style fallback run (`ggml-cuda.cu:2083` onward). Sync/graph eligibility mirrors this logic at `:2012-2040`; the proposed unfusion still uses an existing graph-safe MMF path.

MMF conditions (`ggml/src/ggml-cuda/mmf.cu:133-190`):

- Nonquantized supported type; contiguous dimension zero; reduction length divisible by `warp_size*(4/type_size)` (BF16: 64 on Orin); higher weight strides divisible by `2*type_size` (BF16: 4 bytes).
- Output-row count divisible by `mmf_get_rows_per_block` (32 on NVIDIA, 64 on CDNA; `mmf.cu:5-10`, `mmf.cuh:9-10`). Both 512 and 1024 qualify.
- For IDs, at most 512 tokens for <=1024 output rows, or at most 128 tokens for >1024 rows. For non-ID MMF generally <=16 columns, with additional AMD exceptions. CDNA3 BF16 is excluded. BF16 requires Ampere MMA or AMD WMMA/MFMA availability.

Consequently both gate/up and down are MMF-eligible at one token. Ordinary `MUL_MAT` instead tries MMVF first (`ggml-cuda.cu:1973-1979`); this helps explain the large non-fused vector bucket but must not be confused with NVIDIA ID dispatch.

### Environment and build controls

- `GGML_CUDA_DISABLE_FUSION=1` globally suppresses compute fusion (`ggml-cuda.cu:170-172`, `:3753-3755`) and fusion-related graph optimization (`:4875-4890`). It is already a broad diagnostic, but affects many other fusions. Numeric flags use `atoi != 0` (`:141-143`). Neither MMVF nor MMF's eligibility function has an upstream env override.
- `GGML_CUDA_DISABLE_GRAPHS` disables graphs by **presence**, including value `0` (`common.cuh:1290-1293`). Keep it unset for comparisons. `GGML_CUDA_GRAPH_OPT` controls graph scheduling/optimization (`ggml-cuda.cu:4995` onward), not a direct MMF selector.
- `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` are compile-time defines in MMQ selection (`mmq.cu:356-358`, `:418-420`), not runtime BF16 ID path switches. `GGML_CUDA_MMQ_PREC` (`mmq.cu:95`) is quantized-matmul precision; `GGML_CUDA_CUBLAS_COMPUTE_TYPE` (`ggml-cuda.cu:1695-1709`) controls cuBLAS compute, not these custom BF16 kernels.
- No integrated-GPU or SM-count check selects MMVF versus MMF here. In fact `ggml_cuda_info().devices[id].integrated` is explicitly forced false (`ggml-cuda.cu:352`). An experiment gated on that field would never activate on this source. Both candidates therefore target `cc==870`, rather than a misleading integrated-device predicate or all CC 8.x GPUs.

## 2. CSV template arguments and observed work

Evidence: `experiments/jetson/J-P001-nsys-decode/stats_cuda_gpu_kern_sum.csv:2-4` and `FINDINGS.md:3-8` (relative to campaign root).

| Kernel | Arguments in declaration order | Meaning |
| --- | --- | --- |
| `mul_mat_vec_f<__nv_bfloat16, float, 1, 256, true, false>` | `T, type_acc, ncols_dst, block_size, has_fusion, is_multi_token_id` | BF16 scalar weights; F32 accumulation; one token/output column; 256 threads (8 NVIDIA warps); fusion compiled in; ordinary/single-token ID addressing |
| `mul_mat_vec_f<__nv_bfloat16, float, 1, 256, false, false>` | Same order | No fusion. These booleans do **not** say whether IDs are present. This kernel can receive a null or non-null ID pointer. |
| `mul_mat_f<__nv_bfloat162, 32, 1, 4, true>` | `T, rows_per_block, cols_per_block, nwarps, has_ids` | Packed BF16 pairs; 32 output rows per block; one token column per tile; four warps (128 threads); ID-aware routing. The final `true` is IDs, **not fusion**. |

Declarations: `ggml/src/ggml-cuda/mmvf.cu:8-14`, `ggml/src/ggml-cuda/mmf.cuh:48-55`. BF16 MMVF uses F32 accumulator dispatch (`mmvf.cu:608-629`). A false `is_multi_token_id` still handles IDs via runtime `ids` (`:32-40`); batch-one ID launches select this specialization at `:535-543`.

MMVF launches one block per output row, selected expert slot, and token (`mmvf.cu:455-457`): gate/up grid is 512 x 8 x 1 = 4096 blocks. It minimizes `ceil(K/(2*block_size))` by a strict-improvement scan from one warp through 256 threads (`:439-451`), choosing 256 at K=1024. BF16 pairs are converted to F32 and multiplied by F32 activations (`:236-271`), with warp/shared-memory reduction at `:310-340`.

MMF packs the reduction length into pairs (`mmf.cu:120-126`), uses 32-row tiles and selects warps minimizing `ceil((K/2)/(2*warp_size*nwarps))` (`mmf.cuh:665-673`). Down's K=512 therefore selects 4 warps; gate/up's K=1024 would select 8. The ID MMF grid enumerates all 32 experts, finds selected slots, and skips unused work (`mmf.cuh:98-123`, `:137-174`, `:682` onward). Batch-one uses the normal ID kernel, not compact-ID preprocessing (only >16 tokens: `mmf.cu:71`, `mmf.cuh:601-626`).

The CSV reports fused BF16 MMVF 1560 calls, mean 396.534 us, 41.1%; BF16 ID MMF 1560 calls, mean 117.680 us, 12.2%. Counts equal 24 layers x 65 evaluations. Non-fused BF16 MMVF has 6305 calls, 40.2%; the template name alone cannot prove its tensors or ID presence. The gate/up vs down attribution follows the supplied findings and the source/shape match. Gate+up touch twice down's selected expert weight bytes (16 vs 8 MiB per layer); the supplied derived bandwidths are roughly 40 vs 68 GB/s. These are profiler ratios, not an apples-to-apples proof that MMF will accelerate gate/up. MMVF already has thousands of blocks; poor bandwidth/occupancy efficiency on Orin is a hypothesis, not evidence of too few blocks or a proven cause.

## 3. Minimal A/B patches

Both diffs apply independently to the pinned source and may be included in ONE build. Neither edits weight/ID ordering or introduces a new kernel. Both preserve upstream dispatch when unset or `0`, and accept any nonzero numeric value like existing numeric flags. Set env before starting each process; static caching means changing env after graph capture is not an A/B mechanism.

### Candidate 1: `patches/01-unfuse-bf16-ids-to-mmf.patch`

`GGML_CUDA_J_MMF_IDS=1` makes the float fusion predicate decline BF16, F32-input/output, batch-one `MUL_MAT_ID` on sm_87 **only when the existing ID MMF eligibility check passes**. This also declines eligible ID matmul+bias float fusion using the same predicate, so the pre-existing matmul and bias/GLU nodes execute normally. The two gate/up projections then independently take standalone MMF, followed by existing GLU. Down is already MMF. No changes to sync prediction, MMF eligibility, node contents, memory handling, or quantized fusion are necessary. It deliberately does not attempt to send fusion arguments to MMF, whose entry point has no fusion parameter (`mmf.cuh:44`).

Expected trace: the fused BF16 ID vector bucket disappears for these projections; two MMF kernels plus GLU replace each fused call. Gate/up MMF should have `<__nv_bfloat162,32,1,8,true>` (K=1024), while down retains `<...,32,1,4,true>`. Extra intermediate memory traffic and kernels could erase any bandwidth benefit.

### Candidate 2: `patches/02-fused-bf16-ids-block-128.patch`

`GGML_CUDA_J_MMVF_128=1` caps the existing block-size search at 128 for BF16, fused gate, single-token ID launches on sm_87 with K<=1024. The `!is_multi_token_id` guard excludes multi-token launches even though those also instantiate template `ncols_dst=1`. Gate/up K=1024 chooses 128 instead of 256; each thread processes more pairs. One row per block and all fusion operations remain intact. Existing instantiations, shared-memory allocation, and reductions handle 128 already. Unfused/non-ID calls, other types/architectures, and larger reductions retain upstream settings.

Expected trace: fused gate/up name becomes `<__nv_bfloat16,float,1,128,true,false>`, with the same call count. Smaller blocks may improve residency/resource use, but may also worsen performance; only measurement can decide.

### Validation performed here

Both commands returned exit status 0 with no diagnostics:

```sh
git -C ref/llama.cpp apply --check ../../cc/moe_dispatch_sm87/patches/01-unfuse-bf16-ids-to-mmf.patch
git -C ref/llama.cpp apply --check ../../cc/moe_dispatch_sm87/patches/02-fused-bf16-ids-block-128.patch
```

A combined `apply --check` also passed. Source diff checks remained clean for the inspected CUDA files. Manual review covered the local code-review skill's scope, security, general, and backend checks: narrow dtype/architecture/token guards, existing supported kernels, unchanged strides/allocations/ID pointers, no new device writes or graph mutation. No compilation or numerical/performance validation is claimed; those are reserved for Opus.

## 4. Correctness risks and tests for Opus

1. **MMF is not merely a different summation order.** Existing MMVF expands BF16 weights to F32 and retains F32 activations (`mmvf.cu:257-268`). Existing BF16 MMF converts F32 activation pairs to BF16 before MMA (`mmf.cuh:202-226`, `convert.cuh:54-60`), with F32 accumulator tiles (`mmf.cuh:76-78`) and BF16 x BF16 -> F32 MMA (`mma.cuh:1337`). Candidate 1 preserves the intended real-number projection/GLU math but adds activation rounding as well as changing reduction order. It is not bitwise equivalent and may measurably change logits/ppl. Candidate 2 retains the original operand precision and changes only reduction partition/order. Prefer candidate 2 if strict F32-activation semantics are required.
2. **Fusion semantics:** Candidate 1 materializes gate/up separately and uses standalone GLU; check gate/up roles, SwiGLU/nonlinear/clamp operation and parameters, and any bias ordering. Candidate 2 preserves the fused operation. Compare intermediate gate, up, and post-GLU F32 tensors on identical inputs in addition to logits. If testing beyond this no-bias Granite workload, cover eligible ADD_ID bias fusions and each supported GLU variant. Do not expect unsupported/swapped forms to be newly fused.
3. **IDs, slots, expert weights:** MMVF selects weights using `ids[slot]` and broadcasts activations with slot modulo activation-channel count (`mmvf.cu:36-51`). MMF finds the selected slot for each expert (`mmf.cuh:137-174`) and uses it to gather input/store output. Candidate 1 relies on this existing path, including gate/up broadcast input vs down's per-slot input (`mmf.cu:55-69`). Test non-monotonic expert IDs including 0 and 31, permute unique selected slots and their associated routing weights together, and compare per-slot outputs and weighted reduction. Include both broadcast and per-slot activations. Duplicate IDs are not the Granite top-k contract; MMF's slot-map search is not a promise of duplicate-slot equivalence. Neither patch reorders routing weights.
4. **End-to-end gate:** On the same patched build run baseline (both switches unset/0), candidate 1 only, and candidate 2 only, in separate processes with identical model/input/context/seed/KV settings and CUDA graphs enabled. Leave `GGML_CUDA_DISABLE_FUSION` and `GGML_CUDA_DISABLE_GRAPHS` unset. Enabling both selects candidate 1, leaving candidate 2 inactive for the removed fusion; this is not a useful fourth performance experiment. Verify env-off dispatch matches baseline and the intended env-on kernel signatures appear under nsys.
5. Run the planned **ppl2** and a teacher-forced logits/top-k comparison (same token sequence, not independently sampled continuations). Record max/mean absolute and relative logit error, nonfinite counts, top-k set/rank agreement, top-1 disagreements and margins, plus ppl2 delta versus repeat-baseline noise. Use agreed numerical tolerances; do not invent a pass threshold here. Inspect routed IDs/intermediates at the first divergence, since rounding can legitimately change near-tied routing choices and amplify later differences. Check first execution and graph replay for several decode steps; include prefill/multi-token controls to verify the batch-one guards. Only after correctness passes compare unprofiled decode throughput, with repeated warm runs, and profile to confirm dispatch. No speedup or correctness result is asserted by this assignment.
