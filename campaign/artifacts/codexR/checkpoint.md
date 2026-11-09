# T-R final checkpoint - resumable GPU handoff

Completed CPU assignment work on 2026-11-09, before the 60-minute checkpoint
threshold (started approximately 22:26 UTC, final verification at 22:47 UTC).
No time-limit cutoff occurred. All intentional writes stayed under
`cc/hybrid_split/` and `artifacts/codexR/`; source `ref/` remains clean.

Base: `31288c9c66e59377f7788d920678a565bb89dd37`.
Patch: `cc/hybrid_split/hybrid-split.patch`.
Patch SHA256: `1063ac102a14bff798d56b5bf292d21455f3f4c847f0e2043d98d58126bd1efb`.

## Completed

- Lossless packed-GGUF expert partitioner, real trace hot sets, router permutation
  metadata and three original-ID lookup tensors per layer.
- Optional Granite loader and split FFN graph; CPU negative IDs skip expert dot
  products and zero reused outputs; ordinary unsplit path retained.
- Original routing order restored before softmax/top-k. Host hot banks stay
  batched; CUDA hot banks dispatch slots separately to avoid duplicate null IDs.
- Existing -ot placement only; --no-op-offload and --no-repack required. No CUDA
  kernel, attention or KV modifications.
- CPU-only Release compilation at -j4: llama-cli, llama-completion,
  test-backend-ops and test-hybrid-split. Build CC test-target configuration also
  compiled with LLAMA_BUILD_TESTS=OFF and LLAMA_BUILD_HYBRID_SPLIT_TESTS=ON.
- Eight pytest cases pass. Eight focused configurations x three routing modes
  pass, with scalar references, nontrivial permutation, router restoration,
  negative IDs (-1 and -7), reused NaN-poisoned output, and backend assignment.
- Real n18/n12/n6 expert-payload matches created: 168/360/552 real hot experts,
  plus 24 null experts, giving 162/324/486 MiB hot payload. All three independently
  match original mixE4K for 64 greedy continuation tokens on chat.txt, t=4, with
  max and RMS logit difference 0.
- Patch applied to a clean git archive in verify-tree; all 12 patched files
  byte-identical to the compiled tree. No second clean-tree compilation needed.

## Artifacts and commands

Read `cc/hybrid_split/README.md` for format, ownership/layout/sync, invariants,
null-ID justification, exact build/transform/run commands and graph limitations.
`artifacts/codexR/verification.json` is the machine-readable final summary.
`patch.json`, `patch-apply.log`, `patch-stat.txt`, `build-final.log`,
`build-cc-mode.log`, `pytest.log`, `focused-test.log`, `numerical-n{18,12,6}.json`,
`models.sha256`, and `deliverables.sha256` retain evidence and identities.
The transformed GGUFs are only under `artifacts/codexR/models/`.
Auxiliary trace-192 validation is exact too. Discarded first-design files marked
v0 are retained as failed development evidence; do not use them for final claims.

## Resume on the GPU orchestrator

1. Use the README Build CC command with --patch and both extra test targets.
   No Docker or CUDA compilation was run here. Build CC currently uses parallel
   10 internally; the worker's actual local CPU builds all used at most -j4.
2. Run test-hybrid-split --backend CUDA0. Cold skips must remain on CPU; the hot
   slot and router restoration graph must execute on CUDA.
3. Compare full mixed-device greedy tokens/logits before performance testing.
4. Measure shim and nvidia-smi footprint at c4096, trim hot payload as needed to
   match true total n18/n12/n6 GPU MiB, trace transfers and CUDA graph reuse.
5. Benchmark against static layer placement and engine LRU at equal measured MiB.

Open risks: hot per-slot launches/concatenations, unreduced cold activation
transfers, graph rebuild map readbacks, GPU rounding/routing sensitivity,
backend capture/fallback behavior, and complete GPU footprint changes. No claim
of throughput improvement, CUDA correctness or exact total GPU MiB matching has
been established. Other prompts/models and repacked CPU buffers were not tested.
