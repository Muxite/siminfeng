# Q8 activation reuse campaign patch

Apply only to llama.cpp `31288c9c66e59377f7788d920678a565bb89dd37`. See [DESIGN.md](DESIGN.md) for the safety rule and exclusions. This directory contains the patch and its handoff documentation; no reference-source files were edited.

## Build CC

From the campaign root, the orchestrator can build with the existing Build CC:

```bash
python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp --commit 31288c9c6 \
  --patch cc/q8_reuse/q8-activation-reuse.patch \
  --tag campaign/llamacpp:c1-q8-reuse \
  --out artifacts/codexH/build-q8-reuse
```

Use a fresh/empty output directory and the Build CC's required locally available, digest-pinned CUDA bases. Build a separate unpatched image from the same commit/toolchain for baseline comparison. The Build CC archives the source and applies the patch to its staging copy. Docker and CUDA were not run for this assignment.

## A/B switch

Reuse defaults ON. The environment setting is read once per process. Start separate processes for each run:

```bash
GGML_CUDA_Q8_REUSE=0 /app/llama-bench -m "$MODEL_Q8" -ngl 99 -p 0 -n 128 -b 1 -ub 1
GGML_CUDA_Q8_REUSE=1 /app/llama-bench -m "$MODEL_Q8" -ngl 99 -p 0 -n 128 -b 1 -ub 1
```

These commands assume the patched tools and model are available at the indicated runtime paths. When launching through Docker, forward the switch with `-e GGML_CUDA_Q8_REUSE=0` or `=1`. An unset switch enables reuse; other values, including `false`, also enable it. Keep model, workload, clocks, threads and all flags identical across runs. Record warmup and repeated decode measurements; do not infer a gain from the patch alone.

## Required validation before acceptance

1. Compile the patch with nvcc in the target Build CC. Run `test-backend-ops` for CUDA/CPU matmul correctness, including one-token Q8_0 and control types/batches. No CUDA compile result is available yet.
2. Compare the unpatched Q8_0 baseline, patched OFF and patched ON with identical model/corpus/options. The assignment requires mean KLD **0** and **100% same top-1** using `llama-perplexity --kl-divergence-base`. Force `-b 1 -ub 1` so the check exercises the changed decode path, rather than only prompt MMQ. For example, with host paths to the built binaries:

   ```bash
   "$BASELINE_BIN/llama-perplexity" -m "$MODEL_Q8" -f "$CORPUS" \
     -ngl 99 -c 128 -b 1 -ub 1 --chunks 4 \
     --kl-divergence-base artifacts/codexH/baseline.kld

   GGML_CUDA_Q8_REUSE=0 "$PATCHED_BIN/llama-perplexity" \
     -m "$MODEL_Q8" -f "$CORPUS" -ngl 99 -c 128 -b 1 -ub 1 --chunks 4 \
     --kl-divergence-base artifacts/codexH/baseline.kld --kl-divergence

   GGML_CUDA_Q8_REUSE=1 "$PATCHED_BIN/llama-perplexity" \
     -m "$MODEL_Q8" -f "$CORPUS" -ngl 99 -c 128 -b 1 -ub 1 --chunks 4 \
     --kl-divergence-base artifacts/codexH/baseline.kld --kl-divergence
   ```

   **Tool limitation:** this revision saves clipped, uint16-quantized log probabilities, not lossless raw logits (`tools/perplexity/perplexity.cpp`, the two `log_softmax` overloads). A baseline-versus-itself KLD run may therefore report a small nonzero mean and quantization can obscure top-1 ties. Run that self-comparison too. Do not declare bit identity from a rounded KLD number or silently waive the requested zero gate. Report discrepancies and the self-comparison floor to the orchestrator.

3. Independently compare every raw F32 logit byte against baseline and patched OFF using a lossless logits-dumping harness (`llama_get_logits_ith` for each evaluated token). Require byte-for-byte equality, not a tolerance. Include repeated one-token evaluations with changing inputs/positions, enough calls for graph warmup/capture/replay, and graph/shape switches that grow scratch sizes then return to an older captured graph. Repeat with `GGML_CUDA_DISABLE_GRAPHS=1` and with its variable **unset** to enable graphs; this backend treats any set value, including `0`, as disabling graphs.
4. Exercise allocator alias hazards in backend graphs: repeated use of one input, distinct tensor identities sharing an address, partial-overlap output views, an intervening writer reusing input storage, and successive evaluations that reuse the same tensor object/address with new values. Verify that overlapping RoPE, successful fusion, other operations, and concurrent-stream graphs cannot reuse a stale entry. Check MoE and fused paths as unchanged controls.
5. Profile kernel launches and decode throughput for ON versus OFF. In an unfused Q/RoPE/K/V safe window with the same input, expect one quantization instead of three. Fused operations and other writers end the window, so model-level savings must be measured. Check no regression in disabled, non-Q8_0 and multi-token paths. Capture/replay and pointer-lifetime checks are mandatory even if direct execution passes.

## Checks performed here

- Confirmed reference HEAD is the target commit and its tracked working tree is clean.
- `git apply --check --verbose` passed on a `git archive` copy under `artifacts/codexH/verification-tree`, with Git discovery stopped before the enclosing campaign repository. All three intended files were checked.
- Applied the patch to that copy and compared all three resulting files against the edited work copies; reverse application check passed.
- Host smoke checks passed with C++17, `-Wall -Wextra -Werror`, AddressSanitizer and UndefinedBehaviorSanitizer, for unset, `0`, `1` and `false` switch settings. LeakSanitizer was disabled because it cannot run under the sandbox's ptrace. These mocks do not validate CUDA compilation or GPU arithmetic.

Evidence and repeatable host checks: `artifacts/codexH/validation.txt`, `host_check.py`, `host_check.cpp`, and `make_patch.py`. Runtime numerical/performance gates remain pending with the orchestrator.
