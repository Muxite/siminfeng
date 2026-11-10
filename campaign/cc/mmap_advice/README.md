# Router-informed expert mmap prefetch (J-OM7)

`patches/02-router-prefetch.patch` applies after `patches/01-mmap-advice-random.patch` to `ref/llama.cpp-up1108` at `cf34d3113`. The reference checkout is unchanged. This is a Linux-only, opt-in experiment; no build or GPU execution was performed for this assignment.

```bash
cat cc/mmap_advice/patches/01-mmap-advice-random.patch \
    cc/mmap_advice/patches/02-router-prefetch.patch > /tmp/om7-combined.patch
git -C ref/llama.cpp-up1108 apply --check /tmp/om7-combined.patch
```

## Mechanism

Set `LLAMA_EXPERT_PREFETCH=1`. The loader supplies the file ranges of `ffn_*_exps.weight` tensors as mmap advice exclusions, separately from the existing lazy-loading placement policy. These ranges receive `MADV_RANDOM` and are excluded from initial `WILLNEED` and `MAP_POPULATE`; non-expert ranges retain default sequential file advice and initial `WILLNEED`. Existing lazy ranges remain excluded. Expert mode takes precedence over `LLAMA_MMAP_ADVICE=random`, so that setting both does not make the entire file random. With expert mode off, patch 01 retains its original behavior. NUMA and other explicit loading policies still apply.

The scheduler's new CPU-only before-node callback observes the first `MUL_MAT_ID` that uses a layer's original expert weights. Its `src[2]` is the same `ffn_moe_topk-L` I32 tensor named in `src/llama-graph.cpp:2367`, or the scheduler's host copy of that tensor. CPU graph prefixes finish before observation. The callback reads `{K,T,1,1}` with the actual strides, validates IDs, and deduplicates all selected IDs across the ubatch. Before that expert matmul starts, it issues `madvise(MADV_WILLNEED)` for each ID's gate/up/down slice (also supports merged gate-up). Each slice uses `data + id * nb[2]`, length `nb[2]`, aligned outward to OS pages. Qwen's observed gate/up slices are 589,824 bytes and down slices 720,896 bytes; the implementation derives sizes from tensors, including exceptional layers.

Only contiguous three-dimensional expert tensors whose entire actual data span is inside a model-owned mmap are eligible. A host buffer alone is insufficient proof. Unsupported tensors or router layouts are skipped with a once-per-context log. Biases/scales, transformed router IDs, and MoE-cache slot IDs are not interpreted as original expert IDs. No model weights, router IDs, or floating-point operations are modified.

Each layer retains at most 32 advised IDs in an LRU, with a 32-token expiry measured in that layer's ubatch token count. Cache hits update recency without extending the last-advice timestamp. Failed advice is counted and does not suppress a later retry. A large ubatch is one deduplicated observation; expiry is evaluated at its end, not separately for each token. This is a syscall limiter, not a page-residency tracker.

`LLAMA_EXPERT_PREFETCH_NEXT=1` additionally advises the same union of IDs in layer L+1, subject to its expert bounds and the same LRU. It has no effect without the primary switch. When L+1 is observed in the same ubatch, each distinct prediction counts once in `total`, and membership in its actual selected union counts as a hit. This is prediction precision over ubatch unions, not per-token accuracy; large prompts can inflate it. Predictions whose next layer is never host-observed are excluded from the ratio and are cleared at the next ubatch. No last-layer wraparound occurs.

At context destruction, after synchronization, `LLAMA_LOG_INFO` reports attempted advice calls, attempted page-rounded bytes, syscall failures, and next-layer prediction hits/total plus percent. Bytes advised are hints, not measured disk bytes. Startup logs identify both mapping policy and the host-consumer hook; zero calls must not be treated as evidence of active prefetch.

## Applicability and synchronization

Works with mmap-backed CPU experts under CPU-only execution or any GPU/CPU layer split, including `-ngl 99 -cmoe`. `-cmoe` moves expert weights/operations to CPU; it does **not** force router computation to CPU. With GPU-offloaded layers, router top-k can be GPU-computed. The scheduler already copies these IDs to the CPU expert split and waits for that copy (including its existing asynchronous prefetch event). Observation occurs at this first host consumer, after those existing waits. It does not request another readback or synchronize a GPU backend. The added synchronization is CPU-only; the existing CPU backend's graph compute is already synchronous and its synchronize function is null.

The regular scheduler eval callback is preserved, including its existing synchronization behavior. Installing an ordinary eval callback just to read GPU top-k would introduce a synchronization; this patch avoids that path. On all-CPU execution it finishes the prefix containing the router before reading IDs. On GPU routing it deliberately defers observation from top-k production to the first CPU expert consumer. Device-only expert paths are skipped instead of creating a new device-to-host dependency. GPU-uploaded weights, non-mmap loads, unsupported layouts, and cached/remapped IDs receive no exact host prefetch. Non-Linux builds do not enable the feature.

## Risks

`WILLNEED` is a best-effort kernel hint and may start late, block briefly, or fail; it does not guarantee expert-sized physical IO. Page rounding can include neighboring weights. Predictions and stale LRU entries can waste IO or suppress useful rereads after eviction. Even exact advice may compete with current reads and evict useful cache pages. The CPU graph prefix boundary adds planning/threadpool overhead. Applying RANDOM to experts also changes initial model upload behavior for offloaded weights; measure startup separately. The feature has not been compiled or performance-validated. D033's speedup estimate is a hypothesis, not a result.

## A/B/C for the build/run owner (jetson-arm)

Use one executable containing 01+02 for all three arms and the same Qwen model, fixed prompt, seed, context, thread counts, ubatch size, and generation length. Reproduce D033 with `-lm mmap -ngl 99 -cmoe`, 64 generated tokens, and its recorded p1 prompt. Keep the existing telemetry collector running and quiesce unrelated NVMe work. Example arm invocations, after setting `BIN`, `MODEL`, and `PROMPT_FILE`:

```bash
env -u LLAMA_MMAP_ADVICE -u LLAMA_EXPERT_PREFETCH -u LLAMA_EXPERT_PREFETCH_NEXT \
  "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n 64 -lm mmap -ngl 99 -cmoe --seed 1 \
  > base.stdout 2> base.stderr
env -u LLAMA_MMAP_ADVICE -u LLAMA_EXPERT_PREFETCH_NEXT LLAMA_EXPERT_PREFETCH=1 \
  "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n 64 -lm mmap -ngl 99 -cmoe --seed 1 \
  > prefetch.stdout 2> prefetch.stderr
env -u LLAMA_MMAP_ADVICE LLAMA_EXPERT_PREFETCH=1 LLAMA_EXPERT_PREFETCH_NEXT=1 \
  "$BIN" -m "$MODEL" -f "$PROMPT_FILE" -n 64 -lm mmap -ngl 99 -cmoe --seed 1 \
  > next.stdout 2> next.stderr
```

Use unique output directories per repetition. Run balanced orders such as base/prefetch/next, next/prefetch/base, and repeat. Apply the same established cold-cache procedure before every run, and report warm-cache runs separately. Capture startup, prompt, and decode timing separately, plus a comparable whole-run IO window and a decode-only window. Verify the expert-mode startup logs and nonzero advice counts, and compare output tokens/hashes against base; advice should not change computation.

Snapshot the same physical NVMe device's `/sys/class/block/<device>/stat` at each window boundary (do not sum a partition and its parent). With deltas in completed reads (field 1) and sectors read (field 3), compute `NVMe GiB = sectors * 512 / 2^30` and `avg KiB/IO = sectors * 0.5 / reads`, undefined when reads is zero. Linux block-stat sectors are 512 bytes regardless of logical sector size. Report both metrics alongside decode tokens/s, advice calls/bytes/failures, prediction hits/total, refaults, minimum MemAvailable, and output equality. Include temperature/power and the exact image/source/patch identity. Compare with D033's roughly 56 GiB and 93-95 KiB/IO baseline versus 24 GiB and 4 KiB/IO whole-file RANDOM result; do not infer disk traffic from advice counters. A useful result needs lower NVMe bytes while retaining larger IO and improving decode speed.

Validation performed here: combined `git apply --check` against the pinned reference and whitespace checking of the resulting patch. Compilation, execution, IO behavior, output equality, and prediction quality remain for the build/run owner, as required by the no-build/no-GPU assignment.
