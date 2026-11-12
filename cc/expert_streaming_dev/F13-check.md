# F13 checks

Source base: `6e9d96cf4`, branch `xs-dev`. Only this clone and artifacts under `/home/muk/projects/xs-dev` were used. No network, Docker, GPU, other repository, or factory directory was used.

## Delivery limitation

The sandbox mounts the original `.git` directory read-only. The requested branch commit failed with `fatal: Unable to create '/home/muk/projects/xs-dev/llama.cpp/.git/index.lock': Read-only file system`. Therefore A1 is NOT satisfied: `xs-dev` remains at `6e9d96cf4`, with the implementation in its working tree.

Two real commit objects with the requested subjects and `Assisted-by: Codex` were created using separate object/index storage in `f13-artifacts`, reading the original repository objects without modifying its metadata. They are authored by `Codex <codex@openai.com>` because no Git author identity was configured. `git format-patch` generated the requested patch files from these objects. These objects are not attached to `xs-dev`.

- Patch 10 commit object: `de9c76edd0c56891c80b372c5592f872a2ccdde7`.
- Patch 11 commit object: `2876f707c79681180161d072845769d85d2af7b8`.

Patch 11 declares `Depends-on: 10-expert-readiness.patch`. It shares patch 10's dispatch tracking and statistics integration; the runtime join switch is independent of readiness and was tested with readiness off. A standalone patch 11 would need to carry those shared changes.

`git worktree add --detach` was attempted and blocked by the same sandbox restriction:

```text
fatal: could not create leading directories of '.git/worktrees/worktree-check': Read-only file system
Preparing worktree (detached HEAD 6e9d96cf4)
```

Fallback verification used `git archive 6e9d96cf4` to create a disposable source tree, checked and applied patch 10, then checked and applied patch 11, and compared all three resulting files byte-for-byte with the working sources. The disposable source tree was removed. This is sequential patch validation, but is not the requested worktree validation.

```text
git apply --check /home/muk/projects/xs-dev/10-expert-readiness.patch : PASS
git apply /home/muk/projects/xs-dev/10-expert-readiness.patch : PASS
git apply --check /home/muk/projects/xs-dev/11-inflight-join.patch : PASS
git apply /home/muk/projects/xs-dev/11-inflight-join.patch : PASS
Final patched sources byte-identical to working sources: PASS
Exported verification tree removed.
```

## CPU build

```sh
cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
CCACHE_DISABLE=1 cmake --build build-cpu -j4 --target llama-completion
```

The first build attempt failed because ccache tried to create temporary files under the read-only `/run/user/1000/ccache-tmp`. Disabling ccache allowed the build. A final incremental build compiled the final source state successfully. Logs: `f13-artifacts/configure.log`, `f13-artifacts/build-final.log`.

## Functional checks

Each run starts with every inherited `LLAMA_EXPERT_*` variable removed. All runs use this exact argument list:

```sh
build-cpu/bin/llama-completion -m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf -p "The capital of France is" -n 32 -c 512 --temp 0 -s 1 -no-cnv -cmoe -lm mmap -t 4
```

| Run | Additional LLAMA_EXPERT_* settings | Exit | Stdout SHA256 |
| --- | --- | --- | --- |
| a | All unset | 0 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` |
| b | FILL=1; FILL_RANDOM=decode; FILL_MAX_TOKENS=1; PREGATE=1; PREGATE_K=8 | 0 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` |
| c | FILL=1; FILL_RANDOM=decode; FILL_MAX_TOKENS=1; PREGATE=1; PREGATE_K=8; READINESS=1 | 0 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` |
| d | FILL=1; FILL_RANDOM=decode; FILL_MAX_TOKENS=1; PREGATE=1; PREGATE_K=8; READINESS=1; PREGATE_JOIN=1 | 0 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` |
| e | READINESS=2 | -6 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Stdout of a-d is byte-identical (SHA256 and direct byte comparison). Run e terminates with SIGABRT (Python return code -6, shell convention 134). Raw stdout/stderr are preserved in `f13-artifacts/{a,b,c,d,e}.{stdout,stderr}`.

Run a:

```text
expert_config: effective fill=0 random=0 max_tokens=0 threads=8 chunk_kib=1024 pregate=0 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=0 join=0
expert_config: effective fill=0 random=0 max_tokens=0 threads=8 chunk_kib=1024 pregate=0 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=0 join=0
```

Run b:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=0 join=0
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 join_waits=0 join_wait_ns=0 join_bytes=0
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=0 join=0
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 join_waits=0 join_wait_ns=0 join_bytes=0
```

Run c:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=1 join=0
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 join_waits=0 join_wait_ns=0 join_bytes=0
expert_readiness: stats decode_tokens=0 decode_selected=0 decode_selected_predicted=0 decode_ready_bytes_predicted=0 decode_missing_bytes_predicted=0 decode_ready_bytes_unpredicted=0 decode_missing_bytes_unpredicted=0 decode_wasted_async_bytes=0 decode_duplicate_demand_bytes=0 prompt_selected=0 prompt_missing_bytes=0 probe_ns_total=0
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=1 join=0
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 join_waits=0 join_wait_ns=0 join_bytes=0
expert_readiness: stats decode_tokens=0 decode_selected=0 decode_selected_predicted=0 decode_ready_bytes_predicted=0 decode_missing_bytes_predicted=0 decode_ready_bytes_unpredicted=0 decode_missing_bytes_unpredicted=0 decode_wasted_async_bytes=0 decode_duplicate_demand_bytes=0 prompt_selected=0 prompt_missing_bytes=0 probe_ns_total=0
```

Run d:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=1 join=1
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 join_waits=0 join_wait_ns=0 join_bytes=0
expert_readiness: stats decode_tokens=0 decode_selected=0 decode_selected_predicted=0 decode_ready_bytes_predicted=0 decode_missing_bytes_predicted=0 decode_ready_bytes_unpredicted=0 decode_missing_bytes_unpredicted=0 decode_wasted_async_bytes=0 decode_duplicate_demand_bytes=0 prompt_selected=0 prompt_missing_bytes=0 probe_ns_total=0
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 readiness=1 join=1
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 join_waits=0 join_wait_ns=0 join_bytes=0
expert_readiness: stats decode_tokens=0 decode_selected=0 decode_selected_predicted=0 decode_ready_bytes_predicted=0 decode_missing_bytes_predicted=0 decode_ready_bytes_unpredicted=0 decode_missing_bytes_unpredicted=0 decode_wasted_async_bytes=0 decode_duplicate_demand_bytes=0 prompt_selected=0 prompt_missing_bytes=0 probe_ns_total=0
```

Run e:

```text
expert_config: invalid LLAMA_EXPERT_READINESS=2 (range 0..1)
```

The CPU expert-fill callback is inactive on this ARM CPU build: it repacks expert tensors into `CPU_REPACK`, so the existing mmap-only fill guard rejects them. The exact requested runs use the default log threshold, which hides the core library's informational inactive message. An additional diagnostic invocation with `--verbosity 4` and `-n 1` confirmed the cause (captured below). The existing Qwen35MoE-only pregate architecture check is not reached because fill initialization fails first; Granite would also be excluded by that check. Enabled readiness and requested pregate stats print once per context even when fill initialization is inactive, with zero counters. No readiness active line is emitted when fill is inactive. This CLI invocation constructs two contexts (two effective-config lines), so c and d each show two readiness stats lines, exactly one per context. These model runs establish output identity, validator behavior, and configured stats visibility, but do not test live fill or join IO. The focused test below exercises nonzero readiness and join paths. No global cache eviction was performed; the small model may already be cached, so resident or near-zero counters in any live-fill run would not demonstrate cold-cache performance.

```text
0.00.415.468 I load_tensors:   CPU_REPACK model buffer size =  1300.50 MiB
0.02.514.477 I expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

## Focused filler regression test

`f13-artifacts/test-filler.cpp` compiles directly against the final filler header, replacing `mincore` with a deterministic resident/missing oracle and wrapping the first async `pread` with a condition-variable gate. Real reads use a temporary file under the project artifact directory. It checks partial overlap, join on/off, readiness on/off, async EIO fallback, queued claiming, wasted-byte completion accounting, per-expert page union and shared pages, and unchanged mmap bytes. Its missing-page oracle ensures nonzero paths are tested even with a warm model cache.

```sh
c++ -std=c++17 -O2 -pthread f13-artifacts/test-filler.cpp -o f13-artifacts/test-filler
f13-artifacts/test-filler f13-artifacts/test-data
```

```text
inflight observe=0 join=0 error=0 PASS
inflight observe=0 join=0 error=1 PASS
inflight observe=0 join=1 error=0 PASS
inflight observe=0 join=1 error=1 PASS
inflight observe=1 join=0 error=0 PASS
inflight observe=1 join=0 error=1 PASS
inflight observe=1 join=1 error=0 PASS
inflight observe=1 join=1 error=1 PASS
queued claim PASS
readiness per-expert union, shared pages, ready/missing, predicted/unpredicted PASS
```

## Counter semantics

Counters are per context and cumulative, including any warmup execution. Readiness measurements follow the existing fill token eligibility limits and deduplicate selected IDs within each layer/ubatch. Decode means `n_tokens == 1`; prompt means `n_tokens > 1`. Predictions are the union of both lookahead distances for that layer, with layer 0 and pregate-off selections unpredicted.

- `decode_tokens`: number of fill-eligible one-token ubatches processed, once per ubatch, not once per layer.
- `decode_selected`: sum of unique selected expert IDs at fill-eligible decode layer arrivals.
- `decode_selected_predicted`: those selected expert IDs found in that layer's prediction union, counted once per selected expert per arrival.
- `decode_ready_bytes_predicted`: resident bytes in the page-rounded union of all ranges for each selected predicted expert, sampled before that file's demand reads or join wait.
- `decode_missing_bytes_predicted`: nonresident bytes in the same per-expert ranges at that snapshot.
- `decode_ready_bytes_unpredicted`: resident bytes for selected experts outside that layer's prediction union at the same snapshot.
- `decode_missing_bytes_unpredicted`: nonresident bytes for selected experts outside that prediction union at the same snapshot.
- `decode_wasted_async_bytes`: actual async pread bytes for predicted experts absent from the selected set at their target arrival, including bytes completed after arrival; resident skips and canceled unread bytes contribute zero.
- `decode_duplicate_demand_bytes`: union of demand read ranges overlapping currently running async chunk extents at demand publication, counted once per demand byte; successful joined bytes are removed before this count.
- `prompt_selected`: sum of unique selected expert IDs at fill-eligible prompt layer arrivals, deduplicated across the prompt ubatch's tokens.
- `prompt_missing_bytes`: sum of nonresident bytes across per-selected-expert page-rounded ranges at fill-eligible prompt arrivals.
- `probe_ns_total`: cumulative wall nanoseconds for readiness metadata preparation, the reused residency buffer allocation/mincore, and per-expert residency attribution, for prompt plus decode; excludes demand preads, join waits, and job construction.
- `join_waits`: number of overlapping running chunks for which demand entered a condition-variable completion wait, including failed chunks.
- `join_wait_ns`: cumulative wall nanoseconds inside those completion waits, excluding demand fallback reads.
- `join_bytes`: missing demand bytes removed from demand jobs after successful completion of overlapping async chunks, counted once per demand byte; errors contribute zero and cause demand rereads.

The same file page shared by two selected experts is counted once for each expert's readiness range; within one expert it is counted once across overlapping tensors. The final partial page is clamped to file size. Demand and join counters operate on the union of demand ranges instead.

## Disabled cost and limitations

- Both switches default to 0 and are validated strictly as 0 or 1, even when fill/pregate is inactive. The effective-config line includes `readiness=` and `join=`.
- Readiness-off adds branch checks only to the existing loops: no selected-range allocations, extra mincore, readiness timers, running-range updates, or wasted-byte accounting. The normal `Range` representation is unchanged. Empty local metadata containers require no heap allocations. Per-worker tracking storage is initialized with the pool.
- Join-off adds branch checks and null chunk handles; there are no per-chunk state allocations, completion waits, or queue scans for join. Async queue entries carry an additional null shared-pointer field.
- Readiness is a page-cache snapshot, not a guarantee against later eviction, and is not evidence that prediction caused residency. No extra full mincore pass is added. Split-file models are sampled before each file's demand fill, not atomically across all files; earlier files may already have been filled when later files are sampled.
- Duplicate bytes describe scheduled overlap with an async chunk, not proven duplicate physical device IO: the kernel may serialize reads, workers may skip pages already resident, and a chunk may finish before the demand worker starts. On demand IO failure the scheduled counter can exceed successfully returned demand bytes.
- Wasted bytes are attributed by prediction batch/expert, not by unique file page or later reuse. Repeated predictions can read the same bytes; an expert unused at its target may be useful at a later layer/token. Predictions whose target arrival is never executed are not assigned a selected-set outcome.
- Join waits for the whole overlapping chunk, including portions beyond the selected pages. There is no timeout or spinning. Async errors retain the missing ranges for the normal demand-read fallback. Queued overlapping chunks are canceled/claimed under the dispatch mutex; needed missing portions become ordinary demand jobs. Existing cancellation counters include these claimed chunks.
- Completed successful chunks are trusted after joining, matching the demand path's existing post-read semantics; subsequent eviction is handled by ordinary mmap faults. No weight writes, routing changes, sampler changes, or compute changes were introduced.
- No cold-cache benchmark, Qwen35MoE model run, sanitizer run, or GPU test was performed. Synthetic gate/oracle tests exercise join and error handling but cannot establish device-level speedups.
- All source edits are limited to `src/llama-expert-fill.h`, `src/llama-context.cpp`, and `src/llama-context.h`. `git diff --check` passes. Repository scope/security/general review found no remaining issue in the changed paths.
