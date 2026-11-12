# F16 checks

Base: `6e9d96cf4`; private clone; no Docker, GPU, network, push, or PR actions.

**B1 is blocked:** the sandbox mounts `llama-opt/.git` read-only. The initial `git switch -c p12-trace 6e9d96cf4` failed with `Unable to create .../.git/index.lock: Read-only file system`. No branches or commits were created. The four `.patch` files are independent source diffs, not `git format-patch` outputs. Each passes `git apply --check --whitespace=error` against base files in a throwaway extracted source directory. A Git worktree also requires writes to the protected metadata and could not be created.

`finish_f16_git.py` performs the remaining local branch/commit/format-patch/worktree checks when run with Git metadata write access. It never pushes. Source snapshots are in `llama-opt/build-f16/p12-source` through `p15-source`. The main checkout is restored to the original base after validation.

## Build and run setup

Each separate build used the original checkout as the source directory, with the corresponding independent patch source copied into place. Builds ran one at a time with `-j2`. ccache and compiler temporary files were confined to opt-artifacts.

```sh
export CCACHE_DIR=/home/muk/projects/xs-dev/opt-artifacts/ccache
export CCACHE_TEMPDIR=/home/muk/projects/xs-dev/opt-artifacts/tmp
export TMPDIR=/home/muk/projects/xs-dev/opt-artifacts/tmp
cmake -B build-trace -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-trace -j2 --target llama-completion
cmake -B build-populate -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-populate -j2 --target llama-completion
cmake -B build-pin -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-pin -j2 --target llama-completion
cmake -B build-temporal -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-temporal -j2 --target llama-completion
```

The functional command for every comparison was:

```sh
LLAMA_EXPERT_FILL=1 LLAMA_EXPERT_FILL_RANDOM=decode LLAMA_EXPERT_FILL_MAX_TOKENS=1 LLAMA_EXPERT_PREGATE=1 LLAMA_EXPERT_PREGATE_K=8 \
build-<name>/bin/llama-completion -m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf -p "The capital of France is" -n 32 -c 512 --temp 0 -s 1 -no-cnv -cmoe -lm mmap -t 4
```

Both sides also set `LLAMA_ARG_LOG_VERBOSITY=4` to expose library INFO logs; this does not change stdout. Initial P12 runs at default verbosity also produced the same prescribed-command hash. New required trace/pin/pregate summaries and the populate fill summary print directly to stderr.

**Important fixture limitation:** the exact Granite command on this ARM CPU chooses `CPU_REPACK` even with `-cmoe`, because the existing loader considers extra CPU buffer types. Fill is inactive: `expert tensors not mmap-backed or file offsets unavailable/unsupported layout`. This is not a zero-IO performance result. Supplemental mapped comparisons add `LLAMA_ARG_REPACK=0` on both sides, with all command arguments unchanged. These exercise the actual mmap paths. Repacking changes the model output relative to plain mmap on this fixture; hashes must be compared within each pair.

## B3: stdout comparisons

| Patch | Switch on | Off stdout SHA-256 | On stdout SHA-256 | Result |
| --- | --- | --- | --- | --- |
| P12 | `{'LLAMA_EXPERT_TRACE': '/home/muk/projects/xs-dev/opt-artifacts/p12-trace.csv'}` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | PASS |
| P13 | `{'LLAMA_EXPERT_FILL_BACKEND': 'populate'}` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | PASS |
| P14 | `{'LLAMA_EXPERT_PIN_MIB': '64'}` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | PASS |
| P15 | `{'LLAMA_EXPERT_PREGATE_MODE': 'temporal'}` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | PASS |

Supplemental mapped comparisons (`LLAMA_ARG_REPACK=0`):

| Patch | Off stdout SHA-256 | On stdout SHA-256 | Result |
| --- | --- | --- | --- |
| P12 | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | PASS |
| P13 | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | PASS |
| P14 | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | PASS |
| P15 | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | `a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933` | PASS |

### P12 stats

off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 trace=(unset)
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 trace=/home/muk/projects/xs-dev/opt-artifacts/p12-trace.csv
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
expert_trace: wrote records=907 path=/home/muk/projects/xs-dev/opt-artifacts/p12-trace.csv
```

mapped_off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 trace=(unset)
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=53722553 max_single_fill_ns=662418 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=53722553 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=53722553 demand_wait_ns_per_token=1732985.581
```

mapped_on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 trace=/home/muk/projects/xs-dev/opt-artifacts/p12-trace.csv
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_trace: wrote records=907 path=/home/muk/projects/xs-dev/opt-artifacts/p12-trace.csv
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=57688247 max_single_fill_ns=440716 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=57688247 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=57688247 demand_wait_ns_per_token=1860911.194
```

trace_only: exit=0, stdout_sha256=a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933

```text
expert_config: effective fill=0 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=0 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 trace=/home/muk/projects/xs-dev/opt-artifacts/p12-trace-only.csv
expert_trace: wrote records=907 path=/home/muk/projects/xs-dev/opt-artifacts/p12-trace-only.csv
```

Invalid switch value and unknown `LLAMA_EXPERT_UNRECOGNIZED=1` each abort with SIGABRT (-6), with core files disabled.


### P13 stats

off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 backend=pread
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 backend=populate
expert_fill: stats calls=0 ranges=0 bytes_requested=0 bytes_skipped_resident=0 bytes_read=0 read_syscalls=0 total_fill_wall_ns=0 max_single_fill_ns=0 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=0 decode_bytes_read=0 decode_fill_wall_ns=0 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 populate_bytes=0 populate_fallbacks=0
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

mapped_off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 backend=pread
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=87766413 max_single_fill_ns=4144463 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=87766413 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=87766413 demand_wait_ns_per_token=2831174.613 populate_bytes=0 populate_fallbacks=0
```

mapped_on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 backend=populate
expert_fill: stats calls=0 ranges=0 bytes_requested=0 bytes_skipped_resident=0 bytes_read=0 read_syscalls=0 total_fill_wall_ns=0 max_single_fill_ns=0 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=0 decode_bytes_read=0 decode_fill_wall_ns=0 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 populate_bytes=0 populate_fallbacks=0
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=0 bytes_read=0 read_syscalls=0 total_fill_wall_ns=382798762 max_single_fill_ns=4271666 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=382798762 populate_calls=17091 populate_ns=1117508500 populate_errors=0 demand_wait_ns_total=382798762 demand_wait_ns_per_token=12348347.161 populate_bytes=10003319648 populate_fallbacks=0
```

Invalid switch value and unknown `LLAMA_EXPERT_UNRECOGNIZED=1` each abort with SIGABRT (-6), with core files disabled.


### P14 stats

off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=0 pin_warmup=8 pin_every=32
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=64 pin_warmup=8 pin_every=32
expert_pin: stats pinned_mib=0.000 pinned_experts=0 repins=0 mlock_failures=0 mode=inactive
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
expert_pin: inactive (pin expert layout/mmap unavailable)
```

mapped_off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=0 pin_warmup=8 pin_every=32
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=9829528320 bytes_read=173791328 read_syscalls=598 total_fill_wall_ns=722990448 max_single_fill_ns=42865749 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=173791328 decode_fill_wall_ns=722990448 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=722990448 demand_wait_ns_per_token=23322272.516
```

mapped_on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=64 pin_warmup=8 pin_every=32
expert_pin: stats pinned_mib=0.000 pinned_experts=0 repins=0 mlock_failures=0 mode=inactive
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_pin: stats pinned_mib=62.590 pinned_experts=39 repins=1 mlock_failures=0 mode=mlock
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=81249651 max_single_fill_ns=3762209 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=81249651 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=81249651 demand_wait_ns_per_token=2620956.484
```

locked: exit=0, stdout_sha256=a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=4 pin_warmup=8 pin_every=4
expert_pin: stats pinned_mib=0.000 pinned_experts=0 repins=0 mlock_failures=0 mode=inactive
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_pin: stats pinned_mib=3.211 pinned_experts=2 repins=6 mlock_failures=0 mode=mlock
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=90033367 max_single_fill_ns=8453243 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=90033367 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=90033367 demand_wait_ns_per_token=2904302.161
```

willneed: exit=0, stdout_sha256=a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=64 pin_warmup=8 pin_every=32
expert_pin: stats pinned_mib=0.000 pinned_experts=0 repins=0 mlock_failures=0 mode=inactive
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_pin: mlock failed errno=1; pin_mode=willneed (pages remain evictable)
expert_pin: stats pinned_mib=0.000 pinned_experts=0 repins=1 mlock_failures=1 mode=willneed
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=82119650 max_single_fill_ns=4135019 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=82119650 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=82119650 demand_wait_ns_per_token=2649020.968
```

pin_only: exit=0, stdout_sha256=a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933

```text
expert_config: effective fill=0 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=0 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pin_mib=4 pin_warmup=8 pin_every=32
expert_pin: stats pinned_mib=0.000 pinned_experts=0 repins=0 mlock_failures=0 mode=inactive
expert_pin: stats pinned_mib=3.211 pinned_experts=2 repins=1 mlock_failures=0 mode=mlock
```

Invalid switch value and unknown `LLAMA_EXPERT_UNRECOGNIZED=1` each abort with SIGABRT (-6), with core files disabled.


### P15 stats

off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pregate_mode=router
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pregate_mode=temporal
expert_pregate: inactive (fill unavailable or predictor unsupported)
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 recall_router=0.000000 recall_temporal=0.000000
expert_fill: inactive (expert tensors not mmap-backed or file offsets unavailable/unsupported layout)
```

mapped_off:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pregate_mode=router
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=77023731 max_single_fill_ns=4186379 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=77023731 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=77023731 demand_wait_ns_per_token=2484636.484
```

mapped_on:

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pregate_mode=temporal
expert_pregate: inactive (fill unavailable or predictor unsupported)
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 recall_router=0.000000 recall_temporal=0.000000
expert_pregate: active k=8 mode=temporal router=unused order=after
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_pregate: stats predictions=713 arrivals=690 hits=2347 actual=5520 predicted=5520 recall_at_k=0.425181 precision=0.425181 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=690 sync_calls_without_prediction=54 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=5520 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=2347 recall_at_k_d1=0.425181 recall_at_k_d2=0.000000 actual_d1=5520 actual_d2=0 hits_d1=2347 hits_d2=0 demand_wait_ns_total=63834702 demand_wait_ns_per_token=2059183.935 recall_router=0.000000 recall_temporal=0.411466
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=63834702 max_single_fill_ns=3659166 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=63834702 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=63834702 demand_wait_ns_per_token=2059183.935
```

hybrid: exit=0, stdout_sha256=a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=1 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pregate_mode=hybrid
expert_pregate: inactive (fill unavailable or predictor unsupported)
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 recall_router=0.000000 recall_temporal=0.000000
expert_pregate: inactive (pregate requires qwen35moe without groups, LoRA, or folded rotations)
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=63957988 demand_wait_ns_per_token=2063160.903 recall_router=0.000000 recall_temporal=0.000000
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=63957988 max_single_fill_ns=995673 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=63957988 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=63957988 demand_wait_ns_per_token=2063160.903
```

lookahead2: exit=0, stdout_sha256=a8c1bff8c4256b3c5c2256a1985b4952e659c4ff7a8d347dafe91a438c6ae933

```text
expert_config: effective fill=1 random=decode max_tokens=1 threads=8 chunk_kib=1024 pregate=1 k=8 order=after max_inflight=4 lookahead=2 min_tokens=0 diag=0 populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0 pregate_mode=temporal
expert_pregate: inactive (fill unavailable or predictor unsupported)
expert_pregate: stats predictions=0 arrivals=0 hits=0 actual=0 predicted=0 recall_at_k=0.000000 precision=0.000000 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=0 sync_calls_without_prediction=0 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=0 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=0 recall_at_k_d1=0.000000 recall_at_k_d2=0.000000 actual_d1=0 actual_d2=0 hits_d1=0 hits_d2=0 demand_wait_ns_total=0 demand_wait_ns_per_token=0.000 recall_router=0.000000 recall_temporal=0.000000
expert_pregate: active k=8 mode=temporal router=unused order=after
expert_fill: active threads=8 chunk_kib=1024 random=decode files=1
expert_fill: random advice applied at decode
expert_pregate: stats predictions=1395 arrivals=1350 hits=4574 actual=10800 predicted=10800 recall_at_k=0.423519 precision=0.423519 async_bytes_read=0 async_reads=0 async_errors=0 dropped=0 cancelled_jobs=0 cancelled_bytes=0 inflight_at_arrival=0 inflight_reads=0 pending_reads=0 sync_bytes_with_prediction=0 sync_bytes_without_prediction=0 sync_calls_with_prediction=690 sync_calls_without_prediction=54 skips=0 pred_done=0 pred_inflight=0 pred_queued=0 pred_resident_at_predict=10800 hit_done=0 hit_inflight=0 hit_queued=0 hit_resident=4574 recall_at_k_d1=0.425181 recall_at_k_d2=0.421780 actual_d1=5520 actual_d2=5280 hits_d1=2347 hits_d2=2227 demand_wait_ns_total=61089971 demand_wait_ns_per_token=1970644.226 recall_router=0.000000 recall_temporal=0.409857
expert_fill: stats calls=744 ranges=17856 bytes_requested=10003319648 bytes_skipped_resident=10003319648 bytes_read=0 read_syscalls=0 total_fill_wall_ns=61089971 max_single_fill_ns=3893252 prompt_calls=0 prompt_bytes_read=0 prompt_fill_wall_ns=0 decode_calls=744 decode_bytes_read=0 decode_fill_wall_ns=61089971 populate_calls=0 populate_ns=0 populate_errors=0 demand_wait_ns_total=61089971 demand_wait_ns_per_token=1970644.226
```

Invalid switch value and unknown `LLAMA_EXPERT_UNRECOGNIZED=1` each abort with SIGABRT (-6), with core files disabled.

## P12 simulator summary

Records=1814; decode_tokens=62; K=8; expert_bytes=1918894.

The trace contains separate prescribed/repacked and supplemental/mapped runs; cache and predictor state resets between them. Phase uses ubatch size (1=decode). CLI warmup is included. Final-layer prompt pruning produces fewer prompt rows there. Router coverage is zero: the inherited router predictor supports Qwen35MoE, not Granite.

Overall recall@8:

```json
{
  "frequency": 0.43649193548387094,
  "previous_frequency": 0.43674395161290325,
  "previous_token": 0.4116263440860215,
  "router": 0.0,
  "router_frequency": 0.43649193548387094,
  "router_previous": 0.4116263440860215,
  "router_previous_frequency": 0.43674395161290325
}
```

| Budget GiB | Policy | Hit rate | Bytes read per decode token (estimated) |
| --- | --- | --- | --- |
| 2 | belady | 0.966818 | 12225211.774 |
| 2 | lfu | 0.966818 | 12225211.774 |
| 2 | lru | 0.966818 | 12225211.774 |
| 2 | lru_frequency | 0.966818 | 12225211.774 |
| 3 | belady | 0.966818 | 12225211.774 |
| 3 | lfu | 0.966818 | 12225211.774 |
| 3 | lru | 0.966818 | 12225211.774 |
| 3 | lru_frequency | 0.966818 | 12225211.774 |
| 4 | belady | 0.966818 | 12225211.774 |
| 4 | lfu | 0.966818 | 12225211.774 |
| 4 | lru | 0.966818 | 12225211.774 |
| 4 | lru_frequency | 0.966818 | 12225211.774 |
| 5 | belady | 0.966818 | 12225211.774 |
| 5 | lfu | 0.966818 | 12225211.774 |
| 5 | lru | 0.966818 | 12225211.774 |
| 5 | lru_frequency | 0.966818 | 12225211.774 |

The 2-5 GiB budgets exceed this fixture trace working set at the default Satay expert-size estimate. Equal cache results reflect compulsory misses, not a replacement-policy ranking. Use a Satay trace or smaller budgets to compare eviction policies. `p12-simulator.json` includes every per-layer recall and both reuse histograms.

```json
{
  "distinct_experts_between_uses": {
    "128-255": 4924,
    "256-511": 4910,
    "512-1023": 1675,
    "cold": 395
  },
  "token_gap": {
    "1": 4900,
    "16-31": 137,
    "2-3": 3430,
    "32-63": 11,
    "4-7": 1972,
    "8-15": 1059,
    "cold": 395
  }
}
```

## Additional checks

- Sparse C API output-mask test: output positions 0 and 2 in a 4-token prompt; final MoE trace rows are indexed 0 and 2, other layers 0/1/2/3: PASS.
- P13 worker fixture: demand and async populate, injected EINVAL and ENOSYS pread fallback, injected EIO propagation, read-only mapping contents unchanged: PASS (`p13-worker-test.log`).
- P14 mapped locking, repeated repins at PIN_EVERY=4, pin-only without fill, and forced RLIMIT_MEMLOCK=0: stdout preserved. Zero memlock falls back to WILLNEED and releases all locks. Its pages remain evictable.
- P15 temporal mode and lookahead=2 preserve stdout. Hybrid mode is accepted but inactive on unsupported Granite; real Qwen router/hybrid execution was not tested.
- P15 temporal recall independently recomputed from the final mapped P12 trace:

```json
{
  "actual": 5704,
  "hits": 2347,
  "expected_recall": 0.4114656381486676,
  "reported_recall": 0.411466,
  "result": "PASS"
}
```

- Added C++ lines checked for ASCII and trailing whitespace. Independent patch whitespace/apply checks pass. Python modules compile; stdlib tests pass. `ty` and `editorconfig-checker` are unavailable; no packages were installed.

## B2 (verbatim command)

```sh
python3 /home/muk/projects/xs-dev/opt-artifacts/test_expert_trace_sim.py
```

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 0.069s

OK
```

## Remaining limits

B1 branch/commit/format-patch/worktree packaging remains incomplete solely because Git metadata is read-only. CPU builds, independent source apply checks, B2, and B3 stdout comparisons completed. The model fits in page cache; no cold-storage or GPU/Satay performance claim is made. No real Qwen router/hybrid run was possible with the provided offline fixture. See `F16-design.md` for mechanisms, counter semantics, and bandwidth/memory tradeoffs.
