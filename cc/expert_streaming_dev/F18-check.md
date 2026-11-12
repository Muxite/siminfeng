# F18 checks

Source checkout: `/home/muk/projects/xs-dev/llama-obs`, branch `xs-obs`, base `6e9d96cf4e007c74aab91bb59767247737a80f0f`. No network, Docker or GPU was used. All builds use `-j2`. No commits were made in siminfeng.

## Part A

Implemented `/home/muk/projects/siminfeng/cc/io_bench/io_bench.py`, `test_io_bench.py`, and `README.md`. The README contains the full quick-run stdout and instructions for the exclusive model sweep. Raw quick evidence is in `quick-results/results.json`, `quick-results/results.md` and individual fio cell directories. The fsynced, allocated 1 GiB temporary fixture was removed. The model storage sweep was not run.

Acceptance commands:

```sh
python3 -m pytest /home/muk/projects/siminfeng/cc/io_bench -q -p no:cacheprovider
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py --help
```

Result: 10 tests passed (see `F18-pytest.txt`); help exited 0 (see `F18-io-help.txt`). The tests cover synthetic quantized GGUF extraction including alignment and metadata arrays, truncated inputs, size parsing, range bounds, aligned direct plans, buffered reads/residency, real fio psync offset replay and pooled latency logs, repeated output-directory use, and a tiny CLI sweep.

Quick sweep: 11.665 seconds, 40 cells including unavailable-mode rows. Every successful cell sampled zero resident pages before reading. The observed 2 MiB/16-worker buffered maximum was 431.5 decimal MB/s. These are smoke-test numbers under concurrent compilation, not an exclusive drive ceiling. io_uring initialization failed with `Operation not permitted`; both variants were recorded and subsequent cells skipped. Buffered Python/psync and direct psync succeeded.

## Part B packaging and apply check

Patch: `/home/muk/projects/xs-dev/obs-artifacts/16-phase-rusage.patch`.

The source checkout's `.git` is mounted read-only by the sandbox. Packaging therefore used a local, shared clone at `llama-obs/build-f18-package`; the original checkout retains the three source edits with its branch tip unchanged. Packaging commit: `6be247becc965b94073b793fea170e935914e820` (parent is the exact requested base). The commit includes `Assisted-by: Codex`. No upstream submission was made.

```sh
git -C /home/muk/projects/xs-dev/llama-obs/build-f18-package format-patch -1 --stdout > /home/muk/projects/xs-dev/obs-artifacts/16-phase-rusage.patch
git -C /home/muk/projects/xs-dev/llama-obs/build-f18-package worktree add --detach /home/muk/projects/xs-dev/llama-obs/build-f18-apply-check 6e9d96cf4
git -C /home/muk/projects/xs-dev/llama-obs/build-f18-apply-check apply --check /home/muk/projects/xs-dev/obs-artifacts/16-phase-rusage.patch
```

`git apply --check` passed with exit 0 and no diagnostics. `F18-apply-check.txt` records the worktree's exact base. The throwaway worktree was removed after checking.

## CPU build and runtime

```sh
cd /home/muk/projects/xs-dev/llama-obs
cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
CCACHE_DISABLE=1 cmake --build build-cpu -j2 --target llama-completion
```

The first build attempt failed because the installed ccache used read-only `/run/user/1000/ccache-tmp`. Disabling ccache avoids that sandbox restriction without changing build parallelism or compiler options. Configuration output is in `F18-cmake.log`; build output is in `F18-build.log`.

Build passed with exit 0. A final incremental `CCACHE_DISABLE=1 cmake --build build-cpu -j2 --target llama-completion` also passed, ensuring the warmup reset edit was rebuilt; see `F18-build-final.log`. No compiler warnings/errors were found in either successful build log.

The exact completion command, run from the source checkout, was:

```sh
build-cpu/bin/llama-completion -m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf -p "The capital of France is" -n 32 -c 512 --temp 0 -s 1 -no-cnv -cmoe -lm mmap -t 4
```

For each configuration, `LLAMA_EXPERT_PHASE_STATS=0` and `LLAMA_EXPERT_PHASE_STATS=1` were tested. The stock-equivalent runs had all other `LLAMA_EXPERT_*` variables unset. The fill/pregate runs additionally used exactly:

```sh
LLAMA_EXPERT_FILL=1 LLAMA_EXPERT_FILL_RANDOM=decode LLAMA_EXPERT_FILL_MAX_TOKENS=1 LLAMA_EXPERT_PREGATE=1 LLAMA_EXPERT_PREGATE_K=8
```

All four commands exited 0 and all four stdout SHA256 hashes are identical:

| configuration | stats | stdout SHA256 | stats lines |
|---|---:|---|---:|
| stock | 0 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | 0 |
| stock | 1 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | 1 |
| fill-pregate | 0 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | 0 |
| fill-pregate | 1 | `25d42c39523316ee935c9d59659aef37be4067f8ecd9396542c00c389e2d96f9` | 1 |

Stock-equivalent enabled stats:

```text
expert_phase: stats decode_tokens=31 decode_wall_ns=996522446 minflt=0 majflt=0 nvcsw=1889 nivcsw=763 utime_us=3370625 stime_us=47333 tok_ns_p50=26737078 tok_ns_p95=65107093 tok_ns_max=86920959 first8_ns_mean=26114160 rest_ns_mean=34225831
```

Fill/pregate enabled stats:

```text
expert_phase: stats decode_tokens=31 decode_wall_ns=1377698357 minflt=0 majflt=0 nvcsw=5975 nivcsw=1157 utime_us=4311794 stime_us=48928 tok_ns_p50=31036653 tok_ns_p95=96588972 tok_ns_max=128262220 first8_ns_mean=44809371 rest_ns_mean=44295551
```

Both enabled configurations logged `phase_stats=1` in the effective configuration line; both disabled configurations logged `phase_stats=0`. Enabled stats report 31 successful single-token decode completions after the prompt (32 generated tokens). The checker verified positive ordered p50 <= p95 <= maximum, nonnegative counters, exactly one enabled line and zero disabled lines. Additional fill-disabled runs rejected `LLAMA_EXPERT_PHASE_STATS` values `2`, `-1`, `true`, and the empty string with the strict validator diagnostic. No core files were created.

The reproducer is `F18-run-checks.py`. Exact per-run stdout/stderr files, `F18-runtime.json`, and `F18-runtime.log` are retained beside this note. These are warm small-model smoke checks under shared CPU load; their latency differences are not controlled performance comparisons. In particular, both measured phases had zero page faults and cannot explain the Qwen campaign's non-I/O gap.


## Instrumentation boundaries

`LLAMA_EXPERT_PHASE_STATS` defaults to 0, is strictly validated as an integer in 0..1 even when fill/pregate are disabled, and appears in the effective configuration line. The disabled decode dispatch adds one branch; the template specialization removes the measurement hooks from that path. Enabled storage is allocated and initialized before decoding (8 MiB for 1,048,576 interval samples). There are no per-token allocations or rusage snapshots. Only `clock_gettime(CLOCK_MONOTONIC)` runs once per successful single-token decode completion; the first single-token ubatch also takes the initial clock/rusage snapshots.

The first interval runs from the first single-token decode ubatch to its successful completion; later intervals run between successful single-token decode completions. They include intervening sampling, output access and caller time. The wall/rusage phase ends at context teardown after backend synchronization, before stats sorting/printing and filler destruction. It therefore includes the caller's final output/performance reporting before context destruction. A performance reset discards warmup statistics and starts a fresh phase at the next single-token decode ubatch. Failed calls do not add samples. Quantiles use nearest rank and the first 1,048,576 samples; counts, maximum and first8/rest means cover all completions. This is intended for single-token decode, not batched/speculative throughput attribution.

RUSAGE_SELF includes all process threads (including filler/pregate workers), not just the calling thread. Fault and switch counts do not measure fault service time or blocked duration; user/system CPU time sums thread CPU usage and can exceed elapsed wall time. The stats provide evidence for the non-I/O time but cannot by themselves attribute all of it to faults, waits or compute. A profiler or scheduler trace is needed for that partition. No claim about the Qwen 420 ms/token gap is made from the small Granite smoke test.
