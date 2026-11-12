# F18 storage pattern benchmark

`io_bench.py` measures cold, read-only file access on Linux, with no root privileges or Python runtime dependencies. Run the model sweep only as an exclusive operator queue job. This implementation was checked on a fully written 1 GiB temporary file; the model sweep has not been run.

## Full sweep

```sh
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py \
  --output-dir /home/muk/projects/siminfeng/cc/io_bench/full-results
```

The default model is `/home/muk/models/campaign/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`. Override it with a positional path or `--model`. The defaults are request caps 256K,640K,1M,1.83M,2M,4M; concurrency 1,2,4,8,16,32; random-expert and sequential; Python buffered pread, fio psync buffered/direct, fio io_uring buffered/direct. There are 360 possible cells, each limited to 64 MiB (configurable up to 512 MiB). A 570 second sweep budget skips remaining cells; fio subprocesses have a 15 second timeout bounded by remaining budget. Quick mode selects two sizes, two worker counts and 32 MiB/cell with a 75 second budget. Normal local storage should finish within these budgets; a kernel-blocked pread or very slow eviction cannot be forcibly interrupted by Python.

```sh
# Extract only the GGUF header, without reading tensor payloads:
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py --extract-ranges ranges.json
# Reuse those exact ranges:
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py --ranges ranges.json --output-dir full-results
# Focus on the engine's chunk size and worker count:
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py --sizes 2M --concurrency 16 --pattern random-expert
```

Extraction writes absolute tensor `[offset,size]` information as objects in `tensors`, including `nb2`, and per-expert `[offset,size]` pairs in `slices`. The stdlib reader supports little-endian GGUF v2/v3 and standard GGML quantizations, derives exact sizes from block geometry, and rejects unsupported expert layouts/types. Custom range files may be an extracted object or a bare list of pairs. Non-GGUF fixtures require `--raw-file`, which creates synthetic 8 MiB expert spans.

## What is measured

Python creates a persistent worker per stream for a cell and times each `os.pread` including buffer allocation/copy and thread scheduling. Its throughput is completed bytes divided by worker wall time, excluding setup, eviction and residency probes. fio jobs use one thread per stream, depth 1, with replay iologs containing the same requested offsets, and `--readonly --output-format=json`. fio throughput is summed completed bytes divided by the slowest job's read runtime; p50/p95 are nearest-rank percentiles pooled from per-request total latency logs (nanoseconds). Generated `cell-NNN/job.fio`, replay files, raw fio JSON and latency logs are retained for inspection or standalone replay. JSON records actual request count/min/max/bytes as well as the nominal request cap.

Random-expert shuffles expert slices without replacement per cycle, then reads each slice in chunks up to the requested cap, including short tails. A cap larger than `nb[2]` therefore still issues one slice-sized request, as the engine does. Sequential uses contiguous file chunks; workers stripe successive requests, so the combined workload is sequential but each worker has strides at concurrency > 1. The default random working set is distinct unless the supplied ranges are smaller than the cell byte budget. Duplicate/overlapping supplied ranges and repeated cycles can produce cache hits within a cell. Direct I/O rounds slice boundaries inward and request lengths down to 4 KiB, so direct and buffered byte patterns can differ; JSON records actual sizes. The ceiling summary takes the best successful random-expert buffered cell at a 2 MiB cap and 16 workers across Python and fio; it is an observed benchmark ceiling, not a hardware guarantee.

Before each cell the tool merges and page-aligns its planned read extents and calls `posix_fadvise(DONTNEED)` on those extents. It samples up to 256 first/last request pages with `mincore` before/after without touching them. Dirty pages, other processes or pinned mappings can prevent eviction, and sampling cannot prove every page cold: inspect `resident_fraction` before accepting a cold result. Direct reads normally leave the page cache cold. Read-ahead, Python overhead, fio logging/setup, cache churn, CPU pressure and shared-drive contention affect comparisons. Decimal MB/s is used; K/M/G inputs use powers of 1024. Unsupported or denied io_uring/direct modes are recorded as errors once and then skipped, never silently replaced by buffered I/O.

The random-expert workload isolates request sizes and streams; it does not reproduce router/layer ordering, an 84% cache-hit ratio, or concurrent demand/pregate traffic.

These measurements do not partition the engine's non-I/O time. P16 phase stats measure process-wide minor/major fault counts, voluntary/involuntary context switches, user/system CPU time and completion-interval latency over the decode phase. Fault counts do not directly give fault cost, context switches do not give blocked duration, and CPU times sum over all process threads. Compare these totals with existing fill wait counters; use a separate profiler if attribution is still needed.

## Checks and quick fixture run

```sh
python3 -m pytest /home/muk/projects/siminfeng/cc/io_bench -q -p no:cacheprovider
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py --help
```

The run below used an fsynced, non-sparse 1,073,741,824 byte temp file on the same filesystem. It ran while other workers compiled, so the values are smoke-test evidence, not the drive's exclusive ceiling. The fixture was removed after the run. The io_uring initialization error was `Operation not permitted` in this sandbox. All successful cells sampled 0 resident pages before reading. Elapsed sweep time: 11.665 seconds.

```sh
python3 /home/muk/projects/siminfeng/cc/io_bench/io_bench.py /home/muk/projects/siminfeng/cc/io_bench/f18-temp-1g.bin --raw-file --quick --output-dir /home/muk/projects/siminfeng/cc/io_bench/quick-results
```

Captured stdout:

```text
| pattern | engine | direct | request cap | workers | MB/s | p50 ms | p95 ms | resident before/after | status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| random-expert | python | 0 | 262144 | 1 | 81.8 | 2.112 | 9.611 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 262144 | 1 | 32.5 | 0.901 | 8.203 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 262144 | 1 | 84.9 | 1.261 | 9.376 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 262144 | 1 | - | - | - | 0.0/? | error |
| random-expert | io_uring | 1 | 262144 | 1 | - | - | - | 0.0/? | error |
| random-expert | python | 0 | 262144 | 16 | 206.0 | 19.747 | 33.110 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 262144 | 16 | 268.4 | 10.592 | 28.763 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 262144 | 16 | 322.6 | 12.238 | 15.036 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | io_uring | 1 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | python | 0 | 2097152 | 1 | 170.7 | 8.121 | 43.012 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 2097152 | 1 | 213.7 | 8.797 | 27.223 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 2097152 | 1 | 246.7 | 4.622 | 21.378 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | io_uring | 1 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | python | 0 | 2097152 | 16 | 431.5 | 73.817 | 76.881 | 0.0/1.0 | ok |
| random-expert | psync | 0 | 2097152 | 16 | 60.3 | 526.444 | 554.998 | 0.0/1.0 | ok |
| random-expert | psync | 1 | 2097152 | 16 | 399.5 | 49.189 | 83.562 | 0.0/0.0 | ok |
| random-expert | io_uring | 0 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| random-expert | io_uring | 1 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 262144 | 1 | 134.0 | 1.322 | 5.839 | 0.0/1.0 | ok |
| sequential | psync | 0 | 262144 | 1 | 212.4 | 0.719 | 3.149 | 0.0/1.0 | ok |
| sequential | psync | 1 | 262144 | 1 | 108.9 | 1.813 | 4.790 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 262144 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 262144 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 262144 | 16 | 284.7 | 12.956 | 18.708 | 0.0/1.0 | ok |
| sequential | psync | 0 | 262144 | 16 | 329.0 | 10.940 | 21.615 | 0.0/0.98046875 | ok |
| sequential | psync | 1 | 262144 | 16 | 424.7 | 6.849 | 17.757 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 262144 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 2097152 | 1 | 64.6 | 35.596 | 64.368 | 0.0/1.0 | ok |
| sequential | psync | 0 | 2097152 | 1 | 220.8 | 9.145 | 22.418 | 0.0/1.0 | ok |
| sequential | psync | 1 | 2097152 | 1 | 162.1 | 12.989 | 25.052 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 2097152 | 1 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | python | 0 | 2097152 | 16 | 259.0 | 110.632 | 129.138 | 0.0/1.0 | ok |
| sequential | psync | 0 | 2097152 | 16 | 294.3 | 107.493 | 113.031 | 0.0/1.0 | ok |
| sequential | psync | 1 | 2097152 | 16 | 244.9 | 71.374 | 136.634 | 0.0/0.0 | ok |
| sequential | io_uring | 0 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |
| sequential | io_uring | 1 | 2097152 | 16 | - | - | - | ?/? | unsupported (see earlier error) |

ceiling at engine pattern (2 MiB/16 workers buffered): 431.5 MB/s
Results: /home/muk/projects/siminfeng/cc/io_bench/quick-results/results.json
```
