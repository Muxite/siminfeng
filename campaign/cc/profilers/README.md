# Profiler wrappers

Python standard-library command generators and parsers. Run these commands from
the campaign repository root. Generators print scripts or JSON argv; they do not
invoke Docker, sudo, or NVIDIA tools. Tests use the real checked-in captures and
small synthetic cases for unit conversions and boundaries.

## Parse existing captures and run tests

```bash
export PYTHONDONTWRITEBYTECODE=1
python cc/profilers/bpf_split_timer.py parse artifacts/bpf1/bpf.txt \
  --seconds 5 --tokens-per-second 23.1 > artifacts/codexPROFW/bpf-summary.json
python cc/profilers/ncu_summary.py artifacts/ncu1/ncu.csv \
  > artifacts/codexPROFW/ncu-summary.json
python -m pytest -q cc/profilers/tests \
  -o cache_dir=artifacts/codexPROFW/pytest-cache \
  --basetemp=artifacts/codexPROFW/pytest-tmp
```

BPF: **5626 splits**, **48.70996 splits/token** at 23.1 t/s over 5 seconds,
**0.89034 CPU time fraction**, **791.273 µs mean**. The median falls in
[512, 1024) µs; histogram K/M suffixes mean powers of 1024. D042 §3's 48.9
splits/token is approximate; its stated inputs give 48.71.

NCU: **600 launches**, **2226.88 µs** summed kernel duration,
**221.852565 GB/s** effective DRAM read bandwidth, reproducing D041's 222 GB/s.
The exact assigned launch-bound rule (`time < 5 µs && dram < 5%`) marks
**401 launches, 45.770944% of kernel time**, consistent in scale with D041's
“about 40%” description. The parser reports the exact result without rounding it
to 40%. Time share uses summed GPU kernel durations, not inference wall time.

Each class groups template instantiations by function name. Its `GB_s` is summed
bytes / summed time / 1e9; `dram_pct` and `sm_pct` are duration-weighted means.
`launch_bound` means every launch in that class meets the rule; counts and time
for mixed classes are also retained. The fixture contains read bytes only, so
`byte_scope` is `read`. If write metrics are supplied for every launch, bandwidth
includes them; partial write coverage is rejected. Required metrics are duration,
read bytes, DRAM%, SM%. Missing metrics, duplicate launch metrics, unsupported
units, and nonfinite values fail explicitly. IDs are scoped by process ID.
Supported duration units: ns/us/µs/ms/s; byte units: byte/Kbyte/Mbyte/Gbyte, decimal.
Percentages use `%`. CSV quoted commas and numeric thousands separators work.

As D041 warns, ncu serializes kernels and flushes caches. Its small-kernel times
are inflated compared with CUDA graph replay; these results are profiling
measurements, not unprofiled throughput or a prediction of fusion speedup.

## Generate a BPF capture

Substitute the name of an already-running engine container:

```bash
python cc/profilers/bpf_split_timer.py container --container ENGINE_CONTAINER \
  --seconds 5 --per-thread > artifacts/codexPROFW/capture-bpf.sh
```

On a qualified Docker/bpftrace host with sudo access, run the emitted recipe:

```bash
bash artifacts/codexPROFW/capture-bpf.sh > artifacts/codexPROFW/bpf-live.txt
python cc/profilers/bpf_split_timer.py parse artifacts/codexPROFW/bpf-live.txt \
  --tokens-per-second 23.1 > artifacts/codexPROFW/bpf-live-summary.json
```

The recipe uses `docker inspect -f '{{.State.Pid}}' NAME`, preserves the
`/proc/PID/root/app/` namespace path, selects one actual `libggml-cpu.so.*` file
(skipping symlink aliases), and filters probes by the container's host PID.
If multiple versions exist, select a library explicitly:

```bash
python cc/profilers/bpf_split_timer.py script --pid 12345 \
  --library /proc/12345/root/app/libggml-cpu.so.0.27.0 --seconds 5 --per-thread \
  > artifacts/codexPROFW/split-timer.bt
```

D042 §3 pitfalls: `pgrep` can select the host Docker client. The CPU MoE inner
matmul is a templated static function; use the C-exported `ggml_graph_compute`
(as here) or `ggml_compute_forward_mul_mat_id_tiled` instead. Reading the container
root and attaching uprobes require suitable host permissions; the backend must
exist and export the selected symbol. No rebuild is needed.

Entry timestamps are keyed by host TID; returns without a captured entry are
ignored. Incomplete calls are cleared at exit. `--per-thread` adds counts and
elapsed-time sums for caller threads, not worker-thread CPU utilization. Calls
must not recurse on a single TID. The stable output fields are `@graphs`,
`@graph_tot_us`, `@graph_us`, optional `@thread_graphs[TID]` / `@thread_tot_us[TID]`,
and `PROFW window_s=...`. The window is the configured timer duration, rounded
to milliseconds; use `--seconds` when parsing an interrupted capture to override
it with the actual observed duration. Legacy fixtures require `--seconds`.
`cpu_time_frac` sums completed call elapsed times / window; concurrent callers
can exceed 1. `splits_per_token` divides calls by window × supplied decode t/s
(or is null if t/s is omitted). Match the throughput and capture windows and
exclude prompt/warmup to interpret this as per-token decode work.

## Generate engine extraction and GPU capture argv

This uses the release-1 local image ID recorded in
`builds/results/release-1/build-record.json`. The extraction creates a stopped
container, copies the contents of `/app`, then removes the temporary container;
it never starts the engine. The destination directory must exist. Use a fresh
container name and destination for a different engine image.

```bash
mkdir -p artifacts/codexPROFW/app artifacts/codexPROFW/ncu artifacts/codexPROFW/nsys
python cc/profilers/capture.py extract \
  --image sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e \
  --directory "$PWD/artifacts/codexPROFW/app" --name profw-extract \
  > artifacts/codexPROFW/extract-argv.json
python cc/profilers/capture.py ncu --app "$PWD/artifacts/codexPROFW/app" \
  --models /home/muk/models/campaign/gguf --output "$PWD/artifacts/codexPROFW/ncu" \
  --launch-skip 2000 --launch-count 600 -- \
  -m /m/granite-3.1-1b-a400m-instruct-Q8_0.gguf -ngl 99 -fa 1 -p 0 -n 32 -r 1 \
  > artifacts/codexPROFW/ncu-argv.json
python cc/profilers/capture.py nsys --app "$PWD/artifacts/codexPROFW/app" \
  --models /home/muk/models/campaign/gguf --output "$PWD/artifacts/codexPROFW/nsys" \
  --graph-trace node -- \
  -m /m/granite-3.1-1b-a400m-instruct-Q8_0.gguf -ngl 99 -p 0 -n 128 -r 3 \
  > artifacts/codexPROFW/nsys-argv.json
```

`extract-argv.json` contains three argv arrays in order. Each capture file contains
one argv array. On a qualified GPU host, execute the generated arrays directly:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path
base = Path('artifacts/codexPROFW')
for argv in json.loads((base / 'extract-argv.json').read_text()):
    subprocess.run(argv, check=True)
for name in ('ncu', 'nsys'):
    subprocess.run(json.loads((base / f'{name}-argv.json').read_text()), check=True)
PY
```

If extraction fails after creation, remove the named stopped container with
`docker rm profw-extract` before retrying. Docker capture runs use
`campaign/cuda-tools:c1`, `--gpus all`, and `LD_LIBRARY_PATH=/app`. Both `/app` and
models at `/m` are mounted read-only; only `/out` is writable. D004's ncu
qualification requires Docker `--cap-add SYS_ADMIN`; this is a Docker flag,
not an ncu flag. The metric set and default launch-skip/count match D041.
D011's filtered matmul screen can use `--launch-skip 600 --launch-count 12`
(the generator captures all kernel classes). Nsys disables CPU sampling and
context-switch tracing per D004, and supports D011's node view or a graph view
with `--graph-trace graph`. Reports are `/out/capture.nsys-rep` and
`/out/ncu.csv`. Discard or flag warmup; D004 observed a large first-launch cost.

Live Docker/GPU/sudo execution was not performed for this assignment. Validation
covers fixture parsing, generated argv, quoting, shell syntax, and CLI behavior.
