# nsys_decode: per-token decode breakdown from nsys SQLite

Read-only Python standard-library profiler analysis; no CUDA, Docker, nsys
installation, external packages, database mutation, or compilation required.
The tests require pytest. Runtime is under two seconds on the supplied traces.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 cc/nsys_decode/analyze_nsys.py \
  artifacts/x4-hs/prof/static-n12.sqlite artifacts/x4-hs/prof/split-n12.sqlite \
  --warmup 5 --marker-bytes 196620 --output artifacts/codexPROF/breakdown.json

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q cc/nsys_decode/test_analyze_nsys.py \
  --basetemp=artifacts/codexPROF/pytest-tmp \
  -o cache_dir=artifacts/codexPROF/pytest-cache
```

Each input must contain one CUDA process, one logits D2H copy per token on the
selected `--device` (default 0), and the standard nsys StringIds, memcpy-enum,
runtime and memcpy tables. Kernel/graph/memset tables may be absent. Set
`--marker-bytes` to this model's logits size; **check the repeating pattern**
before interpreting results. Multiple requests, chunked logits, concurrent
inference, and unrelated same-sized D2H copies require a different marker.

The first marker ends prefill; the following intervals are decode steps. Remove
the first `--warmup` intervals (default 5); the final marker closes the final
complete token. Wall time includes sampling and scheduling between logits copies.
Intervals are clipped to boundaries, counts belong to the interval of their
start, and memcpy sizes/API times are joined using CUDA correlation IDs.
API timing therefore includes a boundary-crossing call even if its start count
belongs to the preceding token. The last synchronization tail after the final
DMA completion is outside the window.

JSON includes every retained token, numeric mean/median/min/max, API counts and
durations, copy kind/size/device/API durations, graph-exec frequencies, and
kernel rankings. Times are microseconds, bytes are bytes, raw timestamps ns.

GPU busy means the **union of observed graph envelopes, explicit kernels,
memcpy and memset intervals**, including all streams on the selected device.
Graph envelopes include internal device gaps; this is not hardware utilization.
Idle gaps are the complement within each token (boundary gaps included);
`--gap-us` additionally reports gaps at least that long (default 10us).
The four exclusive wall-time buckets are compute envelope, transfers outside
compute, GPU idle inside sync/copy API calls, and remaining GPU idle. Their sum
is wall time. API duration totals overlap GPU work and must not be added to it.

Graph-level traces hide constituent kernel counts and times. Steady-state
explicit kernel rankings may correctly be empty. The separate **first decode**
ranking exposes initialization/uncaptured kernels when present; it is diagnostic,
not a steady-state kernel measurement. Host idle outside observed APIs includes
CPU compute, scheduling, launch submission, sampling and untraced work, and is
not a CPU-compute measurement. Compare profiled results with unprofiled timing.
