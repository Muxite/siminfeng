# Resource observation contract (draft for x86 and Jetson)

`observe.py` is read-only, stdlib-only Python 3.12. It does not launch workloads,
resolve containers, modify clocks/power, or reset cumulative counters. It is
independent of `bin/lab`. The initial explicit backends are `nvidia-smi` (discrete
NVIDIA) and `linux` (procfs and cgroup v2). Both are enabled by default. Backend
selection is explicit through the Python API or repeated CLI `--backend` options.
Unsupported platforms retain the common fields with unavailable/error status.

## API and output

```python
from observe import sample, sample_series

reading = sample(backends=('nvidia-smi', 'linux'), gpu='0',
                 cgroup_path='/sys/fs/cgroup/my-job', pid=123)
for record in sample_series(.25, duration=10, cgroup_path='/sys/fs/cgroup/my-job'):
    consume(record)
# Alternative: until=time.monotonic() + 10 (absolute monotonic seconds).
```

`sample()` returns a JSON-serializable dictionary:

- `schema`: `resource-sample/0`.
- `ts`: `{utc: <timezone-aware ISO 8601 string>, monotonic: <seconds>}`,
  recorded at collection start. Monotonic values are local to this host/boot;
  they must not be compared between hosts or across reboots.
- `host_id`: OS hostname (operator-assigned; not a globally unique machine ID).
- `backends`: ordered list of enabled/attempted backend names. This list does
  not imply that those backends succeeded; inspect field statuses.
- `gpu_id`: selected GPU UUID, or null if not obtainable. `gpu` selects one
  zero-based CSV row index or exact UUID; default `0`. It never sums devices.
- `fields`: mapping from the common names below to field envelopes.
- `collection_seconds`: monotonic elapsed collection time. The sample is a
  sequence of reads, not an atomic snapshot. GPU query subprocesses each have
  their own timeout (default two seconds).

Every metric has exactly this envelope:

```json
{"value": null, "unit": "B", "source": "/sys/fs/cgroup/job/memory.peak", "scope": "cgroup-subtree", "status": "unavailable", "reason": "not found: /sys/fs/cgroup/job/memory.peak"}
```

`status` is `ok`, `unavailable`, or `error`. Successful fields have a real value
and null reason. Missing files, tools, columns, disabled backends, unspecified
PID/cgroup, empty GPU output, `[N/A]`, and `[Not Supported]` are unavailable
with null value and an explanatory reason. Nonzero query exit (including an
unusable driver), timeouts, permission errors, and malformed numeric readings
are errors with null value and reason. Missing data is never converted to zero.
Actual measured zero is valid. An empty, successfully read cgroup `io.stat`
means no block-device entries, so its sum is a genuine zero. No fallback from
missing `memory.peak` to `memory.current` is made.

## Common metrics

| Field | Unit | Source | Scope / meaning |
| --- | --- | --- | --- |
| `gpu.memory.used` | MiB | NVIDIA `memory.used` | `device-total`; all allocations observed by the driver |
| `gpu.memory.total` | MiB | NVIDIA `memory.total` | `device-total`; device framebuffer capacity |
| `gpu.util` | % | NVIDIA `utilization.gpu` | `device-total`; NVIDIA's utilization sampling window |
| `gpu.temp` | degC | NVIDIA `temperature.gpu` | `device-total` |
| `gpu.power` | W | NVIDIA `power.draw` | `device-total`; reported draw, not integrated energy |
| `gpu.clocks.sm` | MHz | NVIDIA `clocks.sm` / CSV `clocks.current.sm` | `device-total` |
| `gpu.clocks.memory` | MHz | NVIDIA `clocks.mem` / CSV `clocks.current.memory` | `device-total` |
| `gpu.compute_apps` | records | NVIDIA compute-apps CSV, filtered by UUID | `per-process-on-device`; see below |
| `host.mem.total` | B | `/proc/meminfo:MemTotal` | `host`; kernel-visible capacity |
| `host.mem.available` | B | `/proc/meminfo:MemAvailable` | `host`; kernel estimate, not simply free memory |
| `host.mem.swap.total` | B | `/proc/meminfo:SwapTotal` | `host`; configured swap capacity |
| `host.mem.swap.free` | B | `/proc/meminfo:SwapFree` | `host`; free swap capacity |
| `cgroup.memory.current` | B | given cgroup's `memory.current` | `cgroup-subtree`; current charged memory |
| `cgroup.memory.peak` | B | `memory.peak` | `cgroup-subtree`; kernel peak since creation/reset, not series peak |
| `cgroup.memory.swap.current` | B | `memory.swap.current` | `cgroup-subtree`; current swap charge |
| `cgroup.memory.swap.peak` | B | `memory.swap.peak` | `cgroup-subtree`; kernel swap peak where supported |
| `cgroup.io.rbytes` | B | `io.stat` sum of device `rbytes` | `cgroup-subtree`; cumulative block read bytes, not a rate |
| `process.rss` | B | `/proc/<pid>/status:VmRSS` | `process`; kernel RSS estimate, including shared pages |

The `gpu.clocks` family uses separate `.sm` and `.memory` envelopes so one clock
can be unsupported without hiding the other. Source strings identify the exact
query field or file; procfs kB and NVIDIA MiB use binary multiples (1024 and
1048576 bytes respectively). GPUs retain native MiB; host/cgroup/process values
are converted to bytes. Counts and byte totals retain integer precision.

`gpu.compute_apps.value` is a list of `{pid: <positive integer>, used_memory:
<field envelope in MiB>}`. Per-process used memory can be unavailable while the
PID remains known. A successful header-only apps CSV yields an empty list;
empty output without a header is unavailable. The global apps query is filtered
by the selected GPU UUID, preventing attribution to a different device. No GPU
UUID means unavailable attribution. These are compute processes only, not an
exhaustive list of graphics/driver allocations. PIDs are as reported by the
NVIDIA driver and can differ from PIDs visible in a container namespace.

## Attribution and interpretation

Device totals include other processes, contexts, and driver overhead. A difference
from a baseline is not proof of per-job GPU allocation; jobs and background
activity may overlap. Do not label device-total peaks as per-process peaks.
Compute-app memory does not necessarily sum to device-used memory, and it must
not be added to that total.

Cgroup observations require the caller to provide the actual cgroup v2 path.
They describe the subtree, not exclusively a single PID. They do **not** account
for driver/GPU allocations as a complete GPU accounting mechanism. Do not add
process RSS to cgroup current, or child cgroup counters to an ancestor already
including the child. RSS includes shared pages and cannot safely be summed as
unique physical memory. PID reuse, process exit, and cgroup deletion can race
sampling; no stable process identity or container resolver is provided.

On Jetson, CPU and GPU share DRAM. Host RAM use and GPU-associated shared DRAM
must **not** be summed as independent physical consumption. NVIDIA discrete
framebuffer fields must not be filled with Jetson RAM values. Keep those common
GPU memory fields unavailable unless a backend can provide their stated
semantics. Add shared-memory observations under platform fields with
`scope=shared-dram` and explicit overlap metadata/documentation.

## Series and loss accounting

`sample_series(interval, duration=None, *, until=None, sampler=sample, ...)`
is a lazy iterator. Exactly one of `duration` (nonnegative seconds) and `until`
(absolute monotonic deadline) is required; interval must be finite and positive.
It yields each sample as soon as collected, followed by one
`resource-series/0` summary. Fully consume the iterator to obtain the summary.
A zero duration/past deadline produces only a zero-count summary.

Ticks are scheduled from the start at fixed intervals in `[start, end)`; the
first tick is immediate. Delays skip elapsed ticks rather than replaying a burst
of stale measurements. The latest due tick receives the next collection. Sample
starts must precede the deadline; a collection started before it can finish
afterward. A short window can end collection after its last scheduled tick
without sleeping to the end. Slow JSONL consumers contribute to delay and loss.

The summary contains:

- `start_monotonic`, `end_monotonic` (requested deadline), `finished_monotonic`,
  and `interval_seconds`.
- `expected_samples`, `obtained_samples`, and `lost_samples` where expected =
  obtained + lost. Expected counts all scheduled ticks, including those missed
  during overruns and failed sampler calls.
- `max_gap_seconds`: maximum separation between obtained collection starts;
  null with fewer than two samples. `leading_gap_seconds` and
  `trailing_gap_seconds` expose boundary coverage, including a full window
  without samples. The trailing gap is relative to the requested deadline.
- `collection_errors`: slot and reason for sampler exceptions. A returned
  sample with unavailable/error fields still counts as an obtained sample;
  this counter measures delivery loss, not metric availability.

No interpolation, invented measurements, or burst catch-up is performed.
Interrupted/abandoned iterators and CLI Ctrl-C do not produce a final summary;
consumers must treat a missing summary as incomplete. SIGINT exits 130.

## CLI

```sh
python3 -B cc/resource_observe/observe.py once --out jsonl
python3 -B cc/resource_observe/observe.py watch --interval .25 --duration 10 --out jsonl
python3 -B cc/resource_observe/observe.py once --backend linux --pid 123 --cgroup-path /sys/fs/cgroup/job
```

`--out jsonl` chooses the format; records go to stdout. The observer does not
open an output path. Redirect stdout only to an authorized location. `watch`
defaults to ten seconds and supports `--until` instead of `--duration`.
Field errors remain data and do not fail the CLI. Invalid arguments exit 2.

## Platform extension and Jetson handoff

The Python API accepts `extra_backends={name: zero_arg_callable}` with the name
included explicitly in `backends`. Each callable returns a field mapping under
`platform.<name>.*`; every field must use the six-key envelope. Extensions cannot
overwrite common fields. Example:

```python
sample(backends=('linux', 'jetson'), extra_backends={
    'jetson': lambda: {
        'platform.jetson.ram.used': field(123, 'B', 'tegrastats:RAM', 'shared-dram')
    }
})
```

A backend must handle its own tool failures and return status envelopes. An
uncaught extension exception fails `sample()` and becomes a collection error
when called through `sample_series()`. The current CLI exposes only the two
built-in backends; the Jetson lead should register its backend and CLI choice
when it lands. Common field semantics require a contract/schema revision if
changed; new namespaced fields are additive under `resource-sample/0`.

Jetson should implement a read-only `tegrastats` backend with bounded process
lifetime and timeout, real host captures, malformed/missing version variants,
and tests. Suggested platform fields (preserve raw label/source and unsupported
status by Jetson model/version):

| Family | Suggested observations | Unit / scope |
| --- | --- | --- |
| `platform.jetson.ram.*` | RAM used/total, largest free block count/size | B (count for block count), `shared-dram` |
| `platform.jetson.swap.*` | used/total/cached | B, `host` |
| `platform.jetson.gr3d.*` | GPU activity and each reported frequency | %, MHz, `device-total` |
| `platform.jetson.emc.*` | memory-controller activity/frequency | %, MHz, `shared-dram` |
| `platform.jetson.cpu.*` | per-core activity, frequency, offline state | %, MHz, boolean; `host` |
| `platform.jetson.temperature.*` | named thermal sensors | degC, named sensor scope |
| `platform.jetson.power.*` | named rails, instantaneous and reported average | W after explicit mW conversion, named rail scope |
| `platform.jetson.accelerator.*` | optional VIC/NVENC/NVDEC/NVDLA/PVA activity/clocks | %, MHz, named engine scope |

Do not infer GPU allocated bytes from GR3D utilization or shared RAM use. Keep
rail averages distinct from instantaneous power; do not sum overlapping rails.
The Jetson lead must define exact available labels, overlap relationships, unit
conversions, and any common-field mapping using its target's tool output. This
handoff lists implementation targets, not verified Jetson capabilities.

## Fixtures and verification

Run locally, without network or GPU workloads:

```sh
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider cc/resource_observe/tests
```

`tests/fixtures/capture.json` records the UTC capture time, this host's hostname,
exact NVIDIA argv, and exit codes. `meminfo-real.txt` is this host's unmodified
`/proc/meminfo`. `nvidia-*-real.csv` and their `.stderr` companions are the exact
query outputs from this host, including failures; a `.csv` suffix does not imply
a successful CSV response. On this capture both NVIDIA queries exited 9 because
the driver could not be contacted. Successful real GPU telemetry was therefore
unavailable. No real success capture was fabricated or substituted from another
host. `*-synthetic.csv`, `*-unsupported.csv`, `*-malformed.*`, `empty.txt`, and
`nvidia-absent.json` are explicitly synthetic parser/failure cases. Re-capture
successful real GPU output on a host with a usable driver before treating the
NVIDIA success path as hardware-validated.
