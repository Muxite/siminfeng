# Host DRAM calibration (T-CAL2)

Writes are restricted to `cc/host_calibrate/` and `artifacts/codexCAL2/`.
`calibrate.py` only plans and parses; `run_plan.py` explicitly executes the
serial Docker plan. Original `artifacts/codexCAL/` captures and calibration
(including H2D ~28.7 GB/s and residual handoff ~199 us) are preserved.
The H2D source, transfer selection, and handoff algebra are unchanged.
No replacement full host calibration is claimed without a successful new run.

The old backend perf measurements reuse cache-resident weights. The copied real
GPU-host fixtures under `tests/fixtures/cal2/` now fail plausibility checks:
GPU Q4_K ~1384 GB/s exceeds RTX 5070 Ti's stated 896 GB/s ceiling; CPU Q4_K
expert ~97 GB/s exceeds the conservative 60 GB/s fallback. Every legacy row
is screened, with device/type/value/ceiling/reason in the failure. Even legacy
captures below the ceilings cannot be accepted as DRAM measurements: new stream
captures are required. Tests read the real manifest and stdout; no timing rows
are fabricated for those rejection tests.

## Exact commands for Opus on the GPU host

Run from the campaign root. Use the immutable release engine reference that
contains `/app/llama-bench`, `nvidia-smi`, `lscpu`, and `timeout`. The tools image
must already exist locally as `campaign/cuda-tools:c1`, with nvcc, gcc, libgomp,
and timeout. No packages, host tuning, sudo, or image substitutions are needed.
The CPU probe requires x86 AVX2; it is not an ARM/Jetson CPU implementation.

```bash
export PYTHONDONTWRITEBYTECODE=1
export RUN_REF='sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e'
export MODEL='/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf'
mkdir -p artifacts/codexCAL2
python3 -B cc/host_calibrate/calibrate.py plan \
  --image "$RUN_REF" --model "$MODEL" --cpu-layers 0 12 \
  --out artifacts/codexCAL2/gpu > artifacts/codexCAL2/command-plan.json
python3 -B cc/host_calibrate/run_plan.py artifacts/codexCAL2/gpu
python3 -B - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('artifacts/codexPL4/model-stats.json').read_text())['mixE4K']
Path('artifacts/codexCAL2/model-stats.json').write_text(json.dumps(s, indent=2)+'\n')
PY
python3 -B cc/host_calibrate/calibrate.py parse \
  --capture artifacts/codexCAL2/gpu --stats artifacts/codexCAL2/model-stats.json \
  --out artifacts/codexCAL2/host-calibration.json
python3 -B cc/host_calibrate/apply.py artifacts/codexCAL2/host-calibration.json \
  --out artifacts/codexCAL2/planner-calibration.json
```

If populated memory channels and configured DDR transfer rate are known from
host inventory, add e.g. `--memory-channels 2 --memory-rate-mts 3200` to `plan`.
These are explicit host identity inputs, recorded in the manifest/report; do
not infer channel count from DIMM count or CPU cores. The ceiling is channels
x MT/s x 8 / 1000 decimal GB/s (51.2 in that example). lscpu does not provide
those DRAM details, so absence uses the stated conservative 60 GB/s ceiling.
No dmidecode/root access is attempted. The small GPU name table accepts RTX
5070 Ti (896) and Jetson Orin Nano (68). Unknown/ambiguous/missing GPU identity
fails closed, rather than choosing an unlimited ceiling. Jetson's nvidia-smi
availability and its name spelling are **UNVERIFIED**; missing identity fails.
A ceiling is a rejection threshold, not a measured bandwidth claim.

The ten jobs capture identity, build/run H2D, build CUDA and CPU streams, run
both streams, and run the two K-coordinate benches. All build outputs go to
the requested CAL2 capture directory. Exact compiler payloads in the plan:

```text
campaign/cuda-tools:c1 timeout -k 1s 25s nvcc -O2 /src/h2d.cu -o /out/h2d
campaign/cuda-tools:c1 timeout -k 1s 25s nvcc -O3 /src/bw.cu -o /out/bw
campaign/cuda-tools:c1 timeout -k 1s 25s gcc -O3 -mavx2 -fopenmp -Wall -Wextra /src/bw_cpu.c -o /out/bw_cpu
```

## Probe behavior and limits

`bw.cu` obtains actual L2 bytes from cudaGetDeviceProperties. It allocates a
512 MiB minimum weight bank partitioned into eight rotating buffers, grows
it to at least 8x L2, and refuses if this needs more than 1 GiB. One complete
pass visits every buffer before reuse; each sample measures eight complete
passes with CUDA events, three repeats. Coalesced FP32 matrix reads multiply
by a cached 1024-float vector and reduce to observable block sums. This is a
GEMV-like read screen, not a copy/store bandwidth benchmark. Output writes
are 4 KiB per kernel; bandwidth counts weight reads only, in decimal GB/s.

`bw_cpu.c` sums unique L3 shared_cpu_list domains from read-only sysfs (all
visible CPU caches, conservatively including CPUs outside the allowed affinity).
Unknown cache data or total L3 above 128 MiB causes refusal. It allocates eight
aligned buffers totalling at least 512 MiB and 8x aggregate L3, rounded to
32 MiB, with total <=1 GiB. Full passes rotate the buffer order and use AVX2
loads and two independent accumulation chains with observable checksums.
The plan sweeps **every thread count 1..8**, three samples each, >=80 ms per
sample; standalone `bw_cpu N` supports 1..64. Dynamic OpenMP teams are disabled;
if the runtime cannot create the requested team, the probe fails. Eight-thread
samples match the llama-bench `-t 8` setting. Allocation and first-touch writes
are outside the timed region; repeated full passes keep the entire bank in
circulation. Read bytes and elapsed milliseconds determine GB/s independently
of the printed GB/s field, which the parser checks.

Both new probes have a 24 s process alarm. Every planned command, including
H2D and inference, has a 25 s timeout plus 1 s kill grace; runner uses a 29 s
Docker deadline and 1 s forced cleanup. Failed or partial sweeps are rejected.
All explicit allocations are <=1 GiB: new weight banks <=1 GiB (CPU split into
eight allocations), CUDA vector/output 4 KiB each, H2D <=256 MiB per host/device
allocation. Small Granite model qualification is unchanged (file <=1 GiB,
24 layers, 32/8 experts, embedding1024/expert FFN512), tg64 with small batches.
The containers have a 2 GiB RAM cap, read-only root filesystem, no network,
read-only source/model/probe inputs, and only build output mounted writable.
There is no host configuration change. Large-cache hosts fail rather than
silently running a cache-resident probe. Per-probe deadlines do not imply a
30 s deadline for the whole serial plan.

## Interpretation, H2D, and handoff

Successful parse emits `host-calibration/1` with identity, stated ceilings,
raw samples, raw stdout, image/pin/manifest provenance and observed spreads.
CPU thread scaling is retained in raw samples; CPU constants use eight-thread
median, GPU constant uses three-repeat stream median. **All quant types use
the same CPU DRAM proxy**; these probes do not measure dequantization cost or
expert dispatch and cannot establish effective Q2/Q3/Q4/Q6/Q8 throughput.
Do not mistake a plausible number for a validated whole-model coefficient.

The existing H2D sweep still measures pinned transfers from 1 through 256 MiB,
three repeats and eight copies per sample. Its constant is the median of
64–256 MiB samples; original ~28.7 GB/s is not invalidated by the cache issue.
With exact first-K tensor stats, handoff is still computed from:

```text
t(K) = G(K)/(1000*BWgpu) + sum_q Cq(K)/(1000*BWcpu,q) + K*h
h = [t1 - t0 - sum_q (Cq1-Cq0)/(1000*BWcpu,q)
     - (G1-G0)/(1000*BWgpu)] / (K1-K0)
```

Times are microseconds/token and byte counts are active bytes/token from
`decode_model.traffic`. Pairwise repeat residuals produce median/spread;
negative/nonfinite residuals fail rather than being clamped. New bandwidth
proxies can change the fitted handoff, so the old ~199 us is retained only
in the untouched historical calibration. The optional independent CPU expert
bandwidth check is not supplied: the two-K difference alone cannot identify
both expert bandwidth and unknown handoff without another assumption.
Sample spread does not bound systematic error or shared-host load.
`apply.py` writes a separate planner object under CAL2; it never edits the
bundled planner calibration. Consume with the existing Python API `cal=`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider \
  cc/host_calibrate/tests --basetemp=artifacts/codexCAL2/pytest-tmp
```

Real copied CAL GPU/CPU timings, lscpu, nvidia-smi, and H2D formats are tested;
the cache-inflated full legacy capture fails before tensor/model algebra.
Stream accounting/footprint/partial-output rejection and complete handoff
algebra also have explicitly synthetic tests. Native gcc CPU compilation and
execution are recorded under `artifacts/codexCAL2/`; this is not a tools-image
build. Docker socket access is denied in the author environment.

**UNVERIFIED:** CUDA stream compilation/execution/output parser (schema tested
synthetically only); tools-image gcc/libgomp execution; Jetson identity parsing;
manual DDR inventory accuracy; new GPU-host stream-to-handoff full calibration.
The optional upstream literal GB/s perf parser branch remains unverified by a
real capture (actual matrix outputs print FLOP/s; parser computes weight BW).
No synthetic output is presented as a measured host calibration.
