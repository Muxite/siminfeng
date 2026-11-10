# Host calibration (T-CAL)

Offline planner/parser and operator-run probes for llama.cpp pin
`31288c9c66e59377f7788d920678a565bb89dd37`. All generated files belong under
`artifacts/codexCAL/`. No Docker/GPU calls were executed by the author.
Existing planner files and their shared in-progress changes were left untouched.

`calibrate.py plan` emits **a JSON list of Docker argv**, plus a named manifest.
`calibrate.py parse` separately reads captured stdout/status and emits
`host-calibration/1`: decimal GB/s constants, raw samples and output text,
image/pin provenance, CPU/GPU identity text, observed uncertainty and caveats.
No fabricated calibration JSON is supplied as a host measurement.

## Exact operator commands (Opus, GPU host)

From the campaign root, set `RUN_REF` to the immutable **release engine** image
reference. It must contain `/app/test-backend-ops`, `/app/llama-bench`, `timeout`,
`nvidia-smi` and `lscpu`; missing tools fail explicitly. Do not silently substitute
another engine image. The pinned factory recipe excludes tests by default, so
check the release packaging before running. The tools image is only for `nvcc`.

```bash
export PYTHONDONTWRITEBYTECODE=1
export RUN_REF='sha256:REPLACE_WITH_RELEASE_ENGINE_RUN_REF'
export MODEL='/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf'
python3 -B cc/host_calibrate/calibrate.py plan \
  --image "$RUN_REF" --model "$MODEL" --cpu-layers 0 12 \
  --out artifacts/codexCAL/gpu > artifacts/codexCAL/command-plan.json
python3 -B cc/host_calibrate/run_plan.py artifacts/codexCAL/gpu
python3 -B - <<'PY'
import json
from pathlib import Path
# Actual first-K layer tensor headers; normalized stats already used by T-PL4.
s = json.loads(Path('artifacts/codexPL4/model-stats.json').read_text())['mixE4K']
Path('artifacts/codexCAL/model-stats.json').write_text(json.dumps(s, indent=2)+'\n')
PY
python3 -B cc/host_calibrate/calibrate.py parse \
  --capture artifacts/codexCAL/gpu --stats artifacts/codexCAL/model-stats.json \
  --out artifacts/codexCAL/host-calibration.json
python3 -B cc/host_calibrate/apply.py artifacts/codexCAL/host-calibration.json \
  --out artifacts/codexCAL/planner-calibration.json
```

The supplied stats must describe the supplied model, including the exact
first-K expert types and byte totals. Use another pair of K values only with
matching actual tensor stats. Do not use a large production model: planning
rejects model files over 1 GiB. The provided Granite mixE4K is ~780 MiB;
bench generates only 64 tokens with batch/ubatch 64, leaving more than 1 GiB
for small KV, compute and CUDA overhead within the 2 GiB allocation envelope.
The source shapes use at most ~204 MiB weight banks plus ~768 MiB transient
float initialization; dense screens use <60 MiB weights. H2D explicitly
allocates at most 256 MiB host + 256 MiB device. Containers have a 2 GiB host
cgroup cap, no swap allowance beyond that cap, read-only rootfs, no network,
read-only inputs, and only the build artifact directory mounted writable.
These are bounded workloads, not a claim of a hard physical VRAM partition
(the D021 interposer also cannot account for driver/context allocations).
Planning also reads GGUF headers and rejects architectures outside the small
Granite 1B/A400M shape (24 layers, 32/8 experts, embedding1024/expert FFN512).
Arbitrary architectures of the same file size have not had their compute
allocations qualified and are refused.

Each probe has an in-container 25 s timeout with a 1 s kill grace. The optional
runner has an outer 29 s deadline plus a 1 s container-removal deadline; it
records failures and stops rather than parsing partial results. All probes run
serially. Expected warm-path total is about 30–60 s: 16 matrix cases at at
least 1 s each, two tg64/r2 benches, a short copy sweep and build/startup.
Cold image pulls/model page faults and slow CPUs can exceed 60 s; this is a
budget target, not a measured runtime. No global 60 s deadline truncates data.
No sudo, host settings, package installation, or memory-filling screens.

The exact build argv is also in the emitted plan:
`docker run ... campaign/cuda-tools:c1 timeout -k 1s 25s nvcc -O2 /src/h2d.cu -o /out/h2d`.
The CUDA source uses a pinned-memory `cudaMemcpyAsync` sweep of 1, 2, 4, 8,
16, 32, 64, 128, 256 MiB, a warmup and three repeats with eight copies per
CUDA-event timing window. `ms` is time per copy. The constant is the median
of the nine samples at 64–256 MiB; all 27 samples remain in the report.

## Actual pinned output and bandwidth semantics

`artifacts/codexCAL/test-backend-ops-pinned.cpp` was extracted with
`git -C ref/llama.cpp show 31288c9c6:tests/test-backend-ops.cpp`.
Relevant source: `print_perf_console` lines 987–1027; `eval_perf` lines
1580–1719; cases lines 12303–12335; CLI lines 13059–13123.
The literal printf statements include:

```cpp
printf("    %8d runs - %8.2f us/run - ", result.n_runs, result.time_us);
printf("%s/run - \033[1;34m%sS\033[0m", format_flops(op_flops_per_run).c_str(),
       format_flops(result.flops).c_str());
printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m", result.memory_kb, result.bandwidth_gb_s);
```

**The requested MUL_MAT and MUL_MAT_ID do not print GB/s at this pin.** They
have nonzero `op_flops`, print FLOP/s, and even SQL's `bandwidth_gb_s` is zero.
CSV omits performance fields entirely. Inventing a matrix GB/s output format
would be wrong. The parser accepts the actual ANSI console rows and calculates
active compressed weight bytes / measured microseconds / 1000, in decimal GB/s.
It also extracts literal GB/s when present; upstream's label actually means
GiB/s (source divides by 1024 cubed), so the report retains both original and
converted decimal values. That literal branch is **UNVERIFIED** by a real capture.

Dense screens use m4096/k14336/n1; expert screens use m768/k2048/n1,
128 experts/8 selected. Quant blocks (bytes per block): Q2_K 84/256,
Q3_K 110/256, Q4_K 144/256, Q6_K 210/256, Q8_0 34/32.
Expert bandwidth counts selected experts, not the entire bank.
The pin's expert perf cases omit Q2_K/Q3_K; those types explicitly use dense
CPU GEMV as a proxy. Q4_K/Q6_K/Q8_0 CPU constants use expert results. GPU
constant uses Q4_K dense results; other GPU types and expert screens are raw
diagnostics. These operations include dispatch and dequantization, and repeated
weights may hit cache. Do not label them hardware DRAM bandwidth or validated
whole-model coefficients. Raw H2D excludes the cache dispatch captured in T-PL4's
fitted H2D term (D041); cache miss latency remains unmeasured.

## Handoff algebra and uncertainty

Using `decode_model.traffic` on actual tensor headers at K0 and K1:

```text
t(K) = G(K)/(1000*BWgpu) + sum_q Cq(K)/(1000*BWcpu,q) + K*h
h = [median(t1) - median(t0)
     - sum_q (Cq1-Cq0)/(1000*BWcpu,q)
     - (G1-G0)/(1000*BWgpu)] / (K1-K0)
```

Times are microseconds per token (`1e6 / samples_ts`), bytes are active
bytes/token. The GPU delta is negative as experts move to CPU. Omitting that
term misattributes removed GPU work to handoff. Both bench runs use the same
image, model, batch, threads and token count. The implementation takes the
median of pairwise repeat residuals (equivalent for the default two repeats).
Negative handoff fails calibration rather than being silently clamped.
It indicates an incompatible/noisy bandwidth model or insufficient signal.

Uncertainty includes count, observed range and sample standard deviation for
bandwidth, plus pairwise residual handoff ranges. Single case estimates have
null standard deviation. Pairwise residuals are correlated, not independent
measurements or confidence intervals. Systematic model error, shared-host load,
placement-dependent numeric routes, attention and cache residency are not
bounded by those ranges. This is a per-host initial screen, not D039/D041
cross-model validation. Failures and unavailable identity information do not
become zero measurements (consistent with resource_observe's contract).

## Apply to planner without changing planner files

`apply.py` writes a complete independent calibration object with
`physical_decode.constants` replaced. It drops stale CPU screen values and
clears old physical fit evaluation/fold summaries. It retains historical memory,
prefill, quality and host applicability rules; these were not recalibrated.
Consume it with the planner's existing **Python API** `cal=` argument:

```python
import json, sys
sys.path.insert(0, 'cc/moe_planner')
import plan
cal = json.load(open('artifacts/codexCAL/planner-calibration.json'))
result = plan.plan(stats, host, budget, cal=cal, allow_extrapolation=True)
```

Or use `decode_model.predict(stats, family, coord,
cal['physical_decode']['constants'])` directly. There is no existing planner
CLI calibration-file flag. The API suffices, so no write outside the two
assignment roots or new planner hook was needed. `--out` checks resolve symlinks
and prohibit escaping `artifacts/codexCAL/`.

## Verification status and tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider \
  cc/host_calibrate/tests --basetemp=artifacts/codexCAL/pytest-tmp
```

* **Verified real formats:** MUL_MAT and MUL_MAT_ID console timing parsers
  (actual CPU outputs captured here from the existing pinned-source build);
  llama-bench JSON parser (copied unchanged T-PL4 tg32 CPU output, tested with
  n_gen32); lscpu identity text (actual capture here).
* **UNVERIFIED:** CUDA backend execution/timing, literal GB/s branch, H2D
  CUDA compilation/output parser (only explicit synthetic sweep test),
  nvidia-smi successful device identity (retained verbatim), GPU tg64 paired
  handoff estimation and total wall-clock target. The pipeline/algebra test
  uses explicitly synthetic times; it does not claim a real handoff sample.

Real fixtures are in `tests/fixtures/`, raw new captures/stderr and source in
`artifacts/codexCAL/`; `capture-provenance.json` records paths, commands and
hashes. Captured CPU numbers are shared-host observations used to validate
format, not a GPU host-calibration result. No synthetic success is passed off
as an instrument measurement.
