# Inference-engine experiment methods — jetson-arm

Read with CLOSEOUT.md (all, especially §§5–6), `c2/PLAN.md`, and the cited
Jetson D-notes. Qualification below distinguishes protocol tests, compilation,
live execution and scored engine evidence; one does not imply the others.

## Lab map

- **jetson-arm = satay**: Orin Nano 8GB Super, sm_87, six Cortex-A78AE cores,
  7.4 GiB shared DRAM, L4T 36.5.2 / JetPack 6.2.3 / CUDA 12.6.68.
  CPU, GPU and page cache compete for the same physical memory (UMA).
  NVML memory is N/A; declare memory domain `shared-dram`, not VRAM.
- GPU devfreq is unpinned, nvhost_podgov 306–1020 MHz. Keep power/clocks
  unchanged; warm up, counterbalance, measure A/A and retain frequency samples.
- Six zram devices total about 3.8 GiB swap. Swap bursts can be benign;
  monitor available DRAM and cumulative swap as well as swap rate.
- X15 Lite SSD 128GB, PCIe Gen3 x4: about 536–573 MiB/s at large queued reads,
  only 19.6 MiB/s at random 4 KiB qd1. Access granularity matters (D031/D033).
- Native builds and engine runs execute here with target `jetson-orin`.
  **x86-main** is the separate discrete RTX 5070 Ti / sm_120 lane; default Build
  CC target `x86-5070ti` belongs there. No SSH transport between lanes;
  operator-managed git branches exchange recipes/evidence. Do not pool t/s.
- Sources: ref/llama.cpp @ 31288c9c66e59377f7788d920678a565bb89dd37;
  ref/llama.cpp-up1108 @ cf34d311332e362198f0a410c466d1e940287d02.
  Keep reference trees read-only; patch staged snapshots or separate copies.
- Models: ~/models/campaign/gguf (Granite), ~/models/campaign/qwen36-35b-a3b
  (22,360,456,160-byte Qwen), data and kl subdirectories for corpus/base logits.
  CLOSEOUT §§1.4/1.6 contain hashes/download revisions; some hashes remain missing.
- Raw engine results: `experiments/jetson/`; build records/archives:
  `cc/build_engine/results/`; operation logs: artifacts/op-* and
  evidence/jetson-arm/*.jsonl; lab attempts: lab/jobs; queue state: lab/jobq.
  Worker logs/finals: artifacts/codex; work ledger: `c2/worklog.jsonl`.
  Models, Docker images, ccache and build trees are local-only, not a remote backup.

## Tool catalog

Qualification is evidence already recorded, not a fresh run of the tool.
All top-level bin tools and cc capabilities are included; helpers/tests/patches
belong to their capability row. UNTESTED means no qualified Jetson engine result.

| Tool | Path | Purpose | Inputs / outputs | Qualified by | Known limits |
|---|---|---|---|---|---|
| jobq | `bin/jobq` | FIFO/dependency scheduling, class locks | argv/deps → job JSON, events, logs | H1: 10 tests + smoke; c2/worklog.jsonl check-in | Only queued commands protected; worker required |
| run_op | `bin/run_op` | Recorded command lifecycle | argv → intent/started/outcome + hashed stdout/stderr | CLOSEOUT §2 J-EV1; Jetson D021 | Survivors are observed, not automatically all cleaned; env allowlist only |
| cc_hook | `bin/cc_hook` | Claude hook evidence sink | stdin JSON → per-host JSONL | CLOSEOUT §5.1 live use | Fail-open logging; no billing/token accounting |
| lab | `bin/lab` | Image/admission, jobs, attempts, reconcile/register | image/model/test policies → lab state/results | CLOSEOUT §§5.7, 2 Jetson lab jobs; D023 | No GPU scheduling lock; use jobq; older policies may omit context |
| lab_run | `bin/lab_run` | Legacy request-attempt runner | request JSON + attempt → states/results | Source only; no Jetson qualification located | Legacy x86 Docker/NVML assumptions; prefer lab + jobq |
| perf_summary | `bin/perf_summary` | Attempt/sample spreads | lab job IDs → terminal summary | CLOSEOUT §5.1 live use | No A/A-aware gate; prefer abrun for OM |
| index_evidence | `bin/index_evidence` | Merge raw ops | evidence JSONL → index/ops.jsonl | CLOSEOUT logging workflow; source | Missing outcome stays unknown; not recovery |
| cost_report | `bin/cost_report` | Recorded wall/token accounting | index + Codex streams → report | Source; no standalone qualification recorded | Overlap double-counts wall; excludes Claude billing |
| ledger_appendix | `bin/ledger_appendix.py` | Evidence tables | build/experiment records → HTML appendix | CLOSEOUT §5.1 use | Campaign-specific parser, not an acceptance gate |
| cuda_compile_check | `bin/cuda_compile_check` | Patch compile screen | patch → temp source, .cu objects/logs | Source; UNTESTED on Jetson | Hardcoded sm_120 and x86 image; do not use unchanged here |
| abrun | `cc/abrun/` | Preflight, serial counterbalanced run, summarize/gate | spec JSON → summary.json, verdict.json, REPORT.md | H2: 31 tests; c2/worklog.jsonl real OM7 replay check | om_run only; dry-run skips preflight; hashes are byte equality |
| build_engine | `cc/build_engine/` | Isolated build and runtime image | clean commit + patch → image/archive/build-record | `cc/build_engine/results/jetson-c1-rebuild/build-record.json`; D026 | Not bit reproducible; build success is not correctness |
| engine_handoff | `cc/engine_handoff/` | Offline receipt/archive inspection | archive + requirements → observations/receipt | `cc/engine_handoff/example-real5/`; fixture tests in directory | No load/run; executable regular files only; not ABI proof |
| resource_observe | `cc/resource_observe/` | Scoped telemetry + watchdog | backend/PID/path/limits → JSONL, summary | H1 34 tests; `experiments/jetson/OM/watchdog-killtest/` | Sampling misses peaks; cgroups omit Tegra GPU pages |
| om_run | `cc/om_run/om_run.sh` | Guarded owned-container run | OUT IMAGE ENV_FLAGS TOOL args → logs/result/telemetry | CLOSEOUT 48 runs; live killtest above; H1 classifier checks | Fixed model/data mounts; OM_MAX_S bounds sampler, not workload |
| OM legacy pipelines | `cc/om_run/pipeline*.sh` | Historical orchestration | hardcoded arms → run dirs | CLOSEOUT §5.6, pipeline4 deadlock | RETIRED pipeline2–6: superseded by jobq/abrun |
| build_queue | `cc/om_run/build_queue.sh` | Historical build chain | hardcoded builds → logs | CLOSEOUT PR31469 dropped/no outcome | RETIRED: use jobq build dependencies |
| OM sweeps | `cc/om_run/sweep_*.sh` | Historical baselines/version/trace/cache runs | fixed recipes → OM evidence | `experiments/jetson/OM/`; D030–D033 | Reuse recipes through jobq/abrun, not scheduling loops |
| fio_ceiling | `cc/om_run/fio_ceiling.sh` | Read-only drive ceiling | model file → fio results | `experiments/jetson/OM/fio/`; D031 | Device-specific; legacy pgrep wait must be replaced with jobq before reuse |
| om_spec | `cc/om_spec/om_spec.py` | Guarded server speculation comparison | OUT IMAGE mode round → requests/IO/text | `experiments/jetson/SPEC2-20261110-1634/` | Copy-heavy only; original raw chat arm invalid; different harness from completion |
| kernel_screen | `cc/kernel_screen/` | Per-op/type screen with A/A | binary/image + preset + arms → JSON/CSV/report | `experiments/jetson/KS-20261110-1620/` | Kernel-local only; graph-fusion prediction failed; q4_K arm unrun |
| routing_trace | `cc/routing_trace/` | Router IDs/weights capture + validation | model/prompt → metadata/rows/footer | 56 tests + 5 subtests (README); `experiments/jetson/OM/trace-20261110-0504/` live captures | Protocol tests do not validate C++ on every architecture; legacy replay Granite-only |
| expert_replay | `cc/expert_replay/` | Cache/profile/spec-window simulation | validated traces, capacity, bytes/BW → JSON/CSV | CLOSEOUT 10 original tests; SPEC1 14 + 5 subtests; `experiments/jetson/SPEC1/` | Simulation, not speed evidence; expert slots differ from page cache |
| model_manifest | `cc/model_manifest/` | GGUF hashes/layout | GGUF → manifest/layout JSON | `experiments/jetson/bf16-gguf-manifest.json`; `experiments/jetson/OM/qwen36-layout.json` | Whole-file hash may differ with equal tensors; compare both |
| memory_pareto | `cc/memory_pareto/` | Memory/quality/speed frontier | CSV or lab attempts → Markdown/JSON | `cc/memory_pareto/test_pareto.py`; saved pareto.json | No significance/equivalence; fallback unclassified; memory domains differ |
| hybrid_expert | `cc/hybrid_expert/` | Real-weight expert placement microbench | Granite weights + routes → latency/numerics | `cc/hybrid_expert/RESULTS.md`; D019 | Component, not full engine; discrete placement model |
| moe_dispatch_sm87 | `cc/moe_dispatch_sm87/` | Per-type gate/up fusion switches | staged patch + env → engine | D025/D027; `experiments/jetson/J-E007-unfuse-by-type/` | Orin Granite batch1 only; q8_0 regresses |
| moe_cache_profile | `cc/moe_cache_profile/` | Profile pin/dedupe/direct-fill candidates | profile + patch → counters/engine | `experiments/jetson/OM/mc-20261110-0602/`; D032 | Negative on UMA; direct-fill inactive; changed output lacks KL |
| mmap_advice | `cc/mmap_advice/` | Readahead and router-prefetch hooks | patches + env → activation/stats | D033; `experiments/jetson/OM/om7-20261110-1555/` | Whole-file random halves speed; zero-lead prefetch -38% |
| expert_fill | `cc/expert_fill/` | Synchronous mincore + buffered parallel pread | slice ranges → resident pages/counters | `cc/expert_fill/COMPILE-CHECK.txt`; c2/worklog.jsonl J-OM8 check-in | Unit/syntax validation only; full-engine efficacy UNTESTED |
| q8_reuse | `cc/q8_reuse/` | Activation reuse candidate patch | patch/env → engine | README apply + host-mock checks | UNTESTED CUDA/Jetson engine; D012 demoted speed-only exercise |
| gpu_budget | `cc/gpu_budget/` | Cooperative CUDA interposer budget | build/run scripts + allocation policy → probe | D023/D028 identify candidate | UNTESTED on Jetson; not a physical isolation guarantee |
| smoke | `cc/smoke/` | CUDA runtime/allocation probes | CUDA binaries → success/allocation logs | D022/D028 live probes | Not inference/quality; cgroup cap unreliable |

## Methods

### Counterbalanced A/B and A/A drift gate

Use for causal speed claims. Fix model hash, prompt, context, batch, offload,
seed, toolchain and image; prefer one image with env-only arms. Write a spec per
`cc/abrun/README.md`: base plus candidate, ABBA; two candidates, ABCCBA;
more arms, mirrored or explicit order. Repeat whole rounds, record n per arm.
For llama-bench, keep all samples and use `tg_steady` (mean excluding sample 1)
when claiming steady decode; `eval_tps` otherwise uses benchmark avg_ts.
D025/D027 show why warmup and per-type qualification matter under floating clocks.

Read `cc/abrun/gate.py`, verdict.json and REPORT.md. Its rule verbatim is:

    threshold = max(0.03, 2 * aa_gap, 2 * (drift or 0.0))

Equivalently `max(0.03, 2*aa_gap, 2*drift)` for measured drift. aa_gap is the
largest within-arm (max−min)/mean; drift is the absolute difference of first/
second base-half means divided by full base mean (odd middle goes in second).
B/A uses arithmetic arm means, NOT a median per-round ratio. Improvement requires
strict excess above threshold; equality is null. Missing/nonpositive/nonfinite
metrics or unsuccessful runs invalidate scoring. Unmeasured gaps contribute
zero but remain null in detail: lack of replication is not stability evidence.
Accept only valid, uncontaminated runs plus quality/claim gates below; do not
replace the implementation with the older CLOSEOUT proposed median rule.

### Switch-active verification

Use for every env/flag-gated component. Identify an activation log/counter at
its actual consumer, add `switch_regex`, mark candidate `expect_switch: true`,
and keep base inactive. Inspect startup AND exit counters, including fallback.
Accept only active candidate/inactive base with work performed. abrun aborts on
activation mismatch. D032: `--numa distribute` was inert on one NUMA node;
direct-fill never ran because offsets were unaligned. D033 advice_log proved
real activation. A successful process or changed environment alone proves nothing.

### Flag preflight

Before queueing expensive runs, inspect the exact image/tool help with
`cc/abrun/preflight.py`; abrun automatically preflights before measurement.
Dry-run only checks planning. Help cache keys image string + tool, so preserve
immutable image identity and avoid stale help when retagging an image.
Accept only flags supported by that binary; use separate short flags.
D029/D030 and CLOSEOUT §5.2 record 11 flag failures: llama-bench lacks `-c`,
`-cmoe`, `-mmp` in these builds. Completion supports context/CPU-MoE options;
bench uses `-ncmoe`. Help support does not prove a switch reaches its code path.

### Decode-path quality and output class

For decode-kernel changes, save base logits per model/reference and run paired
llama-perplexity with `-c 128 --chunks 8 -b 128 -ub 1` and
`--kl-divergence-base` for capture, then `--kl-divergence` for comparison.
Reuse corpus/chunks/tokenization; also run A/A, PPL and fixed greedy text.
Record mean/p99/max KL, top-1, reference identity and text hashes. Set the
acceptable quality budget before measurement; compare to A/A storage floor.
D025: batch-512 ppl was blind to decode kernels while ub1 KL measured 2.7e-4;
greedy text identity did not make bf16 activation rounding bit-exact.
Evidence recipes: `experiments/jetson/J-E005-patch-ab/`.

Classify **identical** (measured output bytes equal), **differs** (numerical/text
change needing paired quality evaluation), or **lossy** (intentional accuracy
trade-off such as quantization, with its quality cost). abrun hashes only emit
identical/differs/unknown; lossy is a reviewed scientific classification.
Accept no computation-preserving claim from hashes alone, and no quality claim
from unknown hashes. D027 preserves measured quality within a quantized model;
that model is still lossy against bf16. D032 cache text changes remain ungated.

### Over-memory fair baseline and guarded execution

Use when weights exceed physical memory. Invoke `cc/om_run/om_run.sh` as
OUT IMAGE "ENV_FLAGS" TOOL ARGS; it mounts OM_MODELDIR (default Qwen directory)
read-only at /om and records cmd, result, stdout/stderr, watchdog and telemetry.
Use explicit `-lm mmap -ngl 99 -cmoe` for completion, or `-ncmoe 40` for bench.
Pin `-c` for context-capable tools (D023); bench lacks that flag, so record its
actual workload/context behavior rather than passing it blindly.
Compare like-for-like prompt/tool/settings. Report the failing auto default
separately: D030 auto disables mmap on iGPU and runs out of headroom.
Accept status ok, activation, quality and A/A gates, with NVMe GiB/token,
avg KiB/IO, available-memory minimum and swap maximum; host safety is physical,
not a Docker memory budget (D028). Current defaults: OM_MIN_AVAIL_MIB=600,
OM_MAX_SWAP_MIB=2560, OM_MAX_SWAP_RATE=1000; two consecutive crossings at 250 ms.
Header examples are stale. OM_MAX_S=1800 limits telemetry sampling only; give
jobq an explicit timeout for a workload deadline. Distinguish watchdog trip
survived from trip killed, OOM and ordinary failure in result.json (H1).

### Contamination and ownership

Submit one whole experiment via abrun `--submit` (gpu), builds via jobq build.
A worker must run; use job IDs/dependencies and wait/show/cancel (see
`bin/README-jobq.md`). GPU and build classes mutually exclude; CPU jobs exclude
GPU unless explicitly allowed, and can overlap builds. Never opt a heavy replay
into timed GPU work. Check concurrent_jobs AND external processes: unqueued
work bypasses locks. D029 build overlap and D030/D031 replay overlap make timing
**UNSCORED**, even at nice 19; retain usable counters with that qualification.
Accept scored timings only without competing GPU/build/CPU load. Cancel only
owned job process groups/containers; retries use fresh output/attempt identities.

### Profiling and kernel screening

Use nsys decode inventories to identify hypotheses, not to score engine gains.
Compare kernel names/counts/time under matched workloads; J-P001 refuted launch
overhead and J-P002 explained MMVF → MMF. Follow with untraced full-engine A/B.
Use `cc/kernel_screen/README.md` for kernel-local changes: build test-backend-ops
with perf presets, use matched A/A2/B arms, exact shapes/types and repetitions.
Accept matched nonempty cases above A/A noise plus later model-level validation.
Use exact `--types q4_K`; old README uppercase example fails on this binary.
`experiments/jetson/KS-20261110-1620/` did NOT predict D027 graph-fusion effects:
single-op results cannot establish fusion/launch/graph behavior. Use an actual
fused graph or model experiment for that question. ncu counters are blocked;
no admin workaround is part of this procedure (D004; CLOSEOUT §5.4).

### Routing trace, replay and first-order IO model

Use `cc/routing_trace/README.md` to capture routes, then run
`cc/routing_trace/validate_generic_trace.py` before replay. Require successful
capture, metadata dimensions, completion footer/counts and full expected token/
layer coverage. Legacy Granite trace_replay is not the generic reader.
Capture callbacks synchronize/read back tensors and may inhibit fusion: these
are routing observations, never performance-neutral timing runs.
Use `cc/expert_replay/README.md` for cold-cache capacity sweeps, leave-one-out
profiles, and expert-byte mapping. Record global versus per-layer slots, phases,
warmup, train/test provenance and malformed/truncation diagnostics. Accept only
complete independent inputs; label all predicted speed **simulation**. Belady
is a future-aware bound, not an online policy. D012/D014 and CLOSEOUT §3.D:
static cross-workload sets got 16.6% hits versus LRU 62.4%; hit rate is not speed.

D031 first-order model: decode_tps ≈ disk_BW /
(routed_expert_bytes_per_token × miss_rate). Qwen uses about 615 MB/token;
570 MiB/s with miss=1 gives about 1 t/s. State units and assumptions: bytes held
once, effective large-read bandwidth, no additional compute/copy bottleneck.
Map slot capacity to bytes explicitly. Accept as a bounded prediction only;
validate separately with IO counters and clean engine A/B. D032 did not realize
projected cache gains; D033 reduced bytes 57% yet halved speed with 4 KiB IO.

### Claim classes and build identity

Follow `decisions/D012-objective-memory-footprint.md`: report memory footprint
(weights + KV + buffers), paired quality, prefill/decode t/s and memory domain as
a Pareto trade-off. Label full-engine improvement, quantization/offload trade-off,
component benchmark, simulation, or negative/null result (CLOSEOUT §3).
A dependency-version gain is not our code gain; the 500GB→128GB scale ratio is
an analogy. Routing/byte/quality observations may transfer; discrete PCIe timing
and placement do not transfer to UMA. D028 supersedes D012's proposed cgroup SSD
emulation on Tegra. State host/model/type/batch and untested platforms.
For builds, use `cc/build_engine/README.md`: clean identified source, separate
patch, fresh output, target and toolchain recorded. D026 identity is commit +
patch SHA + toolchain + base digest + build record/archive SHA, not binary hash.
Accept a build only with successful record, then separate runtime/quality checks.

## Profilers, tracers and telemetry

`cc/resource_observe/README.md` defines metric scopes and unavailable reasons.
Use capabilities before choosing backends; om_run samples io/proc/tegrastats at
500 ms. Summarize with coverage/gaps; never add overlapping attribution scopes.

| Backend/instrument | Metric names / output | Question answered and caveat |
|---|---|---|
| tegrastats | ram_used, ram_total, ram_lfb, swap_used, gpu_util, gpu_freq, emc_freq, cpuN_load/freq, temp_*, power_* | Whole shared DRAM, clock ramp, thermal/power context; one record per emitted line |
| proc | mem_available, mem_total, swap_total/free, pid_rss, pid_hwm, cgroup_current/peak/max | Headroom/process/cgroup views; subsets overlap system RAM; GPU pages may escape cgroup |
| sysfs_tegra | gpu_freq_cur/min/max, temp_*, nvpmodel_mode | Actual devfreq range/thermal/power mode; queries only, missing permissions reported |
| nvml | gpu_mem_used, gpu_mem_total, gpu_util and device fields (see source GPU_FIELDS) | Discrete VRAM on x86; Orin reports unavailable/N/A, not zero |
| io | device_read, device_read_ios, device_io_ticks, device_util; pgmajfault, workingset_refault_file, pswpin/out, pgpgin/out; pid_read_bytes; cgroup_rbytes_* | Disk bytes/IO size/refault/swap evidence; device counts include unrelated work |
| watchdog | available_mib, swap_used_mib, swap_rate_mibs; watchdog_trip/action | Safety thresholds and action evidence; trip alone does not prove kill |
| nsys / routing tracer | kernel inventory / selected expert IDs and weights | Mechanism / locality; both perturb execution, so score elsewhere |

Counters have _delta/_rate companions; first sample/reset invalidates rates.
Read sectors are fixed 512 bytes; normalize recorded units before arithmetic.
abrun derives nvme_gib, avg_kib_per_io, nvme_gib_per_token,
mem_available_min_mib, swap_max_mib and gpu_freq_median. Counter deltas include
load/prefill/decode across the capture, not automatically decode-only traffic.
Report capture coverage and window. D031's 825 MiB/s sampled peak exceeded the
536–573 MiB/s fio ceiling and is likely an interval artefact, not a faster drive.
Device_util measures busy time, not queue saturation. Missing metrics are null
with reasons, never zero; sampled high-water is not an allocation guarantee.

## Delegation protocol

Use `assignments/H1-jobq.md` and `assignments/J-OM8-expert-pread-fill.md` as
models. Follow `c2/PLAN.md`: bounded implementation to Codex, decisions to lead,
cheap Haiku acceptance check-in. Assignment structure:

1. **Read-first**: repo/host/pin and exact prerequisite files.
2. **Why**: observed failure, hypothesis and discriminating experiment.
3. **Deliverables**: exact files/interfaces, output identity and expected behavior.
4. **Acceptance commands**: executable criteria A1…An with expected outcomes.
5. **Constraints**: writable paths, no evidence/ref edits, GPU/build/network limits,
   time budget and no commit unless assigned.
6. **Final message**: line budget, files, validation, deviations and risks.

Launch through `c2/launch_codex.sh` TASK ASSIGNMENT_FILE [EXTRA_PROMPT]; when GPU
jobs may queue, submit launcher as jobq **cpu** without --allow-during-gpu.
This prevents agent CPU work from contaminating timing; GPU/build work inside
an assignment needs its own explicit class scheduling and permission.
Check-in prompt: “Read ASSIGNMENT. Run A1…An verbatim in the specified cwd;
report PASS/FAIL per criterion, exact output/counts and deviations. Spot-check
referenced evidence, activation and claim scope. Do not fix or run unassigned
GPU/build work.” Lead reviews failures and assigns bounded rework; checker
acceptance is not proof beyond what the commands/spot checks cover.

`c2/worklog.jsonl` is append-only events keyed by task: start (worker, assignment,
ts); end (rc, wall_s, tokens, ts); checkin (checker, verdict, issues,
rework_rounds, failure_class, ts). Preserve start/end timestamps, elapsed wall,
measured token usage (unknown if absent), checker verdict, rework count and
failure classification; add lead-fix attribution when relevant. Launcher writes
start/end, lead/checker supplies checkin; do not invent missing billing data.

## Anti-patterns and operational gotchas

- Avoid whole-file MADV_RANDOM (D033), cross-workload static hot sets, separate
  GPU expert banks on UMA (D030/D032), or interpreting hit rate as speed.
- Do not use cgroup/--memory as a Tegra CUDA/shared-DRAM cap (D028), NUMA on a
  single node as readahead control, batch-512 ppl as decode qualification, or
  Q8_0 gate/up unfusing on Orin (D027). Keep negative/null results.
- Do not compare raw t/s across hosts or binary hashes as build equivalence.
- Never wait/kill by pgrep -f/pkill -f shell patterns: self-match caused 8.7 h
  idle and killed a closeout shell. Use job IDs/PIDs and owned process groups.
- sudo needs the owner's password; agents cannot supply it. No unassigned
  clocks, power, drivers, runtime configuration or root profiler changes.
- Older shells lack new docker group membership: use DOCKER_CMD_PREFIX='sg
  docker -c' where supported. Keep the entire Docker command one quoted argument;
  D029's unquoted nested kill command ran Docker help instead of killing.
- Host binaries may retain RUNPATH to deleted staging: set LD_LIBRARY_PATH to
  actual retained libraries. Images require --runtime nvidia even for CPU tools;
  generic arm64 images need NVIDIA_VISIBLE_DEVICES=all and
  NVIDIA_DRIVER_CAPABILITIES=compute,utility (D022).
- System numpy is broken here; converters use ~/venvs/convert; stdlib tooling
  avoids that dependency. Strata's x86-only Nix flake is not a Jetson build path.
- nsys stale SQLite exports can yield zero-byte CSV: use --force-export=true and
  inspect rows. Do not hide command exit status behind a pipe to tail.
- D024: never stage a live build directory. Keep archives/build/staging outputs
  gitignored, inspect staged stats before any authorized commit, retain recipes
  and identity records. Retrying means new output paths, not overwritten evidence.
- Test fixtures must be captured from real tool output, never hand-written. kernel_screen's fixture used `q4_k`
  (real ggml name `q4_K`), so tests passed while every q4_K screen matched zero cases (KS-20261110-1620,
  KS-q4K-20261111; c2/worklog.jsonl). Same rule for log-line regexes used as switch-active checks.
- Any worker not started through jobq (Codex, check-in agents) is invisible to the contamination rule: J-OM8 run
  001-base overlapped two Codex sessions. Launch workers as jobq `cpu` jobs (c2/codex_job.sh) and run check-ins
  only when no gpu job is running.

### H5: precise time and phase measurements (2026-11-11)

New abrun specs default to printed `eval_ms_per_token`, direction `lower` (A/B
ratio); old explicitly declared t/s specs retain their metric. Regating reads
stderr and telemetry again; `regate.py OUTDIR --dry` prints a table without writes.
The threshold and A/A/drift rules remain unchanged. t/s remains descriptive.

Relative engine log timestamps have no host clock origin in the old captures.
Decode is therefore estimated on telemetry mono_s by anchoring its end at the
last sample before exit and subtracting printed eval_ms_total. Estimated load
starts at first telemetry and lasts load_ms. The maximum observed per-backend
sampling interval bounds endpoint quantization only if sampling continued through
exit. Unknown startup/teardown lag means no finite absolute alignment bound can
be claimed; summaries explicitly report this unknown bound. These elapsed windows
cannot resolve individual tokens, overlap, or first-token work charged to prefill.

Read counters are interpolated at window boundaries (no extrapolation; resets
invalidate the phase delta), excluding model load from decode GiB/token and
using eval_runs as denominator. Instantaneous VDD_IN is integrated trapezoidally
and converted from mW seconds to J/token; this is system energy. Sequential backend
emission clips integration ends to each metric's last available sample. Decode
samples give GPU util median/p90 (nearest rank), IO device util median, power W
median, temp_tj max, and GPU/EMC MHz median. Public devfreq *.gpu/cur_freq is
sampled in Hz without root, including when tegrastats omits its GPU frequency.
System-wide counters and coarse sampling still limit causal IO/power claims.
