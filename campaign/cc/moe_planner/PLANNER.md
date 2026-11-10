# MoE planner: hypotheses before measurement

`plan.py` is a stdlib-only, offline planner for llama.cpp `31288c9c6` and the
campaign patches. It does not compile, launch inference, reserve resources,
submit jobs, or contact services. Its recommendation is a discriminating lab
experiment, not a demonstrated engine improvement or a hard memory guarantee.
All implementation and scratch writes are confined to `cc/moe_planner/` and
`artifacts/codexPL4/`. Prior T-PL artifacts are read-only inputs. There are no changes to `bin/lab` or `lab/tests`.

## Physical decode calibration (T-PL4)

The current decode predictor is `decode_model.py`. `plan.py` retains the old
D022 fits for memory, prefill, and historical diagnostics only. Decode uses:

```text
t_decode = A_gpu / BW_gpu + sum_q(A_cpu,q / BW_cpu,q) + L_cpu * t_handoff
A_cpu,q = sum(first K layers, expert tensors of type q) bytes * n_expert_used/n_expert
A_gpu   = nonexpert_bytes + active expert bytes remaining on GPU
hybrid: A_cpu,q *= (1 - decode hot coverage), L_cpu = number of layers with cold work
LRU:    A_cpu = 0, A_gpu = all active weights, t_decode += miss_bytes/BW_h2d_effective
```

Bandwidth units are decimal GB/s. Code uses bytes / (1000 * GB/s) to obtain
microseconds. `decode_model` in each plan includes the unmodified component
times and traffic; plan-level D028/D029 factors apply afterward. Static placement
uses actual first-K layer tensors, including Qwen's mixed Q4_K/Q6_K projections,
for both active-byte timing and resident expert memory. Missing tensor headers
use equal-size layers and representation-derived expert type. Stale headers
from sibling representation enumeration are discarded when byte totals differ.
All nonexpert weights are treated as active each token; this is a logical
estimate, without an attention/KV traffic model.

The fitted constants for x86-main are:

| Constant | Value |
| --- | ---: |
| Effective GPU bandwidth | 285.227 GB/s |
| CPU Q4_K | 35.887 GB/s |
| CPU Q3_K | 34.591 GB/s |
| CPU Q2_K | 25.902 GB/s |
| CPU Q6_K | Q4_K fallback, 35.887 GB/s |
| CPU-touching-layer handoff | 136.641 us |
| LRU effective H2D, including dispatch | 9.326 GB/s |

Equivalently `BW_cpu(q) = BW_cpu(Q4_K) * efficiency(q)`, with Q3_K efficiency
0.964 and Q2_K efficiency 0.722. These are end-to-end effective coefficients,
not raw DRAM, PCIe, or GPU hardware bandwidth. An independent Q6_K CPU screen
is missing; Qwen's Q6_K projections explicitly share Q4_K efficiency.

Reproduce with `bash artifacts/codexPL4/reproduce.sh`. `fit_decode.py` needs
NumPy/SciPy; ordinary planner inference remains stdlib-only. It reads every
successful decode measurement from D022, D029, D034, D035, D039, and the D040
quiet PAP rerun: 165 rows, 138 Granite and 27 Qwen. D029 includes completion and
bench endpoints; D039 includes all eight static coordinates plus the FA/KV
control. Result SHA256/job/attempt/test provenance and traffic are recorded in
`artifacts/codexPL4/measured-points.json`; complete fitted and held-out residuals
are in `artifacts/codexPL4/fit-report.json`.

GPU inverse bandwidth is least squares against training all-GPU endpoints from
D022/D029. Remaining constants use bounded least squares in microseconds with
equal weight per job/configuration (all repeated rows retained). D029 ON and
D039 release-1 times are normalized by the existing 1.02 reuse factor. Noisy
D035 observations remain included and explicitly labeled. Residual summaries
report every raw row, while scenario tables use measured coordinate medians.

| Joint-fit residual group | Mean absolute error |
| --- | ---: |
| Granite static, including GPU endpoints | 11.13% |
| Granite LRU | 35.01% |
| Qwen static, including FA/KV control | 5.50% |

Leave-one-**model**-out training excludes every job belonging to the held-out
model, including its all-GPU anchors; it does not borrow the CPU screen.

| Training model -> held-out model | MAPE | Maximum absolute error |
| --- | ---: | ---: |
| Granite -> Qwen | 7.85% | 14.30% |
| Qwen -> Granite, all configurations | 52.82% | 156.48% |
| Qwen -> Granite, static only | 52.23% | 116.58% |

The Granite-only CPU bandwidth and handoff columns are collinear. That fold
pins handoff to the separately recorded D031 extra split cost / 12 = 169.167
us/layer; the report labels this as an assumption, since D031 also includes GPU
dispatch work. Its fitted Q4_K bandwidth is about 43 GB/s. Qwen-only training
has no GPU endpoint or Q2_K/Q3_K/LRU calibration: its GPU estimate extrapolates,
handoff reaches the zero bound, and unsupported types use declared Q4_K/12 GB/s
fallbacks. The reverse fold fails transfer. The joint fit should not be called
independent cross-model validation.

The pre-registered requests were replayed offline with their original budgets,
component sets and statistics. Full plan outputs and exact flags are in
`artifacts/codexPL4/plan-qwen-{12000,8000,4000}.json` and
`plan-granite-900.json`; `scenario-comparison.json` records all comparisons.

| Request | Selected CPU layers | New t/s | Measured t/s | Error |
| --- | ---: | ---: | ---: | ---: |
| Qwen 12000 MiB | 18 | 51.247 | 54.62 | -6.17% |
| Qwen 8000 MiB | 30 | 36.342 | 36.58 | -0.65% |
| Qwen 4000 MiB | 42 | 28.154 | 27.73 | +1.53% |
| Granite 900 MiB | 19 | 143.925 | 139.64 | +3.07% |

The previous Qwen predictions were 8.069, 4.863 and 3.565 t/s. Actual first-K
bytes move the 4000 MiB winner from K41 to K42. K41 has no measurement; its
new physical prediction is retained separately without claiming an exact match.
Qwen's selected q4_0-KV/FA/ubatch128 configurations were not measured exactly by
D039: comparisons use same-K f16/default-batch measurements. D039 measured
GPU deltas can exceed the nominal budgets; the transferred Granite memory guard
is not a Qwen memory guarantee. Granite D034 K19..23 errors are +3.07%, -1.69%,
+1.11%, -2.45%, -1.26%; these points participate in fitting.

`cpu_type_screen.py` runs `llama-bench -ngl 0 -p 0 -n 32 -r 3 -t 8 -o json`
with niceness 10, on mixE4K/E3K/E2K, Q8_0, and all three pap24 variants. The
benchmark is built CPU-only at `artifacts/codexPL4/tree` from the pinned archive
with `-j4`. `build-manifest.json` binds the full source commit/archive SHA256
and binary/library hashes. Archive builds explicitly set the commit label so
it does not inherit the parent campaign repository's commit.

The screen records per-second load average, CPU pressure, affinity and CPU
counters, raw stdout/stderr and all three sample rates in
`artifacts/codexPL4/cpu-screen/report.json`. Median effective full-model rates
are approximately Q4_K 26.07, Q3_K 26.58, Q2_K 26.57 and Q8_0 28.28 GB/s.
These include CPU nonexpert work and are **indicative shared-host** results,
not isolated expert throughput. `calibration.json` stores them separately as
`screen_bw_cpu_gbs`; clean mixed GPU/CPU fits take precedence, and an unsupported
Q8_0 CPU path can use its measured screen with a visible caveat. File-specific
throughput and load are preserved rather than presented as clean host constants.

Remaining gaps: one GPU bandwidth cannot explain Q8_0 being faster than mixed
Q4_K despite more active bytes (up to 37.6% endpoint residual); uniform LRU
coverage misses real routing/admission effects (up to 69.2% residual); strict
reverse model transfer fails; Q6_K CPU efficiency is unmeasured; screen timing
is host-contended; PAP's noisy and quiet runs disagree; hybrid extra GPU dispatch
work, context/KV traffic, TTFT, and host working set need separate calibration.
Hybrid eligibility and quality gates remain in force; fitting PAP timing does
not activate the rejected PAP representation as a recommendation. No Docker or
GPU work was performed.

## Run

From the campaign root:

```sh
python3 -B cc/moe_planner/plan.py \
  --stats artifacts/codexPL/granite-fixture.json \
  --host cc/engine_handoff/hosts/x86-main.json \
  --budget-gpu-mib 900 --ctx 4096 --top 5 \
  --components artifacts/codexPL4/components-baseline.json \
  --emit-lab-tests artifacts/codexPL4/lab-tests.json \
  > artifacts/codexPL4/plan-900.json

PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider \
  cc/moe_planner --basetemp=artifacts/codexPL4/pytest-temp
```

The real T-MI inspection of mixE4K is also saved read-only as
`artifacts/codexPL/granite-inspected.json`; it can replace the hand fixture in
that command. The checked-in example report uses the real inspection.
`--only-input-representation` disables discovery of the other available Granite
GGUFs. Otherwise the six known sibling filenames are considered only if present
in the input model's directory and the input agrees with the calibrated Granite
architecture, dimensions, filename and weight byte counts. Missing quantizations
are never invented. All other inputs enumerate their supplied representation.

Additional controls: `--budget-host-mib`, `--max-kld`, `--quality-ledger`,
`--objective decode|ttft|balanced`, `--allow-extrapolation`, and
`--resource-sample sample.json`, and `--components [registry.json]`. `--prompt-tokens` changes the analytical warm
TTFT proxy and requires extrapolation for values other than 62; lab emission
then refuses because the actual mounted chat fixture contains 62 tokens.
Output JSON includes `plans` (at most `--top` family winners), `family_best`
(all family winners), `pareto_set` (globally nondominated feasible candidates),
every rejected candidate and its reason, fit coefficients/residuals, effective
budgets and limitations. A family is representation × placement mechanism ×
KV type × attention mode × enabled engine components. Layer/cache coordinates
and batch/ubatch sizes compete within a family. Top is a maximum: if only one
calibrated family fits, return one plan and explain the shortage. Use
`--allow-extrapolation` to explore other available representations. That flag
continues to be required for every unmeasured host/model/context/KV/batch regime. Invalid
inputs exit 2; a valid budget with no feasible supported plan returns an empty
list and explicit rejections. An empty list does not prove no engine can run.
`--emit-lab-tests` deliberately accepts only destinations under the two allowed
write roots, including when symbolic links are involved.

## Input adapter

The API is `plan(stats, host, budget, allow_extrapolation=False, top=5, ...)`.
Budget keys are `gpu_mib`, optional `host_mib`, `ctx`, optional
`max_quality_loss` (mean KLD), `max_ppl_change` (API only; default 0.01), and `objective`. KLD and relative PPL change are separate metrics and are never added or numerically equated.

`adapt_stats()` accepts the T-MI producer's `model-stats/0` directly:
`arch`, `n_layer`, `n_expert`, `n_expert_used`, `context_length`, `gguf_path`,
`tensors[{role,bytes,...}]`, and
`kv_cache.cache_types.TYPE.bytes_per_token` (total K plus V).
Shared FFN and unknown roles count toward nonexpert residency, never disappear.
The small hand fixture alternatively supplies `model_path`, `representation`,
`expert_bytes`, `nonexpert_bytes`, and `kv_bytes_per_token` directly. A
`totals.by_role` or `totals_by_role` byte mapping is another supported adapter.
Unavailable KV byte counts are omitted; unavailable f16 KV is an error.
Unknown models have no inferred quality measurement and require extrapolation.
The adapter checks shape/byte compatibility, not full GGUF content identity;
qualified measurement must bind model and image SHA256 values.

The optional `resource-sample/0` adapter consumes successful, correctly unitized
`gpu.memory.total`, `gpu.memory.used`, and `host.mem.available` fields. It limits
available budgets to device free MiB and available host bytes. Unavailable/error
fields remain unknown and generate warnings; they never become zero. This is
only an availability hint: samples are non-atomic, can be stale, and do not
reserve memory. Device totals are never relabeled as process allocations or
added to process/cgroup counters. Host RAM and GPU VRAM are separate budgets on
this discrete host; applying these fits to unified memory requires the explicit
extrapolation flag and does not establish unified-memory accounting correctness.

## Evidence and decision rules

* **Expert precision before placement.** [D013](../../decisions/D013-mixed-precision-frontier.md)
  identifies experts as about 90.4% of weight bytes. Consider available expert
  representations before spending transfer or CPU-compute costs. mixE4K mean
  KLD 0.040017 is substantially below uniform Q4_K_M 0.077526 at similar GPU
  footprints. Q8_0 is 0.003389, mixE3K 0.135039, and mixE2K 0.771232.
  These are representation changes, with measured drift on wikitext, not
  computation-preserving improvements. [D009](../../decisions/D009-representation-screen.md),
  [D010](../../decisions/D010-kl-quality-check.md) and D013 supply the numbers.
* **Never recommend engine LRU below 60% resident-byte coverage on this discrete
  PCIe host.** This is a conservative planner policy, **not a measured universal
  60% breakpoint** and not an assumption that byte coverage equals routing hit
  rate. [D022](../../decisions/D022-static-vs-moecache-matched.md) shows static
  placement winning at every matched footprint, including c384 (~59% of the
  648 MiB expert banks) and c640 (~99%).
  [D019](../../decisions/D019-hybrid-harness-result.md) measures pinned H2D
  admission at 37.9 us/expert versus 30.6 us CPU compute plus ~15 us per layer.
  c640 remains an enumerated control but loses to static in the fitted ranking.
* **Declare context.** [D023](../../decisions/D023-novmm-engine-under-budget.md)
  measures 192 MiB f16 KV at ctx4096, 108 MiB compute and 778.7 MiB weights.
  The model's default/max ctx131072 requires 6144 MiB f16 KV analytically;
  KV then exceeds compressed weights. Context is a required constraint; the
  planner never silently shortens it to fit. KV q8_0 and q4_0 use the inspector's
  exact block accounting (102 and 54 MiB at ctx4096 here), require `-fa on`,
  and remain extrapolated for speed; quality uses rules c/d below. D028 observed both
  quantized types aborting before the fix: neither is enumerated unless the
  `kv_quant_fix` component is explicitly enabled (or accepted in the registry).
  Pending status alone does not enable it. At ctx >=8192, f16 also defaults to
  `-fa on`, citing [D028](../../decisions/D028-kv-precision-protocol.md). Its
  measured 32k f16 memory delta is -984 MiB, decode ratio 627.2/570.2, and
  prefill ratio 7001/5865. FA changed text; D033 PPL supplies rule d's caveat rather than a quality block.
* **Auto-fit is not a baseline.** D023's virtual 900 MiB budget caused auto-fit
  to place only 155 MiB on GPU and decode at 86 t/s, below explicit static n24.
  Every emitted plan sets `-ngl 99 --fit off` and declares placement explicitly.
  This also avoids the hybrid graph-before-weight-load failure in
  [D027](../../decisions/D027-hybrid-split-protocol.md).
* **Verify mechanism engagement before timing.**
  [D024](../../decisions/D024-static-hot-rejected.md) rejected static-hot because
  its allocated banks were never used. Every candidate gets a short engagement
  test first. Require counters/backend logs or a qualified fingerprint; successful
  process exit alone is insufficient. Greedy-output identity does not bound KLD.
* **Do not predict normal RAM speed below the working set.**
  [D015](../../decisions/D015-ssd-tier-emulation.md) demonstrates a page-cache
  cliff and severe read amplification. The planner requires a conservative
  complete file-backed weight set plus 400 MiB anonymous allowance. It rejects
  tighter host budgets rather than interpolating the contention-affected SSD
  timing. D022 warm cgroup peaks (~412–418 MiB) do not include every residency
  scenario or all preexisting page-cache charges and cannot certify this budget.
* **Do not scale throughput inversely with bytes.**
  [D011](../../decisions/D011-bandwidth-and-q8-profile.md) and D009 show Q8_0
  improving clean decode by ~32.5% over BF16, but Q4_K_M is slower than Q8_0.
  Kernel/dequantization and launch costs matter. CPU representation transfers use per-type effective bandwidths;
  unsupported types fall back to an indicative CPU screen or Q4_K, with a visible note.

## Fits and calibration boundaries

`calibration.json` preserves all 27 D022 samples from
`lab/jobs/job-ad883810b3/att-1/result.json`, with decision, job, attempt and test
IDs. GPU MiB means **baseline-subtracted device-used peak**, not the shim cap
and not the absolute device total. Each coordinate has three independent rounds;
medians are fitted with equal weight. Source data are immutable inputs. Supporting
records include D023's probe paths, D014's static calibration job, D019's operation
IDs (not invented lab job IDs), D011/D009 clean screen jobs, D015's job and its
conflicting repetition caveat, exact header-derived weight bytes for all six
representations, and the three existing hybrid transform manifests. A null job ID
is explicitly explained where the source uses an operation or probe instead.

The retained historical D022 diagnostic (not the current decode predictor) is ordinary least squares:

```text
GPU MiB(K)       = 1387.600000 - 26.933333 K
Decode us(K)     = 1575.237082 + 268.831188 K
Prompt ms(K)     = 174.238070 + 2.933955 K
```

The retained D022 residual diagnostic for LRU is ordinary least squares of log time:

```text
GPU MiB(C)       = 742 + C
log Decode us(C) = 10.290775 - 0.004155091 C
log Prompt ms(C) = 5.474944 + 0.000019467 C
```

The log model preserves positive time and captures the cache trend with two
parameters; its error is large. No polynomial or perfectly interpolating fit
is used to hide that error. Decode t/s = 1e6 / decode us. The prefill model is
separate from decode. It fits median microseconds/token
from D022 `prefill_tps` (or measured prompt time/count if a rate is missing)
against placement coordinates. Static reproduces the prompt-ms OLS above; LRU
uses its own linear prefill fit rather than borrowing decode's log-time fit.
These are completion prompt-eval measurements, not bench pp512. TTFT is
**warm prompt ms + one decode step**, excluding load,
queueing, tokenization and sampling. Actual first-token arrival was not measured.
A different prompt length is a linear-scaling hypothesis requiring the flag.
At ctx >=8192 the independent D028 prefill endpoint (7001 t/s FA on, 5865 off,
1560-token prose) replaces D022's all-GPU intercept. D022's placement penalty
in us/token is added to that endpoint. This is an explicit extrapolation across
prompt/context/placement; a 62-token long-context prompt is not a new measured
prefill result. Output carries `prefill_tps`, `prefill_ms`, and `prefill_source`;
`ttft_ms = prefill_ms + 1000/decode_tps`.

| Family | Max fitted GPU error | Max held-out GPU error | Max fitted decode error | Max held-out decode error |
| --- | ---: | ---: | ---: | ---: |
| Static | 0.800 MiB | 2.000 MiB | 7.201% | 16.246% |
| LRU | <0.001 MiB | <0.001 MiB | 10.327% | 39.263% |

Leave-one-**coordinate**-out removes all three samples of one layer/cache setting
before refitting. It does not leak other repetitions of the held-out setting.
Endpoint folds test extrapolation for validation only; normal inference refuses
out-of-domain predictions. The complete per-fold residuals and coefficients are
returned in JSON. Prompt-time held-out errors reach ~2.203 ms static and ~5.311 ms
LRU; these are not TTFT confidence intervals. Budget checks add the maximum
held-out GPU residual to predicted device memory. This is an empirical guard,
not a statistical guarantee or a bound on other-process interference.

The calibrated GPU/decode domain is the exact x86-main CPU/GPU/discrete-memory
profile, matching Granite mixE4K byte layout, f16 KV, ctx4096, batch2048,
ubatch512, eight threads, the D022 prompt and K in [0,24] or C in [128,640].
Static intermediate integer K is interpolation. Every other representation,
model, host, context, KV type, prompt length, batch/ubatch, and all hybrid GPU
predictions require `--allow-extrapolation` and have `confidence=extrapolated`.
The overall `calibrated` label describes the GPU/decode regime only;
`metric_confidence` separately identifies host as an analytical working-set
hypothesis and TTFT as a proxy. No flag creates missing measurements.

Outside that domain, byte accounting adjusts the D022 footprint by changed
resident expert/nonexpert bytes, KV bytes relative to 192 MiB, and a hypothetical
compute allowance `108 * ubatch/512`. This assumes linear buffer scaling and
ignores alignment/backend algorithm changes. Static decode transfers use clean
D009 GPU endpoints where available and scale CPU penalty with expert bytes;
missing endpoints fall back to D022's ~684 t/s, explicitly unvalidated. The
other-representation LRU transfer scales fitted time with expert bytes. These
coarse proposals exist to select tests; they are not calibrated speed results.

Long-context memory retains the exact KV/token byte accounting from model-stats.
For ctx >=8192, use the D028 all-GPU f16 endpoint to model attention workspace:
let `scale = (ctx-4096)/(32768-4096)` and let `off_delta` be 3746 minus
(D022 all-GPU fitted MiB + the 4096→32768 f16 KV delta). Add
`(off_delta - 984) * scale` to the D022-plus-KV footprint. At ctx32768 this
recovers 2762 MiB for mixE4K all-GPU at the baseline ubatch (within input byte
rounding). The correction is +30.4 MiB relative to the existing linear KV model;
subtracting 984 again from that model would double-count a nonexistent KQ buffer.
Use the f16 workspace correction as an explicitly unmeasured transfer for
quantized KV. Decode uses D028's FA ratio multiplicatively; other placement,
prompt and context effects remain unmeasured. Batch/ubatch buffer scaling is
still a hypothesis. No context is shortened to make a plan fit.

D030 rejected hybrid at all three measured Granite placements. D031 measured
2030 us/token extra cost at n12, with 12 additional handoff layers: an effective
169.17 us/layer, including dispatch, exposed synchronization/transfers and host
cost. D019's 30.6 us per 884736-byte expert supplies a CPU us/byte estimate.
The reported crossover is `fixed_us / cpu_us_per_byte = 4.665 MiB/expert`.
A conservative 2x margin implements the assignment's “much greater” condition:
only experts above ~9.329 MiB satisfy the necessary amortization condition.
Granite's 0.844 MiB experts fail it, so all real hybrid transforms are rejected
with D030/D031 reasons, even with extrapolation enabled. This is a scale model,
not proof of a universal threshold or sufficient evidence to enable a new model.

The hybrid accounting path retains existing n18/n12/n6 transform manifests,
fixed TRAIN hot IDs, per-layer null expert bytes and decode trace evaluation.
Its current CPU cost uses the physical per-type bandwidth fit and CPU-touching-layer cost; D031 still gates crossover eligibility;
CUDA correctness/quality qualification remains required. Larger models do not
inherit Granite split files. Tests isolate the accounting under a hypothetical
crossover calibration without promoting those files into valid recommendations.

### Components and pending PAP calibration

`--components` reads `cc/factory/components.json` by default if it exists.
Absence is tolerated; an explicit missing alternative path fails. A JSON object
may contain a `components` mapping/list or directly map IDs to records. List
records accept `id` or `name`. `status: accepted` defaults enabled; an explicit
boolean `enabled` overrides that. Pending/rejected entries default disabled;
`enabled: true` is the caller's explicit opt-in. An empty object disables all
components. The planner does not edit the registry or build/apply patches.
For example, an exploratory registry may contain:

```json
{"components": {
  "q8_reuse": {"status": "accepted"},
  "kv_quant_fix": {"status": "pending", "enabled": true}
}}
```

Enabled q8_reuse multiplies decode t/s by **1.02** (D029); memory and prefill
stay unchanged. Runtime environment `GGML_CUDA_Q8_REUSE=1` and the patch
requirement accompany the plan and emitted lab definitions. Transfer of that
factor to placements/contexts beyond D029 remains explicit in metric confidence
and notes. q8_reuse inherits established base quality only through rule a below; its speed transfer remains extrapolated.

`calibration.json:planner_v2.techniques.pap` is a pending representation trade-off
entry, with empty calibration points, a lab/jobs result source, explicit future
point fields (including model bytes, CPU layers, GPU MiB, decode, prefill, KLD
and job/attempt/test provenance), and D032 acceptance criteria. No PAP speed,
quality or nonexistent model is emitted. Future calibration must bind the actual
transform/model and same-placement control before activating this technique.

## Ranking, quality and the measurement handoff

Plans must fit `predicted GPU + guard <= GPU budget` and conservative host
working set `<= host budget`, each capped by physical capacity and optionally
resource availability. Each family is ranked first; its best candidate survives. Ranking first
prefers the calibrated regime. Within each
confidence tier, decode maximizes decode t/s, ttft minimizes the warm TTFT proxy,
and balanced maximizes decode t/s divided by TTFT. Ties prefer smaller GPU
footprint and stable candidate ID. No cross-tier speed claim is implied. `family_best` contains every winner;
`plans` takes the first `--top` of those. `pareto_set` is independent of this
ranking and considers **all** feasible candidates, minimizing GPU MiB and
quality risk while maximizing decode t/s. Quality risk is measured or evidence-inherited screening KLD. Unknown/failing risk
is JSON null and treated as +infinity for dominance, never zero. Tied
nondominated points remain available; the full Pareto set is not truncated by
`--top`.

### Quality ledger and inheritance (T-PL3)

Rebuild the bundled ledger with:

```sh
nice -n 10 python3 -B cc/moe_planner/build_quality_ledger.py
```

The script reads real `lab/jobs/JOB/att-1/result.json` metrics and frozen job
test definitions, requiring completed, clean results. D013 predates frozen
definitions, so its read-only `lab/tests/mem2.json` definitions and SHA256 are
recorded too. No metric value is copied from decision prose. Each entry has
representation, placement family/parameters, KV type, FA, runtime components,
context, chunk count, metric/value, job/attempt/test, decision, result path/SHA256
and job envelope SHA256. Counts:

| Decision | Job | Metric entries |
| --- | --- | ---: |
| D013 | job-aa57700ee5 | 15 |
| D034 | job-56115e2bf1 | 15 |
| D035 | job-3d2fbbd061 | 27 |
| D033 | job-a945fa8656 | 3 |
| D029 | job-34c09bec98 | 4 |

D035 PAP/imatrix measurements are recorded without activating nonexistent
model variants or supplying missing timing/placement calibration. BF16 has no
synthetic zero entry: absent ledger evidence remains unknown.
`load_quality_ledger()` loads the bundled JSON or `--quality-ledger PATH`;
an empty ledger cannot fall back to `calibration.json` quality priors.

The rules apply to matched Granite representations and static placement.
LRU, hybrid, mismatched models, new KV types, unproven components, and
representation transfers without evidence remain unknown. Directly recorded
representation/placement KLD is `measured` screening quality. Other admissible
configurations are `inherited (rule x; evidence y)`:

* **Rule a — computation-preserving components.**
  [D029](../../decisions/D029-c1-q8reuse-accepted.md), job-34c09bec98,
  provides paired OFF/ON complete stdout SHA256 identity and exact PPL
  identity independently for Q8_0 and mixE4K. The builder verifies both from
  result.json and records paired test names. Only those component/representation
  pairs inherit base quality; either text identity or PPL identity alone is
  insufficient. This is an explicit screening transfer across placement/context,
  not new combined-KLD measurement.
  [D036](../../decisions/D036-binary-component-swap.md) corroborates the Q8_0
  binary swap using `artifacts/x7-comp/swap-gen.out` SHA256, which the builder
  compares to D029. That manual run has no lab job/result.json or paired PPL;
  it supplies no independent inheritance permission or fabricated job ID.
* **Rule b — static placement-only changes.**
  [D034](../../decisions/D034-planner-validated.md), job-56115e2bf1,
  measures mixE4K at K=19,20,21,22,23, each KLD 0.040017, agreeing with
  [D013](../../decisions/D013-mixed-precision-frontier.md) and D035 all-GPU
  controls. A representation needs at least two distinct measured static
  placements with identical KLD across the available base evidence before
  another integer `--n-cpu-moe K` in [0,24] inherits it. Changed weights,
  hybrid arithmetic and LRU are outside this rule. Other representations
  do not inherit mixE4K's placement evidence.
* **Rule c — KV quality and scheduler fix.**
  [D033](../../decisions/D033-kv-quant-fixed-and-measured.md), job-a945fa8656,
  measures paired FA-on PPL at ctx4096, six chunks: f16 7.5385, q8_0 7.5392,
  q4_0 8.1368. Compute relative PPL change from those ledger values. q8_0
  inherits mixE4K base screening KLD when within the default ±1% relative
  PPL budget and `kv_quant_fix` is enabled. This is a policy inference, not
  a measurement or mathematical bound on combined KLD. q4_0 is
  `quality-failing` under that PPL budget and never acquires KLD even if
  the PPL budget is loosened. Unknown KV/representation pairs stay unknown.
  The scheduler capacity correction is covered by D033's f16 control and
  quantized variants; no general permission for other fixes is inferred.
  A 0.08 KLD ceiling does not mean an 8% PPL budget.
* **Rule d — FA evidence caveat.**
  D033 paired PPL has FA on only. Emit “PPL measured with FA only; no
  non-FA PPL comparison” as a caveat, not a block, for supported FA
  combinations. Do not claim measured FA equivalence or transfer this
  exception to representations without D033 evidence.

`quality` contains status, description, rule IDs, decision/job/test evidence,
PPL relative change/status, and caveats. `kld` and Pareto `quality_risk` use
the inherited base value only after all applicable rules succeed. Finite
`--max-kld` rejects unknown, failing, or above-ceiling values, regardless of
`--allow-extrapolation`. Without a ceiling, unknown/failing candidates remain
exploratory hypotheses with visible labels and null risk. Caveats retain the
measurement scope: KLD is 20x512 wikitext vs BF16 logits, D033 PPL is six
chunks at ctx4096; neither certifies long-context or general-task quality.
GPU/decode confidence remains independent of quality.

The emitted JSON is exactly a mapping of test names to `bin/lab` definitions,
with `tool`, `parser`, `model_file`, `docker_args`, `args`, and, for quality,
`needs_data`. Tests exercise the real `load_tests()` in isolation and compare
consumer-required fields with the repository's definitions. No new schema
fields or automatic modifications of `lab/tests` are introduced.

For each plan, test prefix is `pl_` plus the first 12 hex characters of SHA256
of the plan ID. Suffixes are `_engagement` (8 generated tokens), `_timing` (160),
and, for known Granite variants with the existing logits present, `_quality512`.
Completion tests mount the exact D022 chat prompt read-only; model flags match
the plan, with `-m` provided once by the runner. Quality uses the same
placement/KV flags at **ctx512** with 20 chunks and the existing
`artifacts/kl/bf16.kl`, as required by that reference. It tests placement drift
at ctx512; it cannot certify ctx4096/long-context KV quality. Provide the exact
wikitext-2 `wiki.test.raw` from the D009 job as the job's data input. To validate
quality at ctx4096, first make matching BF16 reference logits and a paired test
at that context; this task does not run or invent that measurement.

The owner must import definitions into a permitted lab location, choose an
immutable qualified image, and submit with `--model` in the directory containing
the test's `model_file`. Standard representations share the original GGUF
directory; hybrid models live in `artifacts/codexR/models`, requiring a separate
job/model directory and the hybrid image. Do not mix those directory groups in
one job. Before any scored timing: pass engagement with a backend observation,
resolve the hybrid CUDA failure if applicable, pass paired quality, bind
image/model/prompt hashes, and confirm the predicted footprint. Run three
alternating forward/reverse timing rounds under the established one-GPU-job,
no-compile policy. Record shim peak and device delta separately; shim limits are
cooperative intercepted-allocation limits and exclude ~310 MiB driver/context
overhead (D023). Replan after measured failure or material prediction error.
No tests were submitted or executed on the GPU by this worker.

## Example and remaining gaps

With components disabled explicitly, the real mixE4K inspection at 900 MiB,
ctx4096 recommends static n19 with f16 KV:
875.867 MiB predicted GPU plus 2.000 MiB guard, 1178.695 MiB host working-set
allowance, 143.925 decode t/s, and 236.931 ms warm TTFT proxy. The hand-derived
fixture differs by 0.0055 MiB from exact nonexpert header bytes. Exact flags:

```text
-m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf
-ngl 99 --fit off -c 4096 -b 2048 -ub 512 -ctk f16 -ctv f16
-t 8 -tb 8 --n-cpu-moe 19
```

With accepted factory runtime components (`q8_reuse`, `kv_quant_fix`),
900 MiB / ctx4096 / max-KLD 0.08 returns mixE4K static19, f16 KV,
875.867 MiB GPU + 2 MiB guard, 146.804 predicted decode t/s, KLD 0.040017
inherited through rules a/c (D029/D033), with direct placement evidence D034.
No extrapolation flag is needed for its baseline GPU/decode regime.

The required **1200 MiB / ctx32768 / max-KLD 0.08** rerun returns **no feasible
plan** under the existing memory model. This is a memory constraint after the
quality repair: the smallest quality-admissible candidate is mixE4K static24,
q8_0 KV, FA on, batch512/ubatch128, **1314.600 MiB GPU + 2 MiB guard =
1316.600 MiB**. The D028 FA anchor is ~2762 MiB all-GPU/f16 at 32k;
q8 KV saves 720 MiB, all 24 expert layers on CPU save the fitted placement
amount, and ubatch128 saves 81 MiB. The result is still above 1200 MiB.
Changing quality rules cannot remove those bytes. The earlier q4_0 example
was rejected by D033 and is withdrawn. No smaller memory prediction or new
unmeasured placement is invented to satisfy this scenario.

At **1317 MiB** the same ctx32768 / max-KLD 0.08 request has a feasible q8_0
candidate with inherited quality (rules a/b/c/d; D029/D034/D033). Memory,
decode and prefill remain extrapolated; quality is a screening prior.
The historical `artifacts/codexPL3/` contains exact commands, both prior reruns,
a 1317 MiB diagnostic rerun, ledger counts, emitted lab definitions, and CPU-only
pytest output. No Docker/GPU operation was performed. Long-context quality,
actual TTFT/host working set and post-fix combined performance need measurement.
