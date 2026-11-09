# MoE planner: hypotheses before measurement

`plan.py` is a stdlib-only, offline planner for llama.cpp `31288c9c6` and the
campaign patches. It does not compile, launch inference, reserve resources,
submit jobs, or contact services. Its recommendation is a discriminating lab
experiment, not a demonstrated engine improvement or a hard memory guarantee.
All implementation and scratch writes are confined to `cc/moe_planner/` and
`artifacts/codexPL/`. There are no changes to `bin/lab` or `lab/tests`.

## Run

From the campaign root:

```sh
python3 -B cc/moe_planner/plan.py \
  --stats artifacts/codexPL/granite-fixture.json \
  --host cc/engine_handoff/hosts/x86-main.json \
  --budget-gpu-mib 900 --ctx 4096 --top 5 \
  --emit-lab-tests artifacts/codexPL/lab-tests.json \
  > artifacts/codexPL/plan-900.json

PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider \
  cc/moe_planner/test_plan.py --basetemp=artifacts/codexPL/pytest-temp
```

The real T-MI inspection of mixE4K is also saved read-only as
`artifacts/codexPL/granite-inspected.json`; it can replace the hand fixture in
that command. The checked-in example report uses the real inspection.
`--only-input-representation` disables discovery of the other available Granite
GGUFs. Otherwise the six known sibling filenames are considered only if present
in the input model's directory and the input agrees with the calibrated Granite
architecture, dimensions, filename and weight byte counts. Missing quantizations
are never invented. All other inputs enumerate their supplied representation.

Additional controls: `--budget-host-mib`, `--max-kld`,
`--objective decode|ttft|balanced`, `--allow-extrapolation`, and
`--resource-sample sample.json`. `--prompt-tokens` changes the analytical warm
TTFT proxy and requires extrapolation for values other than 62; lab emission
then refuses because the actual mounted chat fixture contains 62 tokens.
Output JSON includes ranked feasible plans, every rejected candidate and its
reason, fit coefficients/residuals, effective budgets and limitations. Invalid
inputs exit 2; a valid budget with no feasible supported plan returns an empty
list and explicit rejections. An empty list does not prove no engine can run.
`--emit-lab-tests` deliberately accepts only destinations under the two allowed
write roots, including when symbolic links are involved.

## Input adapter

The API is `plan(stats, host, budget, allow_extrapolation=False, top=5, ...)`.
Budget keys are `gpu_mib`, optional `host_mib`, `ctx`, optional
`max_quality_loss` (mean KLD), and `objective`.

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
  and remain uncalibrated for speed and combined quality.
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
  Kernel/dequantization and launch costs matter. Representation transfers use
  measured screen endpoints when available and are explicitly extrapolated.

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

The static model is ordinary least squares:

```text
GPU MiB(K)       = 1387.600000 - 26.933333 K
Decode us(K)     = 1575.237082 + 268.831188 K
Prompt ms(K)     = 174.238070 + 2.933955 K
```

The LRU model is ordinary least squares of log time:

```text
GPU MiB(C)       = 742 + C
log Decode us(C) = 10.290775 - 0.004155091 C
log Prompt ms(C) = 5.474944 + 0.000019467 C
```

The log model preserves positive time and captures the cache trend with two
parameters; its error is large. No polynomial or perfectly interpolating fit
is used to hide that error. Decode t/s = 1e6 / decode us. Prompt ms comes from
D022's measured 62-token prompt-eval counts/rates, not the much faster bench
pp512 workload. TTFT is **warm prompt ms + one decode step**, excluding load,
queueing, tokenization and sampling. Actual first-token arrival was not measured.
A different prompt length is a linear-scaling hypothesis requiring the flag.

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

Hybrid candidates use the **existing** n18/n12/n6 split files, exact per-layer
hot counts/IDs from the transform manifests, and one null expert per layer.
Their real-hot totals 168/360/552 plus 24 nulls require 162/324/486 MiB of expert
GPU payload. If the input includes exactly one complete decode trace group,
fixed TRAIN-selected IDs are evaluated against its frequencies; they are never
reranked on evaluation. Otherwise uniform routing is an explicitly stated
fallback. The serial hypothesis adds `30.6 us * expected misses + 15.2 us * 24`
to the fitted GPU baseline. This comes only from D019's exploratory harness,
with D014's independent 33.3 us calibration as a cross-check; it omits the real
split engine's extra launches and synchronization. The prediction is likely
optimistic. D027 records a focused CUDA correctness failure: candidates remain
exploratory until repaired and requalified. CPU equality is insufficient.

## Ranking, quality and the measurement handoff

Plans must fit `predicted GPU + guard <= GPU budget` and conservative host
working set `<= host budget`, each capped by physical capacity and optionally
resource availability. Ranking first prefers the calibrated regime. Within each
confidence tier, decode maximizes decode t/s, ttft minimizes the warm TTFT proxy,
and balanced maximizes decode t/s divided by TTFT. Ties prefer smaller GPU
footprint and stable candidate ID. No cross-tier speed claim is implied.

Without `--max-kld`, measured representation KLD is a risk prior; changed
CPU/GPU placement, hybrid arithmetic and quantized KV have **unknown combined
KLD**. They get `quality_class=combined-quality-unmeasured` and `kld=null`.
A finite quality constraint fails closed on unknown combined KLD even with the
extrapolation flag. Only a matched original all-GPU representation with f16 KV
can use its measured representation drift as the gate. That measurement remains
a wikitext screening prior, not a guarantee for other tasks or context lengths.
The 900 MiB example consequently has no admissible plan under a strict KLD
ceiling until placement quality is measured. Unknown quality is never zero.

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

The real mixE4K inspection at 900 MiB, ctx4096 recommends static n19 with f16 KV:
875.867 MiB predicted GPU plus 2.000 MiB guard, 1178.695 MiB host working-set
allowance, 149.633 decode t/s, and 236.666 ms warm TTFT proxy. The hand-derived
fixture differs by 0.0055 MiB from exact nonexpert header bytes. Exact flags:

```text
-m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf
-ngl 99 --fit off -c 4096 -b 2048 -ub 512 -ctk f16 -ctv f16
-t 8 -tb 8 --n-cpu-moe 19
```

The remaining top five use K20..23 with progressively lower footprint and
throughput. They are predictions between the measured K18 and K24 points,
with unmeasured placement KLD. Gaps: actual TTFT/load time, long-context and
quantized-KV quality, uncontented host working-set/SSD behavior, other model and
host calibration, batch/prefill buffer measurements, GPU hybrid correctness,
launch/transfer costs and engagement, disjoint task-level quality, and prediction
uncertainty beyond the observed residuals. Measurements must decide acceptance.
