# MoE inference-engine factory

`factory.py` connects existing campaign capabilities. It inspects an MoE GGUF,
samples current resources once, requests budget-constrained planner hypotheses,
records model/build/runtime recipes, delegates builds to Build CC v2, submits
the planner's tests to the existing lab, and compares observations with predictions.
It does not infer an accepted memory/accuracy improvement from successful execution.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  nice -n 10 python -B cc/factory/factory.py run \
  --model /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf \
  --host x86-main --budget-gpu-mib 900 --ctx 4096 --dry-run \
  --run-dir artifacts/codexFAC/factory/real-dry-run
```

The assignment names `artifacts/factory/<run-id>/` but explicitly permits writes
only to `cc/factory/**` and `artifacts/codexFAC/`. The allowed-write rule takes
precedence: the default is `artifacts/codexFAC/factory/<stable-run-id>/`.
`--run-dir` must resolve under one of those two permitted roots; symlink escapes
are refused. No shared lab definitions, image registry, job, reference source,
or component files are changed. Python bytecode writes are disabled.

The exact required CLI works without additional arguments. A host can be a
profile name or JSON path. Optional controls are `--budget-host-mib`, `--max-kld`,
`--top`, `--objective decode|ttft|balanced`, `--allow-extrapolation`,
`--include-pending`, `--components`, `--data`, repeatable `--result`, and
`--error-pct`. `--backends linux` requests only the observer's Linux backend;
the default also samples `nvidia-smi`. Neither backend launches inference.

## Stage artifacts and resume

Stages run in this order:

```text
inspect -> system -> plan -> recipe -> build -> measure -> report
```

Each writes `<stage>.json` with `inputs`, canonical `inputs_sha256`, `data`,
`outputs.data_sha256`, and full SHA256 identities for emitted files. The inspector
still reads metadata only; the factory separately streams full model hashes.
Inputs include component implementation files, registry/patch contents, host
profile, planner calibration, discovered representation files, mounted test
inputs, build records, and supplied/discovered result evidence as applicable.
The factory reuses a completed artifact only when both its input identities and
its output content identities match. Failed stages are recorded and retried.
Inputs are assumed immutable during an invocation; hashes are memoized only
within that invocation. Artifacts are written by atomic rename. Concurrent writers
to the same run directory are unsupported; use separate run directories.

`--stages inspect,system,plan` stops before building. Requesting a later stage
also validates its dependencies, running missing or changed dependencies first.
This prevents a stale plan from silently surviving changed model/budget inputs.
The resource sample is deliberately reused on resume; start a fresh run directory
or remove `system.json` to refresh availability. A sample does not reserve memory.
Dry-run/live mode participates in build and measurement identity, so dry-run
artifacts cannot masquerade as completed builds or submitted jobs.

* `inspect.json`: unchanged `model-stats/0` from `cc/model_inspect`.
* `system.json`: host profile and one `resource-sample/0` from `cc/resource_observe`.
* `plan.json`: `cc/moe_planner.plan()` output, safety-gated by the registry;
  `plan-tests.json` is the planner's emitted lab definition mapping. Current
  planner versions return technique-family winners rather than adjacent placements.
  At 900 MiB/ctx4096, the calibrated mixE4K static family can be the only winner.
  A strict KLD ceiling may admit no plans because combined placement quality is
  unknown; an empty plan produces all remaining stage artifacts without a build.
* `recipe.json`: each selected plan's model input/output hashes, transform
  steps, pinned engine commit, ordered component patch list, CMake definitions,
  extra targets, linkage requirement, runtime argv/environment, tests, and
  prerequisites. Existing representations are reused with identical input/output
  hashes. With `--include-pending`, eligible Granite K12/K18/K24 plans also get
  unselected PAP Q2_K/Q3_K alternatives copied from `cc/pap`'s recipe: BF16 input,
  TRAIN imatrix, first-K regex before the all-expert Q4_K regex, Q8_0 base type.
  Their input/output hashes remain explicitly unfilled until materialized.
  They must be inspected/replanned and evaluated before selection. The factory
  never requantizes a low-bit model or silently applies baseline predictions to PAP.
* `build.json`: delegates to `cc/build_engine/build_engine.py`, forwarding
  `--patch`, `--cmake-define`, `--extra-target`, and `--linkage`. Build CC accepts
  one patch, so `ordered-components.patch` concatenates registry patches in order
  and records individual and bundled hashes. Build CC checks application inside
  its private archived source; combinations are not qualified by component acceptance.
  Builds use `nice -n 10`. An invalidated build with an existing evidence directory
  fails instead of overwriting that evidence; choose a fresh run directory.
* `measure.json`: creates a private `$LAB` under the run directory and submits
  separate engagement, quality512 and timing jobs using **the existing `bin/lab`**.
  Its submit path hashes the inputs and embeds the actual test definitions.
  Model directories stay separate per plan. Image registration uses the Build CC's
  recorded immutable config digest, with a distinct name per digest. Planner
  `-e` switches are preserved through the lab's native `env` field. Timing has
  three occurrences; scheduling alternating forward/reverse rounds remains the
  operator's responsibility. Quality submission needs `--data` pointing to the
  exact WikiText TEST input; missing data is recorded as blocked. Submission does
  not start a GPU job. Dry-run prints commands and writes definitions/artifacts,
  without calling Build CC, registering images, or submitting jobs.
* `report.json`: joins only exact emitted test names (or explicitly supplied
  `plan.test_names` for historical fixtures), retaining all occurrence and attempt
  identities. Results discovered under the private lab or passed with `--result`
  are read. `cc/engine_handoff.envelopes.make_result()` supplies the shared durable
  evidence projection when a job record is present. Owner/journal inconsistencies
  suppress scored metrics. Incomplete, failed and partial tests supply no metrics.

## Components and evidence gates

`components.json` is an ordered registry. Entries have accepted/pending/rejected
status plus evidence decision IDs and scope. Default: q8_reuse accepted D029;
gpu_budget accepted infrastructure D021/D023; flash attention recommended D028.
Accepted since: kv_quant_fix (D033; release-1 D037/D040) and imatrix_quant (D035). Rejected: PAP (D040).
Rejected: hybrid_split in the small-expert regime D030/D031, static-hot D024.
`--include-pending` opts into pending patches and PAP proposals; it never includes
rejected components. Quantized KV plans require the (now accepted) kv_quant_fix component; q4_0 KV is quality-failing (D033).
New planner APIs receive exactly the same enabled components as the recipe.
The cooperative budget shim is documented infrastructure, not an automatically
enabled cap or a hard VRAM guarantee. It requires a no-VMM engine and verified
interception. Planner flags include flash attention for quantized KV and current
planner versions recommend it for long context.

Removing `--dry-run` authorizes the ordinary Build CC path, which requires Docker
and locally available pinned CUDA bases. No live build or GPU execution was
performed for this assignment. Run submitted jobs separately only after validating
mechanism engagement, paired quality, immutable inputs/image identity, and the
campaign's one-GPU-job policy. The private lab is an instance of the existing
temporary reference adapter, not a remote receiver. No new GPU admission lock,
engine executor, memory enforcement, or numerical acceptance rule is introduced.

## Report a finished historical job

The additional offline report command accepts either a bare planner JSON or a
factory stage wrapper. Historical jobs require an explicit `test_names` mapping;
similar placements or basenames do not establish identity. Write the prediction
fixture inside `artifacts/codexFAC/` and run:

```sh
python -B cc/factory/factory.py report \
  --predictions artifacts/codexFAC/report-predictions.json \
  --result lab/jobs/job-ad883810b3/att-1/result.json \
  --out artifacts/codexFAC/report-real.json
```

GPU prediction errors compare **baseline-subtracted device delta** with planner
GPU MiB; absolute peak, baseline, cgroup peak and shim peak stay separate. The GPU
flag threshold is the larger of the planner's guard and 2 MiB. Decode/prefill and
warm-TTFT-proxy flags default to 20% absolute relative error (`--error-pct` changes
this); KLD flags use 0.01 absolute error when both values exist. Missing predictions
or measurements produce no invented error. Host working-set allowance is not
compared against cgroup peak. Actual TTFT stays null unless explicitly measured;
`warm_ttft_proxy_ms = prompt_tokens/prefill_tps*1000 + 1000/decode_tps` is separate.
Quality512 remains scoped to ctx512, and quality results are never copied onto
timing tests or other attempts. Reports do not automatically accept accuracy.

Rows have campaign claim categories (`prediction`, `failed candidate`,
`representation trade-off`, `runtime-placement/offload trade-off`) plus D012's
transfer class. Device placement/timing observations are `discrete-only`;
predictions remain `prediction`. Byte-accounting and matched KLD can be
transferable metrics, but the enclosing measured timing result remains local.
An accepted q8 component alone never changes a plan result into a computation-
preserving engine gain: that needs a qualified matched A/B experiment.
Historical jobs without full input identities are marked `inputs_bound=false`.

## CPU-only verification

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  nice -n 10 python -B -m pytest -q -p no:cacheprovider \
  cc/factory/test_factory.py --basetemp=artifacts/codexFAC/pytest-tmp
```

Checks cover all seven stages on the real GGUF and x86-main profile, resume,
changed quality constraints, emitted-output corruption, registry/environment
gating, PAP recipe ordering, Build CC argument forwarding, an actual private-lab
submission with embedded definitions and input hashes (no run), exact joins on
the finished D022 job with fabricated predictions, partial-result exclusion,
and write-root/symlink enforcement. The real-model test skips explicitly if the
campaign model is absent. Scratch files must stay in the specified basetemp.

Remaining measurement work: actual Docker build and patch-combination CUDA
qualification, engagement evidence, paired quality at requested context,
timing/memory/load/actual TTFT, and prediction validation on newly built images.
PAP transforms are proposals, not automatically materialized outputs. Optional
nsys breakdowns remain available through `cc/nsys_decode`; profiling is not part
of the automated factory stages.
