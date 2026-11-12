# abrun

Python 3 standard-library tools for guarded, counterbalanced experiments.
Run from the repository root; no package installation is needed.
```sh
python3 cc/abrun/abrun.py cc/abrun/specs/example-om7-replay.json --dry-run
OM_MAX_S=2400 python3 cc/abrun/abrun.py cc/abrun/specs/example-om7-replay.json
python3 cc/abrun/run_summary.py RUN_DIR --json --no-write
python3 cc/abrun/gate.py labelled-summaries.json --metric eval_tps
python3 cc/abrun/regate.py OUTDIR --exclude 001-base --reason "overlap" --no-write
python3 cc/abrun/preflight.py IMAGE llama-bench -- -n 32 -o json
python3 -m pytest -q cc/abrun
```

## Spec schema

A spec is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `id` | Required safe experiment name; no path separators |
| `outdir_base` | Required parent path; relative paths resolve from repo root |
| `runner` | Required; currently only `"om_run"` |
| `tool` | Required binary name inside `/app/` |
| `argv` | Required array of literal string arguments |
| `switch_regex` | Spec-level stderr activation regexes (OR); default `[]` |
| `metric` | Numeric summary field to compare; default `"eval_tps"` |
| `arms` | Required object: arm name -> `{image, env, expect_switch, switch_regex}` |
| `order` | Required `"ABBA"`, `"ABCCBA"`, `"mirrored"`, or `"explicit"` |
| `explicit_order` | Array of known arm names, required for `"explicit"` |
| `rounds` | Positive integer repeating the entire order; default 1 |
| `cache_protocol` | `"carryover"` (default), `"cold"`, or `"warm"`; host env `OM_CACHE_PROTOCOL` |
| `cache_warm_bytes` | Nonnegative prefix length for warm; host env `OM_CACHE_WARM_BYTES` |
| `cache_warm_ranges` | Host JSON path for warm; host env `OM_CACHE_WARM_RANGES`; exclusive with `cache_warm_bytes` |

`base` is mandatory and is A; remaining arms follow JSON insertion order.
Mirrored order visits every arm then reverses, including both end arms.
Explicit order must include all arms. Images are required; `env` defaults to
`{}` and maps env names to string values. Values cannot contain whitespace
because om_run splits its environment-flags argument. `expect_switch` defaults
to false; true requires spec-level or per-arm regexes. Base must not expect a switch.
Per-arm `switch_regex` is an array whose patterns must ALL match at least one
stderr line when `expect_switch` is true. Spec-level patterns retain OR semantics;
when both lists are present, both requirements apply. Any signature from any arm
or the spec in an `expect_switch: false` run is a leak (including non-base arms).
Each summary and the per-run report include `switch_matches` (regex -> matched
line count, including zeroes); a line counts once per pattern.
The OM7 example uses the original image, argv, env switches, and ABCCBA design.

Page-cache settings are top-level fields shared by every arm, e.g.
`"cache_protocol": "warm", "cache_warm_bytes": 1073741824`.
Alternatively warm settings may be supplied in the launcher's host environment.
Ranges use a JSON list of `{ "name": "expert", "offset": 4096, "size": 8192 }`.
om_run resolves `-m /om/...` against `OM_MODELDIR`; set `OM_CACHE_MODEL` to
the host model path for other container locations. Carryover only measures;
cold requests file eviction; warm evicts then reads the chosen prefix/ranges.
The runner always writes `page_cache.start.json` and `page_cache.end.json`;
start contains residency after preparation, plus before/after eviction evidence
for cold/warm. Recording errors are explicit; preparation errors stop launch.
Summaries and REPORT.md include `cache_start_resident_pct` and
`cache_end_resident_pct`. Verdicts state the protocol and add a descriptive
`cache_confound` note if any arms' observed mean start residency differs by
more than 10 percentage points; this never changes gate decisions.
Missing observations are omitted from these means. Measurements are snapshots;
other mappings, dirty pages, readahead and memory pressure can defeat eviction
or change residency. Warm reads beyond RAM capacity cannot stay fully resident.

## Execution and artifacts

Output is `outdir_base/id/`; an existing destination is refused. `--dry-run`
prints the order and shell-quoted commands without preflight or output writes.
Preflight runs once per image/tool before any measurement. Help is cached in
`lab/preflight/<sha256(image + NUL + tool)>.txt`. The real command uses
`$DOCKER_CMD_PREFIX` (e.g. `sg docker -c`) and Docker's NVIDIA runtime.
Unsupported flags fail before any run. Negative numbers are treated as values;
`--flag=value` is checked as `--flag`. Use separate short flags, not clusters.

Runs execute serially through `bin/run_op --agent abrun --task ID` and om_run.
Each numbered run directory gets `summary.json`; the experiment gets the
saved spec, `verdict.json`, and `REPORT.md`. Missing activation, activation in
any arm that does not expect a switch, failed runs, missing gate metrics, or preflight failure produce INVALID
and a nonzero exit. Activation failures stop immediately. Existing watchdog
settings are inherited. `--submit` queues the whole spec as one `gpu` job via
`bin/jobq`; a queue worker must execute it. Use the queue to exclude builds.

## Metrics and gate

CLI perf lines use their last occurrence. Benchmark JSON preserves pp/tg
`avg_ts`, `samples_ts`, and `steady` (mean excluding sample 1); `prompt_tps` and
`eval_tps` use avg_ts. Select `tg_steady` to gate steady benchmark throughput.
Telemetry read units are normalized to KiB (missing unit defaults to KiB).
Read/IO counters use max-minus-min. Memory and swap are MiB; GPU frequency is
MHz. Missing metrics are null with entries in `reasons`. `--no-write` prevents
summary writes; otherwise summaries are saved. Repeat `--switch-regex` as needed.

The gate accepts `{arm: [summary_object_or_path, ...]}` in execution order.
All metrics are higher-is-better. B/A uses per-arm arithmetic means. A/A gap
is the largest arm range/mean. Drift is the absolute difference of base-half
means divided by the whole base mean (odd middle run goes to second half).
Unmeasured gaps contribute zero; their per-arm fields and drift remain null.
Threshold is max(0.03, 2*A/A gap, 2*drift); strict excess gives improved/regressed.
`secondary_metrics` in verdict JSON and the report table show per-arm arithmetic
means of `nvme_gib_per_token`, `prompt_tps`, and `eval_tps`. These are descriptive,
use available finite values (null if none), and do not add gate criteria.

`regate.py OUTDIR` reads saved `spec.json` and numbered `*/summary.json` files,
recomputes the gate in run order, and writes/prints `verdict-regate.json`.
`--exclude RUN_ID [RUN_ID ...]` can be repeated; unknown IDs are rejected.
The output records `excluded_runs`, `included_runs`, and the supplied `--reason`.
`--metric` overrides the spec metric; `--no-write` only prints JSON.
Original summaries, verdicts, reports, and logs are preserved. For old summaries
without signature counts, regate reads stderr to reconstruct counts in memory.

Hashes classify outputs as identical/differs (with run indices), or unknown
when absent. This is byte equality, not a numerical-quality claim.

Tests can replace execution with `ABRUN_RUNNER` (receives om_run arguments,
bypasses run_op), and help with `ABRUN_PREFLIGHT_RUNNER` (IMAGE TOOL --help).
`ABRUN_PREFLIGHT_CACHE_DIR` relocates the cache. Python callers can inject
`preflight.check(..., runner=callable)` returning help text without Docker.

H5 measurement fields (2026-11-11): new specs default to `eval_ms_per_token`
with `direction: "lower"`; explicitly declared `eval_tps` specs retain higher-is-better
behavior. `direction` accepts `lower` or `higher`. Gate ratios are A/B for lower
and B/A for higher, with the same A/A noise, drift, and strict threshold rule.
The Python `gate.evaluate` API retains its legacy `eval_tps` default; callers may
pass `metric='eval_ms_per_token', direction='lower'`. Printed ms/token is parsed
from the last common_perf_print line (716.54 in the captured J-OM10c/003 fixture),
not reconstructed from rounded t/s. `eval_ms_total`, `eval_runs`, `load_ms`, and
`prompt_ms_per_token` preserve the other performance fields.

`regate.py OUTDIR --dry` prints the secondary table and writes nothing.
`--no-write` retains JSON output. Both recompute raw stderr/telemetry metrics even
when an old summary.json exists. Existing specs keep their original primary metric;
use `--metric eval_ms_per_token` to regate an old spec on the precise metric.

Decode uses the telemetry `mono_s` clock. Current relative stderr timestamps
(e.g. `2.20.928.587`) have no recorded absolute origin, so the fallback sets decode
end to the final telemetry sample before exit, and start to end minus
`eval_ms_total / 1000`. Load is estimated from the first telemetry sample through
`load_ms / 1000` seconds later. These are approximate elapsed phase windows:
eval time may contain inter-token gaps and the model's first generated token is
accounted in prompt evaluation. Teardown, startup, or unmeasured gaps can shift
the windows. `phase_windows` records endpoints, method and error information.
The observed maximum per-backend sampling gap is a conditional endpoint sampling
bound (roughly 0.5 s normally, occasionally 1 s in J-OM10c); unknown engine-to-exit
lag prevents a finite absolute alignment bound. No exact per-token IO claim follows.

NVMe phase deltas linearly interpolate cumulative read-only counters at the phase
boundaries, with no extrapolation. Sequentially emitted backends end microseconds
apart: each integration clips its end to that metric's last available sample.
`nvme_gib_per_token_decode` divides decode-only reads by `eval_runs`; legacy
`nvme_gib_per_token` remains the whole-run value. Counter resets invalidate phase
IO. Power uses trapezoidal integration of instantaneous `power_VDD_IN_instant`
mW, converts to joules, then divides by `eval_runs`; it includes system power.
Decode samples supply gpu_util median and nearest-rank p90, device_util median,
instant VDD_IN watts median, temp_tj maximum, and GPU/EMC frequency medians (MHz;
raw sysfs telemetry is Hz). Missing timing/coverage/metrics yield null plus a reason.
All IO, device busy, and power values remain system-wide, subject to contamination.

J-OM11 uses per-arm `switch_regex_is_config: true` on its baseline: every own
`switch_regex` must match even with `expect_switch: false`, because disabled
components print their effective defaults. Those own config matches are exempt
from leak detection; signatures belonging to other arms still count as leaks.
The flag defaults to false, preserving existing activation checks. It requires
a nonempty per-arm regex list and a boolean value. Treatment arms keep the
ordinary ALL-match checks for effective values and component activation.

J-OM12 adds optional top-level `"threads": true`. It places `OM_THREADS=1` in
all planned runner commands; om_run starts an unprivileged host proc sampler
following container startup and writes `threads.jsonl`, `threads.summary.json`,
and `threads.log`. With the variable unset, the Docker run command is unchanged.
Per-arm `"extra_argv": ["-t", "4"]` appends to common argv and is included in
image/tool preflight validation. `run_summary` recomputes the thread decode
window from the last eval timing and includes `threads.cpu_workers_r_fraction`,
`threads.main_workers_d_fraction`, and `threads.majflt_per_token` in the descriptive
secondary table. See [thread sampler](../thread_sampler/README.md) for the schema,
measured overhead, and phase alignment limits.
