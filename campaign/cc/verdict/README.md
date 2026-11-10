# Frozen-criteria verdict tool

`verdict.py` uses Python's standard library, reads recorded evidence, and emits
`verdict/1` JSON to stdout. Exit status is 0 for PASS, 1 for FAIL (including
missing data, invalid protocols, and unreadable files). It performs no writes.
All criteria are conjunctive; an empty criteria list cannot pass.

Run from any directory:

```sh
python3 -B cc/verdict/verdict.py cc/verdict/criteria/D029-q8_reuse.json
python3 -B cc/verdict/verdict.py cc/verdict/criteria/D033-q8_0.json
python3 -B cc/verdict/verdict.py cc/verdict/criteria/D033-q4_0.json
python3 -B -m unittest discover -s cc/verdict -p 'test_*.py' -v
```

Use absolute script/criteria paths outside the campaign directory. `--root`
defaults to the campaign directory containing this script; it resolves legacy
sidecar paths. `--lab` overrides `$LAB`, which otherwise defaults to `ROOT/lab`.
No external script is executed by the tool. Tests also compare its aggregation
with the original parent-project `.claude/skills/frozen-experiment/scripts/agg.py`.
Test temporary directories are created and removed under `artifacts/codexVER/`.

## Real-record outcome and scope

| Criteria file | Real job | Result | Evidence |
| --- | --- | --- | --- |
| D029-q8_reuse.json | job-34c09bec98 / att-1 | PASS | Q8_0 tg median 791.2678675 → 807.629565, +2.067782%; text and printed PPL equal; CUDA MUL_MAT 2445/2445 |
| D029-mixE4K.json | same | FAIL | +1.782166%, below its separate 2% threshold |
| D033-q8_0.json | job-a945fa8656 / att-1 | PASS | Reported PPL gate: 7.5385 → 7.5392, +0.009286% |
| D033-q4_0.json | same | FAIL | Reported PPL gate: 7.5385 → 8.1368, +7.936592% |
| D033-q8_0-frozen.json | same | FAIL | Full D028 gate: greedy stdout is different; KLD pending |
| D033-q4_0-frozen.json | same | FAIL | Full D028 gate: PPL too high and greedy stdout differs |

The D033 q8_0 acceptance reproduces **only the PPL-based decision reported in
D033**, not an unconditional acceptance under D028. D028 says PPL within +1%
AND identical greedy text or (if not) KLD pending. D033 summarizes a ±1% rule;
the reported-gate files use absolute percentage delta. Both interpretations
produce the same PPL verdicts for these jobs. Actual full stdout hashes begin
`96310158` (f16), `4e8a7ed7` (q8_0), and `df477f77` (q4_0). No KLD measurements
exist in this job, and D028 supplies no KLD threshold. The full-protocol files
fail on nonidentity without inventing either a KLD value or a waiver.

D029 includes primary ON/OFF identity, PPL identity for both model groups,
and the secondary base/OFF/ON identity from `artifacts/x4-c1/gen-*.out`.
Backend correctness comes from `artifacts/x4-c1/tbo-mulmat.out`, an external
legacy log, explicitly marked as such. Its association with the job is supplied
by the criteria and D029, not independently established by lab metadata.
Sidecars have frozen SHA256 values in criteria. `build_criteria.py` is an
optional authoring utility which rebinds those hashes to current bytes: do not
rerun it as part of evaluating a frozen protocol. Review and preserve the
criteria file before observing new results. Output records its SHA256.

D029's host policy checks the available `cc=0` compiler counter, with full
execution coverage and no sampling gap over 15 seconds. It reports 24 samples
bracketing this job, whereas D029's 38 samples describe the larger monitor log
spanning two jobs. Downloads and unrelated CPU inference were not measured by
this counter; they are not asserted absent. Add required counters to future
protocols; a missing counter fails instead of defaulting to zero. The 15-second
sampling tolerance is an explicit tool coverage policy, not a scientific
threshold claimed by D026.

## Criteria/1

A JSON object has `schema: "criteria/1"`, an optional `id`, `scope`, `sources`,
and a nonempty `criteria` array of objects with unique `id` fields. Paths in
criteria are read-only. Numeric criterion example:

```json
{
  "schema": "criteria/1",
  "id": "example",
  "criteria": [{
    "id": "decode gain",
    "metric": "tg_ts",
    "candidate": {"job": "job-34c09bec98", "attempt": "att-1", "test": "c1_q8_tg_r1", "count": 4},
    "baseline": {"job": "job-34c09bec98", "attempt": "att-1", "test": "c1_q8_tg_r0", "count": 4},
    "mode": "delta_pct", "op": ">=", "threshold": 2,
    "alternating": {"group": ["c1_q8_tg_r0", "c1_q8_tg_r1"], "rounds": 2, "repeat": 2}
  }]
}
```

Numeric metrics: `decode_tps`, `prefill_tps`, `ppl`, `kld`, `top1_pct`, `tg_ts`,
`pp_ts`, `gpu_delta_mib`, `host_cgroup_peak_mib`. Modes are `value` (default),
`delta_pct` (100 × (candidate / baseline − 1)), `abs_delta_pct`, `abs_delta`.
Comparators are `>=`, `<=`, `==`, `>`, `<`. Percentage thresholds use percentage
points (2 means 2%, not 0.02). For PPL absolute delta use metric `ppl`, mode
`abs_delta`; for KLD use metric `kld`, mode `value`, `<=` the frozen threshold.
Zero or negative percentage baselines and nonfinite numbers fail.

The per-test metric is the median over invocations, matching `agg.py`. Bench
`tg_ts`/`pp_ts` uses each invocation's last `avg_ts` for generation/prefill;
`samples_ts` is not pooled. Memory metrics use max, as in `agg.py`. Every
selected invocation must have a finite value; missing values are never dropped.
Each reference requires an exact sample count and explicit attempt. Optional
`alternating` verifies the actual submitted sequence filtered to its group,
using `make_order.py`'s forward/reverse rounds and repeat semantics. For groups
with more than two members, list the entire group in its original forward order.

Boolean metrics require `op: "=="`, `threshold: true`:

- `text_sha`: `groups` contains test references. Hashes full stdout bytes with
  full SHA256 (the `text_fp.py` rule without its 8-character display truncation),
  checking all selected invocations across all groups. At least two are needed.
  Optional `files` holds `{path, sha256}` objects for pinned legacy stdout files.
  Compare models/prompts separately: differing test groups must be separate
  criteria if identity is not expected between them.
- `backend_ops`: requires `job`, `attempt`, pinned `path`/`sha256`, `backend`
  (e.g. CUDA0), and `operation` (e.g. MUL_MAT). Checks that backend's pass count
  equals a positive total, all operation lines are present and OK, and its final
  status is OK. A CPU or other backend's summary cannot satisfy CUDA correctness.
- `host_quiet`: requires `job`, `attempt`, pinned `path`/`sha256`, `counters`
  (e.g. `{"cc": 0, "dl": 0}`), `max_gap_s`; optional `max_load_1m`. Reads the
  timestamped loadmon log (`timestamp load1 load5 load15 key=value ...`),
  summarizes samples bracketing the running→assessing interval, and compares
  counter maxima to configured limits. Missing counters, missing endpoint
  coverage, nonincreasing timestamps, and excessive gaps fail. Summaries retain
  observed load range, counter maxima, count, interval, and log/state hashes.

Every referenced job must match its explicit job, attempt, image, submitted
order, full coverage, successful execution, all_ok, and every clean row. All
recorded stdout/stderr files must exist and match the lab result's evidence
hashes, including files for unselected tests. This is deliberately stricter
than `agg.py`'s warning header. Results preserve per-sample values, ranges,
aggregates, result/job hashes, job IDs, and external evidence provenance.
A metric failure carries its observed value; unavailable evidence carries an
error and FAIL. The JSON is a deterministic evaluation, not a signed attestation
that the recorded result metadata or frozen criteria are authentic.

## Other protocol checks

Tests read real D035 and D040 jobs too: PAP KLD 0.07264 passes 0.08, noisy D035
speed +9.8421% passes 5%, but the D040 rerun's +1.4968% fails. Both monitor logs
end before their final test completes; this tool fails their full-run host
coverage checks. It does not promote a provisional acceptance to PASS by
ignoring host evidence. No missing scientific results have been synthesized.
