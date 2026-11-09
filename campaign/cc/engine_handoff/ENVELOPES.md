# Shared job envelopes (T-S)

`envelopes.py` is a thin offline projection of Build CC records and `bin/lab`
job/attempt/export records. It does not create a new executor, provision a
sandbox, contact a receiver, inspect processes, run Docker, or use the network.
The other team's lab agent is not exercised by these checks.

## Python and CLI contract

The three entry points are `make_proposal(build_record, job_spec, host_profile,
policy)`, `check_admission(proposal, host_profile)`, and
`make_result(job_dir, attempt, claim_category="unclassified", *,
manifest_path=None)`. They return JSON-compatible dictionaries. Inputs to the
first two functions are dictionaries; `job_spec` accepts either the immutable
`job.json` wrapper or its `spec`. No argument dictionaries are mutated.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 cc/engine_handoff/envelopes.py proposal \
  --build-record builds/results/b0-baseline-5/build-record.json \
  --job-spec lab/jobs/job-ad883810b3/job.json \
  --host-profile cc/engine_handoff/hosts/x86-main.json \
  --policy cc/engine_handoff/fixtures/envelopes/policy.example.json \
  --out artifacts/codexS/proposal-real.json

PYTHONDONTWRITEBYTECODE=1 python3 cc/engine_handoff/envelopes.py admit \
  --proposal artifacts/codexS/proposal-real.json \
  --host-profile cc/engine_handoff/hosts/jetson-arm.TEMPLATE.json

PYTHONDONTWRITEBYTECODE=1 python3 cc/engine_handoff/envelopes.py result \
  --job-dir lab/jobs/job-8df3c7bb27 --attempt att-1 \
  --claim-category interrupted-observation \
  --out artifacts/codexS/result-interrupted.json
```

All commands write JSON to stdout unless `--out` is supplied. `admit` returns
0 for accepted and 2 for rejected or needs-information. Malformed files or
inconsistent result ownership return 2 with a JSON error on stderr. Generation
of an unresolved proposal returns 0: inspect `dispatchable` and `unresolved`.

The example policy is a **new requester declaration**, not recovered historical
policy. It specifies whole-job wall time and capture bytes; the original lab's
per-test `resources.timeout_s` is deliberately not relabelled as a job budget.
Policy supplies `campaign_id`, `experiment_id`, `attempt_id`, `release_role`,
`build_record_ref`, `wall_time_s`, `capture_bytes`, `memory`, and `evidence`.
`request_id` defaults to the recorded job ID or spec key. Explicit spec IDs take
precedence over policy IDs. Unknown IDs are null, never inferred from filenames.

## Proposal fields

| Field | Meaning |
|---|---|
| `format` | `campaign-job-proposal/0` |
| `campaign_id`, `request_id`, `experiment_id`, `attempt_id` | Requester/controller identities |
| `target.host_id`, `target.platform`, `target.required_capabilities` | Selected receiver and image's required platform/runtime |
| `target.platform_compat` | `cuda_arch` from CMake, plus optional declared `jetpack_l4t`, `driver_min` from `spec.platform_compat` |
| `release.role`, `release.source` | Release role, recorded source commit and tree |
| `release.patch_sha256`, `release.cmake_defines` | Patch identity (null means unpatched), all parsed CMake flags plus schema 2 defines |
| `release.transport` | `kind: docker-archive`, recorded SHA-256 and bytes |
| `release.image_config_id`, `release.index_digest` | Separate config/index identities; unavailable index may be null |
| `release.executables_sha256`, `release.build_record_ref` | Recorded `/app` file hashes (libraries included), record locator |
| `inputs` | `spec.inputs` model, prompt/data, and canonical test-definition hashes |
| `invocation.tests` | Ordered test occurrences with argv, environment, declared cache state |
| `invocation.mounts` | Role, access, optional container target; no host bind paths |
| `policy` | Whole-job wall/capture budgets, memory domain/budget/guarantee/accounting, `gpu_jobs_on_host: 1` |
| `evidence` | Controller outside sandbox, durable destination reference, expected output names |
| `unresolved`, `dispatchable` | Recomputed identity/policy/receiver gaps; dispatchable only when empty |

Nested build-record schemas 1 and 2 are accepted. `image.config_digest` is used
for config identity; Docker's reported `Id` is never substituted for it. A null
index digest is retained rather than fabricated from an archive catalog or an
untyped Docker ID. If the job's image ID is not any recorded config/reported/
index identity, the build/job pairing is unresolved. This occurs in the specified
real records: they are useful fixtures, but are not a matching dispatchable pair.

The specified jobs are also legacy submissions without `spec.inputs`. Their
`model_sha256` is not promoted into a pretend full input manifest. Missing input
identities reject admission; missing transport SHA requests more information.
Other unknown required declarations result in needs-information. A known
platform, CUDA architecture, driver floor, capability, or memory guarantee
mismatch rejects admission even if other fields remain unknown. `dispatchable`
and any sender-supplied flag are never trusted by admission.

A newly pinned proposal can supply `spec.test_definitions`, using the existing
`bin/lab` definition mapping. Definition identity is the same canonical SHA-256
as `bin/lab`: sorted keys, compact separators, ASCII JSON. The adapter constructs
`/app/<tool> -m /m/<model_file>`, optional `-f /d/<data_file>`, and definition args.
It preserves occurrence order, including repeated tests. Read-only `-v` and
`--volume` forms become input roles. Other Docker flags stay unresolved.
`spec.invocation` may instead explicitly declare the complete invocation.

Host paths from `inputs.mounts` become `prompt-0`, `prompt-1`, etc. in sorted host
locator order; file manifests retain relative names and hashes. Data locators
become basenames. Model/data default container targets are `/m` and `/d`; results
is `/results`. Prompt roles retain their declared container target, e.g. `/tin`.
Absolute argv paths must resolve to a hashed input in these mounts (or a results
output). Receiver staging resolves roles to its own paths. Build-record and
durable-store references remain locators, not bind instructions.

**Trust boundary:** proposal values are copied declarations. This adapter does
not re-observe archive bytes or prove model completeness, provenance, authenticity,
or invocation safety. The existing `engine_handoff.py` archive observation and
requirements receipt must still bind delivered bytes before execution. An
accepted compatibility receipt is neither execution nor test assessment. The
requested GPU concurrency of one is not a new GPU lock or capacity reservation.
Cooperative interception requires the right engine/shim and demonstrated runtime
interception (D021/D023); a host capability declaration does not prove these.
Cgroup host RAM accounting is not a discrete VRAM guarantee.

## Result fields and evidence limits

`campaign-job-result/0` keeps `execution_outcome`, `test_assessment`, and
`claim_category` separate. Claim category is caller-supplied, defaults to
`unclassified`, and is not inferred from `quality.all_ok`. There is no top-level
success boolean. Historical `quality` is retained as an assessment declaration,
not re-scored or upgraded into a correctness/performance claim.

`evidence_completeness` contains numeric `expected`, `started`, `clean`, and a
`missing` list of expected test occurrence IDs lacking clean completion. It also
reports `manifest_present` and `journal_errors`. Clean means a durable per-test
assessment with exit code zero and explicit `partial: false`; it does not mean
the test assessor passed. Interrupted legacy attempts with no per-test assessment
have zero established clean tests, even when stdout files exist. Partial evidence
is never rescored; incomplete assessments are marked `assessed: false` while
retaining recorded quality. Nine tests started in the interrupted real attempt;
ten completed cleanly in its new attempt. Attempts never share coverage or tokens.

`last_durable_sequence` records physical state-journal line count and the last
valid record timestamp. Torn/invalid lines are listed and suppress assessment;
line count is an observed local-file position, not a remote acknowledgement or
proof of fsync/host-loss durability. `resource_generation` contains attempt and
SHA-256 of the owner token, never the raw token; conflicting recorded token hashes
or attempt/job IDs are refused. `stop_confirmed` requires recorded cleanup
confirmation and an explicitly empty owned-container list. A cancellation request
or terminal journal entry alone cannot confirm stop; no new process probe occurs.

`artifact_refs` lists path + SHA-256 **from export MANIFEST.json**, not recomputed
hashes of an attempt directory. Default lookup follows `lab/jobs/<job>` to
`lab/export/<job>/<attempt>/MANIFEST.json`; `--manifest` overrides the locator.
Missing manifests yield no artifact refs and `manifest_present: false`. These are
recorded identities, not a fresh export verification. `bin/lab` exports before
appending its final terminal event, so an exported state journal may validly have
a different hash/length from the live journal read for `last_durable_sequence`.
Controller/evidence on the same host protects against sandbox/process loss only.

## Fixture provenance and verification

`fixtures/envelopes/build-record.json` is an identity-field excerpt of
`builds/results/b0-baseline-5/build-record.json` (large Buildx provenance omitted).
The two jobs' job/owner/result/state files and export manifests are byte-for-byte
copies of the requested real records; stdout/stderr payloads are not copied.
`test-definitions.json` copies relevant current lab definitions as supplemental
declarations, not as retrospective proof of legacy input identity. Tests explicitly
add hashes and change the proposed image binding in memory for their accepted
case. The synthetic prompt hash in that case tests schema plumbing only.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q cc/engine_handoff \
  --basetemp=cc/engine_handoff/.envelopes-test-tmp \
  -o cache_dir=cc/engine_handoff/.pytest_cache
```

## Questions for the lab-agent team

- What JetPack/L4T, driver/runtime compatibility, and memory accounting/enforcement
  domains can the Jetson receiver attest to? Fill the template before admission.
- How will the receiver stage role-bound inputs and independently verify the
  archive/config/executable identities against delivered bytes?
- What receiver acknowledgement, owner-generation binding, and stop-confirmation
  evidence will precede retries and prevent stale result delivery?
- Which service enforces whole-job wall/capture budgets and GPU concurrency, and
  what durable destination survives receiver host loss?
