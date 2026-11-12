# F02 — Factory evidence store + scoped API (T02)

Repo: `/home/muk/projects/jiuwenswarm` (branch `ai4r/iterative-cc-factory`, venv `.venv`, Python 3.12). Read first: `AGENTS.md`, `docs/architecture/factory/AMENDMENT.md`, `docs/architecture/ccv2/README.md`, `jiuwenswarm/ccv2/ccv2.py` (native packet: JSON on stdin `{call, attempt, capability, inputs, output_dir, resources}`; process must print one JSON `{outcome: ok|partial|error, outputs, reason?, error?}`; env is inherited from the MCP server), `artifacts/factory/intake.json`.

## Build
New package `jiuwenswarm/factory/` (stdlib + already-installed deps only):

1. `store.py` — `class Store(home: Path)`; files `home/factory.sqlite` (WAL, `synchronous=FULL`) and content-addressed `home/artifacts/<sha256[:2]>/<sha256>`. Records (all with `task_id`):
   - `tasks(task_id, objective, constraints json, allowed_effects json, target json, budget json, revision, current_decision_rev, evidence_cursor, created)`
   - `decisions(task_id, rev, parent_rev, evidence_ids json, observations json, hypothesis, uncertainty, action json, expected_evidence, budget json, stop_condition, reason_for_change, created)` — `advance_decision(task_id, expected_rev, record)` is compare-and-swap: if `current_decision_rev != expected_rev` raise `StaleRevision`; on success current pointer = expected_rev+1.
   - `artifacts(artifact_id, task_id, sha256, size, media_type, producer_attempt, location, complete, created)` — immutable; `put_artifact(task_id, artifact_id|None, bytes|path, media_type, producer_attempt)`; re-put of same id+same bytes returns the existing record (idempotent); same id + different bytes raises `ArtifactConflict`. Verify sha on every read.
   - `observations(obs_id, task_id, source_artifacts json, kind, values json, error json, host_run json, confidence, limitations, created)` — append-only (no update/delete API). Keep interpretation out (that belongs in decisions).
   - `attempts(attempt_id, task_id, kind, idem_key, fingerprint, request json, state, jobq_id, owner json, started, ended, resources json, result_refs json, cleanup_state, history json)` — state-transition helper validates the allowed set: `queued, running, succeeded, failed, cancel_requested, cancelled, unknown`.
   - `evaluations(eval_id, task_id, contract_id, contract_version, contract json, inputs json, execution_outcome, correctness, evidence_completeness, created)` — three separate fields, never collapsed.
   - `uploads(upload_id, task_id, lab_run_id, sha256, ack_state, created)` for Manager receipts (used later by F06).
   - Durability: every write in a transaction; `put_artifact` writes temp file → fsync → rename → fsync dir → then DB commit; return an ack **only after commit**. Injectable failure hook (e.g. `Store(home, fault=callable)`) so tests can force a persistence failure and assert no ack + nothing half-visible.
   - Scope: `Store.scoped(task_id, readable=frozenset())` returns a view whose methods reject any other task id with `ScopeDenied` unless that id is in `readable` (explicit cross-task authorization).
2. `api.py` — `python -m jiuwenswarm.factory.api <capability>` entry used by native CC scripts: reads the ccv2 packet from stdin, resolves `FACTORY_HOME` (default `~/.jiuwenswarm-factory`) and `FACTORY_TASK_ID` (required; scope) and optional `FACTORY_READ_TASKS` (comma list) from env, dispatches, prints exactly one ccv2 process-result JSON. Implement now the evidence capabilities:
   - `evidence.query` inputs `{kinds?:[...], text?:str, since?:int, limit?:int<=50}` → outputs `{items:[{id, type, kind, summary(≤300 chars), created}], cursor}` — bounded, never dumps whole logs.
   - `evidence.read` inputs `{id}` → the full record (artifacts: metadata + up to 64 KiB text preview + sha256; never absolute host paths beyond the store-relative location).
   - `evidence.publish` inputs `{type: observation|decision|artifact, record}`; decisions require `expected_rev` and go through `advance_decision`; artifacts accept `{path}` that must be inside `output_dir` or the agent workspace (`FACTORY_WORKSPACE` env), or inline `text`.
   - Errors → `{"outcome":"error","error":{"code":..., "message":...}}` (codes: `SCOPE_DENIED, STALE_REVISION, ARTIFACT_CONFLICT, NOT_FOUND, INVALID_INPUT, PERSISTENCE_FAILED`).
   - Also a small human CLI `python -m jiuwenswarm.factory.store {init-task,show,dump}` (init-task takes a JSON task file).
3. `tests/factory/test_store.py` (+ `test_api.py` running `api` as a subprocess with a packet on stdin).

## Acceptance (run verbatim; all must pass)
- A1 `.venv/bin/python -m pytest tests/factory --no-cov -q -p no:cacheprovider` passes, including tests named: `test_restart_roundtrip` (close/reopen store, same bytes+sha), `test_stale_decision_rejected`, `test_artifact_conflict_rejected`, `test_idempotent_artifact_put`, `test_persistence_failure_no_ack`, `test_cross_task_denied`, `test_cross_task_allowed_when_granted`, `test_observations_append_only` (no update/delete path exists), `test_api_query_bounded` (limit>50 → INVALID_INPUT; summaries ≤300 chars), `test_api_publish_decision_cas`.
- A2 `.venv/bin/python -m pytest tests/ccv2 --no-cov -q -p no:cacheprovider` result is unchanged from baseline (93 passed, 10 known host failures listed in `artifacts/factory/intake.json`). Do not "fix" those here.
- A3 Only new files under `jiuwenswarm/factory/` and `tests/factory/`. No commits, no network, no Docker, no edits elsewhere.

Final message: list files, test counts, and any design choice the next task (jobs/builder/manager/capsules) must know (exact function signatures).
