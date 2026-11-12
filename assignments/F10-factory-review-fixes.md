# F10 — Review fixes + one config/redaction module (no behaviour drift)

Repo: `/home/muk/projects/jiuwenswarm` (`ai4r/iterative-cc-factory`, `.venv`). Factory home `~/.jiuwenswarm-factory` is READ ONLY for you. An independent review found the defects below; each was verified by reading code. Fix them with tests. Keep diffs small and the code elegant — this code is shown in a talk.

## Defects to fix (each needs a regression test in tests/factory/)
1. jobs.py ~165: never downgrade a terminal attempt (succeeded/failed/cancelled) to `unknown`; a vanished jobq job file = "no change". Route every state change through the TRANSITIONS table.
2. api.py `evidence.publish` / store: agents must not claim host-owned ids. Reject agent artifact/obs ids containing `:` or starting with host prefixes (`admission`, attempt ids known to the store); agent writes cannot set `host_run` or host kinds (`build_result`, `lab_result`, `evaluation`, `admission`, `operator_intervention`).
3. controller/store `advance_decision`: every `evidence_ids` entry must exist in this task's evidence (or a granted read_tasks task); a terminal action (`accept`, `reject`, `report_inconclusive`…) with dangling ids is refused with a clear error.
4. jobs.py ~121: `jobq submit` CalledProcessError keeps `stderr[-2000:]` in `resources.submit_error` and fails the attempt with that reason; only timeout/OSError go to the recover path.
5. jobs.py ~199 `cancel`: bounded wait (default 60 s), then return current state.
6. Secret patterns: `sk-` must be anchored (`(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}`); email pattern must require a real `user@domain.tld`. `scrub_bytes(b'task-abcdefgh disk-space-low')` must be unchanged.
7. export.py: remove the `operator-` special case; add `--extra-file PATH` (repeatable) and use it from docs instead.
8. Controller.cancel routes lab attempts through `Manager.cancel` (container reap).
9. manager.retry_spool: per-manifest try (parse inside), error text `str(exc)[:200]`.
10. build_job.py: `root.mkdir` inside try/finally (reuse an existing empty dir, else suffix); SIGTERM → SystemExit so the `finally` records a `build_result` observation.
11. manager.freeze_reference crash-atomic (artifact first; re-drive missing transitions when the attempt exists).
12. inspector preflight cache under `store.home`, not the env/default home.
13. tests: conftest monkeypatches `shutil.disk_usage` for builder tests so they don't depend on host disk.
14. contract.load: treat id as a path only if it contains `/` or ends in `.json`; `freeze` writes to an explicit path argument.

## Polish
- New `jiuwenswarm/factory/config.py`: one place for host constants, env-overridable with today's defaults: `SIMINFENG_ROOT` (default `~/projects/siminfeng`), jobq bin, build_engine, om_run, probe model, `DOCKER_PREFIX` (`sg docker -c`), min avail MiB, timeouts, default baseline image. Replace every `/home/muk` literal in jiuwenswarm/factory (grep must return nothing outside tests/fixtures and docs).
- New `jiuwenswarm/factory/redact.py`: single redactor used by controller.scrub, export, api.input_reason.
- One effective-config/ENV_KEYS module shared by inspector and lab_runner.
- Drop unused module-level wrappers (builder.submit/status/cancel/result if no callers outside tests — update tests/demo scripts that use them; controller.init_task/run/resume module functions; `_init_task` indirection). Move `builder_reuse_keys` DDL into the Store schema (with migration for existing DBs: CREATE TABLE IF NOT EXISTS).

## Do NOT change (intentional, operator-owned)
- `Store.query` calls `read_artifact` on every artifact row of the page. This is a deliberate, documented seed for a later improvement experiment. Leave the behaviour and its cost exactly as is; do not add caching anywhere in the query path; do not mention it in comments or docs.
- Contracts under `jiuwenswarm/factory/contracts/`, exported bundles, the factory home.

## Docs
Fix contradictions in `docs/architecture/factory/DEMO-10MIN.md` vs `artifacts/factory/COMPLETION-REPORT.md`: Docker identity is live in re-exported bundles; attempt 2 also had one inspector input-shape rejection (repaired); say plainly it is a scripted walk-through of recorded runs, not a timed rehearsal.

## Acceptance
- G1 `.venv/bin/python -m pytest tests/factory --no-cov -q -p no:cacheprovider` all pass (with real disk < 8 GiB too).
- G2 `grep -rn "/home/muk" jiuwenswarm/factory` → only config.py defaults expressed via `Path.home()` (i.e. no literal) .
- G3 `factory story`/`status`/`capsules` from F09 still work on `artifacts/factory/evidence/factory-d4b71bcce337406b96f627d469bec092`.
- G4 No commits/pushes/builds/lab runs; no writes to `~/.jiuwenswarm-factory`.
Final message: per-defect status with test name; anything not done.

## Story/status nits (from F09 output on the real bundle)
- `eval_ms_per_token`, `max_tokens`, `min_tokens` are scrubbed to `<redacted>`: the sensitive-key rule matches any key containing "token". Restrict it to real credential keys (`access_token`, `refresh_token`, `id_token`, `api_key`, `authorization`, `password`, `secret`, `*_token` only when the value looks like a credential). Re-export the three bundles in `artifacts/factory/evidence/` with the fixed redactor (export is read-only on the store; this overwrite of bundle dirs is allowed) and verify them.
- story: the terminal decision line must end on a word/sentence boundary (wrap up to 3 lines, then `…`), never mid-token like `:stdo`.
- story: the effective-config ✓ line shows only `k=8 · pregate=1 · fill=1 · matches request` style (keys that differ from engine defaults or that the request set), ≤ 1 line.
