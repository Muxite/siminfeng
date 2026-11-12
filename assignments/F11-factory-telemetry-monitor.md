# F11 — Telemetry stream for RSI + live web monitor

Repo: `/home/muk/projects/jiuwenswarm` (`ai4r/iterative-cc-factory`, `.venv`). Read `jiuwenswarm/factory/{api,controller,jobs,store,story,cli}.py`, `docs/architecture/factory/CONTROLLER.md`. Factory home `~/.jiuwenswarm-factory` READ ONLY for you (tests use temp homes).

Context: a separate RSI system (someone else's) will consume factory telemetry + workflow data to analyse and improve capsules (tools/skills/components). It needs a stable, documented, append-only event stream. Audiences also need a simple live monitor to watch the factory work. Stdlib only.

## 1. Telemetry (`jiuwenswarm/factory/telemetry.py`)
Append-only JSONL at `<home>/telemetry/events.jsonl` (fsync per line, file lock; never raises into the caller — failures counted in a sidecar `telemetry/errors.jsonl`). Envelope: `{schema: "factory.telemetry/1", event, ts, task_id, turn, seq}` + fields:
- `cc.call` — emitted by the `api.main` dispatch wrapper for EVERY native capsule call: `capsule` (id), `capsule_version`, `release_sha256` (sha of the capsule folder tree, cached per process), `capability`, `duration_ms`, `in_bytes`, `out_bytes`, `outcome`, `error_code`, `task_evidence_records`, `task_artifact_bytes` (store totals at call time — generic context).
- `cc.material` — material reads, if observable from the agent session stream in the controller (tool name `cc_material`/`cc_list`, capsule id, path, bytes); otherwise document as not observed.
- `agent.turn` — controller per turn: start, `duration_ms`, tool calls by name (from the stream events), token usage if the Codex adaptor stream reports it (look for usage/token fields; record `null` when absent — do not invent), outcome.
- `job.transition` — jobs.py on each attempt state change: attempt_id, kind, from, to, `queue_wait_s`, `run_s` when known.
- `decision` — rev, action type, evidence id count, reason length.
- `build.result` / `lab.result` / `evaluation` — wall_s, cache_hit, ms_per_token, correctness, execution_outcome, engagement x/y.
All events carry task_id; the controller sets `FACTORY_TURN` env for capsule subprocesses so `cc.call` gets `turn`.

`jiuwenswarm factory telemetry [TASK] [--json] [--since TS]`: per-capability table (calls, p50/p95/max ms, mean out_bytes, errors), per-turn table (duration, tool calls, tokens), job table (queue wait/run), top-3 "where time went" lines. Export (`export.py`) includes the task's telemetry slice as `telemetry.jsonl` in bundles (+ manifest).

`docs/architecture/factory/TELEMETRY.md`: schema per event, guarantees (append-only, ordering by seq, what is NOT captured), how an external improver proposes a capsule revision (new capsule version folder → new release sha → compare `cc.call` distributions before/after on the same workflow), and the replay recipe: `python -m jiuwenswarm.factory.replay TASK` re-issues the recorded `cc.call` inputs of read-only capabilities (`evidence.query`, `evidence.read`, `*.status`, `*.result`, inspector `inspect`) against a copy of the store and emits fresh `cc.call` events tagged `replay:true` — the cheap offline benchmark for capsule changes. Implement replay (read-only capabilities only; refuse others).

## 2. Monitor (`jiuwenswarm/factory/monitor.py`, `jiuwenswarm factory monitor [--port 8787] [--bind 127.0.0.1]`)
`http.server` ThreadingHTTPServer, read-only. Endpoints: `/` single self-contained HTML (inline CSS/JS, no external URLs), `/api/tasks`, `/api/story?task=`, `/api/telemetry?task=&since=`, `/api/job?attempt=` (state + last 20 lines of the job log, scrubbed via redact). Page polls every 2 s. Layout for a projector (dark, large type, readable at 1280×720, works at 400 px wide):
- Header: task short id, objective (one line), outcome badge, turn/budget, elapsed.
- **Engine stack** column: the base engine and each component applied, in order, with state (integrating / built / measured / kept ✓ / reverted ✗) and its measured ms/token + speedup vs base when present (derive from decisions/evaluations/lab results; generic, not hardcoded).
- **Timeline**: story events (same data as `factory story --json`), newest at bottom, auto-scroll, ✗ events red.
- **Now**: latest decision action + reason; active job card with elapsed timer and log tail.
- **CC calls**: last 30 `cc.call` events: capsule.capability, duration bar, outcome.
Escape all text (no innerHTML of record data). Task selector when several tasks exist; default newest.

## 3. Tests
Telemetry: every api dispatch emits exactly one `cc.call` with correct capsule/capability/outcome (incl. error path); job transitions emitted; telemetry failure never breaks a call; export includes slice; replay refuses mutating capabilities and reproduces read-only outputs byte-identically. Monitor: start on port 0 in a thread, GET each endpoint with a temp store, HTML contains no `http://`/`https://` external refs, JSON shapes.

## Acceptance
- G1 `.venv/bin/python -m pytest tests/factory --no-cov -q -p no:cacheprovider` all pass.
- G2 `.venv/bin/jiuwenswarm factory telemetry --help` and `monitor --help` exit 0.
- G3 `jiuwenswarm factory monitor --home <tempcopy-of-~/.jiuwenswarm-factory> --port 0` serves `/api/story?task=factory-d4b71bcce337406b96f627d469bec092` with the real chain (copy the home to /tmp first with `cp -a`, then delete the copy).
- G4 No commits/pushes/builds/lab runs; no writes to `~/.jiuwenswarm-factory`. Do not touch `Store.query`'s behaviour.
Final message: files, event schema summary, what the adaptor stream does/doesn't expose (tokens), anything not done.
