# RSI seed: deliberate inefficiency in the factory `evidence` capsule (operator-only)

Kept out of the jiuwenswarm repo on purpose, so an improver must find the problem from telemetry and workflow data, not from docs. Share it with the RSI owner, who decides how much to reveal.

## Seed
- **Capsule:** `factory.evidence`, capability `query`. It is the most-called capability: the agent queries evidence before every decision.
- **Defect:** `jiuwenswarm/factory/store.py` `Store.query`, `for row in rows: if row['type']=='artifact': self.read_artifact(...)`. Each query reads and re-hashes every artifact blob in the result page, up to 50, and then throws the bytes away. The result does not depend on that work.
- **Why it's realistic:**
  - It started as integrity checking on the read path.
  - The cost is invisible on small tasks: 21 ms on the accepted demo task, about 11 MB of artifacts.
  - The cost grows with artifact bytes. Each lab or A/B run adds about 11–90 MB: telemetry, threads.jsonl and watchdog files. Over an evolve task with many runs, query latency climbs to seconds.
  - One corrupt blob also makes every query fail. That is a robustness smell as well.
- **Telemetry signature** (factory.telemetry/1):
  - `cc.call` with `capsule=factory.evidence, capability=query`: `duration_ms` rises with `task_artifact_bytes`, while `out_bytes` stays flat.
  - The other capabilities' latency stays flat.
  - The signal is clear in the evolve run and in `factory telemetry TASK`.
- **Expected fix (easy):** don't read or hash blobs in `query`; verify on `read`/`read_artifact` only, or check existence and size. Outputs stay byte-identical.
- **Acceptance for an improvement:**
  - `python -m jiuwenswarm.factory.replay TASK` gives byte-identical query outputs.
  - p95 `duration_ms` for `evidence.query` is flat in `task_artifact_bytes`.
  - `pytest tests/factory` passes.
  - Integrity is still enforced on `evidence.read` (the tamper test still fails reads).
- **Do not fix in the main line.** F10/F11/F12 assignments explicitly preserve it.

## Secondary, organic candidates (not seeded, found in review)
These are real but not planted:
- The agent polls `*.status` between turns; there is no long-poll capability.
- Three redactors existed before F10.
- `lab_client.result` returns large artifact lists.
