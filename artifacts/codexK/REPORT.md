# T-K handoff archive observation

Implemented only under `cc/engine_handoff/`; run evidence is also retained under
`artifacts/codexK/`. No network, Docker, image import, workload execution, or
wholesale extraction was used.

## Changes

- `cc/engine_handoff/engine_handoff.py`: streaming classic/OCI observation,
  config/blob/diff-id integrity, merged executable hashes and whiteouts;
  independently observed facts and record discrepancies; direct re-observation
  during validation; requester YAML and per-field declared/observed provenance;
  partial requirements stay null/unresolved, with profiling-false null allowed.
- `cc/engine_handoff/test_engine_handoff.py`: retained 48 original admission
  checks, updated for the new observed generation behavior and `/app/run`.
- `cc/engine_handoff/test_archive_observation.py`: synthetic archive/regression
  coverage without the real 2.2 GB artifact.
- `cc/engine_handoff/fixtures/{engine.tar,build-record.json}`: inert executable
  moved from `/engine/run` to `/app/run`, with consistent identities.
- `cc/engine_handoff/example/requirements.granite.yaml`: requested sample.
- `cc/engine_handoff/README.md`: usage, provenance and certification boundaries.
- `cc/engine_handoff/example-real/`: retained facts JSON/Markdown, requester YAML,
  workload, generated handoff, verdict and timings.
- `artifacts/codexK/{run_real.py,real-run.log,pytest.log,REPORT.md}`: reproduction
  and verification evidence.

The earlier integration result in `experiments/handoff-real/verdict.json` and
its accompanying facts/handoff is the decision evidence for independently
observing archive contents rather than promoting declared build claims.

## Verification

`PYTHONDONTWRITEBYTECODE=1 python -m pytest -q cc/engine_handoff --basetemp=cc/engine_handoff/.test-tmp -o cache_dir=cc/engine_handoff/.pytest_cache`

All 66 tests pass (48 original checks plus 18 new cases). The tests include
single-byte layer payload corruption with precise layer SHA-256 rejection and
handoff-only identity edits while original transport hash/size remain unchanged.

## Real run

The real `builds/results/b0-baseline-3/engine-image.docker.tar` was inspected,
generated and validated once using the retained reproduction script. The GGUF
header was read without tensor processing. Total wall time: **46.785 seconds**
(facts 25.137 s, generation <0.001 s, validation 21.631 s). The final verdict is
**rejected**, with two distinct unresolved identity discrepancies. Exact emitted
reasons are in `cc/engine_handoff/example-real/verdict.json`.

1. `build_record.image_config_id: differs from observed archive`.
   The record's `image.Id` is
   `sha256:cf383f82e33130ce1be424b4c88a78ed12d8a693406b0c3b7ddb61afed0612d1`,
   which names the OCI index. The independently observed config blob identity is
   `sha256:7c2b92adbe4474052b3210f1361ee286f678445efa1a3ef81c244a8271f1e7a4`.
2. `build_record.executable_sha256: differs from observed archive`.
   All seven recorded tool hashes match. The archive additionally contains 12
   regular shared-library files with executable mode under `/app/`; the declared
   map therefore differs from the complete 19-entry observed executable map.
   The additional paths are:
   `/app/libggml-base.so.0.27.0`, `/app/libggml-cpu.so.0.27.0`,
   `/app/libggml-cuda.so.0.27.0`, `/app/libggml.so.0.27.0`,
   `/app/libllama-bench-impl.so`, `/app/libllama-cli-impl.so`,
   `/app/libllama-common.so.0.7.0`, `/app/libllama-completion-impl.so`,
   `/app/libllama-perplexity-impl.so`, `/app/libllama-server-impl.so`,
   `/app/libllama.so.0.7.0`, `/app/libmtmd.so.0.7.0`.

Validation separately reports those two record/handoff claim mismatches in
addition to the two carried unresolved entries (four emitted reasons total).
There are no other unresolved entries. Platform is `linux/amd64`, there are 11
layers, archive hash/bytes match the record, and requested requirements are
filled. No identity discrepancy was cleared or silently reconciled.

## Open issues and limits

The Build CC record needs the config digest and a complete executable map (or an
explicit future schema distinguishing tool subsets). Its location is outside
this assignment's write scope, so it was not changed. The generated handoff
already contains the observed values. Source/build provenance, runtime and GPU
availability, requester authority and actual test success remain outside this
receipt inspector's certification boundary.
