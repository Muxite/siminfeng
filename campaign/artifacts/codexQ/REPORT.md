# T-Q: lab input identity

Base: `2cf2925`. Changed `bin/lab`; added `bin/test_lab_inputs.py` and this report. No Docker, GPU, or network access was used. Existing unrelated working-tree changes were left alone.

New submissions preserve the original spec fields and add `inputs`:

- `model_files`: each distinct model basename actually selected by the tests, mapped to its file SHA-256. Test overrides must name files in the default model's directory.
- `mounts`: absolute host directory to a manifest of referenced relative paths and SHA-256 values. Supports `-v host:ctr:ro`, attached `-vhost:ctr:ro`, `--volume host:ctr:ro`, and `--volume=host:ctr:ro`, including comma-separated options containing `ro`. Finds literal absolute paths in test arguments, including `--option=/path`; nested mounts use the most specific container directory. Unreferenced directory contents do not affect identity.
- `data_files`: explicit spec data path and SHA-256, when provided.
- `tests`: test name to SHA-256 of canonical JSON (sorted keys, compact separators, ASCII escapes). Formatting and object key order do not affect identity; array order does.

Submission rejects unavailable files and missing required data before creating a job. Preparing verifies the original default model hash and every recorded input hash before GPU queries or process launch. A changed or missing input terminates the attempt with `execution=not_started`, exit code 4, and a diagnostic identifying the file or test definition. Assessors and the measurement loop are unchanged.

## Migration rule

An existing key whose saved spec lacks `inputs` is compared against only the original spec fields. An equal submission returns the existing job without collecting new inputs, rewriting the immutable job, or changing its saved spec hash. Changes to original fields still reject the key. Such jobs retain their historical run behavior: only `spec.model` is verified, and the preparing state records `inputs_unverified: true` (also recorded in terminal state on preflight failure). To obtain the stronger guarantee, submit under a new key. Existing keys with `inputs` compare the complete spec including the freshly computed input identity.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider --basetemp=artifacts/codexQ/pytest-tmp bin/test_lab_inputs.py bin/test_lab_assess.py
```

Result: **48 passed in 0.71s**. Fixtures create fake models, test JSON, image registrations, prompts, jobs, and attempts under the temporary lab directory. External process launch fails tests unless explicitly mocked. Coverage includes every model/prompt/data file changing or disappearing, test-definition changes, canonical JSON stability, missing submit inputs, immutable idempotency for new and legacy jobs, unreferenced files, nested mounts, path boundaries, required data, legacy model verification, and successful new/legacy attempts with mocked processes. Existing assessment regressions also pass.

## Gaps

Reference discovery covers literal absolute argument paths and `--option=/path`, not paths hidden in environment variables, inline scripts, lists embedded in a single argument, or files transitively opened by the engine. Docker `--mount`, writable mounts, and implicit dependencies are outside the manifest. Read-only `-v` sources that are not directories are explicitly rejected rather than silently treated as verified. No snapshot or lock prevents inputs changing after preparing verification; Docker/GPU integration was not exercised. Legacy jobs intentionally cannot establish identities for inputs they never recorded.
