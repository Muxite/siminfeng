# Engine handoff receipt tool

`engine_handoff.py` inspects delivery bytes and requester requirements offline.
It never imports an image or launches a workload. Python 3 and PyYAML are
required; pytest is used for tests. Optional GGUF header inspection uses the
bundled gguf-py and its existing NumPy dependency, without loading model tensors.
No network or Docker is used.

## Archive observations

`observe_archive(path)` returns `image_config_id`, `platform`, `layer_count`, and
`executables` (absolute `/app/...` names mapped to SHA-256). It supports classic
Docker `manifest.json` archives and OCI `index.json`/`blobs/sha256` archives,
including nested OCI indexes and Docker attestation descriptors. It admits one
runnable image; ambiguous multi-image deliveries are rejected.

The outer archive is indexed by member headers without extraction. Layers are
streamed as plain or gzip tars. Only executable regular files under `app/` are
hashed into the merged view. Later entries override earlier entries; whiteouts
remove lower files/subtrees, and opaque whiteouts hide lower directory contents
without hiding files added in the same layer. Non-executable replacements and
symlinks remove previous regular-file identities. Symlinks and hardlinks are not
followed or included as regular executables. Executable shared libraries count.
Memory holds headers, paths, hashes, and bounded stream buffers, never a whole
layer or executable. Layer tar headers are discarded as they are consumed.
Metadata documents have an 8 MiB limit. Archive path traversal and duplicate outer
members are rejected.

The config's bytes must hash to its filename. This config digest is the image
config ID; an OCI index or manifest digest is a different identity. Content-addressed
layer blob names are verified, and config `rootfs.diff_ids` are checked against
the uncompressed layer bytes when present. This detects corruption outside
`app/` too. A classic archive without layer digests or diff_ids can still be bound
to its recomputed outer archive hash, but lacks those additional integrity checks.

## Facts, declarations, and generation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 cc/engine_handoff/engine_handoff.py facts \
  --build-record builds/results/b0-baseline-3/build-record.json \
  --archive builds/results/b0-baseline-3/engine-image.docker.tar \
  --model-config /home/muk/models/campaign/granite-3.1-1b-a400m-instruct/config.json \
  --gguf /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf \
  --out-dir cc/engine_handoff/example-real

PYTHONDONTWRITEBYTECODE=1 python3 cc/engine_handoff/engine_handoff.py generate \
  --facts cc/engine_handoff/example-real/facts.json \
  --requirements cc/engine_handoff/example/requirements.granite.yaml \
  --out cc/engine_handoff/example-real/handoff.yaml \
  --invocation-argv /app/llama-completion -m MODEL -n 24

PYTHONDONTWRITEBYTECODE=1 python3 cc/engine_handoff/engine_handoff.py validate \
  cc/engine_handoff/example-real/handoff.yaml
```

`facts` writes JSON and Markdown. Facts have a value, `observed`/`declared`/`missing`
tag, and source. Nested Build CC schema-version-1 records are normalized, while
all original claims remain declared. Image config ID, platform, layer count, and
executable hashes are observed independently from the archive, with its path as
source. Archive hashes and sizes are recomputed. Identity mismatches go into
`unresolved`. Other build metadata stays declared without being represented as
an independently verified identity or an admission requirement.

`generate` fills release identities only from archive observations. Optional
`--requirements FILE.yaml` supplies handoff ID, release role, model references
and manifest, workload artifact, GPU/runtime/resource/network requirements, test
requests, test policy, and profiling request/tool. Each copied requester value
has a `declared` entry in the handoff's `provenance`; archive identity provenance
remains `observed`. The workload hash is computed from its bytes and tagged
observed, while its requester-provided path stays declared. No requester value
can override an observed release identity. Requirements workload paths resolve
relative to the requirements file. Other artifact references resolve relative
to the handoff at validation; absolute manifest references are convenient.

`--workload` remains supported when requirements do not supply the artifact.
Requirements take precedence for fields they supply. Missing fields remain null
and listed in `unresolved`. A null profiling tool is allowed when profiling is
explicitly false. Put options before `--invocation-argv`; everything after it is
a literal argv element. No shell expansion or execution occurs.

Without GGUF input the six unavailable header facts are listed in `unresolved`.
HF config bytes serve as a minimal metadata manifest and do not bind model
weights/tokenizer. Requirements and facts are provenance records, not signed
attestations or proof of requester authority.

## Validation and boundaries

Validation prints a JSON receipt verdict and returns 0 for `accepted`, 2 for
`rejected` or `needs-information`. It rehashes archive, model manifest and workload
and re-observes every locally readable archive. It compares the handoff's config
ID, platform and full executable map directly with those observations, even if
the outer hash/size still match. A blob integrity failure identifies the layer
or config and failed SHA-256 check. It also rereads the normalized build record
and compares its identity claims. A record's incomplete executable map remains
a discrepancy, even when its listed tools match a subset of the observed map.

Invalid identities, malformed/ambiguous YAML, forbidden execution keys/strings,
shell launchers, unsupported platform, network access after staging, nonallowlisted
mounts and policy violations are rejected. Missing readable artifacts or explicit
unresolved entries require information unless another check rejects the receipt.
The allowed mounts are model:ro, workload:ro and results:rw. The lab retains
provisioning/isolation ownership and the external durable-evidence/crash/resume
contract. Acceptance is a receipt check, separate from job execution or test
success; the existing `bin/lab` remains the executor.

Archive checks establish byte identity and consistency, not authenticity,
provenance, numerical correctness, model completeness, executable safety, GPU
availability, runtime compatibility or actual isolation/resource enforcement.
Keep artifacts immutable between inspection and execution; this tool does not
hold them for a later executor.

## Tests and retained real run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q cc/engine_handoff \
  --basetemp=cc/engine_handoff/.test-tmp \
  -o cache_dir=cc/engine_handoff/.pytest_cache
```

The original 48 checks remain, updated to expect archive-derived identities.
The inert 20 KiB fixture now uses `/app/run`. Additional small tarfile-built
fixtures cover both layouts, gzip/plain layers, whiteouts and replacements,
config/layer tampering, edited handoff identities with untouched transport
hash/size, requester provenance, partial requirements and record discrepancies.

The integration evidence motivating this change is
`experiments/handoff-real/{facts.json,handoff.yaml,verdict.json}`: generation had
left identities null because it treated all build-record identities as claims.
The T-K run is retained in `example-real/`, including facts, requirements,
workload, generated handoff, verdict and timings. Reproduction is available in
`artifacts/codexK/run_real.py`; findings are in `artifacts/codexK/REPORT.md`.
Older captures in `example/` predate this follow-up; use `example-real/` for the
retained real-archive observation result and `example/requirements.granite.yaml`
for the sample requester declaration.
