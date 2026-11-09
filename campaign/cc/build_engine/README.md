# build_engine

Use this capability to turn a clean, identified local llama.cpp checkout and an optional Git patch into a local, runtime-only CUDA image and an evidence directory. It targets linux/amd64 and CUDA architecture 120 (RTX 5070 Ti). Model weights are supplied separately at runtime.

Requires Python 3.12+ (stdlib only), Git, Docker daemon access, and Docker Buildx/Bake. pytest is needed only for tests. No GPU is needed to compile or collect hashes. The builder installs Ubuntu build packages with apt; an actual build needs cached packages/base layers or repository/registry access. This implementation does not download source, patches, models, or web UI assets. The existing `campaign/cuda-tools:c1` image is not used as the builder because the required builder base is the digest-pinned NVIDIA devel image.

## Run

Run from the campaign directory. Give every attempt a new or empty output directory. For assignment T-N, keep outputs under `cc/build_engine/` or `artifacts/codexN/`. The CLI uses the Dockerfile and Bake recipe in `cc/build_engine/recipes/`; it does not modify or use the older recipes under `builds/llamacpp/`.

```bash
# Supply the real runtime image digest from trusted/local image evidence.
export BASE_RUNTIME='nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:<64-hex-digest>'
python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp --commit 31288c9c6 \
  --tag campaign/llamacpp:c1 --out cc/build_engine/results/baseline
```

The placeholder must be replaced; it is not a working digest. If BASE_RUNTIME is unset, the CLI inspects the local runtime tag and selects its NVIDIA RepoDigest without pulling it. Missing local images or RepoDigests fail at `base_images`. BASE_DEVEL defaults to the devel digest already recorded in `artifacts/op-309e1f95c716/stdout`; override it with a full `nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:...` reference when needed. Both bases are validated and supplied as separate name and digest Dockerfile arguments. The CLI never builds from a floating base tag.

To add a patch, save a unified Git diff to a file and add `--patch path/to/change.patch`. Source HEAD must resolve to `--commit`; tracked changes and untracked files cause rejection. The source checkout remains read-only: Git optional locks are disabled, a `git archive` snapshot is extracted under the output directory, and `git apply --check` runs there before `git apply`. Git discovery stops before the enclosing campaign repository. Binary patches supported by Git are accepted. Submodules are rejected. Git archive honors committed export attributes; ignored and untracked files and `.git` are omitted. Docker ignore files in the staged snapshot are emptied to prevent omission of tracked source. Temporary staging is removed after either result. Keep the supplied patch and original commit for rebuilding; patch bytes are identified by SHA256 but are not retained as a separate output artifact.

The image contains only the seven requested tools in `/app`, their generated shared libraries and the OpenMP runtime, on top of the CUDA runtime base. It has no compiler, source checkout, or weights. Its entrypoint is cleared, so invoke tools by full path, for example `/app/llama-bench`. Release/shared CUDA compilation uses the flags in `build-record.json`; CPU-native tuning, tests, downloaded UI, and OpenSSL are disabled. `LLAMA_CURL=OFF` is passed as requested, although this checkout treats LLAMA_CURL as deprecated. Source and patch identity are recorded as image labels; binaries' built-in Git version may say `unknown` because `.git` is not staged.

## Engine variants and linkage

Use repeatable `--cmake-define KEY=VALUE` options. Keys must match `^[A-Z0-9_]+$`; values must be nonempty and contain only ASCII letters, digits, or `_./:+,@%=-`. Whitespace, shell metacharacters, empty values, duplicate keys (even with identical values), and all built-in keys are rejected before output creation. This also reserves the built-in compiler-launcher and executable-linker flags. Definitions are sorted by key, passed as the `CMAKE_DEFINES` build argument as space-separated `-DKEY=VALUE` words, and stored in the runtime image label `campaign.cmake.defines` in that same format. The CLI overwrites inherited `CMAKE_DEFINES` environment values.

`recipes` records the SHA256 and size of each recipe file. `recipe_sha256` hashes canonical JSON (sorted keys, compact separators) containing those file identities, all built-in flags, the definitions, and extra targets. Definition values therefore change recipe identity even when recipe bytes are unchanged; reordering definitions does not.

For the GPU budget shim, build with VMM disabled at compile time. From the campaign root, with a clean source checkout at the shown commit and local base images available:

```bash
python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp --commit 31288c9c66e59377f7788d920678a565bb89dd37 \
  --tag campaign/llamacpp:no-vmm \
  --out artifacts/codexN/no-vmm-build \
  --cmake-define GGML_CUDA_NO_VMM=ON \
  --linkage
```

`--linkage` enables a final check after saving the image and collecting provenance. It runs one additional `docker run --rm --network=none --read-only --entrypoint /bin/sh` against the built image's config ID, with a 60-second client timeout and no GPU. It discovers `/app/libggml-cuda.so` or a versioned `/app/libggml-cuda.so.*`, runs `ldd`, and selects `readelf -Ws` or, if unavailable, `nm -D`. A pass requires a dynamic `libcudart.so*` dependency and undefined imports of exactly `cudaMalloc`, `cudaFree`, and `cudaMemGetInfo` (including versioned symbols). Other CUDA API names and defined symbols do not satisfy the check. `ldd` showing `libcudart.so.12 => not found` still establishes a dynamic dependency; this check does not establish that runtime dependencies resolve on the GPU host.

The record's `linkage` object contains `status` (`not_requested`, `passed`, `failed`, or `unavailable`), the discovered library, config ID, `libcudart_dynamic`, `undefined_imports`, each tool's return code/reason, and output paths. Missing tools, a missing library, nonzero tool exits, Docker failures, timeouts, and malformed probe output are `unavailable`; unavailable measurements are JSON null rather than false. Runtime images may lack both readelf and nm: the check explicitly records this rather than installing tools or silently claiming imports are absent.

A failed or unavailable requested check sets top-level `failure_stage="linkage"` and exits 1, while preserving the saved archive and other evidence. `--linkage-optional` enables the same check (even without `--linkage`) but permits a failed/unavailable result with overall success and exit 0; inspect `linkage.status` in that case. With neither flag, linkage is `not_requested`.

The probe saves `linkage.raw.txt` (framed stdout), `linkage.stderr.txt` (Docker/probe stderr), `linkage.ldd.txt`, and `linkage.symbols.txt`. Tool stderr is merged into its corresponding raw tool output. On a Docker error or malformed probe, only the first two files may be available. This ELF check does not prove allocation interception, actual pool selection, or GPU execution; the orchestrator must verify `NO_VMM = 1` and allocation events under the budget shim.

## Outputs and failure handling

`build-record.json` uses schema version 2. Version 1 evidence remains untouched; this capability has no build-record reader. Schema 2 adds `cmake_defines`, `recipe_sha256`, and `linkage`; `cmake_flags` includes all built-in and extra definitions. It records status, failure stage/error, requested and resolved commit, Git tree ID and dirty status, patch SHA256 or null, tag/platform, recipe SHA256 and sizes, base references/digests, Docker and Buildx versions, CMake flags, Buildx metadata, image config ID/RepoDigests/Created/Architecture/Os, tool SHA256 values, saved archive SHA256/bytes, build duration and total duration. Fields remain null or empty when a failure occurs before collection. `build_wall_seconds` measures the Bake invocation, including export; `wall_seconds` covers the whole attempt.

The other files are:

- `build.stdout` and `build.stderr`: full subprocess stdout/stderr, including compiler output and diagnostics on failure. Binary Git archive output is directed to staging rather than the logs.
- `buildx-metadata.json`: Buildx metadata, when the build produces it; retained even when the build exits nonzero.
- `engine-image.docker.tar`: `docker save` of the immutable Docker-reported image ID, collected before runtime probes. This ID may identify an OCI index in the containerd store. Its bytes and SHA256 are recorded after saving.

The CLI exits 0 only after all required evidence collection succeeds, otherwise nonzero; `--linkage-optional` permits a recorded linkage failure. Failures include source mismatch/dirty tree, unsupported submodules, malformed or inapplicable patch, unavailable Docker/base image, build/export failure, incorrect architecture, missing binaries, save failure, or malformed archive. A failure after building can leave the tagged local image and partial evidence; inspect `failure_stage` and retry in a fresh output directory. Invalid CLI arguments, an unwritable output path, or a nonempty output directory fail before a record can be created. Existing evidence is never overwritten. The CLI does not push, prune, mount the Docker socket in an image, or operate on unrelated containers. A short-lived `docker run --rm --network=none --read-only --entrypoint /bin/sh` container hashes the tools and libraries without requesting a GPU. After resolving the tag once, saving uses the immutable Docker-reported ID. Before hashing or linkage, the archive inspector follows the OCI catalog and nested indexes to the linux/amd64 image manifest and config, skipping other platforms and attestations. It verifies the SHA256 of referenced JSON blobs and requires exactly one matching config. Classic Docker archives are supported by hashing the config bytes named in `manifest.json`. Hashing and optional linkage then use this config digest as the local `sha256:...` Docker image ID; local-only `repo@digest` references are not used. Buildx `containerimage.config.digest` is optional and is cross-checked against the archive when present. Index metadata is cross-checked when the archive retains the index. Missing, ambiguous, or inconsistent archive identities fail before runtime probes. The record keeps `Id`/`docker_reported_id`, `config_digest`, `manifest_digest`, `index_digest`, `index_manifests`, and `docker_id_in_archive_index` separate; an unavailable manifest or index digest is null. `index.json` is the archive catalog and is not assumed to be the original image index.

Bake requests maximum provenance and saves metadata separately. This does not imply provenance survives image loading and `docker save`. The archive inspector checks `index.json` and reachable OCI manifests for actual in-toto SLSA provenance statements. A classic Docker archive without an OCI index records `retained: false`; an SBOM or an attestation annotation alone does not count. A save or inspection failure leaves provenance unconfirmed. No signature or attestation trust verification is performed.

## Validation without a build

The CLI stages its recipes in `.campaign-recipe/` inside the archived source context. Bake receives digest-pinned base references, source and patch identity, extra targets, and extra definitions from the CLI. Prefer the CLI for validation and evidence collection; direct builds using the older `builds/llamacpp/` recipes do not support these variant options.

From the campaign directory, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/build_engine/test_build_engine.py \
  --basetemp=cc/build_engine/.pytest-tmp \
  -o cache_dir=cc/build_engine/.pytest_cache
```

Tests use temporary Git repositories and mocked Docker subprocesses. They cover patch rejection, successful patch isolation, successful record schema, build failure logs/metadata, dirty or mismatched source, archive/tool hashing, provenance versus SBOM, and refusing to overwrite evidence. They also cover define validation, variant recipe identity/build arguments/labels, schema 2, captured ELF linkage fixtures, shell tool selection, config-ID versus index-ID targeting, and fatal/optional linkage failures. The T-N2 regression uses the real containerd Buildx metadata fixture without config metadata, plus constructed content-addressed OCI index/manifest/config archives (with another platform and an attestation). It checks config metadata mismatches, missing or corrupt configs, ambiguous configs, and index metadata mismatches before any runtime probe. They use no network, GPU, or real engine build.

This evidence does not prove bitwise reproducibility, runtime correctness, model accuracy, GPU compatibility/performance, or the absence of vulnerabilities. Apt package versions, compiler behavior, timestamps, and upstream build scripts can vary. Base digests and input hashes identify inputs; they do not establish equivalence to an upstream image. The orchestrator must perform the real build and runtime checks. No Docker execution or network access was performed for T-N.
