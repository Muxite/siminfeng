# build_engine

Use this capability to turn a clean, identified local llama.cpp checkout and an optional Git patch into a local, runtime-only CUDA image and an evidence directory. It targets linux/amd64 and CUDA architecture 120 (RTX 5070 Ti). Model weights are supplied separately at runtime.

Requires Python 3.12+ (stdlib only), Git, Docker daemon access, and Docker Buildx/Bake. pytest is needed only for tests. No GPU is needed to compile or collect hashes. The builder installs Ubuntu build packages with apt; an actual build needs cached packages/base layers or repository/registry access. This implementation does not download source, patches, models, or web UI assets. The existing `campaign/cuda-tools:c1` image is not used as the builder because the required builder base is the digest-pinned NVIDIA devel image.

## Run

Run from the campaign directory. Give every attempt a new or empty output directory. While executing assignment T-D, keep outputs under `cc/build_engine/` or `builds/llamacpp/`.

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

## Outputs and failure handling

`build-record.json` uses schema version 1. It records status, failure stage/error, requested and resolved commit, Git tree ID and dirty status, patch SHA256 or null, tag/platform, recipe SHA256 and sizes, base references/digests, Docker and Buildx versions, CMake flags, Buildx metadata, image config ID/RepoDigests/Created/Architecture/Os, tool SHA256 values, saved archive SHA256/bytes, build duration and total duration. Fields remain null or empty when a failure occurs before collection. `build_wall_seconds` measures the Bake invocation, including export; `wall_seconds` covers the whole attempt.

The other files are:

- `build.stdout` and `build.stderr`: full subprocess stdout/stderr, including compiler output and diagnostics on failure. Binary Git archive output is directed to staging rather than the logs.
- `buildx-metadata.json`: Buildx metadata, when the build produces it; retained even when the build exits nonzero.
- `engine-image.docker.tar`: `docker save` of the built immutable image config ID, on success. Its bytes and SHA256 are recorded after saving.

The CLI exits 0 only after all evidence collection succeeds, otherwise nonzero. Failures include source mismatch/dirty tree, unsupported submodules, malformed or inapplicable patch, unavailable Docker/base image, build/export failure, incorrect architecture, missing binaries, save failure, or malformed archive. A failure after building can leave the tagged local image and partial evidence; inspect `failure_stage` and retry in a fresh output directory. Invalid CLI arguments, an unwritable output path, or a nonempty output directory fail before a record can be created. Existing evidence is never overwritten. The CLI does not push, prune, mount the Docker socket in an image, or operate on unrelated containers. A single short-lived `docker run --rm --network=none --read-only --entrypoint /usr/bin/sha256sum` container hashes the tools without requesting a GPU. Subsequent hashing and saving use the immutable config ID so a later tag change cannot change the evidence target.

Bake requests maximum provenance and saves metadata separately. This does not imply provenance survives image loading and `docker save`. The archive inspector checks `index.json` and reachable OCI manifests for actual in-toto SLSA provenance statements. A classic Docker archive without an OCI index records `retained: false`; an SBOM or an attestation annotation alone does not count. A save or inspection failure leaves provenance unconfirmed. No signature or attestation trust verification is performed.

## Validation without a build

From `builds/llamacpp/`, run `docker buildx bake --print`; the retained `bake-print.json` is the validated default expansion. Its runtime digest is intentionally empty until supplied; print mode validates Bake syntax, not base availability or compilation. For a direct Bake build, supply digest-pinned BASE_DEVEL and BASE_RUNTIME plus SOURCE_DIR, SOURCE_COMMIT and PATCH_SHA256, and use a prepared staging snapshot. Prefer the CLI, which verifies those inputs. Bake uses `BAKE_CMD_CONTEXT` to locate the recipe Dockerfile outside the source context; invoke it from the recipe directory.

From the campaign directory, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/build_engine/test_build_engine.py \
  --basetemp=cc/build_engine/.pytest-tmp \
  -o cache_dir=cc/build_engine/.pytest_cache
```

Tests use temporary Git repositories and mocked Docker subprocesses. They cover patch rejection, successful patch isolation, successful record schema, build failure logs/metadata, dirty or mismatched source, archive/tool hashing, provenance versus SBOM, and refusing to overwrite evidence. They use no network, GPU, or real build.

This evidence does not prove bitwise reproducibility, runtime correctness, model accuracy, GPU compatibility/performance, or the absence of vulnerabilities. Apt package versions, compiler behavior, timestamps, and upstream build scripts can vary. Base digests and input hashes identify inputs; they do not establish equivalence to an upstream image. The orchestrator must perform the real build and runtime checks. No full build was performed for T-D.
