# build_engine

Use this capability to turn a clean, identified local llama.cpp checkout and an optional Git patch into a local, runtime-only CUDA image and an evidence directory. The default target `x86-5070ti` uses linux/amd64 and CUDA architecture 120 (RTX 5070 Ti); `--target jetson-orin` uses linux/arm64 and sm_87. Model weights are supplied separately at runtime.

Requires Python 3.10+ (stdlib only), Git, Docker daemon access, and Docker Buildx/Bake. pytest is needed only for tests. No GPU is needed to compile or collect hashes. The builder installs Ubuntu build packages with apt; an actual build needs cached packages/base layers or repository/registry access. This implementation does not download source, patches, models, or web UI assets. The existing `campaign/cuda-tools:c1` image is not used as the builder because the required builder base is the digest-pinned NVIDIA devel image.

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

`build-record.json` uses schema version 1 for amd64 and version 2 for Jetson. It records status, failure stage/error, requested and resolved commit, Git tree ID and dirty status, patch SHA256 or null, tag/platform, recipe SHA256 and sizes, base references/digests, Docker and Buildx versions, CMake flags, Buildx metadata, image config ID/RepoDigests/Created/Architecture/Os, tool SHA256 values, saved archive SHA256/bytes, build duration and total duration. Fields remain null or empty when a failure occurs before collection. `build_wall_seconds` measures the Bake invocation, including export; `wall_seconds` covers the whole attempt.

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


## Jetson Orin target

Requires an Orin host with L4T 36, JetPack >=6.1 (qualified toolchain: JetPack
6.2.3, nvcc 12.6.68), native GCC, CMake, Ninja, Git, Docker and Buildx.
ccache is optional and enables C, C++ and CUDA compiler launchers when found.
Python 3.10 is supported, including safe snapshot extraction on versions without
`tarfile` extraction filters. No packages are installed by this target.

Compilation is host-native to use the device's JetPack CUDA toolchain and avoid
pulling the roughly 9 GB l4t-jetpack development image. Source verification,
archive isolation and patch checking are the same as for amd64. CMake configures
`<staged-source>/.campaign-build` with Ninja, Release/shared libraries, CUDA architecture 87,
`GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`,
`LLAMA_CURL=OFF` and `LLAMA_BUILD_TESTS=OFF`. `BUILD_JOBS` defaults to 3 and
must be a positive integer. The seven standard tools and any `--extra-target`
executables are compiled. Generated `lib*.so*` files retain their symlinks, and
GCC's libgomp.so.1 is copied into the runtime context alongside the tools.

Run from the campaign directory, after any other native build has finished:

```bash
DOCKER_CMD_PREFIX='sg docker -c' BUILD_JOBS=3 python3 cc/build_engine/build_engine.py \
  --target jetson-orin --source ref/llama.cpp \
  --commit 31288c9c66e59377f7788d920678a565bb89dd37 \
  --tag campaign/llamacpp:jetson-c1 --out cc/build_engine/results/jetson-c1
```

Use a fresh output directory for each attempt. `DOCKER_CMD_PREFIX` is optional;
when set, it is parsed as a launcher and receives the entire shell-quoted Docker
command as one final argument. `sg docker -c` is needed in this login session
because its group membership predates the Docker group change. Every Docker
invocation, including version queries, inspection, hashing and saving, uses it.
Native build and Git commands do not use this prefix.

Buildx packages only the prebuilt runtime context with `--platform linux/arm64
--load`. The default base is
`nvcr.io/nvidia/l4t-cuda:12.6.11-runtime@sha256:49853e9fe2a0305efa51fcbb3426eea1bca658085f2d26bb40347711434849b1`.
`BASE_RUNTIME` may override its digest using the same image name/tag;
`BASE_DEVEL` is unused. The Dockerfile clears the entrypoint, sets
`LD_LIBRARY_PATH=/app`, and sets `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` per D022. GPU execution still needs
`--runtime nvidia`; tool hashing explicitly uses `--runtime runc` with no network
or GPU request.

Schema 2 adds `target_profile`, `builder: "host-native"`, `toolchain` (nvcc,
GCC, CMake, Ninja, first `/etc/nv_tegra_release` line and nvidia-jetpack dpkg
version), and `target_requirements` (linux/arm64, L4T major 36, sm_87,
JetPack >=6.1, NVIDIA runtime with explicit visibility). Source commit/tree,
patch SHA256, toolchain and target profile are also image labels. The native
build directory remains as evidence; its temporary source snapshot and runtime
context are removed on success or failure. Rebuilding requires a fresh attempt.
`build_wall_seconds` covers native compilation plus image packaging. Failures
retain logs, partial toolchain identity, available Buildx metadata and the usual
`failure_stage` field.

Mocked tests verify commands, source isolation, records, packaging inputs,
Docker launcher quoting and failure reporting on Python 3.10. No real build or
Docker invocation was performed for J-T1. Toolchain identities and target
requirements are recorded, not checked against a compatibility matrix. Image
architecture and shipped file hashes are checked during a real attempt; ABI
compatibility, dynamic library closure, GPU execution, memory use, performance,
model correctness and reproducibility still require the orchestrator's real
build and runtime checks. Wait for the existing native compile to finish before
running this command on the 8 GB device.

### Native ccache keying and evidence (J-T1b)

Jetson configure and compile commands run from the staged source with relative
`-S .`, `-B .campaign-build` and `--build .campaign-build` arguments. The build
stays inside `CCACHE_BASEDIR=<staged-source>`, so generated include/output paths
can be normalized across random staging directories. Native commands inherit
`CCACHE_NOHASHDIR=1` and `CCACHE_COMPILERCHECK=content`. The reserved
`.campaign-build` path must not exist in the source snapshot. No stable path
option is needed. On success or compile failure the build directory is moved to
`<out>/build` as evidence before staging cleanup; its absolute CMake cache paths
refer to the removed snapshot, so it is not a resumable build directory.
The amd64 recipe, environment and record schema are unchanged.

Jetson records include `ccache.enabled`, `version`, `before`, `after` and `delta`.
Snapshots bracket native configure/compilation (including failed compilation),
using `ccache --print-stats` plus English `ccache -s -v`; each contains hits,
misses, uncacheable, raw counters and the verbose summary. Delta contains the
three counter differences. Disabled/unreached snapshots are null; failure to
collect final stats is recorded as `stats_error`. These are shared-cache
counters: concurrent users or counter resets can distort the differences.
No statistics are reset and no cache settings are persisted.

Read-only inspection on 2026-11-10 found ccache 4.5.1, zero hits, 456 misses
and about 0.11 GB in `ccache -s -v`; a later machine-readable snapshot had 457
misses and zero unsupported-language/option or failed-compilation counters.
The running build can change those totals. The
[ccache 4.5.1 manual](https://ccache.dev/manual/4.5.1.html#_configuration_options)
explicitly supports compiler type `nvcc`: single-source CUDA `.cu` object
compilations are cacheable. Aggregate counters and compressed cache size do not
prove that any specific `.cu` invocation was cached; no real build was run for
this assignment. Confirm CUDA hits in a later authorized rebuild. Compiler
flags, generated content, absolute paths outside the base directory and embedded
source paths can still affect reuse. Disabling directory hashing can reuse old
debug compilation-directory paths; Release builds here do not request debug info.

## Backend-op screen executable

Add `--with-tests --extra-target test-backend-ops` to enable
`-DLLAMA_BUILD_TESTS=ON` and build/ship `/app/test-backend-ops` on either target.
The target is registered by `tests/CMakeLists.txt:316` at `31288c9c6`.
Requesting this target without `--with-tests` is rejected before building.
`build-record.json` records `with_tests`, `extra_targets`, actual CMake flags and
the executable hash. For x86, the tests flag is changed only in the staged
Dockerfile; `staged_recipes` records its hashes. Tests remain disabled by default.
See [kernel screen](../kernel_screen/README.md) for presets, the optional perf
case registration patch, environment comparisons and limitations.
