# Assignment J-T1 — Build CC: Jetson-qualified linux/arm64 target (host jetson-arm; parent c1; owner Opus-jetson; worker Codex)

Base: branch `jetson/c1`. Existing Build CC: `cc/build_engine/build_engine.py` (+ tests, README) and recipe `builds/llamacpp/{Dockerfile,docker-bake.hcl}` — amd64/CUDA 12.8/sm_120 only, written for Python 3.12; this host has Python 3.10.12.

Allowed files: `cc/build_engine/**`, new `builds/llamacpp-jetson/**`. Do NOT modify `builds/llamacpp/*` behaviour for amd64 (x86 owns it) except trivially-compatible refactors; amd64 tests must still pass. No sudo, no package installs, no nvpmodel/clocks.

## Design (decided by Opus)
Add a target profile `--target jetson-orin` (default stays `x86-5070ti`). For Jetson, the builder is **host-native** (JetPack 6.2.3 nvcc 12.6.68 on the device; a 9 GB l4t-jetpack devel pull is avoided), and the runtime image is built with Buildx from a prebuilt-binary context:
1. Same source checks as today (HEAD == --commit, clean, no submodules, `git archive` snapshot, `git apply --check` then apply patch).
2. Configure+build in `<out>/build` with: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON` (+ `-DCMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER=ccache` when ccache exists), `cmake --build -j${BUILD_JOBS:-3} --target <same tool list as amd64>`. Record toolchain identity: nvcc version, gcc version, cmake/ninja versions, `/etc/nv_tegra_release` line, `nvidia-jetpack` dpkg version.
3. Runtime image Dockerfile `builds/llamacpp-jetson/Dockerfile`: `FROM ${BASE_RUNTIME}` with BASE_RUNTIME = `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime@sha256:49853e9fe2a0305efa51fcbb3426eea1bca658085f2d26bb40347711434849b1`; COPY tools + `lib*.so*` into `/app`, `LD_LIBRARY_PATH=/app`, entrypoint cleared, ENV `NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility` (see decisions/D022), labels for source commit/tree/patch sha/toolchain/target profile. Use `docker buildx build --platform linux/arm64 --load` (or bake) — Buildx plugin v0.29.1 is installed at ~/.docker/cli-plugins; the docker CLI must be invoked as `sg docker -c "<cmd>"` in this session (user is in group docker but the login session predates it) — make the docker command prefix configurable via env `DOCKER_CMD_PREFIX` rather than hard-coding sg.
4. build-record.json: same schema, `schema_version` 2 adds `target_profile`, `builder: "host-native"`, `toolchain{}`, `target_requirements` = {platform linux/arm64, l4t_major 36, cuda_arch sm_87, jetpack ">=6.1", runtime "nvidia (NVIDIA_VISIBLE_DEVICES required)"}. Failed builds are recorded with failure_stage as today. Tool hashing via `docker run --rm --network=none` uses `--runtime runc` (no GPU).
5. Python 3.10 compatibility for the whole module (no 3.12-only syntax/APIs) — keep amd64 behaviour identical.

## Tests
Extend `test_build_engine.py` with unit tests for the jetson profile (cmake argv generation, record fields, Dockerfile args, DOCKER_CMD_PREFIX handling) using the existing fake-subprocess style. Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q cc/build_engine/test_build_engine.py --basetemp=cc/build_engine/.pytest-tmp -o cache_dir=cc/build_engine/.pytest_cache`.
Do NOT run a real build (Opus runs it; a native build is in progress on this 8 GB device and a second compile would exhaust shared memory). Do not run docker.

## Output / limits
Time 40 min. Update README (Jetson section: requirements, why host-native, what is and isn't verified). Final message ≤ 20 lines: files changed, pytest line, the exact CLI command Opus should run for the real build, open risks.
