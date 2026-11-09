# Strata CPU mechanism tests (T-G)

This recipe selects `exchange_storage_test`, `routed_stream_test`, `expert_multi_test`, `gguf_geometry_test`, and `model_family_test` at Strata commit `ffa80d73133ea5c548aefe57889a16e8085a938f`. See [select.md](select.md) for the eight-candidate table, source evidence and coverage limits.

**Status: statically inspected; shell and embedded Python syntax checked. Docker, CMake configure, compilation and tests were not run.**

## Exact invocation

Run on the Docker host. The image must already exist locally; this command cannot pull it. Use an empty `container-build` directory for the first run (the script is mounted separately).

```bash
mkdir -p /home/muk/projects/siminfeng/campaign/builds/strata-tests/container-build
docker run --rm --pull=never \
  --label campaign=c1 \
  --cpus 8 --memory 16g --memory-swap 16g \
  --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env CUDA_VISIBLE_DEVICES= \
  --mount type=bind,src=/home/muk/projects/Strata,dst=/src,readonly \
  --mount type=bind,src=/home/muk/projects/siminfeng/campaign/builds/strata-tests/container-build,dst=/build \
  --mount type=bind,src=/home/muk/projects/siminfeng/campaign/builds/strata-tests/run_cpu_tests.sh,dst=/runner.sh,readonly \
  --workdir /build \
  campaign/cuda-tools:c1 bash /runner.sh
```

No GPU is requested or exposed by the command. Use the ordinary Docker CPU runtime. The source and runner mounts are read-only; the container root is also read-only. All runner outputs, compiler scratch files, temporary synthetic fixtures and logs are directed under `/build`. Container `/build/results.json` becomes host `builds/strata-tests/container-build/results.json`.

The runner refuses an unverifiable or different source HEAD. The source checkout must include its readable Git metadata. Do not reuse a build directory configured for another source/generator; use a fresh empty directory instead. The host UID/GID must be able to write the build directory.

## Build and execution behavior

The script configures the existing project with `cmake -S /src -B /build -G Ninja`, C++20 inherited from Strata, `Release` with `-O2 -UNDEBUG`, `STRATA_BUILD_TESTS=ON`, CUDA/HIP/SYCL/gfx906 OFF, and **`STRATA_NATIVE_EXPERTS=OFF`**. Other optional GPU/engine features are explicitly disabled. Both FetchContent disconnected switches are ON. The exact argument list is in [run_cpu_tests.sh](run_cpu_tests.sh).

Only three selected CMake targets are built, individually with `--parallel 8`; these header-only tests do not link the CPU kernel library. `exchange_storage_test` is compiled unchanged with standalone `g++ -std=c++20 -O2 -UNDEBUG -pthread -I/src/include ...`: its stock CMake declaration is inside the GPU gate although the implementation it tests is header-only. `expert_multi_test` also uses standalone compilation: the stock native-OFF library omits AVX1 definitions referenced by its dispatcher. The runner builds six unchanged CPU source files with individual ISA flags and links their objects, with no pool, engine or third-party library. The exact source/flag list and evidence are in `select.md`. No Strata source is patched, stubbed or copied. Each successful build is followed by a direct executable invocation from `/build` with default arguments. No broad CTest run, benchmark mode, real-model mode or GPU rotation mode is used.

The runner attempts both standalone builds even if CMake configuration fails. A build/test failure does not prevent later independent builds/tests. Configure timeout is 120 seconds, each test's total build timeout 180 seconds, and each test timeout 60 seconds; timeout kills the process group. Tool-version/revision probes each have a 15-second timeout. Normal runtime is **estimated at 1–5 minutes** on an 8-CPU x86-64 host with local storage; this is unmeasured. The sum of the serial timeout budgets is about 24 minutes. Large logs/disk exhaustion or externally killed containers can prevent report completion.

## Results and exit status

`/build/results.json` contains `toolchain` entries for CMake, g++, Ninja, Git, Python, nvcc and `uname`, the source revision check, the complete configure command/result, and a `tests` array. Every test row contains the requested fields:

- `test`: executable name; `built`: boolean for this build attempt.
- `exit_code`: actual executable exit code, or `null` when it was not run. Negative values mean termination by a signal; runner timeout is 124.
- `seconds`: build + run elapsed time for that test, excluding shared configuration; `run_seconds` is also recorded when run.
- `stdout_tail`: final 8 KiB of combined stdout/stderr from the run, or the blocking build/configure log when not run.

Additional fields include `method`, `status`, `build_exit_code`, build/run commands, timings and log paths. Full logs live in `/build/logs/`. Results are saved atomically after each completed test, so prior results survive a later ordinary failure. The three excluded GPU candidates are documented in `select.md`; they are not built or mislabeled as passed in the JSON.

Runner exit status: **0** = all five passed; **3** = all other tests passed but `expert_multi_test` skipped for missing CPU ISA (77); **1** = any build/configure/test failure; **2** = required source revision could not be verified. Missing optional nvcc metadata does not fail CPU tests; missing required CMake/compiler tools appears in the relevant build/configure results.

## Known blockers and limits

- `expert_multi_test` is CPU-only but needs AVX-512/VNNI/VBMI and enabled OS state. A host without them returns 77. Docker does not emulate missing instructions. Its ISA flags are confined to kernel objects in the standalone recipe. The test's skip probe checks CPUID, so disabled OS register state can instead cause a signal, which the runner records as a failure.
- The native-OFF CPU CMake library has a statically identified missing-AVX1-source link blocker. The standalone expert recipe avoids it by explicitly compiling that unchanged source; actual linker success remains unverified.
- Native-expert support defaults ON and normally fetches llama.cpp. This offline recipe disables it. Enabling native tests without a separately supplied compatible full dependency checkout is not an offline recipe. No downloaded dependencies or model files are needed by the selected subset.
- Direct GPU-cache residency/admission, authoritative top-k execution, and the related DeepSeek once-only combination oracle cannot run in this container without a GPU. The CPU ownership test only checks simulated aliases, and the routed-stream test only simulates device copies. No direct CPU test was found for stable-address stale activation reuse, current-layer hit IDs, or `Launch/pool/H2D/Combine` adapter ordering.
- CMake target availability and dependencies were inspected at the pinned source. Actual configure/build success and test verdicts must be established by running the command above; this assignment explicitly forbade doing so during preparation.
