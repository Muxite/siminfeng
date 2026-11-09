# GPU allocation budget shim

`libgpubudget.so` is a process-local `LD_PRELOAD` interposer for Linux and CUDA
12.8. It caps **successful, intercepted allocation sizes**, aggregated across
all devices and contexts in one process. It requires no driver changes, MPS,
network access, or privileged container options. It is an experiment control,
not a security boundary or a limit on physical VRAM reported by `nvidia-smi`.

The assignment's approximate scale is `128/500 * 2.67 GiB ≈ 700 MiB`.
Set `GPU_BUDGET_MIB=700` for that allocation budget. If the 2.67 GB figure uses
decimal units instead, the corresponding result is approximately 652 MiB;
700 MiB is the requested campaign approximation.

## Build and test (orchestrator)

Run these from the campaign root. These are instructions for the orchestrator;
the authoring agent did not run Docker. The source mount is read-only and all
compiler outputs go into the separate writable output mount:

```bash
mkdir -p artifacts/codexL/cuda
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/cc/gpu_budget:/src:ro" \
  -v "$PWD/artifacts/codexL/cuda:/out" \
  campaign/cuda-tools:c1 bash /src/build.sh /out

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/cc/gpu_budget:/src:ro" \
  -v "$PWD/artifacts/codexL/cuda:/out" \
  campaign/cuda-tools:c1 bash /src/run_test.sh /out
```

`build.sh [output-dir]` is the single build command inside the image. It uses
`gcc` and CUDA headers for the shim, then `nvcc --cudart shared` for the test.
The shim depends only on the ELF loader, libc, and pthreads; it does not link
against CUDA libraries. The test links the CUDA driver stub at build time,
without embedding the stub directory in its runtime search path. At execution,
the NVIDIA container runtime must supply the real `libcuda.so.1`.

`run_test.sh` pins budget=250, reserve=0, managed counting=1. It prints a
PASS/FAIL table and exits nonzero if any check fails. It checks 3 × 100 MiB,
third-allocation OOM, free/reuse, exact memory-info values, async allocation,
async denial/free/reuse, pitched/3D/managed accounting, and direct synchronous
and asynchronous driver allocation. It writes `test-budget.jsonl` in the output
directory unless `GPU_BUDGET_LOG` is supplied. A GPU with managed memory and
CUDA memory-pool support is required; unsupported async APIs are explicit test
failures, not silent passes.

## Using with the engine

Set `ENGINE_DIR` to the Build CC build output containing `bin/llama-cli` and
`MODEL_DIR` to the model directory. For example:

```bash
docker run --rm --gpus all \
  -v "$ENGINE_DIR:/engine:ro" \
  -v "$MODEL_DIR:/models:ro" \
  -v "$PWD/artifacts/codexL/cuda:/budget:ro" \
  -v "$PWD/artifacts/codexL:/logs" \
  -e LD_PRELOAD=/budget/libgpubudget.so \
  -e GPU_BUDGET_MIB=700 \
  -e GPU_BUDGET_LOG=/logs/engine-budget.jsonl \
  -e GGML_CUDA_NO_VMM=1 \
  campaign/cuda-tools:c1 \
  /engine/bin/llama-cli -m /models/model.gguf -ngl 999 -p 'Hello'
```

**VMM must actually be disabled in the engine build.** Source inspection of
this checkout found `GGML_CUDA_NO_VMM` is a compile-time setting, not a runtime
environment switch. The environment setting above expresses the requested
campaign intent, but by itself has **no effect** on this revision. Build CC
must configure llama.cpp with `-DGGML_CUDA_NO_VMM=ON` and rebuild its CUDA backend.
`ggml/src/ggml-cuda/common.cuh:232` and `CMakeLists.txt:147` guard VMM at compile
time; `ggml-cuda.cu:725` otherwise chooses the VMM pool when supported.
Check the Build CC `CMakeCache.txt` for `GGML_CUDA_NO_VMM:BOOL=ON`, the backend
feature report for `NO_VMM = 1`, and confirm budget-log `cudaMalloc` allocation
events while loading the model. A device capability line saying `VMM: yes`
alone does not prove which pool was selected.

This requires dynamically linked CUDA runtime calls. The campaign image is
specified to link `libggml-cuda.so` against `libcudart.so.12`; verify the actual
Build CC artifact inside the image with `ldd /engine/bin/libggml-cuda.so` (adjust
the path if needed). Inspect `readelf -Ws` for undefined `cudaMalloc` and check
that the engine produces allocation events in the budget log. Static cudart,
private symbol lookup, or direct driver function pointers can bypass preload.
The image and engine binary were unavailable for execution in this workspace,
so their dynamic linkage and actual pool selection have not been GPU-verified.

Engines using `cudaMemGetInfo`/`cuMemGetInfo_v2` or llama.cpp's
`ggml_backend_dev_memory` see the synthetic budget and free balance. Fit logic
can reduce GPU layers or offload to CPU. Allocators that do not adapt receive
OOM and may abort; llama.cpp pool growth, temporary buffers, and KV cache also
consume this budget. An exact physical-memory simulation is not implied.

## Configuration and accounting

| Variable | Behavior |
| --- | --- |
| `GPU_BUDGET_MIB` | Unsigned integer binary MiB. Absent: no cap and real memory-info values. `0`: no positive-size allocations. |
| `GPU_BUDGET_RESERVE_MIB` | Default 0. Effective allocation capacity is `max(budget - reserve, 0)`. |
| `GPU_BUDGET_COUNT_MANAGED` | `1` (default): charge requested managed bytes immediately, without tracking residency/prefetch. `0`: managed memory is outside the budget. |
| `GPU_BUDGET_LOG` | Optional writable JSONL path, opened in append mode. All events are flushed immediately, including denials, plus a final summary at normal exit/unload. |

Configuration is read once on first intercepted outer call. Malformed integers,
overflow, invalid managed settings, or an unopenable requested log terminate
with exit 125 rather than silently dropping the restriction. Each process has
its own cap and must have its own log path for unambiguous records.

`cudaMemGetInfo` and `cuMemGetInfo_v2` first invoke the real API so normal
initialization and invalid-argument/context errors are preserved. After success,
`total = budget`, `free = max(budget - reserve, 0) - used`. Allocation accounting
uses successful requested bytes, with padding included for pitched storage.
No allocation is issued when that amount exceeds the remaining capacity.
Underlying CUDA allocation failures leave the counter unchanged. Denials return
`cudaErrorMemoryAllocation` or `CUDA_ERROR_OUT_OF_MEMORY`; output arguments on
failure must not be used. A missing real entry point returns NOT_SUPPORTED.
Synthetic failures return the specified error directly; they do not update
cudart's private last-error state for `cudaGetLastError`/`cudaPeekAtLastError`.

A mutex serializes decisions, real calls, and pointer-hash-table updates. A
thread-local nesting guard prevents runtime→driver calls from charging twice.
Bookkeeping storage is reserved before invoking an allocator; metadata OOM is
also an allocation denial. Successful frees of unknown pointers are passed
through and do not change the balance. Failed frees retain their charge.

`cudaFreeAsync` and `cuMemFreeAsync` **synchronize their stream after enqueueing
the free** before returning budget to the counter. They therefore block and
alter async performance/ordering. Stream synchronization errors retain the
charge conservatively, even if the underlying free eventually completes; the
balance may stay lower until process exit. No event polling or background
reaper is used. Async allocation is charged immediately on successful enqueue.

`cudaMallocPitch` and `cudaMalloc3D`, while the cap is enabled, synthesize linear
storage through real `cudaMalloc`. Pitch is rounded to at least 512 bytes and
the device's texture and texture-pitch alignments, and checked against maximum
pitch. The charged size is `pitch * height [* depth]`, checked for overflow.
Returned `xsize`/`ysize` retain requested extents. This avoids speculative
over-budget allocations, but pitch can differ from native cudart. Pitched
allocation with zero extents is rejected. Texture-specific uses and exact
native pitch compatibility require validation on the target GPU; the intended
llama.cpp path uses ordinary `cudaMalloc`.

The JSONL fields are `event`, `api`, `pointer` (hex string), `bytes`, `used`,
`peak`, `budget`, `reserve`, `capacity`, and numeric `status`. Events are
`init`, `alloc`, `free`, `deny`, `error`, and `exit`. `used`/`peak` exclude the
reserve. Opted-out managed allocations still log their requested size without
changing counters; their frees are unknown-size (`bytes=0`) events.
No summary is guaranteed for SIGKILL, `_exit`, or fatal crashes;
already-flushed events survive. Atexit and destructor share an idempotent
finalizer, so ordinary exit logs the summary once.

## Coverage and limitations

Covered exported symbols:

| API family | Symbols |
| --- | --- |
| Runtime allocation | `cudaMalloc`, `cudaMallocAsync`, `cudaMallocPitch`, `cudaMalloc3D`, `cudaMallocManaged` |
| Runtime free/info | `cudaFree`, `cudaFreeAsync`, `cudaMemGetInfo` |
| Driver allocation | `cuMemAlloc_v2`, `cuMemAllocAsync` |
| Driver free/info | `cuMemFree_v2`, `cuMemFreeAsync`, `cuMemGetInfo_v2` |

**Not covered:** `cuMemCreate` / `cuMemRelease` and VMM mappings; arrays and
mipmapped arrays; `cuMemAllocManaged`, driver pitched allocation, explicit
`*FromPoolAsync`; IPC/imported/external allocations; host allocation/mapping
(`cudaHostAlloc`, `cudaHostRegister`, mapped host memory); per-thread-default-
stream `_ptsz` symbols and legacy unversioned driver ABI symbols; direct
`cuGetProcAddress`/`cudaGetDriverEntryPoint` lookups; deep-bound or statically
linked CUDA; and allocation/context cleanup through `cudaDeviceReset` or
context destruction. Do not reset/destroy contexts while tracked allocations
remain live. Start a new process between trials. Forking after CUDA/shim
initialization and CUDA graph capture/replay are unsupported.

VMM is deliberately left unsupported: releasing a generic allocation handle
does **not** release physical storage while mappings survive. This checkout's
llama.cpp maps each allocation then immediately calls `cuMemRelease`, so a
simple handle-only wrapper would falsely restore the budget. Correct coverage
would also require mapping lifetimes, aliases, imports, retained handles, and
partial unmaps. Until that exists, build with VMM disabled as described above.

CUDA context/cuBLAS workspace overhead (approximately 300 MiB, device and
version dependent), internal driver/cudart allocations, kernels' local memory,
allocation granularity, and pool caches/reserved memory are **not counted**.
Pools may retain physical memory after a charged logical allocation is freed.
To leave room for estimated overhead within a 700 MiB experiment envelope, use
`GPU_BUDGET_MIB=700 GPU_BUDGET_RESERVE_MIB=300`: total reports 700 MiB and initial
free is 400 MiB. If 700 MiB should be available specifically to intercepted
allocations, leave reserve at 0 and allow overhead outside that amount.
Measure the real overhead separately; the reserve is an estimate, not telemetry.

## Verification evidence

No network or Docker was used. CUDA headers, nvcc, and a GPU runtime were not
available on the authoring host. The complete test program passed against a
CPU-only fake CUDA backend, which uses actual ELF preload and deliberately
nests runtime→driver calls. Additional checks cover 16 concurrent allocators,
backend allocation failures, unknown/null frees, overflow, reserve, managed
opt-out, sync failure, zero/disabled caps, invalid configuration, and JSONL
invariants. See `artifacts/codexL/host-verification.txt` and reproduce with
`bash artifacts/codexL/host_mock/verify.sh`. These mocks are not CUDA ABI or
GPU correctness evidence. The orchestrator must still run the build and real
CUDA tests above, then validate the actual Build CC engine under the shim.
