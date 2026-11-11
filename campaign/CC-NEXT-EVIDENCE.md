# CC-NEXT-EVIDENCE — read-only evidence pack for choosing the next CC

- **Prepared:** 2026-11-10 by the x86 lead (Claude Opus), from existing records only.
- **No GPU runs, service changes, builds, installs or pushes** were made for this document.
- **Base:** branch `x86/c2-resume` at `ecaa4b0d1ed134d598993ca9f26e21745b0bbd4e`. This file and its ZIP are **uncommitted** working-tree additions.
- **Fresh read-only CPU actions taken for this document:**
  1. Recomputed text hashes from raw stdout.
  2. Reran `cc/profilers/ncu_summary.py` and `cc/verdict/verdict.py` on committed records.
  3. Recomputed medians from `result.json`.
  4. Ran the CPU unit-test suites earlier today (closeout).
- **Labels used throughout:**
  - **OBS** — observed in a raw record (path + sha given).
  - **INF** — inference from records.
  - **PRED** — prediction.
  - **ABSENT** — no record exists.
- **ZIP:** `CC-NEXT-EVIDENCE.zip` sits next to this file. `ZIP-MANIFEST.sha256` inside it lists every member. No weights, images, auth files or raw agent transcripts are included.

### Your +2.1749% (cross-check)
OBS. It matches **release-1** job `job-85805b616c` (`lab/jobs/job-85805b616c/att-1/result.json`, sha `dcbd41b1cc1c…`). The values are Q8_0 tg128 medians of n=3 avg_ts each:

```
c1_q8_tg_r0 (GGML_CUDA_Q8_REUSE=0) median 793.178
c1_q8_tg_r1 (GGML_CUDA_Q8_REUSE=1) median 810.429   -> +2.1749%
```

The D029 number (+2.0678%) comes from the *q8reuse-2* image, job `job-34c09bec98`. `cc/verdict` reproduces it from `result.json` sha `4c1fc620c6a5…`.

---
## 1. Q8 reuse — mechanism and qualification

### 1.1 Identities (OBS)
| Item | Value |
|---|---|
| Patch | `cc/q8_reuse/q8-activation-reuse.patch` sha256 `9943dee44b930f4febd23dce91fcb9c50ff67625ede5fb3fcc8126ea734a0675` (3 files: common.cuh +28, ggml-cuda.cu +71, mmvq.cu +17/−4) |
| Base source | llama.cpp `31288c9c66e59377f7788d920678a565bb89dd37`, tree `158a7ade…`, clean |
| Unpatched image q8base-2 | run_ref `sha256:11be9d67af63bbc624b127e205aa67fa36c418d576b09145002c3a8a94292b27`, config `sha256:0d1263107a1d…` |
| Patched image q8reuse-2 | run_ref `sha256:bd78b473d7b42280ba91e2ae6c2c5c0e44948de5d5ae23f9340fa7cf1a1838ae`, config `sha256:111ddae86bb2…` |
| release-1 (q8_reuse + kv_quant_fix) | run_ref `sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e`, config `sha256:d6d073145e27…`, stack patch `2e340d159bc5…` |
| up1 (stack rebased to upstream 3b4d6ab) | run_ref `sha256:00454e957a5a859275606f811bb9fa7e85b9855b157c9125703ba535f2977e51`, patch `516e21606a20…` |
| Switch | `GGML_CUDA_Q8_REUSE`: exactly `0` disables it. Unset or any other value enables it. Read once per process. |

### 1.2 Case-by-case coverage
| Case | GPU-executed? | Host-mock only? | Evidence |
|---|---|---|---|
| Eligibility (Q8_0, contiguous F32 `[N,1,1,1]`, F32 out, non-overlap, non-COMPUTE weights) | **Indirectly.** It runs inside real models and inside isolated `test-backend-ops MUL_MAT`. No test asserts that the eligibility *decision* was taken. | Yes: `host_check.cpp` extracts the eligibility code with mocked CUDA | `artifacts/codexH/validation.txt` (sha `9f5238f1…`): `PASS: host cache identity, overwrite, window, lifetime and fallback checks; enabled=1` ×3 / `enabled=0` ×1 (switch values unset/0/1/false) |
| Aliasing / overlap (equal pointer, partial view overlap) | **ABSENT** on GPU | Host mock only (same file) | — |
| Stale input (same tensor object/address with new values; intervening writer) | **ABSENT** as a targeted GPU test. INF: every real decode step re-uses addresses with new values, and the text stayed identical, but nothing isolates this case. | Host mock only | — |
| Shape switch (grow scratch, then return to an older captured graph) | **ABSENT** | Not in the mock (the mock covers lifetime/allocation only) | README step 3 lists it as required. It was never run. |
| Concurrent streams | **ABSENT**. By design reuse is disabled when concurrent events are planned. That gate was never exercised. | Partly (fallback path in the mock) | DESIGN.md: "Reuse is disabled … when the evaluator plans concurrent events" |
| CUDA graph capture/replay | **Implicitly, graphs ON** (default) in every llama-bench and llama-completion run of job-34c09bec98 and job-85805b616c. `GGML_CUDA_DISABLE_GRAPHS` was **never set** for any q8 run. | No | test defs `lab/tests/c1q8.json` (sha `3c9aab87…`) |
| Graphs-disabled path | **ABSENT** | — | — |
| Isolated op correctness | Yes: `test-backend-ops -b CUDA0 -o MUL_MAT` | — | See 1.3. INF: an isolated single-MUL_MAT graph can *populate* the cache but cannot *hit* it, because a hit needs two eligible consumers in one window. So backend-ops shows the patch doesn't break MUL_MAT; it does **not** test the reuse-hit path. |
| MoE `MUL_MAT_ID`, fused MMVQ | Out of scope by design ("not covered"). The up1 backend-ops output contains **0** `MUL_MAT_ID` lines (`-o MUL_MAT` matches only MUL_MAT). | — | DESIGN.md "Coverage and open risks" |

### 1.3 Exact GPU commands, return codes and raw paths (OBS, from `evidence/*.jsonl` run_op intent/outcome records)
| op | ts (UTC) | argv (abridged only by line-wrap) | exit | elapsed | stdout sha256 | result |
|---|---|---|---|---|---|---|
| `op-ce5c24c377e0` | 2026-11-09 22:56:15 | `docker run --rm --label campaign=c1 --gpus all --network none --entrypoint /app/test-backend-ops sha256:bd78b473… -b CUDA0 -o MUL_MAT` (reuse **unset = ON**) | 0 | 49.6 s | `8a006d2711db…` | `2445/2445 tests passed` |
| `op-d7fe288cf1fb` | 2026-11-09 23:42:00 | same, image `sha256:5b8a6a22…` (release-1) | 0 | 50.4 s | `8a006d2711db…` (byte-identical to the above) | `2445/2445 tests passed` |
| `op-467dd889324a` | 2026-11-10 04:20:14 | `docker run --rm --gpus all --network=none --entrypoint /app/test-backend-ops sha256:00454e95… test -b CUDA0 -o MUL_MAT` (up1) | 0 | 44.4 s | `78c3633c15d9…` | `2517/2517 tests passed` |
| `op-7fc5ee29a9ae` | 2026-11-09 22:57:05 | `llama-completion` on **q8base-2** (`sha256:11be9d67…`), Q8_0, chat.txt, greedy | 0 | 3.1 s | `658947762b307a3a1610a8e87850484e38c9f6a147d761f3abf44f602c47d4ce` | unpatched reference text |
| `job-34c09bec98` att-1 | 2026-11-09 22:58 | 36 tests: llama-bench `-p 0 -n 128 -r 10 -o json -ngl 99`; llama-completion `-f /tin/chat.txt -n 160 -c 4096 --temp 0 -s 1 -ngl 99 -no-cnv --no-warmup`; llama-perplexity `-c 512 --chunks 8 -b 512 -ub 1 -ngl 99`; each with `GGML_CUDA_Q8_REUSE=0/1`, 2 ABBA cycles | all 0 | — | `result.json` `4c1fc620…` | see below |

The raw stdout/stderr for each op lives in `artifacts/op-<id>/{stdout,stderr}`. All three backend-ops runs print `Backend 2/2: CPU … Skipping` and `2/2 backends passed`. Their stderr is identical (`c27396a8…`): the device line `NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, VMM: yes`.

**Text identity, recomputed today from raw stdout (OBS):**
- **Q8_0:** all 8 `c1_q8_gen_r{0,1}#*.stdout` hash to `65894776…`, the same as the unpatched `op-7fc5ee29a9ae`.
- **mixE4K:** all 8 hash to `950b46b1…`.

**PPL (OBS):** `c1_q8_ppl_r0#32` and `r1#33` both print `Final estimate: PPL = 9.6530 +/- 0.56029` at `-ub 1`.

### 1.4 Engagement counter / paired launch count
- **Direct engagement counter: ABSENT.** The patch has no log line, counter or stat. Its only instrumentation is `getenv("GGML_CUDA_Q8_REUSE")` (patch line 69–71).
- **Paired ON/OFF quantize-launch count: ABSENT.** No nsys or ncu capture exists for the same binary with OFF and ON.
- **Indirect, unmatched signal (INF, do not cite as qualification):**
  - Baseline: stock image `ghcr…@sha256:395d01bb…`, nsys `--cuda-graph-trace=node`, tg128, `artifacts/profiles/q8_node_cuda_gpu_kern_sum.csv`. It shows `quantize_q8_1` 55,825 instances = `mul_mat_vec_q` 55,825, i.e. **1.00 quantize per Q8 matvec**.
  - release-1 under ncu, a 600-kernel window (`artifacts/ncu1/ncu.csv`, reuse ON by default, `-fa 1`): 101 `quantize_q8_1` against 149 `mul_mat_vec_q<8,…>` (99 unfused + 25 + 25 fused variants), i.e. **0.68**.
  - The direction is what the design predicts: q/k/v share one quantization, while fused variants keep their own.
  - **Confounders:** different build, different `-fa`, a different profiler, and a window that is not token-aligned.
- **Strongest component-level evidence:** D036 (OBS). q8base-2 and q8reuse-2 differ in exactly one shipped file, `libggml-cuda.so`, and swapping only that file into the base image reproduces about +2.0%.

### 1.5 Logit parity
- **Raw F32 logit parity: ABSENT.**
- **What exists:**
  1. Greedy text sha equality (base / OFF / ON).
  2. Printed PPL equality at `-ub 1`.
  3. No q8_reuse KLD run was ever made. The README's step 2 KLD-zero gate was not executed. D029 used text + PPL + backend-ops, the D026 frozen rule.
- **Tool caveat:** `llama-perplexity --kl-divergence-base` at this revision saves uint16-quantized, clipped log-probs, not raw logits (`cc/q8_reuse/README.md`, "Tool limitation").
- Raw-logit harnesses exist **for other components only**: `cc/hybrid_split/compare_logits.cpp` and `artifacts/codexR/debug_logits.cpp`. Neither was ever run against q8_reuse.

### 1.6 Smallest still-missing qualification test
**PRED: about 2 GPU minutes, no new code.** Run the same release-1 binary under nsys twice, `GGML_CUDA_Q8_REUSE=0` and `=1`:

```
nsys profile --trace=cuda --cuda-graph-trace=node … /app/llama-bench -m Q8_0 -p 0 -n 32 -r 1 -ngl 99
nsys stats --report cuda_gpu_kern_sum --format csv
```

Then compare `quantize_q8_1` instances per `mul_mat_vec_q` instance:
- **Expected (PRED):** OFF ≈ 1.0, ON < 1.0. The D011 hypothesis was "3 quantizes → 1" per q/k/v window.
- **What it provides:** a paired, direct engagement measurement.
- **The next-smallest:** a raw-F32 logits byte comparison, base vs OFF vs ON, × graphs `unset` vs `GGML_CUDA_DISABLE_GRAPHS=1`, plus one shape switch (two prompt lengths, then back). That needs a ~100-line harness adapted from `cc/hybrid_split/compare_logits.cpp`.

---
## 2. Profiler input for a reliable tool (ncu)

### 2.1 Inventory (OBS)
| File | Bytes | sha256 | Git | Form |
|---|---|---|---|---|
| `artifacts/profiles/tg_matmul_ncu.csv` | 4,458 | `b9823c660647c6e1df96f2a9fa4fb971a4ca74849533b980c7c409bc96f34780` | tracked | **wide** "raw page": 1 header + 1 **units row** + 12 kernel rows |
| `artifacts/profiles/tg_matmul_ncu.ncu-rep` | 7,240,830 | `00ff20a316180b32a77a3d7728c1227ff5f30b48f2b43eb35b7550ff47a1e5d2` | **untracked, local only** (not in ARTIFACT-MANIFEST) | binary report |
| `artifacts/ncu1/ncu.csv` | 874,887 | `1237b06752eef88c1b599b4c0389e55f56ee7b351e9fe53c7eed461dd42c442e` | tracked | **long-form** (one metric per row), preceded by 2 `==PROF==` lines |
| `artifacts/codexPROFW/ncu-summary.json` | 5,379 | `cf8446bb186694fabc130db2295d89c3ac81476a4004b28f87c8d6c8920d01a5` | tracked | parser output |

**The smallest real long-form CSV is `artifacts/ncu1/ncu.csv`; it is the only one.** It has **no `.ncu-rep`**: it was written directly with `--csv --log-file`.

### 2.2 `artifacts/ncu1/ncu.csv` identity (OBS unless marked)
- **Capture op:** `op-b78485045e65`, 2026-11-10 03:01:39 UTC, exit 0, 77.2 s, stdout sha `d43b3f7f369e…`. The argv is verbatim:

  ```
  docker run --rm --label campaign=c1 --gpus all --cpus 4 --memory 8g --pids-limit 256 --cap-add SYS_ADMIN
    -v …/artifacts/ncu1:/w -v /home/muk/models/campaign/gguf:/m:ro -w /w -e LD_LIBRARY_PATH=/w/app
    campaign/cuda-tools:c1 ncu --launch-skip 2000 --launch-count 600
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed
    --csv --log-file /w/ncu.csv /w/app/llama-bench -m /m/granite-3.1-1b-a400m-instruct-Q8_0.gguf -ngl 99 -fa 1 -p 0 -n 32 -r 1
  ```
- **Export argv:** none. The CSV is the direct capture output.
- **Profiler version:**
  - Image `campaign/cuda-tools:c1` (id `sha256:21382da8240c…`, Dockerfile `builds/cuda-tools/Dockerfile` sha `e20bb7e9…`, `FROM nvidia/cuda:12.8.1-devel-ubuntu24.04`).
  - The container banner says `CUDA Version 12.8.1`.
  - **The ncu version string is ABSENT from the record.** INF: the ncu bundled with CUDA 12.8.1 devel, i.e. the 2025.1.x line.
- **Process / device:** `Process ID 99`, `Process Name llama-bench`, `Device 0`, `CC 12.0`, `Stream 14`, `Context 1`; 600 kernels with IDs 0–599; one section, `Command line profiler metrics`.
- **ROI / warmup:**
  - The only ROI is `--launch-skip 2000 --launch-count 600`. There are no NVTX ranges and no token alignment.
  - llama-bench's own warmup run falls inside the skipped 2000 launches (INF, not verified).
  - The llama-bench line under ncu reports `tg32 0.44 ± 0.00` t/s, so profiled time is not inference speed.
- **Engine identity: weak.**
  - The op description says "Granite Q8_0 release-1". The binaries came from `/w/app` = `artifacts/ncu1/app/`, which **no longer exists**.
  - How it was populated was not recorded by run_op, and its library hashes are **ABSENT**.
  - So the release-1 identity rests on the op description only.
- **Model:** `granite-3.1-1b-a400m-instruct-Q8_0.gguf`, sha `e92cd1e63bc63028…` (manifest).
- **Workload:** decode only (`-p 0 -n 32 -r 1`), `-fa 1`, all layers on the GPU.

### 2.3 The D011 capture (wide form)
- **Capture:** `op-c57eaa560eea`, 2026-11-09 20:42:06, exit 0, 5.9 s. ncu **2025.1.1**, host-mounted from `~/models/campaign/tools/nsight-compute/2025.1.1`, entrypoint `/ncu/ncu`. Image: **stock** `ghcr.io/ggml-org/llama.cpp@sha256:395d01bb…`. Command:

  ```
  --kernel-name regex:mul_mat_vec_f|mul_mat_f --launch-skip 600 --launch-count 12
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size
  -o /out/tg_matmul_ncu -f /app/llama-bench -m /m/granite-…-bf16.gguf -p 0 -n 16 -r 1 -ngl 99
  ```
- **Export:** a direct Bash command, logged only by the Claude Code hook and **not** by run_op, at 2026-11-09 20:42:20:

  ```
  ncu -i /out/tg_matmul_ncu.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,dram__bytes_read.sum,…
  ```
  It was redirected to `tg_matmul_ncu.csv`. The exit code is **ABSENT** from run_op evidence.
- **Model:** **BF16**, not Q8_0.

### 2.4 Matched baseline / Q8-OFF / Q8-ON capture
**ABSENT.** The two ncu captures differ in model (BF16 vs Q8_0), engine (stock vs release-1), kernel filter (regex vs all) and form (wide vs long). Neither is an OFF/ON pair.

### 2.5 Unavailable vs zero
- In `ncu1/ncu.csv`, every one of the 2400 metric rows has a numeric value: no `n/a`, no zero values (OBS).
- `dram__bytes_write.sum` was **not collected**.
- `cc/profilers/ncu_summary.py` (sha `de3baae1…`) handles that as follows:
  - It sets top-level `"byte_scope": "read"` (correct).
  - But each class reports `"write_bytes": 0.0` (line 82: `metrics.get(WRITE, 0)`).
  - **This is an "unavailable shown as zero" defect in the per-class output.** A consumer reading `write_bytes` alone would be misled.
  - Line 63 raises `incomplete write-byte coverage` only when *some* launches have writes.
- The wide CSV's units row shows `Mbyte`, `%`, `us`, while the long-form gives `byte`, `%`, `ns`. The parser has a unit-handling test (`test_ncu_units`).

### 2.6 Metrics actually present
- **`ncu1`:** `gpu__time_duration.sum` (ns), `dram__bytes_read.sum` (byte), `dram__throughput.avg.pct_of_peak_sustained_elapsed` (%), `sm__throughput.avg.pct_of_peak_sustained_elapsed` (%).
- **`tg_matmul_ncu.csv`:** the same, in Mbyte/us, plus `launch__grid_size`.
- **Not present:** write bytes, L2, occupancy, instruction counts, launch overhead, CPU-side timing.

### 2.7 Parser tests and latest results (OBS)
- **Tests:** `cc/profilers/tests/test_profilers.py` (sha `ef62868a…`). The ncu-relevant ones are `test_real_ncu` (which reads the real `artifacts/ncu1/ncu.csv`), `test_ncu_units`, `test_ncu_weighting_and_strict_thresholds`, `test_ncu_rejects_bad_or_incomplete_metrics` and `test_capture_qualified_flags`.
- **Latest result, run today (2026-11-10, CPU only):** `cc/profilers/ | 29 passed in 0.10s`. The Codex-time result, `artifacts/codexPROFW/pytest.txt`, was also `29 passed in 0.18s`.
- **Fresh rerun today:** `python3 -B cc/profilers/ncu_summary.py artifacts/ncu1/ncu.csv` gives rc=0, and the output is **byte-identical** to the committed `ncu-summary.json` (sha `cf8446bb…`): `effective_GB_s 221.85`, `launch_bound_count 401/600`, `launch_bound_time_share 0.458`.

### 2.8 Practical failures and format differences (OBS)
- **Permissions:** ncu needs `--cap-add SYS_ADMIN` per container (D004: host `RmProfilingAdminOnly=1`).
- **Distorted timings:** ncu serialises kernels and flushes caches (D041 caveat), which inflates small-kernel time. The 401/600 launch-bound result partly reflects this.
- **Not machine-pure CSV:** `--log-file` output starts with `==PROF==` lines, so parsers must skip non-CSV lines.
- **Two shapes:** `--page raw` output is wide with a units row; `--csv` capture output is long-form. They need different parsers.
- **Throughput shift:** `llama-bench` throughput drops to 0.44 t/s under ncu.
- **First-launch cost:** D004 observed a large first-launch cost; the skip count handles it.
- **Unpreserved app copy:** `artifacts/ncu1/app/` was not kept, so the engine identity is weak.
- **Untracked export:** the export command for the D011 CSV ran outside run_op.

---
## 3. Build CC portability

### 3.1 One complete successful record: release-1 (OBS)
`builds/results/release-1/build-record.json` (sha `53cc452bf58650ccbcfd23aa225a43c7b8110b964b01eacc2cc0621877115cfb`, 85,815 bytes, included in the ZIP).

| Field | Value |
|---|---|
| CLI (op-01a0b4c24cc7, exit 0, 63.6 s) | `python3 cc/build_engine/build_engine.py --source ref/llama.cpp --commit 31288c9c66e59377f7788d920678a565bb89dd37 --patch cc/components/release-stack-1.patch --extra-target test-backend-ops --tag campaign/llamacpp:release1 --out builds/results/release-1 --linkage` (OBS: verbatim run_op intent argv; recorded env keys: `CAMPAIGN_ID`, `CUDA_VISIBLE_DEVICES`, `LANG`) |
| Env | `BASE_DEVEL` and `BASE_RUNTIME` were not set. The CLI resolved them from local tags/defaults to `nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66` and `nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867`. `CAMPAIGN_ID=c1` came from run_op. |
| Patch | `2e340d159bc51d9918bfdf41327f989478daa52cd2dab3e876a8fe722d663031` |
| Recipe | `recipe_sha256 042d4ce5…`; `cc/build_engine/recipes/Dockerfile` `a8db815e…` (2,151 B); `docker-bake.hcl` `361d5f55…` (1,243 B) |
| CMake | `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF …` plus the derived `LLAMA_BUILD_TESTS=ON` (because of `--extra-target`) |
| Tooling | Docker 29.8.2 (API 1.56), buildx v0.37.1 |
| Identity | run_ref = Id = index digest `sha256:5b8a6a22…`; config `sha256:d6d07314…`; manifest `sha256:4615c3b7…` |
| Linkage | `passed`: `libcudart.so.12` dynamic; imports cudaMalloc/cudaFree/cudaMemGetInfo/cuMemCreate present |
| Output | `engine-image.docker.tar`: 2,210,451,968 B, sha `146931d6759f33b5f4046d9d7c2a4ca5c4947a2469707e2aad6ae834fbcb78dc`, at `builds/results/release-1/` (**local only**) |
| Times | `build_wall_seconds 16.0` (**warm** ccache), `wall_seconds 63.5` |

### 3.2 Cold vs warm build times (OBS, all 21 records)
| Category | Records (build_wall_seconds / wall_seconds) |
|---|---|
| Cold or near-cold successes | b0-baseline-3 446.1 / 479.4 s; up0 445.8 / 498.1 s (new upstream checkout); up1 354.7 / 405.1 s; q8reuse-2 241.6 / 290.7 s |
| Warm (ccache) | release-1 16.0 s; kvfix-1 17.4 s; rt1 18.0 s; hsplit-1 36.0 s; q8base-2 57.5 s; novmm-4 0.4 s; b0-baseline-5 0.9 s |
| Failures | b0-baseline (bake entitlements: `ERROR: additional privileges requested`); b0-baseline-2 (cmake, 677.8 s); q8base-1 and q8reuse-1 (cmake: `test-backend-ops` target without `LLAMA_BUILD_TESTS=ON`, later fixed by deriving it); novmm-1 (`inspect_image`: "Buildx metadata lacks an immutable image config digest"); novmm-2 and shot-1 (`hash_tools`: `docker run` exit 125 on the config digest); novmm-3 and shot-2 (`linkage`) |

### 3.3 Source checkout requirements (OBS, README + code)
- A local clone (`ref/llama.cpp`, `ref/llama.cpp-up`). These are **gitignored and not distributed**.
- HEAD must equal `--commit`, with no tracked changes, no untracked files and no submodules.
- The CLI never downloads source.
- Base images must be present locally with RepoDigests, otherwise the build fails at `base_images`.
- apt packages are installed inside the build, which needs a registry/apt mirror or cached layers.

### 3.4 Hidden absolute paths, restrictions, assumptions (OBS)
- **The records embed absolute host paths:**
  - `source.path = /home/muk/projects/siminfeng/campaign/ref/llama.cpp`;
  - buildx provenance `vcs:localdir:context = campaign/builds/results/release-1/staging-02tppnvh/source`;
  - `vcs:source = git@github.com:Muxite/siminfeng.git`;
  - `vcs:revision = 6887259c…` (the *campaign repo* commit, not llama.cpp).
- **Assignment restriction (README):** "For assignment T-N, keep outputs under `cc/build_engine/` or `artifacts/codexN/`". The factory separately refuses writes outside assignment roots (`cc/factory`: "write outside assignment roots").
- **Hard-coded:** `CMAKE_CUDA_ARCHITECTURES=120` (`build_engine.py:21`). Only `linux/amd64` is accepted (`:181, :195, :213, :443`: "Built image is not linux/amd64").
- **Unsupported:** Jetson/arm64/sm_87, submodules, dirty trees, remote source, other CUDA arches, signature/attestation verification.
- **`artifacts/dl2/build_up.sh` line 3** assigns a made-up 40-hex placeholder `U=3b4d6ab3f6f0d2b4e0f3b4b8a5c1f1b1d0d6e9a1`. Line 4 overwrites it with `git rev-parse 3b4d6ab3f`, so the builds used the real `3b4d6ab3f79651935fba1dd93732f29022a7f5f5`. Harmless, but sloppy; preserved as-is.

### 3.5 Surviving build processes after client timeout
- **ABSENT as a test.** All 19 run_op-wrapped build ops record `survivors: []` and `signal: None` (OBS). No build was ever killed by a client timeout.
- The CLI sets a timeout only on the 60 s linkage probe. Bake has no timeout. INF: if the client is killed, BuildKit (inside dockerd) keeps building, and `staging-*` under the output directory may remain. D001 found a run_op orphan-child bug when the wrapper is SIGKILLed; it was fixed with process-group handling for run_op itself, not for BuildKit.

### 3.6 Cleanup and retry gaps (OBS, README)
- **Failure leftovers:** "A failure after building can leave the tagged local image and partial evidence; … retry in a fresh output directory".
- **No GC:** there is no garbage collection. Today there are 16 `engine-image.docker.tar` under `builds/results/*`, totalling **33 GB**.
- **Not in the closeout manifest:** these 16 tars are **not listed in `ARTIFACT-MANIFEST.tsv`**, which covered only the codexFAC tar. Each build record holds its own tar sha. This is a closeout gap.
- **Not retained:** "patch bytes are identified by SHA256 but are not retained as a separate output artifact".

### 3.7 Does run_ref use agree with README prose? **No (OBS).**
- **Code** (`build_engine.py:487–506`, sha `b05e37a2…`):

  ```python
  run_ref = reported_id
  record["image"]["run_ref"] = run_ref
  stage = "hash_tools"
  hashes = run(["docker", "run", "--rm", "--network=none", "--read-only",
                "--entrypoint", "/bin/sh", run_ref, "-c", …
  record["linkage"] = collect_linkage(run, out, run_ref, image_tar)
  ```
- **README** (`cc/build_engine/README.md`, sha `86d93961…`, "Outputs and failure handling"):
  > "Hashing and optional linkage then use this config digest as the local `sha256:...` Docker image ID"
- **Which is right:** the code was fixed in 16c7ff6 ("probes run by Docker reported ID (containerd rejects config digest)"), after novmm-2 and shot-1 failed with exit 125. The README prose was not updated.
- **Other stale README items:**
  - "the seven requested tools": up to 8 ship with `--extra-target`;
  - "tests … are disabled": they are derived ON with `--extra-target`.
- **Tests:** `cc/build_engine` gives 113 passed today, so the tests don't catch the prose.

---
## 4. JiuwenSwarm readiness (read-only look; nothing installed or ported)
- **Repo (OBS):** `~/projects/jiuwenswarm`, origin `git@github.com:openJiuwen-ai/jiuwenswarm.git`, branch `develop` at `438c7ebee3eac08392de8b20ee6f7509cc3cf58d` (2026-11-05). The package name in `pyproject.toml` is `workswarm` 0.2.8.
- **Runtime (OBS):** it imports via `uv run --no-sync`. No `.venv`, no running process (`pgrep jiuwen` empty) and no user config dir (`~/.jiuwen*`) were found.
- **"CC v2" on this machine (OBS):**
  - The only artifacts named "CC v2" are **siminfeng's own Build CC v2** (`cc/build_engine`, schema_version 2). It lives in this campaign repo, not in JiuwenSwarm.
  - A search of JiuwenSwarm (`*.py/md/json/yaml/toml`) and all its refs found **no CC / capability-card / CC-v2 concept**.
  - The nearest concepts are ability "cards" (`agent.ability_manager.list()`, `card.name`) and a native-tool rail (`jiuwenswarm/agents/harness/common/rails/cua_extension_rail.py`).
  - **CC v2 is not installed in JiuwenSwarm.** If you mean a different "CC v2" distribution, it is **ABSENT** here, and I need its repo/branch.
- **Can the current runner expose a finite read-only parser plus passive component material? (INF from code)**
  - **Parser: yes.**
    - `cc/profilers/ncu_summary.py` and `cc/verdict/verdict.py` are stdlib-only and read-only (verdict: "It performs no writes"). They are deterministic (rerun today: byte-identical) and finite: one file in, JSON out, exit 0/1.
    - Exposing either needs only a stdio/MCP shim. JiuwenSwarm's agents have MCP plumbing (`jiuwenswarm/agents/swarm/*`). **The shim does not exist.**
  - **Passive component material: yes.** This means the patch, DESIGN/README, build-record and test definitions, all plain files with shas.
- **What is missing to run the Builder as an explicitly authorized trusted native tool (INF):**
  1. **Docker daemon + buildx access.** This is root-equivalent, so it needs an explicit trust grant.
  2. **Local digest-pinned CUDA base images** (about 10 GB).
  3. **A clean llama.cpp checkout** at the pinned commit (gitignored, so it must be supplied).
  4. **About 2.2 GB of disk per build archive, plus ccache.**
  5. **Cancellation and timeout semantics** for bake (absent).
  6. **Relocation:** de-absolutising source/out paths, or declaring them.
  7. **Policy decisions:** the amd64/sm_120-only constraint stated as a capability limit; the assignment-root write policy mapped to JiuwenSwarm's sandbox model.
  8. **A result envelope** the swarm can consume. `cc/engine_handoff` exists but has never been exchanged with a remote receiver.

---
## 5. Contradictions resolved

### 5a. up1 MUL_MAT backend-ops: D044 is right, CLOSEOUT is wrong (OBS)
- **The resolving raw artifact:** `artifacts/op-467dd889324a/stdout` (sha `78c3633c15d96f9eb1a8434bcbf9ba16e80c2fa5798092860b331e26609de321`, 424,394 B).
  - Command: `docker run --rm --gpus all --network=none --entrypoint /app/test-backend-ops sha256:00454e95… test -b CUDA0 -o MUL_MAT`.
  - Run 2026-11-10 04:20:14 UTC, exit 0, 44.35 s.
  - Excerpt: `2517/2517 tests passed`, `2/2 backends passed`.
- **Scope note:** the op description says "MUL_MAT/MUL_MAT_ID", but argv `-o MUL_MAT` produced **0** MUL_MAT_ID lines. Only MUL_MAT was tested.
- **Correction needed:** CLOSEOUT §1 ("No test-backend-ops run … on up1") and §2b are incorrect. **Not corrected in git yet** (no pushes were requested). Speed requalification of q8_reuse on up1 remains **ABSENT**.
- **up0:** shipped without `test-backend-ops` (no `--extra-target`), so no backend-ops run exists for up0.

### 5b. Demo rehearsal: the exact script WAS live-tested once (OBS)
- **The Codex validation was static only:** `artifacts/codexDEMO1/validation.txt` says `NOT RUN Docker or bin/lab run; live timing/readiness unverified`. Codex had no Docker.
- **The orchestrator then ran it live**, as hook-logged command:

  ```
  CAMPAIGN_ID=c1 timeout 900 bash cc/demo/demo.sh > artifacts/x5-env/demo-rehearsal-1.log 2>&1
  ```
  - It ran 2026-11-09 23:06:24 → 23:06:38 UTC (hook `duration_ms: 13540`) and printed `rc=0`.
  - **Log:** `artifacts/x5-env/demo-rehearsal-1.log` (tracked, sha `6ccaca17b2baa13e0f45b5234d6a7238cbd3ea1b66b752d0080b3085c301c915`, 533 lines).
  - **Run records:** `artifacts/codexDEMO1/demo-20261109T230625.053878644Z-1426862/`, job `job-6cb9c2ac5c`.
  - **Final summary:**

    ```
    4. att-1: sandbox interrupted, partial evidence exported and hashes verified; stop confirmed.
    5. att-2: 2/2 clean completions; att-1 unchanged
    ```
  - Log line 284: `error: no such object: lab-c1-job-6cb9c2ac5c-att-1-m1_static_n12-0`. INF: a container inspect after the kill had removed it; it did not fail the run.
- **Script identity:** `cc/demo/demo.sh` sha `0d81e7804ea761c07d6ba0a689db85f031d4689bcd50d81e32702a13cda4fcdd`.
  - It is unchanged since its only commit, `95c5c1c` (23:07:04, 26 s after the run).
  - INF: the run used these bytes. The run itself did not record the script sha.
- **Not done:** the demo was **not rerun since** (no fresh result).
- **Weak points:**
  - It uses image `llamacpp-novmm4` and old D025-era tests, not release-1 or Qwen.
  - The run was hook-logged, not run_op-wrapped, so there is no run_op exit-code record; rc comes from the echoed `rc=0`.

---
## 6. Judgment: three narrowly scoped candidates

Ranked by usefulness, qualification effort and integration friction. Build CC is **deliberately not ranked first**. It is the most useful in principle, but §3–4 show it has the most friction (Docker trust, 2.2 GB outputs, amd64/sm_120 only, stale docs).

### #1 — `ncu_summary`: ncu CSV → kernel-class bandwidth / launch-bound summary (tool)
- **Replaces:** hand-pivoting ncu CSVs to answer "which kernel classes are bandwidth-bound and which are launch-bound" (done by hand for D011 and D041).
- **Trustworthy output:** per-class launch counts, summed `gpu__time_duration`, read bytes, and DRAM/SM % weighted by duration, under ncu's serialised conditions. It also fails closed on missing metrics and unknown units.
- **Likely misleading results:**
  - Absolute times and `effective_GB_s` for small kernels (ncu serialisation and cache flush). `launch_bound_count` inflated relative to graph replay.
  - **`write_bytes: 0.0` when writes were not collected** (§2.5).
  - Engine identity carried only by the CSV path.
- **Failure modes:** `==PROF==` preamble lines; wide vs long form; unit-scale mismatches; a non-token-aligned window; a missing `--cap-add SYS_ADMIN` at capture.
- **Effort and friction:** smallest qualification effort. It is already tested: 29 tests, including a real-file test. Lowest friction: stdlib, read-only, finite.
- **Fixes to make first:**
  1. Emit `null` for uncollected write bytes.
  2. Accept the wide `--page raw` form, or reject it explicitly.
  3. Require an identity sidecar (engine run_ref, model sha, argv).
- **Smallest demo:** `python3 cc/profilers/ncu_summary.py artifacts/ncu1/ncu.csv` → `effective_GB_s 221.85`, 401/600 launch-bound. Then show it rejecting a truncated CSV.

### #2 — `verdict`: frozen-criteria verdict from lab records (tool)
- **Replaces:** hand-computing medians and deltas and checking a frozen protocol rule against `result.json` (every D-note).
- **Trustworthy output:** PASS/FAIL per criterion, with job/result sha256, sample lists, aggregates and alternating order (§0: D029 reproduced, +2.0678%).
- **Likely misleading results:**
  - It judges only what the criteria file encodes. It would "PASS" a run whose engagement was never shown (D024-type silent non-engagement) or whose host was noisy (D035).
  - Its `backend_ops` criterion reads a *legacy sidecar log associated by the criteria file*. The tool labels this `"association": "legacy sidecar associated by criteria; not a lab test"`. The backend-ops evidence is therefore only as good as the path the criteria author chose (`verdict.py` ~l.197–213).
- **Failure modes:** criteria drift from the D-note; legacy sidecar paths (`--root`); no criteria files yet for D040/D044+.
- **Effort and friction:** low (26 tests + 9 subtests pass; read-only; exit 0/1).
- **Smallest demo:** four commands (D029-q8 PASS rc=0; D029-mix FAIL rc=1; D033-q8_0 PASS; D033-q4_0 FAIL), rerun today with the same outcomes as its README table.

### #3 — `q8_reuse` component CC (patch + qualification pack)
- **Replaces:** nothing a user does by hand. It is an engine component: +2.1% Q8_0 Granite decode, computation-preserving by text/PPL evidence.
- **Trustworthy output:** the patch applies to 31288c9c6 (and, rebased, to 3b4d6ab). It is output-preserving on two Granite variants. The +2.07–2.17% is same-binary on the RTX 5070 Ti.
- **Likely misleading results:**
  - "Engine improvement" for MoE in general. It does **not** touch `MUL_MAT_ID` (the MoE experts) and applies only to unfused Q8_0 matvecs.
  - Gains on Qwen, Q4_K or other GPUs are **ABSENT**.
- **Failure modes:** heavily fused graphs may see no gain; graphs-disabled, shape-switch and concurrent-stream paths are untested on the GPU; there is no engagement counter.
- **Effort and friction:** medium. It needs the §1.6 paired launch count, the raw-logit harness and an up1 speed rerun, all GPU-time.
- **Smallest demo:** same-binary `GGML_CUDA_Q8_REUSE=0/1` llama-bench tg128 on Granite Q8_0 (about 1 min), plus the nsys quantize count (§1.6).

### Data the campaign failed to collect that most limits this choice
1. **No engagement or launch-count telemetry for any engine component.** q8_reuse is judged by its speed delta alone, and the ncu/nsys captures were never paired with an identical OFF/ON binary. This blocks qualifying #3 and limits what #1 can be shown to diagnose.
2. **Tool-output identity:** profiler captures don't record the exact engine binaries (the `artifacts/ncu1/app/` copy is gone) or the profiler version.
3. **No raw-logit parity evidence** anywhere for the computation-preserving claims.
4. **No consumer-side trial:** none of these tools was ever invoked by a second agent or runner (JiuwenSwarm or otherwise). Integration friction is inferred, not measured.

---
## 7. Campaign and service state (observed fresh)
Captured with `nvidia-smi`, `docker inspect`, `systemctl --user is-active` and `pgrep`. The concrete values are in the ZIP's `state-at-writing.txt`. The last-reported values were: GPU used only by the owner's `euaaa-llm-gpu-1` (llama-server) and Sunshine; no campaign processes; `kernel.perf_event_paranoid=1`.

Awaiting the next development assignment. No further work is in progress.
