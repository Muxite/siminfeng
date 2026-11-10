# External survey: MoE larger-than-memory inference, Jetson/x86 tooling (2026-11-10)
Method note: GitHub REST API (repos, search/issues, pulls) queried 2026-11-10; WebSearch (extended/standard);
WebFetch of NGC / NVIDIA forum pages; local probes on this Jetson (R36, kernel 5.15.199-tegra).
Verification levels: [V] = fetched/API-confirmed today, [S] = search-snippet only (verify before relying).
Our pin 31288c9c6 is 519 commits behind master (3b4d6ab3f, 2026-11-10) per compare API.
No version below is recommended unless >= 2 weeks old (cutoff 2026-10-27); newer items are marked "watch".

## A. Engines / forks

### A1. llama.cpp upstream, post-31288c9c6 (MIT, all platforms incl. CUDA aarch64)
Found by: `GET search/issues?q=repo:ggml-org/llama.cpp+<kw>+is:pr+created:>2026-10-01` for kw in
moe offload / moe-cache / cuda graph moe / expert cache ssd / jetson orin / unified memory mmap / lazy-mode;
then `GET pulls/<n>` for bodies; `compare/31288c9c6...master` for ahead_by.
- #29887 `--moe-cache-mib N` GPU LRU cache of host-resident experts (merged 2026-10-07; port of Tether qvac-fabric
  MoE cache; only <=32-token batches; suggests cache ~10% of expert size). Already in or near our pin: verify with
  `git merge-base --is-ancestor`. Verdict: ADOPT (it is the core mechanism for our goal).
- #30966 moe-cache: run missed experts on CPU (`--moe-cache-cpu-share`) + tuning script (merged 2026-11-07).
  Useless where CPU == same DRAM bus as GPU? On Orin (unified LPDDR5) CPU read of expert may beat H2D copy: STUDY/benchmark.
- #31233 LRFU/MQ-like ClassLRU replaces SLRU (merged 2026-11-06): +6-7% decode warm cache, 8% fewer uploads. ADOPT w/ rebase.
- #31888 CPU part no longer waits on cached-expert matmuls (merged 2026-11-08): +7-9% TG on Gemma 4 -cmoe. ADOPT.
- #31286 cpu-share auto controller: closed unmerged ("fixed 0.95 works just as well"). Reject; use fixed value.
- #31850 (open) separate CUDA stream for expert H2D (`GGML_CUDA_UPLOAD_STREAM=1`, `GGML_SCHED_COPIES=N`). WATCH; on UMA
  H2D is a memcpy within DRAM so gain may be nil on Orin, real on 5070 Ti.
- #31469 (open) out-of-core MoE streaming, VRAM slot pool + RAM victim cache, `--moe-direct-io` (NVMe regime beyond RAM).
  The closest upstream analogue to our "model >> memory" goal. STUDY now (read diff, test branch on x86), WATCH merge.
- #31775 (open) sched: start new split at incompatible weight (fixes #31675 with -nkvo hybrids). WATCH.
- #31957 (open, 2026-11-09) "apply the iGPU lazy-mode fallback only when mmap is off": directly about the
  `-lm auto` iGPU behavior we hit. STUDY/test on Orin; likely ADOPT as local patch.
- #27969 (merged 2026-08-29) renamed `--tensor-read-lazy` to `--lazy-mode`/`-lzm`; #30772 (closed) `-lzm direct` pread
  of PLE rows (idea reusable for expert rows). #24156 (open) reclaim mmap pages after CPU_REPACK. STUDY.
- #31268 (open) "CUDA: optimize Orin single-column MMVQ"; #30858 (merged) small_k MMVQ on Orin bs=1; #28285 (merged)
  MMVQ->MMQ crossover for SM87. All Orin-specific decode-kernel work. ADOPT (cherry-pick); #31268 STUDY.
- #29142 (closed) hardcoded 32GB VMM pool cuMemAddressReserve fails on Jetson iGPU; #29499 (open) server hang on Orin NX
  L4T 36.4.7. Check whether our pin has the #29142 fix. Reference for known Tegra bugs.
- #29949 (open FR) GPU-resident LRU where cache decisions run on GPU (CUDA-graph friendly). WATCH.
- #31288 `cuda : add top-k MoE fusion for 384 experts`, #31084 deterministic radix Top-K (merged): MoE routing fusion
  inside graphs. Low relevance to Orin bf16/Q4_K gate/up slowness; WATCH.
- #31452 (open RFC) deterministic layer & MoE streaming, "70B/141B on 16GB RAM". Low maturity; WATCH.
- #31419 (merged) ggml-backend: do not upload unread inputs. Cheap win; ADOPT with rebase.
- Issue #15034 (Gemma 3n broken on Orin Nano CUDA, closed 2025-08) and #28160 (closed; lazy-mode auto halves pp512 on Vulkan
  iGPU) are the two refs in our source; both closed, read their linked fixes.
CUDA-graph note: no MoE-specific CUDA-graph PR surfaced in the window; the search `cuda graph moe is:merged` returned only
top-k fusion PRs (#31288, #31615 Metal). Gap: nobody upstream seems to be graph-capturing expert-cache lookups.

### A2. ik_llama.cpp  https://github.com/ikawrakow/ik_llama.cpp  [V]
MIT, last push 2026-11-09, 3.4k stars, no GitHub releases. llama.cpp fork with extra quants (IQK), fused MoE ops, better
CPU/hybrid MoE. Platform: ARM NEON supported but issue #308 shows arm64 build breakage; CUDA sm_87/sm_120 unverified.
Use: source of fused-MoE and quant ideas (compare to our slow gate/up fused kernels). Verdict: STUDY (read, do not adopt; fork drift).
Found by: GitHub API repo lookup from memory of well-known forks; confirmed active via `repos/` endpoint.

### A3. KTransformers  https://github.com/kvcache-ai/ktransformers  [V]
Apache-2.0, v0.7.2 (2026-10-15), 19.7k stars. CPU-GPU heterogeneous MoE (hot experts GPU, cold CPU; AMX/AVX512/AVX2 kernels;
expert scheduling doc 2026-01). Also has a jetson-containers package (`packages/llm/ktransformers`). x86-CPU-centric: kernels
target x86 AVX/AMX, so Orin (ARM) benefit is doubtful. Use: x86 5070 Ti hybrid baseline for expert placement ideas.
Verdict: STUDY (x86), REJECT for Orin. Found by: README via contents API; jetson-containers `packages/llm` directory listing.

### A4. PowerInfer / PowerInfer-2 / SmallThinker  https://github.com/Tiiny-AI/PowerInfer  [V]
MIT, last push 2026-05-11, 9.9k stars (old SJTU-IPADS URL 301-redirects). Activation-locality neuron-level sparse
offload; PowerInfer-2 (2024, smartphone, NVMe-streamed, arXiv 2406.06282 [S from memory, verify]) plus SmallThinker-21B-A3B
models designed for on-device/SSD. Models (HF PowerInfer/SmallThinker-21BA3B-Instruct) are useful MoE test targets. Code
is a fork of llama.cpp (older base). Verdict: STUDY paper; use SmallThinker as workload; REJECT engine.
Found by: `repos/SJTU-IPADS/PowerInfer` returned "Moved Permanently" -> followed to Tiiny-AI.

### A5. MoE-Infinity  https://github.com/EfficientMoE/MoE-Infinity  [V]
Apache-2.0, v0.2.2 (2026-09-30), 382 stars, active. Sequence-level expert activation tracing + prefetch/cache for
offloaded MoE; PyTorch (not GGUF). aarch64/Jetson unverified. Verdict: STUDY (prefetch policy ideas; trace methodology).
Found by: GitHub repo API from known name.

### A6. Fiddler  https://github.com/efeslab/fiddler  [V]
Apache-2.0, last push 2024-11-18 (stale), 268 stars. Runs missed experts on CPU instead of copying weights: same idea
as llama.cpp #30966. Verdict: WATCH/reference only. Found by: tried RedHatAI/AMD-AIG paths (404) then efeslab.

### A7. qvac-fabric-llm.cpp  https://github.com/tetherto/qvac-fabric-llm.cpp  [V]
MIT, pushed 2026-11-10. Origin of the MoE cache ported to #29887. Found by: citation chain from PR #29887 body.
Verdict: STUDY (may contain further cache work, mobile/Vulkan focus).

### A8. TensorRT Edge-LLM  https://github.com/NVIDIA/TensorRT-Edge-LLM  [V]
Apache-2.0, v0.11.1 (2026-10-28), README: JetPack 7.2.2 support, Thor focus, MLPerf Edge. Also jetson-containers package
`tensorrt_edgellm`. No evidence of SSD-streamed MoE or JetPack 6.x/sm_87 support: the support matrix page
(nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html) needs reading. Verdict: WATCH.
Found by: NVIDIA org repo list in API + jetson-containers dir listing.

### A9. TensorRT-LLM  https://github.com/NVIDIA/TensorRT-LLM  [V]
v1.2.1 (2026-04-20), license NOASSERTION (Apache + NVIDIA terms). Datacenter-first, no Orin Nano 8GB path. REJECT.
### A10. vLLM  https://github.com/vllm-project/vllm  [V]
Apache-2.0, v0.33.0rc3 (2026-11-02). Jetson via jetson-containers `vllm`; needs far more RAM than 8 GB, no GGUF-MoE
streaming. REJECT for Orin; fine as an x86 reference for quality/eval parity.
### A11. SGLang  https://github.com/sgl-project/sglang  [V]
Apache-2.0, v0.5.23 (2026-11-02). Same assessment; has jetson-containers package. REJECT (Orin), watch for expert-offload.
### A12. MLC-LLM  https://github.com/mlc-ai/mlc-llm  [V]
Apache-2.0, 23.3k stars, last push 2026-10-30, no GitHub releases. TVM compiled kernels, CUDA on Jetson via jetson-containers `mlc`.
No MoE disk streaming. REJECT (MoE-on-SSD), maybe STUDY compiled-kernel tuning for sm_87.
Provenance for A8-A12: Search "GitHub repos releases" via API + jetson-containers packages/llm listing (names ktransformers,
mlc, sglang, vllm, tensorrt_edgellm present).

## B. Jetson build / image tooling
- dusty-nv/jetson-containers  https://github.com/dusty-nv/jetson-containers [V]: license NOASSERTION (MIT per README
  historically), 4.9k stars, last push 2026-11-02, no releases/tags (git ls-remote tags empty). Build system with package
  tree (`packages/llm/llama_cpp`, etc.) and `jetson-containers build`; tracks L4T/CUDA matrix. Use: copy the llama_cpp
  Dockerfile/cmake flags (`-DCMAKE_CUDA_ARCHITECTURES=87`, `GGML_CUDA_F16`) and its L4T-version detection; compare to our
  l4t-cuda:12.6.11-runtime packaging. Verdict: STUDY (adopt patterns, not the whole build system).
  Found by: Google-style memory + GitHub API `contents/packages/llm`.
- NGC l4t-cuda  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-cuda [V via WebFetch]: shows tag
  `12.6.11-runtime` as Latest (Oct 8, 2024), plus 12.2.12 devel/runtime; no 12.6 devel tag in the partial list (page truncated; check
  "View all tags"). Our base is current; for builds we use host-native, consistent with the gap. ADOPT status quo.
  Found by: NGC catalog page fetch.
- JetPack 7.2 / 7.2.1 [S]: per RidgeRun & NVIDIA archive pages, JetPack 7.2 = Jetson Linux 39.2 (2026-06-02), CUDA 13.2.1,
  TensorRT 10.16.2, Ubuntu 24.04, kernel 6.8, and it adds Orin (Nano, NX, AGX) to the JetPack 7 stack. Orin Nano has no SD image
  (USB ISO only). Implication: sm_87 + CUDA 13 + kernel 6.8 (BTF/eBPF likelier) is available as an upgrade; JetPack 6.x binaries must
  be rebuilt. Verdict: STUDY as a separate branch/host experiment (CUDA 13 might change UMA/VMM behaviour, nvidia-smi, ncu).
  Found by: WebSearch "JetPack 7.2 Jetson Orin Nano support CUDA 13"; links: developer.nvidia.com/embedded/jetpack/downloads/archive-7.2,
  developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin/JetPack_7.2/Getting_Started/Components.
- NVIDIA Container Toolkit  https://github.com/NVIDIA/nvidia-container-toolkit [V]: Apache-2.0, v1.20.1 (2026-09-19), CSV mode
  (mount lists in /etc/nvidia-container-runtime/host-files-for-container.d/*.csv) is the Tegra path; `nvidia-ctk cdi generate`
  with CDI is the modern route. Use: confirm the CDI route for our images (no runtime hook). Verdict: ADOPT (CDI); docs:
  docs.nvidia.com/datacenter/cloud-native/container-toolkit (not fetched, verify).
  Found by: GitHub API releases.

## C. Profiling / observability
- Nsight Systems on this Orin: 2024.5.4.34 (local `nsys --version`) [V]. NVIDIA forum: only the JetPack-bundled nsys
  can profile Jetson; website releases (2025.x) cannot. Version-matching host needed for GUI.
  Forum thread "GA10B GPU metrics DRAM bandwidth returns all zeros ... Orin Nano" (forums.developer.nvidia.com/t/377013) [V]:
  `mcc__dram_throughput_*` zero under `ga10b-gfxt`; unresolved; nsys 2024.5.4, ncu 2024.3.1, JP 6.2.2. Confirms our ncu/metrics
  limits; fall back to `tegrastats` EMC% and nsys CUDA trace + OS-runtime trace. Verdict: ADOPT nsys (timeline only), no DRAM counters.
  Found by: WebSearch "Nsight Systems Jetson Orin JetPack 6.2 L4T supported version".
- Nsight Compute: forum threads "ncu failure on JetPack 6.2 / CUDA 12.6" (338-342k ids) show perf-counter permission issues;
  matches admin-only finding. Option: run under sudo or set RmProfilingAdminOnly=0 module param (nvidia-tegra: check, unverified).
- jtop/jetson-stats  https://github.com/rbonghi/jetson_stats [V]: AGPL-3.0, 7.2.2 (2026-10-01), 2.6k stars. Python lib
  `jtop` reads tegra sysfs (GPU load, EMC, power rails) and has a Python API; AGPL: use as an out-of-process tool only.
  Verdict: ADOPT for sampling (jtop API in sampler), do not link into published code. Found by: GitHub API.
- tegrastats  [V local /usr/bin/tegrastats]: builtin; `--interval ms --logfile`; has EMC_FREQ, GR3D_FREQ, RAM, SWAP, IRAM. Keep.
- Local kernel config [V, /proc/config.gz]: BPF_SYSCALL, BPF_EVENTS, FTRACE, PERF_EVENTS = y; DEBUG_INFO_BTF, KPROBES,
  BLK_DEV_IO_TRACE not shown as =y by my grep (may be =m or unset; verify). Binaries: perf, bpftrace, blktrace, trace-cmd NOT in PATH.
  Verdict: tracefs/ftrace (`events/block`, `events/filemap`, `exceptions/page_fault_user`) usable with no install; bpftrace
  needs apt (blocked by our "no installs"), BTF missing means `bpftrace` kprobes with explicit struct defs only. STUDY.
  The repo's io_watchdog.py appears to cover /proc/diskstats + PSI; add `/proc/pressure/io`, `/proc/vmstat` pgmajfault.
- fio  https://github.com/axboe/fio [V]: GPL-2.0, 3.43 (2026-09-23); `/usr/bin/fio` already installed. Use: NVMe baselines
  (`--rw=randread --bs=4k..1m --direct=1 --iodepth`, mimic expert-sized reads ~ MiB scale, io_uring and pread). ADOPT.
- NVTX in llama.cpp: no upstream NVTX markers found by API search (`nvtx` query not run; gap). Cheap local patch: wrap
  ggml_backend_sched_compute_splits and moe-cache lookups with nvtx3 header-only (shipped with CUDA 12.6) so nsys shows
  expert-upload vs compute. ADOPT as local patch. (Not verified upstream; do `GET search/code?q=nvtx+repo:ggml-org/llama.cpp`.)

## D. Labs / isolation / reproducibility
- BuildKit  https://github.com/moby/buildkit [V]: Apache-2.0, v0.34.2 (2026-11-03). Reproducible builds: set
  `SOURCE_DATE_EPOCH` build-arg, `--output type=oci,rewrite-timestamp=true`; compile with `-ffile-prefix-map=$PWD=.`,
  `-Wl,--build-id=none`, `-DNDEBUG`, `CCACHE_BASEDIR`; CUDA fatbin embeds paths/timestamps (nvcc `-Xfatbin` flags,
  `--no-host-device-initializer-list`? unverified). Matches our finding "binaries non-reproducible with ccache".
  Verdict: ADOPT rewrite-timestamp + prefix-map; docs: docs.docker.com/build/ci/github-actions/reproducible-builds (not fetched).
  Found by: GitHub releases; flag names from BuildKit docs memory [S], verify.
- Testcontainers (python)  https://github.com/testcontainers/testcontainers-python [V]: Apache-2.0, 4.15.0 (2026-07-24). Disposable
  container lifecycle in tests. Needs Docker daemon; on Tegra GPU access via CDI. Verdict: WATCH (we already have run_op).
- BenchExec  https://github.com/sosy-lab/benchexec [V]: Apache-2.0, 3.35 (2026-07-03). Runs commands in cgroup-v2 with
  CPU/mem limits, measures wall/CPU/memory/IO, resists interference. Uses cgroup memory limits, which do not bound CUDA
  allocations on Tegra, so only useful for CPU/IO accounting. Verdict: STUDY (resource measurement harness model).
- systemd-run scopes: `systemd-run --scope -p MemoryMax= -p IOReadBandwidthMax=/dev/nvme0n1 8M ...` for CPU-page/IO isolation
  and per-run accounting (`systemctl show --property=MemoryPeak`). Works for page cache charge (mmap) but not nvgpu/CUDA
  allocations. Verdict: ADOPT for IO throttling experiments (sim of slower SSD). [S, standard systemd docs]
- Capping GPU memory on Jetson: no mainstream tool found. Search leads: llama.cpp #29142 (VMM pool reservation) and
  `GGML_CUDA_...` env; the realistic cap is app-level (cudaMemGetInfo-based budget + `--moe-cache-mib`, `-ngl`, `-ot`).
  Verdict: build an in-process budget guard; WATCH NVIDIA forum for "cgroup" + "nvgpu" threads.
  Found by: WebSearch not run for this; gap noted.

## E. Papers (2024-2026). arXiv IDs from search results [S]; metadata API calls returned empty (rate-limit), so verify IDs
| arXiv | Title / mechanism | Platform | Code |
|---|---|---|---|
| 2611.04794 | "Bevy": runahead scheduling for offloaded MoE; finds prefetch can hurt when GPU expert cache exists | Mac mini M6, Jetson AGX Thor, PCIe5 NVMe (unified memory) | not found |
| 2609.18063 | "The Other Half of the Memory Wall" (Edge0): 35B MoE from SSD with trained routing predictor to overlap reads | consumer | not found |
| 2609.12978 | SeqMoE: predictive, CUDA-graph-compatible MoE offloading | GPU | not found |
| 2601.17063 | FlashMoE: ML-based cache replacement vs LRU/LFU for SSD-backed edge MoE, up to +51% hit, 2.6x | desktop/edge | not found |
| 2601.21198 | ZipMoE: lossless compression + cache-affinity scheduling for on-device MoE | Jetson AGX Orin 32/64GB | not found |
| 2605.05899 | VisMMOE: visual-expert affinity offload; used swap on Jetson Orin | Orin | not found |
| 2605.27081 | ReMoE: router fine-tuning to improve expert reuse under memory limits | GPU | not found |
| 2406.06282 | PowerInfer-2 (smartphone, NVMe neuron streaming) [from memory] | phone | via PowerInfer repo |
Most relevant to us: Bevy (Jetson unified memory + NVMe, cache vs prefetch interaction), SeqMoE (CUDA-graph compatibility),
ZipMoE (Orin, compression, matches Q4_K/bf16 size problem). Found by: WebSearch (extended) "MoE expert offloading SSD unified
memory arXiv 2026 inference edge Jetson"; follow-up should read each paper's "Code availability" and citing papers.

## Top 10 to act on (ranked)
1. Rebase/cherry-pick moe-cache stack: #29887 (check present), #31233 ClassLRU, #31888 CPU-wait fix, #30966 cpu-share, #31419; re-run Orin decode.
2. Read and test #31957 (iGPU lazy-mode only when mmap off) against our `-lm auto` disables-mmap finding.
3. Evaluate #31469 `--moe-direct-io` on x86 (NVMe out-of-core): closest upstream match to the goal; watch merge.
4. Cherry-pick Orin MMVQ work (#30858, #28285 merged; study #31268) for the decode-bound path.
5. Add local NVTX ranges around splits/expert uploads/moe-cache lookups; use bundled nsys 2024.5.4 timeline (DRAM counters read zero on GA10B).
6. fio baseline matrix on the Orin NVMe and x86 SSD (4k..8M, direct, io_uring/pread); feed into model of expert-fetch cost.
7. Isolation: systemd-run scopes with IO bandwidth limits and MemoryPeak for page-cache/mmap tests; BenchExec as design reference; app-level GPU budget guard since cgroup does not bound CUDA on Tegra.
8. Reproducible builds: BuildKit rewrite-timestamp + SOURCE_DATE_EPOCH + -ffile-prefix-map + nvcc fatbin determinism to fix "non-reproducible binaries".
9. Read Bevy, SeqMoE, ZipMoE, FlashMoE for policy ideas (prefetch vs cache, compression, CUDA-graph compatibility); verify IDs first.
10. Spike JetPack 7.2.x (CUDA 13.2, kernel 6.8) on a spare partition/card: checks UMA/VMM behaviour, BTF/eBPF, newer nsys; do not adopt before validation.

## How to keep finding sources
- Weekly: `GET search/issues?q=repo:ggml-org/llama.cpp+<moe|expert|lazy|jetson|orin|tegra|igpu>+is:pr+created:>YYYY-MM-DD`
  (sort=updated) and `compare/<pin>...master` scanned for file paths ggml-cuda/, llama-moe-cache*, llama-model-loader*.
- Watch (GitHub "custom notifications"/RSS atom feed `/releases.atom`, `/commits/master.atom`) for ik_llama.cpp, KTransformers,
  MoE-Infinity, TensorRT-Edge-LLM, jetson-containers, jetson_stats, BuildKit.
- arXiv: queries "mixture-of-experts offloading", "expert cache SSD", "unified memory Jetson LLM"; sort by date; check
  each paper's references and "cited by" (Semantic Scholar API: api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations);
  record whether code exists (search paper title in GitHub, Papers with Code replacement: HF paper pages link repos).
- Citation chasing from PR bodies (qvac-fabric came from #29887), from "Note wrt #..." cross-references, and from issue
  links; jetson-containers packages/ dir shows which stacks work on L4T; NVIDIA forums (Jetson Orin Nano board) for tool limits.
- Record for each: URL, date, license, platform, query used, verification level ([V]/[S]); re-verify [S] items before acting.
