# Strata over-memory MoE study for Jetson Orin Nano 8 GB (read-only, 2026-11-10)

Strata = campaign/ref/Strata @504b1433 (paths below relative to it). llama.cpp = campaign/ref/llama.cpp @31288c9c6.
Unverified items are marked (U). Nothing was built or run.

## 1. Strata mechanism for models larger than memory

Strata is NOT a generic "stream everything from SSD" engine. It is a 3-tier expert placement: VRAM cache -> RAM (or mmap'd
file pages) -> SSD file tier. Hot experts are on GPU; misses are computed in place by CPU threads in parallel with the GPU.
- Resident on GPU: attention/DeltaNet mixers, routers, shared experts, head, KV (from 64K only its hot part, rest streams
  from RAM), plus an expert cache filling remaining VRAM (docs/HOW_IT_WORKS.md:14-25).
- Resident in RAM: all experts, pinned; CPU computes the non-cached ones "in place" concurrently (HOW_IT_WORKS.md:27-29).
- Over-memory regime (the relevant one): DeepSeek-V4.1-Flash 2.00 bpw, 138 GB experts + 111 GB n-gram table. Experts go
  "as many as fit in graphics memory; the rest computed on the CPU, read in place from the model files"; n-gram table stays
  on SSD, a few rows per token (docs/DEEPSEEK41.md:13-17). "With less memory than that, part of the CPU's experts are read
  from the SSD while the model answers. It runs, slowly" (DEEPSEEK41.md:23-25). Measured: RTX5070 12GB + 64GB RAM + 3.6GB/s
  NVMe ~4.7 tok/s (DEEPSEEK41.md:63-65); Strix Halo 128GB unified, 109 GB experts in pool + 23 GB from SSD, 19-21 tok/s code
  (DEEPSEEK41.md:51-55). RAM-capped runs on 1 GPU: 26 tok/s unlimited -> 21/18/15/8 at 40/32/24/16 GB RAM (DEEPSEEK41.md:89-91).
  Planner: uma headroom = MemAvailable - 6 GiB (`DS4_UMA_HEADROOM_GIB`), because page cache of the model files is most of
  what cudaMemGetInfo counted (DGX_SPARK_LINUX.md:84-86).
- Cache policy: ExpertCache::admit "never evicts" (include/strata/core/expert_cache.hpp:158-160); policy is instead an
  *adaptive tier* that swaps hot experts every N verify windows (`--adapt-every/--adapt-swaps/--adapt-decay`, up to 96 swaps,
  decayed heat counts; EXCHANGE_ROTATION.md:86-92; learned state saved as a profile, expert_cache.hpp:57-65). Optional per-layer
  slot ranges (expert_cache.hpp:171-190: global arrival-order filling measured 2.97% hit vs 21% for 8 slots/layer).
- Static plan = profile.bin: STRP header + (layer,expert) ranked by routing frequency + freq table (expert_cache.hpp:40-50);
  `--expert-profile` fills the cache in rank order at start. Shipped DeepSeek profiles `data/dsv41/profile-*.bin`
  (DEEPSEEK41.md:73-77); a profile for a different expert count is rejected by size (DEEPSEEK41.md:85-86).
- Prefetch/overlap: (a) lookahead predictor thread predicts next-layer misses from votes, scored as recall/precision
  (expert_source.cpp:3132-3200); (b) predicted experts are `pread` into staging on I/O threads (io_enqueue, newest-layer first,
  queue cap 512; expert_source.cpp:1249-1297) or `madvise(WILLNEED)` when mapped (expert_source.cpp:3036-3047); (c) a gate closes the
  read-ahead if <10% of bytes were used or the cache already holds the set (expert_source.cpp:2030-2060; probe every 256
  windows); (d) prompt path streams next layer's experts to GPU over PCIe during attention (HOW_IT_WORKS.md:51-53); (e) KV
  prefetch on a second stream, opt-in `STRATA_KV_PREFETCH=1`, only -4/-5% prefill (KV_PREFETCH.md:9-15,44-47).
- I/O path (Linux): O_DIRECT descriptors + kernel AIO (`io_submit`, 256 requests in flight per thread, 4 KiB-aligned,
  merged only where adjacent, 32 MiB merge cap; expert_source.cpp:2489-2492, 2868-2880). No io_uring found (grep). Or page-cache
  mmap with `POSIX_FADV_RANDOM` (expert_source.cpp:1185). It chooses at start with a probe: 8-thread O_DIRECT 1 MiB random
  preads vs 3-thread mapped first-touch, plus a "room" test on page cache vs cgroup limit (expert_source.cpp:796-801, 913-980;
  `STRATA_FILE_TIER_PROBE=0` disables). Staging buffers are pinned with `cudaHostAlloc` (1903-1914); complement arena
  `cudaHostAlloc(Mapped|Portable)` or paced `cudaHostRegister` (3854, 3962, 4005); THP via MADV_HUGEPAGE (src/core/pinned.cu:457-510);
  layer release by `MADV_DONTNEED` + `FADV_DONTNEED` (expert_source.cpp:4117-4123); `mlock` (src/platform/memory.cpp:167).
- Discrete-GPU-specific: exchange rotation (resident RAM complement <-> VRAM swaps, avoids a host memcpy; +10-15% decode,
  EXCHANGE_ROTATION.md:1-8,93-97), batched DMA (`cudaMemcpyBatchAsync`, CUDA>=13, PCIe; no demonstrated tok/s gain,
  BATCHED_DMA.md:1-7,59-63), disjoint expert slots across 2 GPUs (DISJOINT_EXPERT_CACHE.md:1-15), KV prefetch.
- Unified-memory-specific (DGX Spark): `--mmap-experts` = one copy, the file pages ARE the GPU cache (DGX_SPARK_LINUX.md:40-42,
  70-71); UMA arena left unregistered on iGPU Radeon (include/strata/core/arch_defaults.hpp:50-55); don't mlock a table
  that competes with page cache (26 vs 39 tok/s, DGX_SPARK_LINUX.md:81-82); Spark decode is GPU-launch bound (~930 kernels/token,
  DGX_SPARK_LINUX.md:144-145) - same regime as our Orin finding.

## 2. Does Strata run on aarch64 + sm_87 + JetPack?

No, not as-is; plausibly not at all for our model sizes. Evidence:
- aarch64 is supported only on DGX Spark: "Nothing here is known to work on another ARM machine" (DGX_SPARK_LINUX.md:3-6);
  needs CUDA 13.0 and driver 580.65+, glibc 2.39 (DGX_SPARK_LINUX.md:25-26). spark/build_engine.sh:37-41 aborts unless nvcc
  is 13.x; JetPack here is CUDA 12.6. cmake/backends/cuda.cmake:23-25 default archs 86/89/120; arch >=75 accepted so sm_87 is
  not rejected by the arch check (cuda.cmake:55-61) but kernels were only tuned/measured on 86/89/120/121 (MMVQ_IL_TABLE.md:5-10).
- Build uses NEON ggml-cpu for CPU experts on aarch64 (CMakeLists.txt:72-73; src/kernels/cpu/portable.cpp:1); Q2_0/Gyro CPU
  kernels are x86-only (DGX_SPARK_LINUX.md:12-14). The CPU lane on 6 A78AE cores would be the bottleneck.
- Engine paths are model-hardcoded (Qwen3.8-Flash-Next geometry H=2560, FF=640 compile-time, expert_source.hpp:17-19; separate
  `src/qwen35` for Qwen3.6-35B-A3B; `src/dsv41`). Supported list: Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, Qwen3.6-35B-A3B
  (MODELS.md:49-63, 267-290), all 12 GB+ NVIDIA RTX 30/40/50 (MODELS.md:68-75). Orin Nano 8 GB is far below the stated minimum
  (12 GB GPU + 46 GB RAM, MODELS.md:85-86).
- Portable/reusable: (1) algorithms: per-layer/global LFU-decay vs LRU vs Belady replay (tools/bench_eviction.py:1-30; GPU-free,
  runs on any `--dump-routing` trace); (2) profile format + builders (tools/make_profile.py:1-35, make_profile_mix.py: each trace
  normalised then weighted; note hard-coded `N_LAYER, N_EXPERT = 48, 512` at make_profile.py:~37, needs `--n-expert` and edit for
  40x256 or 48x128); (3) trace collection pattern (collect_profile_traces.py:1-17, 45-prompt corpus bench/profile_corpus.json);
  (4) the tier probe/gate heuristics as design; (5) tools/quant_formats.py (HTTP-range GGUF header reader). Tied to engine:
  cache slot banks, CPU expert pool, adaptive swap, io-prefetch, all CUDA kernels. MIT license allows lifting code.
- Trace format (make_profile.py:read_trace): repeating records `int32 layer, int32 k, k x int32 expert ids, k x float32 weights`.
  We can emit the same from a llama.cpp `cb_eval` hook on `ffn_moe_topk-%d` (common.h:498 has cb_eval) and reuse the tools.

## 3. llama.cpp 31288c9c6 oversubscription features

- `-lm/--load-mode auto|none|mmap|mlock|mmap+mlock|dio` (common/arg.cpp:2695-2709). **TRAP on Orin:** `auto` = mmap unless a
  device lacks `mmap_support` (src/llama-model.cpp:1590-1598); CUDA reports `mmap_support = type != IGPU`
  (ggml/src/ggml-cuda/ggml-cuda.cu:5498) and Orin is `prop.integrated` -> IGPU (cu:5472). So `auto` silently becomes "none"
  (read whole file into RAM) = OOM for a 20 GiB model. Must pass `-lm mmap` explicitly. `dio` (O_DIRECT, llama-model-loader.cpp:561)
  only changes how the load copies bytes; it does not make weights lazily paged, so it is not an oversubscription mode.
  `-lzm` lazy mode (only tensors >4 GiB, needs mmap) is for per-layer-embeddings, not experts (arg.cpp:2711-2722).
- `-ot/--override-tensor`, `-cmoe`, `-ncmoe N` (keeps experts of the FIRST N layers on CPU; arg.cpp:2758-2779). CPU-resident
  experts under mmap are page-cache pages computed by CPU threads.
- `--moe-cache-mib N` EXISTS (arg.cpp:2781-2789, env LLAMA_ARG_MOE_CACHE_MIB; src/llama-moe-cache.{h,cpp}): an LRU of the experts
  of host-resident MoE layers in device buffers ("moe_cache_lru", cpp:15-50). Per-layer slot map; the scheduler copy callback
  uploads only missing experts with `ggml_backend_tensor_set_async` (cpp:560-580), GPU computes everything (misses evict LRU
  head, cpp:72-114; layers with identical tensor layout share a bank group, cpp:167). Requires experts in host memory (`-ncmoe/-cmoe`);
  disables pipeline parallelism (llama-context.cpp:523). Warning if no layer is fully host (cpp:263). This is the closest
  analogue to Strata's GPU tier. Cost on UMA with pageableMemoryAccess=0: a real memcpy page-cache -> cudaMalloc bank, so
  cache MiB are DRAM taken from the 5.9 GiB budget (U: whether bank memory is counted by `--fit`).
- `--fit` default ON (common.h:483), margin 1024 MiB (common.h:488); ggml-cuda's UMA path reads MemAvailable (cu:5445-5450). It
  sizes -ngl/ctx, but with `-lm mmap` plus offloaded experts it cannot see page-cache pressure; set `-fit off` and size by hand
  for experiments. `llama-bench` has -ncmoe, -lm, -fitt but no moe-cache flag (llama-bench.cpp:474-480) -> use llama-cli /
  llama-server / llama-perplexity for cache runs.
- Fair default baseline (what an ordinary user would run): `llama-server|llama-cli -m M.gguf -lm mmap -ngl 99 -ncmoe <N>
  -c 4096 -fa on -fit off`, with N chosen so dense+KV+compute fit and remaining experts sit on CPU/mmap; also baseline B0 =
  `-ngl 0 -lm mmap` (pure CPU mmap). Always `-lm mmap`; never `none` for over-memory.

## 4. Proxy model recommendation

Usable memory 5.9 GiB. Target 14.8-26.6 GiB GGUF.
- **Primary: unsloth/Qwen3.6-35B-A3B-GGUF, file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22.36 GB = 20.82 GiB = 3.5x** (HF tree API,
  sha not fetched). Arch `qwen35moe` (src/llama-arch.cpp:42 present at 31288c9c6). 40 layers (full attention every 4th, rest
  linear-attention/GDN), 256 experts, top-8 + shared expert, expert FFN 512, hidden 2048, 2 KV heads (HF config.json). Also
  Strata-supported (MODELS.md:267-290, Q4_K_XL is its listed 22.9 GB MTP variant). Pros: tiny KV (only 10 attention layers),
  expert = 2048x512x3 ~ 1.7-2 MB Q4 -> small, NVMe-friendly reads, ~3B active. Cons: GDN state; qwen35moe on sm_87 not yet
  validated here (U); experts are ~95% of bytes so dense part (~1 GiB) fits.
  Smaller siblings if memory-tight: UD-Q3_K_M 15.46 GiB (2.6x), UD-IQ4_XS 16.51 GiB (2.8x).
- Pure-llama.cpp-mature alternative: unsloth/Qwen3-30B-A3B-GGUF `Qwen3-30B-A3B-Q4_K_M.gguf` 18.56 GB = 17.28 GiB = 2.9x; arch
  `qwen3moe` (arch.cpp:37), 48 layers, 128 experts, top-8, expert FFN 768, 4 KV heads (HF config). Not in Strata's list.
- **Fallback (~1.9x): ggml-org/gpt-oss-20b-GGUF `gpt-oss-20b-MXFP4.gguf` 11.28 GiB** (HF tree), arch `gpt-oss` (arch.cpp:131), 24
  layers, 32 experts, top-4, hidden 2880 (HF config). No requant noise, native MXFP4. Nearest ~1.5x options are
  worse fits: OLMoE-1B-7B Q8_0 6.85 GiB (1.16x, bartowski; 64 experts/16 layers, U) as a smoke test only.

## 5. First experiment plan (<=12 steps)

Model: Qwen3.6-35B-A3B UD-Q4_K_XL (M); smoke/debug on gpt-oss-20b first. Use fixed prompt set (3 domains) and fixed seed.
1. Preflight: `swapon --show` (JetPack zram counts as memory; record or disable it for the run), `free -m`, record
   MemAvailable, `nvpmodel -q`, clocks fixed (`jetson_clocks`). Download M to NVMe; `fio`/`dd iflag=direct` sequential + 1 MiB
   random read MB/s (this is the ceiling: expect ~1-3 GB/s class; U).
2. Safety harness (every run): `systemd-run --scope -p MemoryMax=6200M -p MemorySwapMax=0` (or `ulimit -v` fallback) + a 1 s
   watchdog that SIGKILLs when MemAvailable < 350 MiB or PSI `memory some avg10` > 40; log `/proc/vmstat`
   (pgmajfault, pgpgin, pswpin/out), `/proc/meminfo` (Cached, Mapped, MemAvailable), `tegrastats`, `iostat -x 1`. Drop caches
   (`echo 3 > drop_caches`, needs root: confirm with user) before each cold run; also report warm run.
3. Reference for quality: (a) placement-invariance reference = same GGUF M on same box, `llama-perplexity -lm mmap -ngl 0
   --kl-divergence-base ref.kld` over ~10 chunks x 512 of wikitext + code + chat (slow, fine; logits file ~ n_vocab(248k)*4B*tokens
   -> keep tokens <= ~5k or use fewer chunks); (b) quant-vs-original reference = Q8_0 (34.4 GiB) via mmap from NVMe (cannot be
   resident; run once, hours) OR run the Q8_0 pass on a larger host and copy the .kld. Compare placements with
   `--kl-divergence` (expect KL~0, differences only CUDA vs CPU kernels).
4. B0 baseline: `-lm mmap -ngl 0` pure CPU mmap (page-cache bound). B1: `-lm mmap -ngl 99 -ncmoe 99(-cmoe)` GPU dense + all experts on
   CPU/mmap. B2: `-ngl 99 -ncmoe N` with N tuned to the minimum that avoids OOM (partial expert-GPU placement; GPU-layer experts
   count against DRAM). Record prefill (pp512, pp2048) and decode (tg128, tg512) t/s, cold + warm, major faults/token, NVMe MB/s.
5. C1: B1 + `--moe-cache-mib` sweep {256, 512, 1024, 1536} with `-fit off`; check DRAM high-water (cache + page cache + dense +
   KV). Expect page cache to shrink as banks grow; find the knee. Record cache hit rate if the stats log prints one (stats
   struct cpp:196-202).
6. Capture routing trace: patch-free `cb_eval` tool (new tool in our tree, not here) writing Strata-format records
   (`layer, k, ids, weights`) over decode of the 3 prompt domains; also record per-token sequences for replay.
7. Offline (no GPU) policy replay: `tools/bench_eviction.py` on the trace (adapt `--n-expert 256`, layers 40) at slot counts
   equal to the MiB sweep / expert bytes; compare fill-only, LRU, LFU-decay, Belady oracle, global vs `--per-layer`, and
   `--prefill` profile start. Gives the hit-rate headroom of `--moe-cache-mib` (LRU) vs a profile+LFU policy before writing code.
8. Profile pinning: `make_profile_mix.py` on traces of domains A/B -> ranked pairs; test on held-out domain C (transfer, as
   Strata did: 60% shipped vs 71% mixed, make_profile_mix.py:22-27). Implement as `-ot` regex lists pinning top-ranked layers'
   experts (granularity is whole layer tensor) or, better, a thin llama-moe-cache patch that pre-fills slots from profile.bin (C2).
9. C2/C3 (only if step 7 shows >10 points hit-rate gain): prefill-from-profile and LFU-decay in llama-moe-cache; re-run step 5 matrix.
10. I/O variants: `-lm mmap` default readahead vs `madvise(RANDOM)`/`posix_fadvise` set via an LD_PRELOAD shim or env patch;
    `MADV_WILLNEED` of next layer's selected experts (Strata mechanism). Measure major-fault stalls (perf trace -F / bpftrace
    `vmscan`), readahead_kb of the NVMe (`blockdev --getra`; test 0/128/2048 KiB).
11. Quality gate: KL/top-1 agreement vs step 3 ref for B0,B1,C1,C2 (should be equal within noise); report Δppl. Any lossy knob (expert
    top-k reduction as in Strata DeepSeek "Faster mode" 5 of 6, DEEPSEEK41.md:90) is a separate arm with its own KL.
12. Report per arm: decode t/s median of 3 (cold/warm), prefill t/s, majflt/s, NVMe MB/s average and p99 latency, DRAM
    high-water (MemAvailable min), cache hit rate, KL, any watchdog trips. Decision: if C1 decode >= 1.5x B1 at equal DRAM, adopt
    moe-cache; else consider profile-driven pinning. Abort any arm that trips the watchdog twice.

Caveats: GPU is Orin 1024-core, ~68 GB/s LPDDR5 shared with CPU; page-cache copies and bank memcpy compete for the same DRAM
bandwidth. Strata's Spark decode (~45 tok/s) shows the unified regime is launch-bound once experts are resident; our over-memory
regime will instead be NVMe-bound (active expert bytes/token ~ 8 experts x 40 layers x ~1.7 MB = ~0.5 GB/token worst case ->
at 2 GB/s ~4 tok/s upper bound with 0% cache hit; U, verify expert size from GGUF).
