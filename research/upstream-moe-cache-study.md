# Upstream MoE-cache study (read-only; pinned 31288c9c6 vs up1108 cf34d3113 vs PR #31469 fc4a5a77a)

Paths: P = ref/llama.cpp, U = ref/llama.cpp-up1108, R = ref/llama.cpp-pr31469. All refs file:line.

## 1. Pinned vs upstream moe-cache

Common skeleton (both trees; P file 639 lines, U 916):
- Eligibility: a layer is cached only if ALL its expert tensors (gate_up|gate,up,down) are WEIGHTS usage in a host buffer
  (`ggml_backend_buffer_is_host`) - U llama-moe-cache.cpp:312-316, 422-444 (P: same helper at ~150-160). Device type GPU or IGPU
  accepted (U:405) so Orin Nano qualifies. mmap'd CPU buffers count as host.
- Granularity: one slot = one whole expert across all its tensors (bank per tensor, `bank ne[2] = n_slots(+n_zero+1)`; U:553-558).
  Slot size = sum of `t->nb[2]` (gate+up+down or gate_up+down) -> for Q4_K_XL ~ 1.3-1.6 MiB/expert/layer.
- Budget: ONE global budget `--moe-cache-mib` (per device, split by free mem/tensor-split, U:462-483). Within a device, layers with identical
  tensor layout form a `group` that shares ONE bank set and ONE LRU across all layers (U:332-343, 503-519), i.e. a global pool across layers,
  not per-layer. `n_slots` = largest N with padded alloc <= budget (U:508). Must be >= n_expert_used else "not cached" (U:511).
  Hybrid layouts (e.g. different quant per layer) become separate groups, each getting budget proportional to host bytes.
- Slot map: per-layer I32[n_expert] in host memory (U:569). Graph does `get_rows(slot_map, ids)` -> `mul_mat_id(bank_view, cur, slots)`
  (llama-graph.cpp U:2664-2713, 1765). Scheduler copy callback `llama_moe_cache::copy` (U:761-856) fires when the slot-map split is
  copied: D2H-reads selected ids (U:799-800, SYNCHRONIZES the backend each layer), runs `plan()`, uploads misses, uploads slot map (U:853).
  Hook: llama-context.cpp:2991 (copy cb), 3044 (large-batch `copy_experts`, device->device copy of cached experts into the scratch).
- Used only for ubatch <= `max_batch`=32 tokens and when unique experts fit (U:368, 694); bigger batches bypass (read-only reuse via copy_experts).
- Upload path: `ggml_backend_tensor_set_async` -> `cudaMemcpyAsync(H2D)` straight from `w->data + e*nb[2]` (U:840; ggml-cuda.cu:2894-2901).
  Source is the model tensor in place: pageable mmap pages unless weights were loaded pinned. NO staging buffer of its own; consecutive
  expert+slot pairs are merged into one memcpy (U:835-843). Sorted miss order gives mergeable runs (U:270).

Policy:
- P: plain LRU (single doubly-linked list; `plan` P:74-109): hits touched to tail, misses sorted, victim = head, no admission control, no
  frequency. Slot-map sentinel -1.
- U (#30637 SLRU -> #31233 ClassLRU): "ClassLRU" U:20-287. Per-expert CRF = decayed use count, half-life 128 tokens (U:27), retained across
  evictions (U:46); class = floor(log2 CRF) in 0..7 (U:116); 8 LRU lists; victim = per-class LRU candidate with max age/2^class (U:152-175);
  experts selected by the current plan are protected (U:132-150). No hard admission filter other than CPU-share rule below. Slot map sentinel =
  n_slots ("empty"), slots >= n_slots point at zero slots (CPU-computed).
- #30966 CPU share (U docs/moe-cache.md; U:187-266, 450-452, 801-827, graph 2599-2603, 2664-2713): only AFTER cache is full (`n_filled==n_slots`, U:807),
  a fraction `--moe-cache-cpu-share F` of misses (lowest CRF first; fractional carry `cpu_acc` U:808-812) are NOT uploaded: the device reads
  zero-slot rows (n_zero = min(n_expert_used,64) extra zero experts, U:452) and a second `mul_mat_id(host_expert_tensor, cpu_ids)` computes
  those experts; result `ggml_add`ed (graph U:2603). `cpu_map` (host I32, -1 = skip) selects them. Runs on the ggml CPU backend via the
  scheduler (threads = `-t`/ggml threadpool; no private pool), as its own split. Overlap with GPU is only what async launch gives: GPU kernels
  of the hit part are queued, CPU split runs, then add. Disabled if `ggml_backend_offload_op` says the GPU would take such small op
  (U:701-719), not with LoRA / weight_before_ffn / Hadamard-rotated experts (graph U:2423). #31888 (cf34d3113, +3 lines graph) just forces
  `ggml_build_forward_expand(gf, cur)` before the lookup so the CPU part's input is ready (does not wait on cache upload).
- #31497 not separately studied (unknown content; not in the moe-cache files).
- Stats: `impl::stats` {hits,misses,cpu,bytes} for ubatch<=8 (decode) and >8, plus `stats_copy` (U:361-374, 846-850). Printed once at
  destructor via `LLAMA_LOG_INFO` (U:858-872): `llama_moe_cache: ubatch <= 8: hits=.., misses=.. (.. on the CPU), hit rate=..%, uploaded=.. MiB`.
  Init log U:626-634 prints size, slots per group, "tokens worth". P prints same minus CPU column (P:585-597). Nothing per-layer.
- Flags (U common/arg.cpp:2747-2766; P arg.cpp:2780): `--moe-cache-mib N` (env LLAMA_ARG_MOE_CACHE_MIB, default 0=off), `--moe-cache-cpu-share F`
  (env LLAMA_ARG_MOE_CACHE_CPU_SHARE, default 0). Env `LLAMA_MOE_CACHE_TRACE=<file>` (U:494, format U:642-678) dumps routing ids
  (binary "MCT1": per-lookup layer + ids) - this IS a ready routing-trace producer on the real model.
- Tuning script `U scripts/moe-cache-tune.py` (630 lines): input = a `llama-server` command (after `--`), options `--max-mib`, `--out` dir
  (cached runs), `--markdown`. Runs server 6x (2 sizes x shares 0,0.5,0.95; N_FIRST 2048/N_CALIB 512 tokens, fixed prompt; py:21-27),
  sets LLAMA_MOE_CACHE_TRACE, reads timings via HTTP, fits upload and CPU speeds, replays trace through a Python copy of the cache logic
  and prints predicted t/s per (size, share) + recommended args (lowest share within 5% of best). Server-only; many minutes at 20 GiB load.
  Replay model assumes bytes/upload-speed; on UMA both speeds are DRAM-bound so fit will be poor. The replay is reusable as simulator.

## 2. UMA behaviour (Orin Nano, integrated=1)

- Bank allocated with the device's own buft (`llama_memory_bufs::alloc_buf(.., d.buft, ..)` U:591) = plain cudaMalloc buffer. On Tegra that
  is a second allocation in the same DRAM. Each upload = CPU-driven pageable cudaMemcpyAsync (cuda stages through its own pinned bounce
  buffer, so for pageable mmap memory it effectively synchronizes the host thread) -> double residency of every cached expert
  (host page + bank), and the host copy is page-cache-backed, i.e. droppable only if mmap. Net: cache of X MiB costs X MiB extra DRAM
  while the full 20.8 GiB model is still referenced; with 8 GB RAM the model can never be fully resident anyway.
- No zero-copy path anywhere in the cache. `ggml-cuda.cu:352 (P)/338 (U)` forces `info.devices[id].integrated=false` ("corrupted output #15034"),
  which feeds only `supports_buft` for cuda_host (U:6381), a debug assert (U:4888-5062), and graph-compute placement; cache device-type test
  uses `prop.integrated` directly (U:5858) so it is unaffected. `cudaHostRegister` exists as `ggml_backend_cuda_register_host_buffer`
  (U:5680-5700; flags Portable|ReadOnly) but is gated by env `GGML_CUDA_REGISTER_HOST` (U:167) and NOTHING in src/ calls it
  (grep: header only) - mmap region never registered. Model-loader pinned host buffers: `-lm none/mlock/dio` use CUDA_Host buft
  (cpu buft list, llama-model.cpp:1245-1258, `--no-host` disables) -> cudaMallocHost; those are device-visible zero-copy on Tegra,
  `docs/moe-cache.md` says mmap (default) is NOT pinned. On UMA a pinned host expert can be consumed by kernels directly (mul_mat_id
  reading host pointer is not done by ggml: host-buffer weights run on CPU backend unless offloaded op, so no GPU read of host weights).
- Implication: on Jetson the cache converts "CPU computes experts from RAM at DRAM BW" into "CPU memcpy + GPU computes". The memcpy
  costs the same DRAM traffic as the CPU compute reading it once, so a cache only wins on re-use (hits), not on cold misses; cpu-share
  is likely good (the doc's own caveat: CPU share wins when upload is not faster than CPU reading host experts - true on UMA).

## 3. Pre-warm / pin from a routing profile

Not supported. No pinning, no warmup; cache starts empty (`slots[e]=empty` U:606; `key_of=-1` U:36,63). P likewise.
Minimal patch `--moe-cache-profile <file>` (text lines "layer expert [score]" ranked, top-N taken in file order):
1. common/common.h:600 add `std::string moe_cache_profile; int moe_cache_pin=0`; common/arg.cpp:2747 new opt (+env); common/common.cpp:1761 copy
   to cparams; include/llama.h:400 + llama-cparams.h + llama-context.cpp:394/4276 (default) plumb; llama-context.cpp:616 pass to ctor.
2. `llama_moe_cache::llama_moe_cache` / `impl::impl` signature (h:17, cpp:401): add profile path + n_pin.
3. After `g.lru.init(...)` (U:516) compute `n_pin = min(profile size for layers of g, n_slots - n_expert_used*max_batch margin)`.
   Pin to the high slot range: keep `g.n_slots` for bank/alloc and add `lru.n_slots = n_slots - n_pin` (use `n_lru` for init U:56-80, victim lists,
   `n_filled==n_slots` test U:807, `slot < n_slots` hit tests U:216,739,743). Slots [n_lru, n_slots) are never in any list so can never be
   victims; "empty" sentinel must move to `n_slots` (>= n_slots) and hit test become `slot < n_slots` unchanged. zero slots stay after n_slots.
4. After `buf_host` is cleared (U:599-609): for each pinned (il,e): `slot_map[il][e]=slot`, `key_of`-equivalent not needed (not in lists),
   then upload once with synchronous `ggml_backend_tensor_set(bank, w->data + e*nb2, slot*nb2, nb2)` for each tensor `ip` (like U:840).
   Also set CRF so the class logic is consistent (optional). Pinned experts then hit via normal path; `plan` `uniq.size()>n_slots` check
   (U:205) must use n_lru-aware count only for misses. `copy_experts` (U:739) works unchanged.
5. In P (plain LRU) the same: init lru over `n_slots - n_pin`, pre-set `slot_map`, touch nothing. Also needs stats line `pinned=N`.
Do on U (policy + CPU share); P gets no CPU share. Patch size ~100 lines. Pre-warm without pin is the same minus step 3 (call plan once with the profile ids).

## 4. PR #31469 (R, "Demand-paged MoE", Codex-assisted)

- Base 8345f3339 (Oct 5), +293 commits, 61 files, +9012/-86 vs base; tested envelope only: Linux x86-64, RTX 3060 12 GB sm_86, 16 GB RAM,
  `qwen4exp` (48 layers x 512 experts, IQ2_XS, custom arch) (R docs/moe-direct-io.md:5-13). Not general.
- Design: `--moe-direct-io` (R common/arg.cpp, env LLAMA_ARG_MOE_DIRECT_IO) forces ffn_*_exps to CPU buffer (mmap) and builds a global VRAM
  slot pool (`--moe-slots N`, default 4096, shared across layers; llama-context.cpp:1413-1437 scales a built-in per-layer table),
  per-layer hot seeding from a baked table `src/moe_hot_profile.h` (top-72 per layer for qwen4exp only; `--moe-profile PATH` can override;
  `data/profile-universal.bin`), GPU "doorbell" kernel (moe_doorbell.cu, 735 lines) resolves expert->slot inside the stream without D2H sync,
  user-space pager `moe_pager.cpp` (1049 lines): io_uring reads (buffered by default despite the flag name; O_DIRECT at pager.cpp:625/660),
  expert pack `experts.bin` (tools/moe_pack.py, +33 GiB for the tested model, one contiguous read per expert), pinned staging ring
  (cudaHostAlloc), `cudaMemcpyAsync` H2D (pager.cpp:870,902,974), "victim cache": anonymous host mmap registered with `cudaHostRegister`
  (pager.cpp:462-480) default 8 GiB, `MOE_CACHE_MIB`, plus a learned router predictor (`moe_predictor.cu`) / prefetch (MOE_PRED,
  MOE_HOT_K, MOE_PREKICK, MOE_RA_*), `--cache-warmup/--auto-warm` (runs a generation to warm).
- Flags: --moe-direct-io, --moe-slots, --moe-profile, --moe-pack-dir, --moe-chunk-max (8), --auto-warm, --cache-warmup N, --cache-warmup-prompt,
  --rs-seq; ~40 MOE_* env vars (llama-context.cpp). Diagnostics MOE_DIAG=1 MOE_IO_DIAG=1 (hit/IO bytes).
- UMA: assumes discrete VRAM + separate system RAM (RAM victim cache sits between NVMe and VRAM; docs envelope 12 GiB cgroup). On Orin
  everything is one DRAM, so VRAM pool + victim cache + page cache all compete (triple copy). Needs liburing, sm_86 kernels, cudaHostRegister
  of anonymous memory (works on Tegra but cudaHostRegister read-only flag etc. unverified). Only qwen4exp graph path is wired
  (src/models/qwen4exp.cpp, llama-graph.cpp:2220-2330); qwen35moe not hooked -> would need porting. Benefit on UMA: the NVMe->RAM
  pager design (expert-aligned reads from a pack, io_uring readahead, predictor) is the relevant idea for 20.8 GiB > 8 GB RAM; VRAM pool
  is not. Reuse as ideas, not code.
- Conflicts vs U: `git diff 8345f333 cf34d3113 -- src/llama-context.cpp src/llama-graph.cpp` = 1557+/552- (graph refactors, ClassLRU,
  hdmd); R rewrites llama-context.cpp by +2860 and llama-graph.cpp by +211 -> heavy, near-certain conflicts; R also edits ggml.h/ggml.c
  (+32/+77), ggml-cuda.cu(+159), ggml-cpu ops. Merge R onto U is a port, not a rebase. R is also orthogonal to moe-cache (does not use it).

## 5. Experiment matrix (Jetson; mem 8 GB, model 20.8 GiB, qwen35moe 40L x 256E)

Fixed: `-m Q4_K_XL.gguf -ngl 99 -ncmoe 40 (or -cmoe) -c 2048 -t 6 -ub 8? -b 8? -fa on -n 256 --temp 0 --seed 1`, fixed prompt (use tune script's PROMPT).
Tool: llama-bench has NO --moe-cache-mib / --moe-cache-cpu-share / -lm? (bench takes -ncmoe/-cmoe/-ot but not cache flags; verify with
`llama-bench -h`) -> use llama-cli/llama-completion/llama-server with `--perf` timings (prompt eval + eval t/s); trace via env var.
Trees: P (pinned, 3 variants), U (up1108), R not run (needs qwen4exp/NVMe pack).
 A0 baseline P/U: `-lm mmap` no cache (the 1 t/s figure; control); A1 `-lm none` / `-lm mlock` with `-cmoe` if RAM allows (it does not: 20.8 GiB
   > 8 GiB -> will thrash/OOM; use only mmap).
 A2 cache sweep (U): `--moe-cache-mib` 512, 1024, 1536, 2048, 3072 (RAM headroom: 8 GB - OS - KV - non-expert weights) with share 0.
 A3 cpu-share sweep (U): MiB best of A2 x `--moe-cache-cpu-share` 0, 0.5, 0.8, 0.95 (is CPU read faster than cudaMemcpy of pageable mmap? On UMA likely yes).
 A4 P vs U same size (LRU vs ClassLRU hit rate), run twice (cold page cache via `echo 3 > drop_caches`, then warm).
 A5 `-lm dio`/pinned host (`--no-host` on/off) once to see pinned zero-copy upload effect on UMA.
 A6 patched U with `--moe-cache-profile` pinned top-N (from LLAMA_MOE_CACHE_TRACE profile) vs unpinned, same total MiB.
 Each: 2 runs, tg 256, report median; also decode with n_batch 1 (spec off).
Counters to log: exit-time `llama_moe_cache:` lines (hit rate, misses, cpu misses, uploaded MiB per ubatch<=8), init line (slots, tokens worth),
 llama_perf (pp/tg t/s), `/proc/meminfo` MemAvailable + Cached + Mapped, `cat /proc/pressure/memory`, `vmstat 1` (bi, si/so),
 `iostat -x` NVMe read MB/s, tegrastats RAM/GR3D/EMC, majflt/minflt of the process (`/proc/PID/stat`), and the MCT1 trace for offline replay.
 Add per-layer hits and wall time inside `copy()` (U:761) in a follow-up patch (sync D2H at U:800 costs per-layer latency; log ns/layer).

## Caveats / unverified
- #31497 content not identified in these files; P line numbers approximate (structure identical to U before ClassLRU/CPU-share).
- Overlap claim for CPU share is inferred from scheduler design, not measured. llama-bench flag set not checked in this pass.
