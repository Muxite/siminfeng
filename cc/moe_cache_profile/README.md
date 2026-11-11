# Profile-pinned upstream MoE cache (component CC)

Standalone patch for `ref/llama.cpp-up1108` at `cf34d311332e362198f0a410c466d1e940287d02`. The reference tree is read-only. Apply the patch to a separate checkout of that revision. No build or GPU run was performed for this component.

## Interface

```sh
llama-cli -m model.gguf -ngl 99 -cmoe --moe-cache-mib 1024 \
  --moe-cache-profile profile.json --moe-cache-pin-frac 0.75
# Alternatively: LLAMA_MOE_CACHE_PROFILE=profile.json
# Optional exit report: LLAMA_MOE_CACHE_STATS_JSON=cache-stats.json
```

`--moe-cache-profile FILE` requires a positive `--moe-cache-mib M`. `--moe-cache-pin-frac F` is finite, in [0,1], and defaults to 1.0. The path is copied into context-owned storage. The public context parameter structure gains fields; rebuild callers against the patched headers.

Accepted profiles, in descending rank order:

* The actual `cc/expert_replay/expert_replay.py profile` output: `{"layers":2,"experts":4,"ranked":[[0,2],[1,1]],"scores":[10,8],...}`.
* A ranked object list: `[{"layer":0,"expert":2,"count":10},{"layer":1,"expert":1,"count":8}]`, or that list under `ranked` in an object.
* Plain text with exactly `layer expert` per nonempty line. Blank lines and full-line `#` comments are allowed.

Array order defines rank; scores/counts are not re-sorted. All entries, including entries beyond the pin budget, are validated before cache allocation. Invalid integer IDs, out-of-range IDs, duplicates, empty profiles, malformed records and unreadable files fail context creation. Optional top-level `layers`/`experts` or `n_layer`/`n_expert` dimensions must match the model. Without dimensions, model identity cannot be established beyond ID bounds. There is no Python format conversion helper.

## Allocation and applicability

The static byte ceiling is `floor(M * F)` MiB. It is divided across upstream device/layout groups using the same device and host-byte proportions as the total cache budget. Within each group, the highest-ranked eligible experts are selected until the group's byte ceiling or total slots is exhausted. Unused quota is not redistributed between groups. One slot contains every expert tensor for one `(layer, expert)` pair; alignment, zero slots and the upstream guard slot remain within the total cache allocation. Consequently pin bytes can be below the ceiling.

Pinned experts occupy the high slot range and never enter an eviction list. The remaining slots use upstream ClassLRU, including retained CRF and CPU-share rounding. With a complete ranking and F=1, all available data slots are static. A shorter profile leaves unused data slots managed. F=0 pins nothing. Eligible profile entries on layers already resident on a device or otherwise not cacheable are skipped; they still undergo dimension and range validation.

Needs a GPU/IGPU backend and host-resident expert tensors assigned to that backend, as upstream does. Works with `-cmoe`, `-ncmoe`, and `--moe-cache-cpu-share`. Pinning does not move device-resident experts back to host memory. Tensor parallelism remains unsupported. Initialization fills pinned slots once with synchronous tensor uploads and logs slot count, bytes and elapsed milliseconds. Allocation-only contexts skip uploads.

Profile mode reserves upstream CPU zero slots even at CPU share zero. Single-token decode sends enough misses through the existing CPU-part graph to avoid exhausting managed slots; CPU share remains a requested fraction, with capacity overflow imposing a minimum CPU fraction. Graphs that cannot use the CPU part (LoRA, weights-before-FFN, rotated experts) fall back to the host path whenever the worst-case unpinned selection exceeds managed capacity. Multi-token batches do the same, avoiding zero-slot coloring exhaustion. Existing large-batch `copy_experts` can still reuse pinned residents. Upstream batch-size and total-slot eligibility limits still apply. The CPU scheduling/offload behavior and performance require backend validation.

On Jetson UMA, the bank is a separate device allocation in shared DRAM. With patch 01 alone, cached experts have double residency with their mmap/page-cache source pages; patch 02 below optionally advises source pages away. Neither patch implements zero-copy. Startup uploads can fault in source pages and increase memory pressure.

## Statistics and compatibility

With a profile, exit logs retain the upstream aggregate counters and add pinned hits versus managed hits, pinned slot count/bytes, and per-layer hit percentages in groups of eight layers (`n/a` for no lookups). Hits and misses count distinct selected experts per layer lookup, following upstream; repeated token references within a lookup are deduplicated. CPU misses are included in misses. Startup bytes are separate from demand-upload bytes.

`LLAMA_MOE_CACHE_STATS_JSON=FILE` writes aggregate small/large ubatch counters, the indexed per-layer array, pinned bytes/slots, and the existing large-batch copy counters. JSON rates are fractions or null for no lookups. Large-batch reuse counts tensor-copy operations as upstream does and is separate from lookup hit rates. Output is written at cache destruction, overwrites the file, and reports write failures in the log. Use distinct files for multiple contexts. Without a profile and without this explicit statistics environment variable, log text, slot capacity, ClassLRU decisions, CPU-share behavior and model computation follow the original path; byte-identical generated output still needs runtime comparison.

## Validation

Performed without builds/GPU: `apply --check` passed at the pinned revision, the reference worktree remained clean, and all 10 existing expert-replay tests passed.

```sh
git -C ref/llama.cpp-up1108 apply --check ../../cc/moe_cache_profile/patches/01-moe-cache-profile.patch
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s cc/expert_replay -v
```

The Python suite checks the producer and replay conventions, not the new C++ parser. The patch uses the already-vendored nlohmann JSON header directly, with no new dependency or common-library linkage. Apply checks establish patch compatibility, not compilation or runtime correctness.

Tests to run later in a separate patched checkout with a suitable model/backend:

1. Compare unpatched and patched runs without the new flags, with fixed prompt, seed, temperature, batch sizes and CPU shares; compare generated bytes, cache aggregate counters, uploads and trace IDs. Repeat for multi-device/layout groups and allocation-only contexts.
2. Load replay JSON, ranked object JSON, and equivalent text; compare pinned bytes/slots and first-token pinned hits. Check F=0, 0.75 and 1, short and complete rankings, whole-MiB rounding and budgets barely large enough for one token.
3. Reject negative, fractional, oversized and out-of-range IDs, duplicates, malformed/empty files, each conflicting dimension alias, missing profile files, profile without a cache budget, and invalid pin fractions (NaN, infinity, trailing text).
4. Route only pinned experts over repeated decode calls; require zero demand uploads. Alternate unpinned experts until managed slots churn; verify pinned slots survive and `hits = pinned_hits + managed_hits`. For a full static cache, cold misses must run through the CPU part without eviction or abort.
5. Exercise CPU shares 0, 0.5 and 1, multi-token fallback, batches above 32, repeated IDs across tokens, LoRA and rotated/weights-before-FFN graphs. Compare results with the host baseline within backend numerical tolerances.
6. Parse exit JSON; compare every logged counter and per-layer rate, verify null rates for unused layers, test unwritable output, and confirm large-batch copies remain separate. Measure startup upload time and Jetson memory pressure before performance claims.


## UMA source-page deduplication (patch 02)

Apply `patches/02-uma-dedupe.patch` **after** patch 01. Opt in with `--moe-cache-uma-dedupe` or `LLAMA_MOE_CACHE_UMA_DEDUPE=1`; the default is disabled. No profile is required. The experimental context parameter `moe_cache_uma_dedupe` is 0 (off), 1 (integrated only), or 2 (force); rebuild API callers after applying this patch. With no cache budget there is no cache and no advice.

Each cache group is enabled only on a backend device of type IGPU. Upstream CUDA derives that type from `cudaDeviceProp.integrated`, including on Jetson; its separate internal `integrated` optimization flag is not used here. `--moe-cache-uma-dedupe=force` bypasses the device check, but still requires Linux and actual mmap source storage. On discrete GPUs, the host copy is the cache's backing tier; releasing it can turn later misses into storage reads. Do not use force as a general performance setting.

After all gate/up/down slices (or the merged gate/up equivalent) of a pinned or managed expert have been uploaded, the patch calls Linux `madvise(MADV_PAGEOUT)` on each slice's full interior pages. Startup tensor uploads are synchronous. Managed fills add one backend synchronization after all slices have been queued, before any advice; this prevents releasing pages while a pending `cudaMemcpyAsync` reads them, at the cost of upload overlap. Slot eviction performs no advice or prefetch. Future source accesses fault pages back through the existing mapping.

Provenance is checked against model-owned file mappings and host tensor buffers; heap buffers, device weights, `--no-mmap`, and allocation-only contexts receive no advice. The model loader uses read-only `MAP_SHARED` file mappings on Linux, so fallback `MADV_DONTNEED` does not discard model modifications. The same fallback would be safe for a read-only private file mapping; writable private/COW mappings must never be added to this path. PAGEOUT needs Linux 5.4 or newer. If the header lacks PAGEOUT, or the syscall returns EINVAL, ENOSYS, or EOPNOTSUPP, the patch tries DONTNEED. Other failures remain visible and do not abort inference. Non-Linux builds disable deduplication with a warning.

Ranges are aligned inward using the system base-page size: the first and last partial pages stay resident, and slices smaller than a page may release nothing. No adjacent expert bytes are advised. Huge pages can prevent reclaim or cause splitting; explicit hugetlb alignment requirements can produce EINVAL even for the fallback. Locked or DMA-pinned pages may fail advice or remain resident. Successful PAGEOUT is best effort. DONTNEED on a shared file mapping can drop this process's PTEs while leaving the file page cache resident, so it does not guarantee physical deduplication. Shared file pages may also be used by other mappings/processes. CPU-side miss computation or other contexts reading an advised expert can immediately re-fault it and increase NVMe I/O; the source mapping remains valid and the file must remain unchanged throughout inference.

Exit logs report dedupe syscall calls, bytes successfully advised, total failed syscalls and counts per numeric errno (with its message). `LLAMA_MOE_CACHE_STATS_JSON=FILE` adds `dedupe: {calls, bytes_advised, failures, errno}`; errno keys are decimal strings. A failed PAGEOUT plus successful fallback counts as two calls and one failure. Byte counts sum successful advised ranges over all fills, including repeated experts; they measure neither unique bytes nor confirmed reclaimed DRAM. JSON counters are present and zero when disabled. No GPU performance or output-equivalence claim is made.

### Jetson A/B to run later

Use the same patched executable, model, fixed prompt, seed, temperature, context and CPU-share settings for all three arms, without a profile initially:

| Arm | Additional arguments |
| --- | --- |
| base | `-cmoe` |
| cache2048 | `-cmoe --moe-cache-mib 2048` |
| cache2048+dedupe | `-cmoe --moe-cache-mib 2048 --moe-cache-uma-dedupe` |

For example, use `llama-completion -m MODEL -f PROMPT -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap` plus each arm's arguments. Set a distinct `LLAMA_MOE_CACHE_STATS_JSON` path per run. Repeat with profile/pin-fraction settings held identical between the cache arms to exercise startup pins as well as managed fills. Include CPU-share 0 and 0.5, enough distinct selections to cause eviction, and verify generated output against base within backend numerical tolerances.

During each run, sample the inference PID with the resource observer's I/O backend:

```sh
python3 cc/resource_observe/resource_observe.py sample --backends io \
  --device nvme0n1 --pid INFERENCE_PID --interval-ms 500 --out ARM.io.jsonl
python3 cc/resource_observe/resource_observe.py summarize ARM.io.jsonl
```

Start observation before measured inference and stop it after exit. Record decode tokens/s, prompt time, startup pin time, cache hit/upload counters, dedupe counters, available DRAM/file cache, swap, the system `workingset_refault_file` delta, and NVMe read MiB (`device_read_delta`, summed over valid intervals). Keep system and device scopes separate; disk reads include other workloads. Compare equal generated-token counts and isolate decode from startup faults. Alternate/randomize arm order, repeat warm runs, and control initial page-cache state consistently for cold runs. Do not infer reclaim from advice bytes alone: require improved memory pressure and throughput without excessive refaults or NVMe reads.

Validation for patch 02: concatenating 01 and 02 and running `git -C ref/llama.cpp-up1108 apply --check` passes at the pinned revision. Only touched source files were staged under `/tmp` to produce the incremental diff; the reference source was not copied or modified. No builds or GPU runs were performed. Inward-range arithmetic was checked against independent ceil/floor boundaries for 4 KiB and 64 KiB pages, including sub-page and unaligned slices.

## Patch 03: direct file fills (Linux CUDA, opt-in)

Apply `03-direct-fill.patch` after 01 and 02. Add `--moe-cache-direct-fill`, or set `LLAMA_MOE_CACHE_DIRECT_FILL=1`, with a positive `--moe-cache-mib`. It covers both startup profile pins and demand miss uploads. The default is off; C API callers can set `llama_context_params.moe_cache_direct_fill` (default false). Rebuild callers against the changed public context structure.

The loader's `weights_map` already supplies `llama_tensor_weight.offs = gguf_get_data_offset(shard) + gguf_get_tensor_offset(shard, tensor)` and `idx`, the source shard index. After loading, the model retains one CLOEXEC duplicate of each relevant source descriptor and a tensor-to-(shard, absolute offset) map. It records only non-view, contiguous mapped tensors whose type, dimensions and strides match the on-disk tensor. Cache contexts reopen `/proc/self/fd/FD` with `O_RDONLY | O_DIRECT | O_CLOEXEC`, keeping the same file identity across rename/chdir and keeping independent file flags. Split GGUF works by using each tensor's own shard descriptor and offset, never the main shard's data offset. Descriptor retention is done at model load even if direct fill is later disabled; model destruction closes those duplicates. Non-Linux builds and backends without the optional CUDA allocation procedures, async transfers or events log a warning and retain the mmap upload path.

For ggml layout `[n_embd, n_ff, n_expert, 1]`, each expert occupies contiguous dimension-0 rows over dimension 1. The slice length is `tensor->nb[2]` (including quantization block sizes), and its file offset is `tensor_file_offset + expert_index * tensor->nb[2]`. Both the tensor offset and slice length must be multiples of 4096. A tensor with different layout, a view, missing/changed source metadata, or unaligned offsets/slices is individually disabled with a clear log and uses the existing mmap path. Sources outside the signed file-offset range are also disabled. No padding or neighboring-expert reads are used to make an incompatible tensor eligible.

Each active CUDA device gets a two-buffer staging ring, each buffer sized for the maximum expert tensor slice on that device plus 4095 bytes of alignment slack. Optional backend procedures call `cudaHostAlloc(..., cudaHostAllocPortable)` and `cudaFreeHost`; llama itself gains no CUDA headers or link dependency. The usable address is rounded up to 4 KiB within each allocation. Pinning failure (including `GGML_CUDA_NO_PINNED`) disables direct fill on that device rather than substituting pageable memory. The ring adds pinned host DRAM outside `--moe-cache-mib`, reported in cache host memory accounting. Allocation-only contexts skip ring allocation and all reads.

`pread` fills one pinned buffer without dereferencing the expert's mmap pages; the existing CUDA `tensor_set_async` submits `cudaMemcpyAsync` into its bank slot on the backend stream. Reading the next slice into the other buffer can overlap the previous transfer. An event after each upload protects the buffer before reuse. Startup fills synchronize before returning; demand fills retain stream ordering with the subsequent slot-map upload and the existing backend synchronization before the next lookup. Ring destruction drains uploads before freeing staging memory and bank storage. There is no disk worker thread: double buffering supplies the requested read/transfer overlap, while disk reads still occupy the calling thread.

If `O_DIRECT` open is unsupported, the context logs and uses buffered `pread` followed by `posix_fadvise(POSIX_FADV_DONTNEED)` on each read slice. A direct-read EINVAL/EOPNOTSUPP/ENOSYS or short positive read switches that shard to buffered mode, logs, and re-reads the full slice. EINTR retries; buffered short reads continue. EOF, other read errors and inability to reopen a failed direct descriptor raise an error rather than upload incomplete data. Advice failure is logged once per shard. Buffered advice is best effort: it can leave page-cache residency, especially for partial system pages on a 64 KiB-page host. The model files must remain immutable during inference. Linux `/proc` must be mounted and permit reopening descriptors; otherwise the affected tensors log and fall back to mmap.

Patch 02's mmap dedupe becomes unnecessary for file-filled slices and skips them, including buffered file fills. In mixed layers it still advises the mmap-filled tensors after their uploads complete. Fully file-filled demand uploads do not add patch 02's advice synchronization. This does not evict pages already prefetched or faulted by model loading, CPU expert computation, another context, or the large-batch host path; those paths are unchanged. Direct fill avoids adding mmap/page-cache source residency specifically during cache-slot fills. It is expected to reduce UMA double residency and small mmap faults on misses, but bypassing a warm page cache can increase NVMe traffic and slow warm runs. It still copies into the cache bank; performance improvement is unmeasured.

Exit logs and `LLAMA_MOE_CACHE_STATS_JSON` report `direct_fill: {reads, bytes, read_us, mean_read_us, fallbacks}`. Reads count completed file-filled tensor slices (both direct and buffered), bytes sum their payloads, and mean latency is total read microseconds divided by those reads; it includes retry/reopen time but excludes waiting for a staging event, advice, and CUDA copies. JSON mean is null for no reads. Fallbacks count uploaded tensor slices using buffered file I/O or the mmap upload path while the option is enabled, once per slice even when mmap copies are coalesced. Startup pins and demand fills are included; CPU-computed misses and large-batch cache reuse are not. Logs distinguish buffered shard fallbacks from mmap tensor/device fallbacks. Disabled counters are zero.

### Direct-fill A/B to run on Jetson

Use the fixed model/prompt/seed/context/output length and observation procedure from the patch 02 A/B above. Compare these arms using the same patched executable:

| Arm | Additional arguments |
| --- | --- |
| no cache | `-cmoe` |
| mmap cache | `-cmoe --moe-cache-mib 2048` |
| mmap cache + dedupe | `-cmoe --moe-cache-mib 2048 --moe-cache-uma-dedupe` |
| direct cache | `-cmoe --moe-cache-mib 2048 --moe-cache-direct-fill` |
| direct cache + dedupe | `-cmoe --moe-cache-mib 2048 --moe-cache-direct-fill --moe-cache-uma-dedupe` |

Run without a profile first, then with identical profile and pin-fraction settings across cache arms to cover startup pins. Include CPU shares 0 and 0.5 and enough expert churn to require demand fills. Use distinct statistics files. Verify generated outputs against the baseline within backend tolerances and unchanged hit/miss/pin decisions between cache arms. Record decode tokens/s separately from startup/prefill, startup fill time, direct reads/bytes/mean/fallbacks, demand-upload bytes, dedupe calls, available DRAM/file cache, swap, file refaults, and NVMe read MiB. Repeat controlled cold and warm runs with alternating arm order. Fully eligible direct arms should have zero fallback slices and zero dedupe calls, even with dedupe enabled; mixed tensor fallbacks explain nonzero dedupe. A split GGUF fixture must match single-file output and expected shard-offset payloads. Also exercise unsupported direct I/O, read-time rejection/short reads, unaligned tensors, disabled pinning, non-CUDA and allocation-only contexts. Measure whether lower residency/refault pressure outweighs direct NVMe reads; no throughput claim follows from apply checks alone.

Validation for patch 03: no builds or GPU runs. The patch is incremental over 01+02; all three concatenated pass `git -C ref/llama.cpp-up1108 apply --check` at `cf34d311332e362198f0a410c466d1e940287d02`. `git diff --check` passes for the staged-source incremental diff. A host-only synthetic split-file I/O check verifies the absolute-offset formula, descriptor reopening after rename, aligned direct reads (when supported), buffered reads plus DONTNEED, and inward 4 KiB staging alignment arithmetic. These checks do not validate compilation, CUDA overlap or output correctness. Only touched source files were staged under `/tmp`; the reference worktree remains unchanged.
