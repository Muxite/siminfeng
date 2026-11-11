# J-OM8: synchronous exact expert fill

Linux, C++17, opt-in `LLAMA_EXPERT_FILL=1`. The CPU scheduler's before-node
observer waits for host router IDs, deduplicates the layer's ubatch union,
and fills gate/up/down (or merged gate-up) slices before expert MUL_MAT_ID.
The loader's `weights_map` supplies source file index and absolute GGUF byte
offset; the model retains buffered file handles and validates each tensor's
mmap address against that offset. Split GGUF files get independent fillers.
Unavailable offsets, non-mmap tensors or unsupported layouts disable filling
with one `expert_fill: inactive (<reason>)` line per context.

`expert_fill.h` is the source of truth: the patch copies its exact contents to
`src/llama-expert-fill.h`. Each filler opens its own O_RDONLY (not O_DIRECT)
fd via the retained model fd's `/proc/self/fd/N` path. Its persistent pool uses
`LLAMA_EXPERT_FILL_THREADS` (default 8) and per-worker scratch buffers sized by
`LLAMA_EXPERT_FILL_CHUNK_KIB` (default 1024). Invalid settings use defaults;
limits are 256 threads and 65536 KiB. `fill` aligns outward, merges adjacent
or overlapping ranges, queries mincore, and reads only nonresident runs in
parallel with buffered pread. It waits for every job, including on IO errors.
Counters include calls, input ranges, requested page-aligned union bytes,
resident bytes skipped, actual bytes read, pread syscalls (including retries),
total fill wall ns and maximum single-fill ns. Final pages stop at file EOF.
Concurrent fills on one filler serialize; mmap lifetime belongs to its caller.

Fill takes precedence over both OM7 prefetch switches, including on failure.
It never calls OM7's per-expert WILLNEED. `LLAMA_EXPERT_FILL_RANDOM=1` separately
applies RANDOM to expert mmap ranges and excludes their initial prefetch;
default-off retains normal loader advice. Buffered fd readahead can still add
bytes even with mmap RANDOM. Other explicit mmap/NUMA/lazy policies still apply.

Expected effect (hypothesis, not performance evidence): CLOSEOUT 3.C uses
615 MB selected expert bytes/token, approximately 586.5 MiB. At 3.D's LRU
hit rates 0.536/0.662/0.694/0.605, misses are about 272/198/179/232 MiB/token.
At the 536-573 MiB/s drive ceiling, IO-only bounds are respectively
1.97-2.11 / 2.70-2.89 / 2.99-3.19 / 2.31-2.47 tokens/s; mean hit 0.624 gives
221 MiB/token and 2.43-2.60 tokens/s. Replay slots are not a measured page-cache
hit rate; these bounds omit CPU, synchronization, metadata and readahead costs.

Risks: page-cache thrash can evict experts needed later in the same token,
including pages just filled. mincore is a racing residency snapshot, not a
pin; it avoids copying already resident bytes into scratch. Missing bytes
still incur a page-cache-to-scratch copy. Pool memory, graph boundaries,
fragmented missing runs, readahead and synchronous waits can erase any gain.

Run `make -C cc/expert_fill test` (no root/GPU). It evicts a 64 MiB file with
FADV_DONTNEED, checks the resident union, measures extra readahead pages,
checks counters and a zero-read repeat, and verifies mmap data and bounds.

A/B spec: `cc/abrun/specs/J-OM8.json`, image
`campaign/llamacpp:jetson-up1108-om8`; A=base, B=fill, C=fill_random (fill+random); ABCCBA.
Use llama-completion p1 with `-n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99
-lm mmap -cmoe`. Require switch regex `expert_fill: active` only in fill arms.
Collect eval_tps, nvme_gib, avg_kib_per_io; verify identical output hashes,
nonzero fill stats, matched cold-cache procedure and no competing NVMe work.
Startup line records threads/chunk/random; exit `expert_fill: stats` has all
counters. Compare disk telemetry, not fill counters, to assess physical IO.
04: `RANDOM=decode` (full name `LLAMA_EXPERT_FILL_RANDOM`) advises once at the first n_tokens==1 ubatch.
`LLAMA_EXPERT_FILL_MIN_TOKENS` / `LLAMA_EXPERT_FILL_MAX_TOKENS` bound fill inclusively (0 = unbounded; invalid = 0).
Stats add prompt_/decode_ calls, bytes_read, fill_wall_ns; decode means n_tokens==1. Header matches combined-01-02-03-04.
`J-OM8b.json`: explicit eight-run mirror x2; both switch regexes use OR: base matches none, fill arms need >=1 (decode marker is not independently enforced).

05 / J-OM9 diagnostics and populate:
`LLAMA_EXPERT_FILL_DIAG=1` enables decode-only (n_tokens==1 ubatch) process
RUSAGE_SELF deltas across synchronous fill, and between fills; RUSAGE_THREAD
is summed over the callback and active fill workers. `thread_majflt/minflt`
are fill-thread totals, not the process total (do not add them together).
Fill timing includes the immediate post-fill mincore probe; between timing
includes CPU compute, other process activity, pre-use/next-layer probes and
sampling. Token boundaries are the next process_ubatch entry, with the final
interval closed after destructor synchronize; the last interval can therefore
include caller time between the last decode and context destruction. No
prompt interval is included. With split files the process snapshot spans the
whole layer fill, avoiding overlapping process-wide measurements per file.

Post-fill, pre-use and next-layer counters count nonresident pages in aligned
unions per mapping, with check counts to interpret repeated observations.
The CPU observer synchronizes the prefix before its non-ask callback
(ggml-backend.cpp compute lambda), so 05 can check immediately before EACH
expert MUL_MAT_ID, including gate/up/down nodes after the layer's initial
fill. `pre_use_nonresident` counts all filled ranges of that layer at each
such hook. `next_nonresident` is a separate probe before the next layer fill;
it measures eviction after consumption, which need not hurt performance.
mincore reports page-cache residency, not installed process page tables;
resident pages can still require minor PTE faults. These probes cannot
attribute faults to a particular tensor or distinguish GPU-driver/KV faults.
Compare process fill/between major faults and pre-use residency to narrow the
origin; this change supplies diagnostics, not new Jetson performance evidence.

Optional `LLAMA_EXPERT_FILL_DIAG_CSV=/path` writes one row per decode ubatch
(1-based token_idx). Every 16th token also prints a compact per-token line to
stderr; the exit `expert_fill: diag` line includes all accumulated counters.
J-OM9 uses stderr since om_run only mounts /om read-only. The runner has no
leak_regex support: frd16p requires all three positive per-arm signatures,
but `populate unsupported` must be checked in stderr manually and invalidates
that arm's populate comparison. No negative lookahead on individual lines
can enforce absence elsewhere in stderr.

`LLAMA_EXPERT_FILL_POPULATE=1` separately opts into MADV_POPULATE_READ (22),
after all preads, over the complete aligned union including warm ranges.
Jobs are page-aligned and distributed across the same fill pool. EINVAL logs
`expert_fill: populate unsupported` once per process and disables further
populate attempts across fillers; already in-flight jobs may complete.
Other errors fail filling explicitly. Stats report calls, summed worker ns
(not wall ns), and failed calls. The regression test requires Linux >=5.14.

Default-off syscall guards (A6):
`const bool diag = expert_diag_enabled && expert_diag_token_active;`
guards process snapshots and post-fill/next-layer probes;
`if (!expert_diag_enabled || !expert_diag_token_active) { return; }`
guards pre-use probes and token finalization;
`batch_diag_ = diag_ && decode;` and
`if (batch_diag_) { getrusage(RUSAGE_THREAD, &before); }` (and after)
guard worker snapshots. With DIAG unset and POPULATE unset, 05 adds no
getrusage/mincore/madvise/file IO syscalls. POPULATE is an independent opt-in
and intentionally adds madvise even when DIAG is unset.

## 06 / J-OM10: pre-gated asynchronous next-layer fill

Opt in with `LLAMA_EXPERT_FILL=1 LLAMA_EXPERT_PREGATE=1`; K defaults to 8,
with `LLAMA_EXPERT_PREGATE_K=8`, `12`, or `16` supported (other values use 8).
Only single-token ubatches inside the configured fill token bounds predict.
The first gate/up expert MUL_MAT_ID hook of layer L predicts L+1 and enqueues
its gate/up/down (or merged gate-up/down) slices before L's existing synchronous
fill. The final executed layer does not predict an MTP block; layer 0 has no
incoming prediction. Per-ubatch prediction sets reset at process_ubatch entry.

Verified model: `~/models/campaign/qwen36-35b-a3b/model.gguf`, metadata
`general.architecture=qwen35moe`, embedding length 2048, 256 routed experts,
8 selected, RMS epsilon approximately 1e-6. `blk.1.ffn_gate_inp.weight` is
F32, shape [2048,256], 2,097,152 bytes; all 40 router tensors have this
same type/shape. The model has 40 blocks and no router/selection bias tensors. The arch's
`src/models/qwen35moe.cpp::graph::build_layer_ffn` calls `build_moe_ffn` with
SOFTMAX, `norm_w=true`, no `exp_probs_b`, and the overload supplies null
`gate_inp_b`. There is no grouped top-k (the runtime also rejects groups >1).
The shared expert's separate sigmoid gate does not select routed expert IDs.

Router math, including norm/bias handling (A6 i): the true router input is
`x_L = RMSNorm(attention_L + residual_L, attn_post_norm_L, eps)`.
True scores are `z_L[e] = sum_j W_L[e,j] * x_L[j]`, followed by softmax across
256 experts and top-8; selected weights are divided by their sum AFTER
selection. Neither a logit bias nor an expert selection bias is applied.
Prediction uses `z_pred[e] = sum_j W_(L+1)[e,j] * x_L[j]` and top-K logits;
softmax preserves their ranking. It deliberately omits the next residual,
attention and next layer's RMSNorm/learned norm weight; it does not apply a
second normalization to the already normalized x_L. This approximation,
not reuse of L's selected IDs, is what recall measures. The CPU sum uses
float accumulation and breaks equal scores by smaller expert ID. Other
architectures, active LoRAs, and folded Hadamard rotations disable prediction
explicitly; rotated expert inputs would not share the router input basis.
This GGUF has no `prism.*` metadata, so no folded rotations are requested.

Host input and router access (A6 ii): `expert_pregate_hook` reads `t->src[1]`
of the FIRST gate/up/merged-gate-up MUL_MAT_ID, a reshape/host copy of that
layer's `attn_post_norm`. It requires F32, contiguous [2048,1,1,1], a nonnull
host buffer (`ggml_backend_buffer_is_host` on its view base), and valid data;
it never mistakes the down projection's intermediate for x_L. The CPU
scheduler copies split inputs and synchronizes the preceding CPU graph
prefix before invoking the non-ask callback (02/05). Thus this read is from
host memory, including CUDA host buffers, and needs no new GPU readback.
The assumption that router tensor data is an mmap pointer is FALSE when
that tensor is GPU-offloaded: 03's strict `expert_file_info` rejects it,
and the loader can unmap offloaded spans. 06 adds a metadata-only mode to
that API, keeping its strict default for experts. It uses the retained
source fd and weights_map offset to create an independent, temporary file
mmap, copies/dequantizes router rows to a per-context float cache, and
unmaps the temporary view. It never dereferences GPU router data. F32 uses
memcpy; F16/BF16/quantized rows use GGML's type-traits CPU `to_float` decoder.
Unsupported decoders/layouts fail explicitly. Router cache initialization
runs only when pregating is enabled, once before graph execution.

Nonblocking submission (A6 iii): `enqueue_async` does no mincore, pread,
condition-variable completion wait, or call to synchronous `fill`. It builds
jobs outside the queue mutex, then splices a whole submission under that
short mutex. The persistent fill workers always choose ready synchronous
jobs before the separate FIFO async queue; already running preads are not
preempted. Queue depth is bounded at 4096 chunks per file; full submissions
are dropped and counted. Worker-side mincore skips resident pages before
buffered preads. Async jobs use the same pool and per-worker scratch, and
have separate counters; the synchronous batch's completion count never
includes speculative work. The prediction hook never calls `wait_async`.
The pre-existing synchronous fill still waits for its own jobs and can be
delayed by an async chunk occupying a worker. Only teardown and tests drain
async jobs. A sync/async residency race can duplicate preads; both read the
same immutable file into separate worker scratch and populate the same
page cache, so this wastes bandwidth but does not alter model results.
No pinning or exclusion of in-flight ranges is attempted.

Logs: startup `expert_pregate: active k=...`; teardown `expert_pregate: stats`.
`predictions` counts L->L+1 attempts submitted; `arrivals` counts actual next
layer hooks with predictions. `hits/actual` is recall@K (fraction of actual
selected IDs predicted), `hits/predicted` is precision, both over arrivals.
`async_bytes_read` and `async_reads` count successful bytes and attempted
pread syscalls, including duplicate reads/retries; errors and dropped
submissions are explicit. `inflight_at_arrival` counts arrivals with unfinished
async chunks, `inflight_reads` sums running chunks at arrivals, and
`pending_reads` sums queued plus running chunks. These are racing snapshots,
not a wait or exact count of kernel IOs. `sync_bytes_with_prediction` and
`sync_bytes_without_prediction` are actual decode fill bytes partitioned by
whether that layer received a prediction; corresponding calls are per-file
fills. The latter cohort normally contains layer 0, not a counterfactual
measurement of all layers. Compare each pg arm's `expert_fill: stats
 decode_bytes_read` with frd16's same counter for the full with/without
prediction comparison. Counterfactual bytes cannot be recovered from a
cache already changed by prefetch. Residency snapshots and pread bytes are
not physical NVMe telemetry. Teardown drains outstanding jobs before logging
async totals; late reads are included even if they missed their deadline.

Default-off: `if (ctx.expert_pregate_enabled)` guards the prediction hook;
router setup returns immediately when LLAMA_EXPERT_PREGATE is unset.
No extra hot-path syscalls, allocations, router math, residency probes or
resource diagnostics run when unset. The queue and residency scratch are
initialized with the filler; no new threads or fds are needed for async IO.
06 preserves 05's separately opt-in diagnostic/populate behavior.

Expected effect, not measured evidence: OM9's approximately 461 ms fill IO
plus 394 ms other work gives 855 ms/token (about 1.17 tokens/s). With recall r,
resident predicted bytes and completely hidden prefetch IO, synchronous IO
falls by roughly fraction r, to `461*(1-r)` ms or `164*(1-r)` MB/token.
An optimistic incremental model is `T = 394 + 461*(1-r)` ms: r=0.5 gives
624.5 ms (1.60 tokens/s), r=0.8 gives 486.2 ms (2.06 tokens/s), and r=1
approaches the compute floor 394 ms (2.54 tokens/s). IO is moved, not erased:
on one shared drive the pipeline also obeys an IO throughput floor, at
least `max(394,461)` ms if the original missing 164 MB still needs reading,
and worse with wrong predictions. Thus the 394 ms limit requires sufficient
bandwidth/cache hits too; one-layer lead time may not hide all fills.
L+1 overlap is bounded by L's remaining compute/IO and next attention.

Risks: wrong predictions add IO and page-cache pressure, evicting useful
experts or even freshly prefetched pages. K=16 may improve recall while
lowering precision and increasing thrash. Six Jetson CPU cores must share
expert matmul, CPU router dot products, and up to 16 fill workers; contention
and page-cache copies can consume the overlap benefit. Float router caching
adds about 2 MiB per predicted layer of host RAM, plus startup file faults.
Low-priority queue order does not imply kernel IO priority. Incomplete jobs
can outlive a layer/token and pollute later cache state. Neither mincore nor
buffered pread pins pages; async errors leave the normal sync path to recover.

`make -C cc/expert_fill test` includes cold async completion, zero-byte sync
repeat, simultaneous disjoint sync/async fills, and invalid async bounds.
`cc/abrun/specs/J-OM10.json` uses image `campaign/llamacpp:jetson-up1108-om10`,
OM9 argv, no diagnostic settings, and exactly
`base frd16 pg8 pg16 pg16 pg8 frd16 base`. Each pg arm independently requires
both `expert_pregate: active` and `expert_pregate: stats` via per-arm regexes.
No Jetson runtime, recall or performance claim is made by these compile checks.

## 07 / J-OM10b: pre-gate scheduling

06's CPU observer enqueued L+1 before filling L. Workers preferred published
sync jobs but could all run speculative chunks, including while demand built
its residency/job list. Queued chunks had no arrival deadline. They did query
residency at consumption, so a stale chunk could skip resident pages, but
could reread pages evicted since demand or race an in-flight demand read.
These code paths support the scheduling diagnosis; no new runtime result is
claimed here.

07 defaults `LLAMA_EXPERT_PREGATE_ORDER` to `after` (unset or any value other
than `before`). `before` preserves the old enqueue-before-demand call order
for A/B; cancellation and concurrency controls still apply in both modes.
In the CPU observer, the reviewable order is:

```cpp
if (ctx.expert_pregate_enabled) {
    ctx.expert_pregate_arrive(il, selected); // cancel before demand
    if (ctx.expert_pregate_before) { ctx.expert_pregate_hook(il, t); }
}
ctx.expert_fill_selected(il, selected); // demand for L first by default
if (ctx.expert_pregate_enabled && !ctx.expert_pregate_before) {
    ctx.expert_pregate_hook(il, t); // ORDER=after (default): enqueue L+1
}
```

The cancel site in `expert_pregate_arrive`, before either demand fill or the
optional old-order prediction, is:

```cpp
for (auto & entry : expert_fillers) { entry.second->cancel_async_through(il); }
```

`enqueue_async(ranges, il + 1)` tags every chunk with its destination layer.
Under the dispatch mutex, `cancel_async_through` erases only queued chunks
whose destination is <= the arriving layer, updates their batch pending
counts, and accounts for cancellation. It never changes running chunks.
At each new microbatch, `cancel_async_through(SIZE_MAX)` retires all queued
work before layer numbering/prediction metadata reset. Running chunks keep
their shared batch alive and finish normally, even across token boundaries.

`LLAMA_EXPERT_PREGATE_MAX_INFLIGHT` defaults to 4 per file's existing worker
pool, clamped to that pool's thread count. Valid values are 1..256; unset,
zero, negative, malformed, or larger values fall back to 4. Demand always
wins dispatch: an exception-safe demand scope suppresses new async starts
from before residency queries through demand completion (including optional
populate). Workers still select demand jobs first. The cap counts running
async chunks across all batches, with a slot released and workers notified
on every completion, including IO errors. Running speculative IO is not
preempted and may still duplicate demand IO; kernel IO priority is unchanged.
With the experiment's 16 workers and cap 4, at least 12 workers remain
available for demand. No additional threads or descriptors are introduced.

`expert_pregate: stats` adds `cancelled_jobs` and `cancelled_bytes` (sum of
page-aligned queued chunk lengths removed, not physical IO saved). Per-batch
and filler counters also expose these fields; filler `max_inflight` is the
observed high-water mark used by tests. Existing arrival counters are sampled
before cancellation. `async_wait_ns_in_demand` is omitted: running snapshots
do not measure overlap duration, and exact same-layer overlap would require
per-layer completion timing/accounting on the worker hot path. No proxy is
reported as actual demand wait time.

The canonical header remains `cc/expert_fill/expert_fill.h`; both patches
are generated using its verbatim bytes for `src/llama-expert-fill.h`.
`test_cancel_before_start` freezes async dispatch at the demand-priority
boundary and verifies current/older cancellation, future preservation,
aligned-byte accounting, idempotence, drain completion, and subsequent reuse.
`test_async_inflight_cap` exercises limits 1 and 2, default 4 and invalid
fallback, with two batches, demand progress, recorded concurrency, and slot
reuse. Existing cold IO, warm skip, async completion, and populate checks
remain enabled.

`cc/abrun/specs/J-OM10b.json` uses image
`campaign/llamacpp:jetson-up1108-om10b`, the J-OM10 argv, and explicit order
`base frd16 pg8b pg12b pg12b pg8b frd16 base`. The pg arms set K=8/12,
ORDER=after, MAX_INFLIGHT=4, and retain J-OM10's per-arm switch regexes.
Compile/application proof is in `COMPILE-CHECK.txt`, section 07. No Docker,
GPU, full engine build, commits, or experiment evidence edits are required.

## 08 / J-OM11: strict configuration, arrival timeliness, lookahead 2

Apply `08-pregate-observe.patch` after 01–07, or apply
`combined-01-02-03-04-05-06-07-08.patch` to cf34d311. The header in this
folder remains the single source for `src/llama-expert-fill.h`.

Every supplied `LLAMA_EXPERT_*` name must be known. All settings are parsed
before the context constructor body, including when fill and pregate are off;
model mmap initialization also validates before applying expert advice. Invalid
values print `expert_config: invalid NAME=VALUE (range ...)`, flush stderr,
and call `std::abort()`. Recoverable fill/predict exception handlers cannot
swallow these failures. Integers accept decimal digits only, including leading
zeroes; signs, whitespace, empty strings, suffixes and overflow abort. No
integer silently falls back to its default. These ranges supersede 03–07:

| Variable (`LLAMA_EXPERT_` prefix) | Accepted values | Default |
|---|---|---|
| `FILL`, `FILL_DIAG`, `FILL_POPULATE`, `PREGATE` | integer 0..1 | 0 |
| `FILL_RANDOM` | `0`, `1`, `decode` | `0` |
| `FILL_THREADS` | integer 1..64 | 8 |
| `FILL_CHUNK_KIB` | integer 64..65536 KiB | 1024 |
| `FILL_MIN_TOKENS`, `FILL_MAX_TOKENS` | integer 0..2147483647; max 0 means unlimited | 0 |
| `FILL_DIAG_CSV` | nonempty path if supplied | unset |
| `PREGATE_K` | integer 1..32, including **10** | 8 |
| `PREGATE_ORDER` | `after`, `before` | `after` |
| `PREGATE_MAX_INFLIGHT` | integer 1..64; workers also bound actual concurrency | 4 |
| `PREGATE_LOOKAHEAD` | integer 1..2 | 1 |
| `PREFETCH`, `PREFETCH_NEXT` (legacy 02) | integer 0..1 | 0 |

One `expert_config: effective ...` line is printed per context, even with no
expert environment variables. It contains every parsed setting and default.
It reports configured settings; runtime support is still checked by the
existing `expert_fill: active` and `expert_pregate: active` messages. J-OM11
requires both value assertions and activation for treatment arms.

At layer L, distance 1 uses x_L with router L+1. With lookahead 2, distance 2
also uses the same x_L with router L+2. All distance-1 jobs are submitted first.
The queue orders by target layer, then distance (1 before 2), then FIFO. Thus a
fresh distance-1 prediction also precedes older distance-2 work targeting the
same layer. Arrival of M cancels queued jobs for all targets <= M before demand;
running preads finish. The token boundary cancels the previous token's backlog.
Distance-1 and distance-2 records are retained separately until their target
arrives; one never overwrites the other. Tail layers only predict valid targets.

Timeliness counts **prediction instances** (target layer, expert, distance),
not chunks, at target arrival before demand. An expert covers all its tensors
and files. Prediction-time mincore probes skip submission only when all its
ranges are resident; a fake oracle exercises this exact path in unit tests.
Completed jobs are not evidence that pages remain resident until demand.
For multi-chunk experts, the conservative classification is:

- `pred_resident_at_predict`: every range was resident at predict time; no jobs enqueued.
- `pred_queued`: any chunk is queued/cancelled, backlog-rejected, or failed; never reported done.
- `pred_inflight`: no queued/cancelled/failed chunks and at least one running chunk.
- `pred_done`: all submitted chunks have completed successfully.

`hit_done`, `hit_inflight`, `hit_queued`, `hit_resident` apply the same
classification only when the expert is actually selected. Each file's batch snapshots
and queued-job cancellation share its dispatch mutex; a chunk sampled queued
cannot start before it is cancelled. Sampling never waits for IO. This occurs
before demand. Worker progress during sampling across files is possible; these
are arrival snapshots, not a global IO barrier. Cancelled
and rejected work cannot become `done` just because its pending count reaches 0.
`pred_done + pred_inflight + pred_queued + pred_resident_at_predict = predicted`;
the four hit counters sum to `hits`.

`recall_at_k_d1` and `recall_at_k_d2` each divide distance-specific correct
predictions by selected experts at layers with that distance's prediction.
`actual_d1/actual_d2` and `hits_d1/hits_d2` expose those denominators/numerators.
The existing aggregate recall includes both distances. The same expert can be
predicted at both distances; each instance is counted for its own prediction
and async work. These counters measure job timeliness, not unused-page eviction.

`demand_wait_ns_total` sums existing steady-clock demand `fill()` wall times,
including residency queries, scheduling, IO and optional populate, over every
filled layer/file and phase. It excludes prediction, diagnostic probes outside
fill and teardown. `demand_wait_ns_per_token` divides by tokens in eligible
ubatches (prompt tokens count individually; one decode token contributes one
across all layers). Both pregate and fill exit stats expose these fields, so
fill-only arms also report them. Existing phase counters remain available.

A6 abort path (`expert_fill.h`):
```cpp
c.k = integer("LLAMA_EXPERT_PREGATE_K", 8, 1, 32);
// integer():
if (!valid || n < lo || n > hi) {
    invalid(name, value, std::to_string(lo) + ".." + std::to_string(hi));
}
// invalid():
fprintf(stderr, "expert_config: invalid %s=%s (range %s)\n", name, value, range.c_str());
fflush(stderr);
std::abort();
```
For K=0 this emits `expert_config: invalid LLAMA_EXPERT_PREGATE_K=0 (range 1..32)`
and terminates with SIGABRT, even when `FILL=0`.

A6 effective-config print site (`ExpertConfig::print()`):
```cpp
fprintf(stderr, "expert_config: effective fill=%zu random=%s max_tokens=%zu threads=%zu chunk_kib=%zu "
    "pregate=%zu k=%zu order=%s max_inflight=%zu lookahead=%zu min_tokens=%zu diag=%zu populate=%zu "
    "diag_csv=%s prefetch=%zu prefetch_next=%zu\n", /* all parsed fields */ ...);
```
The context constructor calls `expert_config.print()` exactly once under Linux.

Verification: `make -C cc/expert_fill test`; named tests include
`test_strict_parser`, `test_lookahead_order`, and `test_classification_counters`.
`COMPILE-CHECK.txt` section 08 records fresh-tree `patch -F0` equivalence,
header byte equality and syntax-only checks. No Docker/GPU/full build is needed.
`cc/abrun/specs/J-OM11.json` has two reps per arm in the required mirrored order.
It uses `eval_ms_per_token`, `direction=lower` because the current gate supports
that direction. Per-arm switch patterns require all effective values, including
K and lookahead, on the same line. The base has no env and uses
`switch_regex_is_config` to assert disabled defaults without treating its own
effective-config line as activation; other arm activation signatures still fail
that control. All ordinary switch checks preserve their previous semantics.

### 09 / J-OM12 thread names

`09-thread-names.patch` adds `pthread_setname_np(pthread_self(), "xfill-N")`
at each expert-fill worker's entry, guarded by `__linux__`. The return value is
ignored: naming is diagnostic and does not affect scheduling, dispatch, or IO.
The canonical header and combined 01..09 patch contain the same change.
Fresh zero-fuzz application, tree/header equality, and all five touched C++
syntax checks are recorded in `COMPILE-CHECK.txt` under 09.
