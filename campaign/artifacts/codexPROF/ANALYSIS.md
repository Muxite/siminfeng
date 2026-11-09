# T-PROF: why hybrid split loses to static layer offload

Split is slower mainly because per-slot GPU dispatch expands the GPU workload,
despite successful CUDA graph reuse. Across 154 steady decode intervals, split
adds **2.032 ms/token**: **1.446 ms (71.2%)** is additional GPU graph execution,
**0.371 ms (18.3%)** is additional exposed sync/copy time, and **0.214 ms (10.6%)**
is additional host-side time while the GPU is idle. There is **no observed graph
capture loss**. These are trace wall-time accounts, not causal intervention results.

## Inputs and method

Read decisions D008, D011, D019 and D030 and `cc/hybrid_split/README.md`.
Inputs are `artifacts/x4-hs/prof/{static,split}-n12.sqlite` and their `.log` files;
the `.nsys-rep` files were not re-exported. Analysis uses Python sqlite3 and the
standard library only; no GPU, Docker, sampling or node-level capture was run.

Each trace has 160 D2H copies of 196,620 bytes (49,155 float logits). The first
ends prompt evaluation; the next 159 delimit complete decode steps. Discard
the first five decode intervals, retaining **154** intervals in each trace.
Boundaries are logits DMA completion to logits DMA completion; the final
logits DMA belongs to the token it completes. Counts use operation start times;
durations are clipped at boundaries, including crossing host API calls.
This includes inter-token sampling/host scheduling; it differs slightly from
llama's eval-only timer. All per-token table values below are arithmetic means,
unless explicitly labeled median. The 13/25 graph pattern and 12/24 layer-copy
pattern occur in **every** retained interval. Raw per-token results are in
`breakdown.json`, not inferred by dividing whole-process initialization totals.

Graph-level nsys supplies `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` execution envelopes.
Take the union of these envelopes, explicit kernels, memcpy and memset across
streams for GPU busy coverage; idle is its complement. **Graph envelopes include
internal kernel gaps**, so busy coverage is not SM utilization or the sum of
individual kernel execution times. Idle-gap counts include token-edge gaps and
all positive gaps; a second metric filters gaps below 10 us.

## Steady-state side-by-side

| Per decode token | Static n12 | Split n12 | Split minus static |
| --- | ---: | ---: | ---: |
| Wall time, ms | 5.346 | 7.378 | +2.032 |
| Median wall time, ms | 5.325 | 7.377 | +2.052 |
| CUDA graph launches | 13 | 25 | +12 |
| Graph-launch host API time, us | 244.17 | 651.72 | +407.55 |
| Kernel-launch APIs outside capture | 0 | 0 | 0 |
| Explicit kernel activity records | 0 | 0 | 0 |
| StreamSynchronize calls | 110 | 194 | +84 |
| StreamSynchronize host API time, us | 1,268.62 | 2,945.79 | +1,677.18 |
| MemcpyAsync calls | 46 | 82 | +36 |
| MemcpyAsync host API time, us | 173.22 | 312.09 | +138.87 |
| D2H bytes | 246,156 | 295,692 | +49,536 |
| H2D bytes | 406,048 | 799,264 | +393,216 |
| GPU graph execution envelopes, us | 1,012.77 | 2,459.06 | +1,446.29 |
| Device copy time outside graph envelopes, us | 47.58 | 86.17 | +38.59 |
| GPU busy coverage, us | 1,060.35 | 2,545.22 | +1,484.87 |
| GPU idle gaps, count | 59 | 107 | +48 |
| GPU idle gaps, total us | 4,285.57 | 4,832.51 | +546.95 |
| Idle gaps at least 10 us, count | 28.63 | 56.20 | +27.57 |
| Idle gaps at least 10 us, total us | 4,051.42 | 4,431.81 | +380.39 |

Total host API times **overlap** GPU execution/copies. In particular, the +1.677 ms
of StreamSynchronize time is largely waiting for the extra GPU work; adding it
to the +1.446 ms GPU difference would double-count the slowdown.

### Transfers by direction and size

Each cell is **calls/token; device us/token; matching MemcpyAsync API us/token**.
The API join uses correlation IDs. All 46/82 memcpy calls are matched. No
synchronous `cudaMemcpy`, `cudaMemcpy2D`, or additional copy kind occurs in the
retained window. There are no steady-state expert-weight-sized uploads.

| Direction | Bytes/call | Static | Split |
| --- | ---: | ---: | ---: |
| D2H selected cold IDs | 32 | 12; 3.68; 35.92 | 24; 7.36; 97.57 |
| D2H input activation | 4,096 | 12; 4.80; 42.45 | 24; 9.58; 73.50 |
| D2H logits | 196,620 | 1; 6.73; 6.85 | 1; 6.74; 6.47 |
| H2D small input | 4 | 2; 0.54; 7.33 | 2; 0.51; 7.74 |
| H2D small input | 8 | 3; 0.77; 9.88 | 3; 0.77; 9.78 |
| H2D small input | 512 | 1; 0.28; 3.18 | 1; 0.28; 3.16 |
| H2D small input | 4,096 | 3; 1.16; 14.43 | 3; 1.16; 14.55 |
| H2D cold slot output | 32,768 | 12; 29.62; 53.18 | 24; 59.77; 99.31 |

The 32,768-byte H2D is the full 8 x 1,024 float cold-slot result. Split doubles
the number of CPU-active layers from 12 to 24, adding 432.38 KiB/token of total
bidirectional traffic. Transfer **bandwidth** is not the main problem: the extra
device DMA time is only 38.59 us/token. The associated handoffs and exposed host
latency matter more. Reducing cold slots on the CPU before upload requires
preserving the original slot reduction order/numerics; it is not a free change.

## CUDA graphs and the available kernel evidence

**Yes, split executes GPU work inside CUDA graphs.** Every retained token launches
the same **25** graph executables, each exactly once: graphExecIds 2,5,...,74,
with roughly 73.7–101.6 us per graph on average. Static uses **13** reused graph
executables (2,5,...,38): the first twelve average 24.9–26.2 us each, and the last
averages 700.8 us. This matches twelve CPU-layer partitions plus the contiguous
GPU-layer suffix in static, versus CPU work interleaved at all 24 split layers.
Layer association is a structural inference from the repeating sequence and
copy counts; the export contains no NVTX layer labels.

There are **zero** retained BeginCapture, EndCapture, Instantiate or GraphExecUpdate
calls and **zero** direct kernel-launch APIs. Both logs report `graphs reused =
158` (llama compute-graph reuse, distinct from the CUDA graph-executable count).
`graphNodeId` is NULL for the explicit kernel records, as expected for these
graph-level exports. The absence of constituent kernel rows in steady decode
does **not** mean the graphs contain no kernels.

The first decode interval, before removal as warmup, contains directly traced
kernels: **583 static, 1,535 split**, totaling **1,164.83 / 2,556.09 us**, with wall
times 13.413/21.418 ms. Split additionally performs one graph capture/instantiate
and graph launch in this interval; static begins graph execution in the next
interval. Thus these are not identical complete inventories of all device work.
These are **one initialization decode's kernel totals**, not measured steady-state
kernel totals. Their difference of 1,391.26 us is close to the steady graph
envelope difference of 1,446.29 us, supporting the per-slot explanation without
pretending to recover hidden graph nodes. Full demangled names are in the JSON.

Top directly visible kernels, ranked separately within each first-decode column:

| Kernel / specialization | Static rank: count, total us | Split rank: count, total us |
| --- | ---: | ---: |
| mul_mat_vec_q type 12, fused flags true,true | 2: 12, 198.69 | 1: 192, 790.75 |
| quantize_q8_1 | 3: 121, 114.11 | 2: 480, 450.94 |
| mul_mat_vec_q type 12, fused flags false,true | 4: 12, 112.86 | 3: 192, 399.55 |
| mul_mat_vec_q type 8 | 1: 97, 285.60 | 4: 96, 213.82 |
| concat_cont | absent | 5: 168, 174.02 |
| rms_norm_f32 | 5: 49, 73.57 | 6: 48, 70.88 |
| k_bin_bcast op_add | 8: 48, 48.38 | 7: 70, 70.43 |
| flash_attn_ext_vec | 6: 24, 67.01 | 8: 24, 67.46 |
| topk_moe_cuda | 7: 24, 50.78 | 9: 24, 50.94 |

The 192 expert calls per projection family are 24 layers x 8 slots; static has
12 GPU expert layers. Gate/up fusion means the actual observed kernel expansion
is not simply three independent matmul kernels per slot. Quantization adds
**359 calls and 336.83 us**, concatenation adds **168 calls and 174.02 us**, and
the two type-12 expert families add **360 calls and 878.75 us**. Together these
three categories explain 1,389.60 us of the 1,391.26 us explicit-kernel difference.
Those deltas include different GPU expert work/placement, not solely dispatch
overhead. Float/int GET_ROWS total 59.30 us in split versus 2.21 us static;
router restoration is a smaller secondary target, not the dominant loss.

## Exclusive attribution of the 2,031.82 us/token difference

| Bucket | Static us | Split us | Difference us | Share of delta |
| --- | ---: | ---: | ---: | ---: |
| GPU kernels/graph execution envelopes | 1,012.77 | 2,459.06 | +1,446.29 | 71.2% |
| Extra syncs/transfers, exposed portion | 440.97 | 812.10 | +371.13 | 18.3% |
| Remaining CPU-side time while GPU idle | 3,892.18 | 4,106.58 | +214.40 | 10.6% |
| Graph capture loss | 0 observed | 0 observed | 0 | 0% |
| Total | 5,345.91 | 7,377.74 | +2,031.82 | 100% before rounding |

The sync/transfer bucket is device-transfer-only time plus the intersection of
GPU idle with host sync/copy APIs: 47.58 + 393.39 us static, 86.17 + 725.93 us split.
The remaining idle time includes CPU cold computation, host graph submission,
sampling, scheduler and untraced work. With `--sample=none --cpuctxsw=none`, CPU
compute and CPU scheduling cannot be separated. Host graph-launch API time is
not itself graph capture loss, and its exposed portion is included in remaining
idle. The first bucket cannot separate kernel arithmetic from internal device
gaps in reused graphs. The table is an exact interval partition under these
definitions; a more precise causal breakdown is not identifiable from this capture.

Why does the deficit grow with more hot experts? Every split layer still pays
eight-slot GPU dispatch, quantization/concat, and a CPU boundary, including null
hot slots and cold skipped IDs. Static gains more contiguous GPU-only layers as
n-cpu-moe decreases, amortizing launches/boundaries. Thus split's fixed costs
become more prominent as CPU arithmetic shrinks. This explains the direction of
D030, but only n12 is traced here: n18/n6 magnitudes are not directly measured.

## Proposed redesigns (not implemented)

1. **Batched CUDA MUL_MAT_ID with negative/skip IDs and duplicate-safe routing.**
   Keep original selected-slot outputs; skip invalid IDs, remove the null expert,
   dispatch gate/up/down over all slots, and eliminate concatenations. Preserve
   gate/up fusion and reuse input quantization across consumers. Forecast
   **0.7–1.1 ms/token saved** at n12. The first-decode ledger offers about
   0.316 ms from eliminating 336 redundant quantizations at ~0.94 us each,
   0.174 ms from concatenations, plus an assumed 0.21–0.61 ms of expert dispatch
   and null-work savings. This is a target, not a steady-state kernel measurement;
   the 1.446 ms graph delta is a useful scale, not guaranteed removable overhead.
   Validate mixed IDs, duplicate misses, original reduction order and GPU numerics.

2. **Concentrate CPU work in K layers, selected jointly with hot residency.**
   Make the other layers fully GPU resident and spend remaining memory on partial
   hot banks only in the CPU-active layers. At K=12, removing half of the split
   boundaries could recover roughly half of its 0.812 ms exposed sync/transfer
   bucket, or **0.25–0.40 ms/token gross** after retaining logits/setup overhead.
   Budget **0.1–0.3 ms/token net** only if the new placement adds at most about
   0.1–0.2 ms CPU work. This assumption requires routing/CPU-cost measurement;
   the trace cannot establish the net benefit and the result can be negative.
   At fixed expert bytes, fully funding layers reduces frequency-selective hot
   coverage elsewhere. Choose K by total latency, not hit rate alone. Static
   n12 is already the measured endpoint demonstrating the boundary advantage.

3. **Launch cold CPU work concurrently with hot GPU FFN after routing.**
   Transfer IDs/input as soon as routing completes, run cold compute on a
   dedicated worker, run hot FFN concurrently, and join via an event immediately
   before ordered output reduction. Use pinned buffers and asynchronous copies;
   merely changing cudaMemcpy to cudaMemcpyAsync is insufficient (already async).
   Forecast **0.4–0.9 ms/token saved** before redesign 1, conditional on achieving
   overlap. The first-decode hot expert/quantization/concat work is about
   1.7 ms/token, versus 4.1 ms remaining host-idle coverage: hiding 25–50% of that
   hot work yields this range. Shared routing/attention and cross-layer
   dependencies prevent hiding the entire GPU envelope. D019 achieved only
   6–33% of hideable work with launch work on the CPU thread; a separate worker
   and an earlier fork are prerequisites. After batching, less GPU work remains
   to hide, so these forecasts **must not be added together**.

## Validation and limits

`pytest.log`: **3 passed**, covering a tiny synthetic SQLite export, overlapping
graph/kernel/copy intervals, correlation joins, first-five-style warmup removal,
missing kernel tables, invalid markers, and boundary-crossing API clipping.
The real-trace tool completes in about one second and opens databases read-only.

Profiled wall throughput is 187.1 static / 135.5 split tok/s (split -27.5%; wall
latency +38.0%). Eval-only logs report 188.72 / 134.72 tok/s. D030 unprofiled n12
medians were 189.8 static, 198.8 static+flags, 160.8 split. Split is perturbed more
here, so **do not extrapolate the 2.032 ms trace gap to unprofiled savings**.
Logs show eight CPU threads; D019's CPU harness calibration is not a matched
eight-thread end-to-end comparison. These runs also produce different greedy
continuations, so their routing/workloads are not token-identical. D030 reports
roughly matched GPU memory (split about 8 MiB higher), not exact byte equality;
the trace itself does not establish memory equivalence.

Both exports warn that CUDA driver 13.2 is unsupported by nsys 2024.6.2 and use
12.8 tracing libraries. No NVTX events were collected. Together with graph-level
visibility and disabled CPU sampling/scheduling, this limits fine attribution;
the observed repeating graph/copy counts and API timings remain the evidence
used here. No quality gate, redesign speedup, other hot fraction, or hardware
utilization claim has been established.

## Reuse

Canonical tool and tests: `cc/nsys_decode/{analyze_nsys.py,test_analyze_nsys.py}`;
`artifacts/codexPROF/analyze_nsys.py` links to that same executable source.
usage and accounting contract: `cc/nsys_decode/README.md`.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 cc/nsys_decode/analyze_nsys.py \
  artifacts/x4-hs/prof/static-n12.sqlite artifacts/x4-hs/prof/split-n12.sqlite \
  --warmup 5 --marker-bytes 196620 --output artifacts/codexPROF/breakdown.json
```

Use `--marker-bytes` for a different vocabulary/logits layout, `--device` to select
one GPU, and `--gap-us` to adjust the additional gap filter. Verify one repeating
logits marker per token before applying this segmentation to a different engine.
