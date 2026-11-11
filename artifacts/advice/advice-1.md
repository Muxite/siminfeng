**The transferable result is your measurement method; the target’s feasibility remains unproven.** First establish whether “500 GB” means BF16 or already compressed weights. If BF16, ideal 4-bit storage is 125 GB before scales, higher-precision tensors, KV, scratch, and OS memory—already too tight for 128 GB.

1. **What transfers**

   Paired quality evaluation, complete routing traces, byte accounting, and separating kernel time from scheduling/I/O time all transfer. Quantization and KV experiments transfer as procedures, not numerical predictions.

   Do **not** extrapolate Granite’s Q8/Q4 ranking, 75% bandwidth utilization, quality loss, cache locality, or tokens/s. Target matrix dimensions, kernels, active-weight bytes, expert count/top-k, and domain specialization change these. The Q4 slowdown is consistent with dequantization overhead, but timing alone does not establish causality.

   CPU placement on your discrete machine measures CPU execution plus PCIe/synchronization effects. Unified memory removes some transfers but introduces shared bandwidth contention; it does not guarantee zero-copy backend behavior. A container limit tests Linux reclaim behavior, not the target’s unified-memory performance. A fully resident model says almost nothing about SSD stalls.

2. **Minimal discriminating experiments**

   - **Build the target budget first:** actual tensor sizes, expert bytes at each precision, always-resident weights, context-dependent KV, scratch, and memory reserve. Obtain target-model routing traces across prose, code, reasoning, and long-context workloads, separating prefill/decode. Without these, cache conclusions remain Granite-specific.
   - **Quantization:** compare uniform supported 2/3/4/8-bit formats, experts-only quantization with protected routers/attention, and equal-byte-budget frequency-based versus sensitivity-based expert precision. Separate calibration and evaluation domains. Frequency is not importance; rare experts may be essential. Sensitivity plus activation frequency is also the premise of [MxMoE](https://arxiv.org/abs/2505.05799).
   - **Residency/I/O:** replay target traces with actual expert sizes across cache budgets; compare static hot sets, LRU/LFU, and offline optimal replacement. Measure bytes fetched/token, read amplification, and p50/p95 token latency. Validate using actual target-sized SSD reads and compute kernels. A necessary throughput bound is  
     `tokens/s ≤ effective SSD bytes/s ÷ fetched bytes/token`; dependency stalls tighten it.
   - **Prefetch:** compare none, implementable prediction, and oracle lookahead. Measure timely coverage, wasted bytes, evictions, and exposed stalls. Exact next-layer routing requires its hidden state; “early router” evaluation is generally prediction. Replay must respect when predictions become available.
   - **Structural compression:** compare expert pruning/top-k reduction and low-rank candidates against quantization at matched bytes and quality, including recovery training where needed. Require kernels that exploit the representation. Validate finalists end-to-end on the actual unified-memory platform.

   Beyond PPL/KL: paired task accuracy with uncertainty; code execution, reasoning, instruction following, multilingual/domain coverage; long-context retrieval and synthesis; per-layer top-k overlap, router cutoff margins, selected-weight divergence, load entropy/skew, and rare-expert regressions. Measure routes on identical teacher-forced prefixes, then separately evaluate free generation. Routing flips are diagnostics, not automatically errors.

3. **Build these reusable components first**

   - **Manifest/budgeter:** checkpoint + formats + context → bytes per tensor/expert and peak-memory estimate.
   - **Trace comparator:** paired runs → complete routes, margins, domain-conditioned frequencies, reuse distances, and routing divergence.
   - **Timed replay engine:** trace + manifest + measured I/O/kernel costs → cache curves, bandwidth demand, prefetch waste, latency bounds.
   - **Benchmark harness:** fixed prompts/configuration → quality, TTFT, decode distributions, faults, physical SSD bytes, residency, and memory pressure.

   Explicitly distinguish cold storage, warm page cache, and warm expert cache. Audit mmap readahead, duplicated buffers, pinning, swap/compression, and cgroup accounting. Rerun traces after quantization: noise changes downstream routes. Pin engine commits; measure sustained unified-memory bandwidth under concurrent CPU/GPU/I/O activity.