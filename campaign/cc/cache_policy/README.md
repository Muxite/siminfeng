# Offline expert-cache policy replay

Run from the campaign root, with Python 3.12, numpy, pytest, and g++ available:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 cc/cache_policy/replay.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m pytest -q cc/cache_policy \
  --basetemp=artifacts/codexPOL/pytest-temp \
  -o cache_dir=artifacts/codexPOL/pytest-cache
```

The entry point writes only below `artifacts/codexPOL`; tests must use the shown
base/cache locations to preserve the assignment's write restrictions. No model
inference, GPU allocation, source-tree changes, or tensor-payload reads occur.
The existing `cc/model_inspect/inspect_model.py:130` API reads the model header.
Its `HeaderReader` bypasses payload construction. The model file is never hashed.
Only the small trace, source, protocol and result artifacts are hashed.

The report is [REPORT.md](../../artifacts/codexPOL/REPORT.md), with full machine
results in [results.json](../../artifacts/codexPOL/results.json). The prereplay
configuration is [protocol.json](../../artifacts/codexPOL/protocol.json). The
configuration was written before the first replay; neither it nor the static
training split was adjusted after seeing results.

## Source definitions and verification

All line numbers below refer to the unmodified archived `.cpp` files. They can
also be reproduced with `git -C ref/llama.cpp show <commit>:src/llama-moe-cache.cpp`.

* **ClassLRU:** pin `3b4d6ab3f79651935fba1dd93732f29022a7f5f5`,
  [classlru.cpp](../../artifacts/codexPOL/source/classlru.cpp:20).
  Lines 20–27 define eight classes and 128-token CRF half-life. Lines 110–128
  define keys, persistent frequency history and layer-clock decay. Lines
  131–174 choose the first unselected candidate from each class list, then
  maximize `age / 2**class`; strict greater-than preserves the lower class on
  score ties. Each key's class stays fixed until its next use. Lines 197–221
  update hits in router rank order before selecting victims. Lines 268–284
  fill misses in ascending expert ID and advance only the current layer's
  clock. Eviction retains CRF and last-use history. Python rounds CRF updates
  to float32, matching C++ float state.
* **CPU-share:** same pin, lines 224–265 select lowest-CRF misses for CPU,
  breaking ties on expert ID; these misses do not enter residency. Lines
  806–811 enable bypass only once the group is full and use a per-layer
  16-bit carry for fractional shares. The primary CPU mode fixes share to 1;
  tests also compare share .37 against upstream. Eight zero slots plus the
  guard slot are reserved for this mode (lines 450–459). This pin includes
  #31853's CRF selection in addition to #30966. No partial share was chosen
  using evaluation data.
* **LRU:** pre-SLRU parent `098ce526b546b5a190327c1da82929cee051b9ac`,
  [lru.cpp](../../artifacts/codexPOL/source/lru.cpp:16).
  Lines 23–65 define the LRU slot list. Lines 94–116 touch all hits in router
  rank order, then admit misses in ascending expert ID. Applying pinned
  CPU-share to this historical policy is explicitly counterfactual.
* **SLRU\*:** historical commit
  `33d6135cbe20e90e67d9db1c874fbb4a9dc64b19` is locally identified as
  "replace MoE LRU cache with a 50/50 SLRU cache (#30637)". Its source blob
  `4d3596b244cb46e429b7bda8f7369c528c02dd1c` is missing from the partial clone;
  the other locally known SLRU tree blob is missing as well. GitHub/raw fetch
  failed. The archived [failure](../../artifacts/codexPOL/source/slru-fetch.stderr)
  demonstrates the limitation. The simulator reconstructs 50/50 probationary
  and protected queues: new entries go to probation, hits move to protected
  MRU, protected overflow demotes its LRU, eviction chooses probation LRU then
  protected LRU, and current-plan selections cannot be evicted. **These are
  explicit reconstruction assumptions, not verified #30637 source lines. The
  assignment's faithful upstream SLRU requirement remains unfulfilled.**
* **Static:** oracle-free TRAIN prose-prefill selection counts for `(layer,id)`.
  Each layout group selects its highest counts up to its allocated slot limit;
  ties use ascending `(layer,id)`. Because all experts within a group have
  the same byte size, frequency ranking equals frequency/byte ranking. The
  fixed set never adapts to evaluation. Static upload misses are modeled as
  transient staging, without eviction of the selected set; CPU misses compute
  on CPU. Both are analytical baselines, not upstream static cache options.

`test_replay.py` extracts the actual ClassLRU struct from archived upstream C++,
compiles it in the permitted test directory, and compares hits, uploads, CPU
misses, every slot identity and class after every plan. The test uses four
capacities, three CPU shares, four layers, deterministic random plans and
working-set shifts. A separate compiled historical LRU oracle checks every
resident slot and hit count. This verifies replay semantics without compiling
llama.cpp or changing the reference checkout. The SLRU reconstruction has no
upstream differential oracle and is deliberately marked with an asterisk.

## Exact bytes and evaluation

Pinned `classlru.cpp:300` compares projection types, shapes and expert strides to
form groups. Lines 421–443 group compatible layers on one GPU. Lines 492–518
split budget in proportion to host bytes and size each bank, including reserved
slots and per-projection alignment. CUDA alignment is 128 bytes at pinned
`ggml/src/ggml-cuda/ggml-cuda.cu:1043`. LRU and SLRU use this same pinned allocation
for controlled comparisons rather than mixing historic allocation rules.

Header inspection finds two layout groups of 24 layers each: 3,059,712 bytes
(2.91796875 MiB) and 2,654,208 bytes (2.53125 MiB) per complete expert
(gate + up + down). Mean expert size is 2.724609375 MiB; all 128 experts in all
48 layers occupy 16,740 MiB. These are header-derived storage sizes, not an
assumed uniform 2.72 MiB. The primary budgets are **decimal** 2, 4, 6, 8 and
10 GB of cache banks. They are not total VRAM limits. D041 cache sizes are
converted from binary MiB exactly.

The group budget is `floor(B * group_host_bytes / all_host_bytes)`. A bank's
allocation is `sum(ceil(stride*(slots+reserve)/128)*128)`; reserve is one for
upload mode and nine for CPU-share mode. Model/KV/compute buffers are excluded.
Static uses the same allocation, including reserved space, for comparability.
Full grouping, slot counts, allocation bytes and static IDs are in results.json.

The old `cc/routing_trace/trace_replay.py` establishes token-entry hit accounting
and unweighted request rates, but rejects Qwen IDs >=32 and uses independent
per-layer caches. This implementation replaces those assumptions with header
counts and upstream shared layout banks. Input validation checks unique
identities, top-8 distinct in-range expert IDs, finite nonnegative weights,
complete layers, and contiguous single-sequence positions. Decode has exactly
64 positions in each capture. All eight hits are determined before miss fills.

Static selection uses **only prose prefill**. Prose decode and chat/code prefill
are excluded. Adaptive caches are cold at each held-out chat/code decode start,
then evolve online across all 64 positions. This avoids inventing prefill
microbatch behavior from token-level traces: pinned `classlru.cpp:693` bypasses
large batches, so tokenwise prefill warming would not faithfully model the
engine. Pooled decode weights each routed expert request equally. Per-domain
and per-layer metrics and trace SHA256s remain available for audit.

## Throughput and calibration

For each decode token, aggregate exact uploaded bytes `U`, CPU bytes `C`, and
CPU-active layer count `A`. The raw physics forecast in seconds is

```
t = 0.004637503 + U / 28.7e9
    + C / (8 * mean_expert_bytes) * 0.000757210 + A * 0.000030
```

H2D is D042 section 5. GPU floor and aggregate CPU-layer time are the D039 fit
recorded in `artifacts/codexQRT/ANALYSIS.md` (section "Decode forecast and
crossover"). The 30 us boundary term is the lower T-PROF estimate in that
analysis; it is an assumption. CPU execution and upload costs are serialized,
with no assumed overlap. CPU-share=1 still uploads initial misses before filling
the group, and pays those costs in both raw and fitted forecasts. A frozen
resident set after filling explains why CPU-mode results are identical among
LRU, SLRU* and ClassLRU under this protocol.

D041's measured LRU chat values are 18.1 t/s at 2200 MiB and 51.3 t/s at 6000 MiB.
With `x=U/28.7e9` and fixed floor `g`, fit just one nonnegative multiplier
`alpha=max(0, sum(x*(1/observed_tps-g))/sum(x*x))`. The calibrated forecast
replaces the upload term with `alpha*U/28.7e9`. RMSE is reported in both t/s and
ms/token, alongside signed residuals and leave-one-out predictions. Fitting in
latency space does not minimize t/s residuals. The observed gap cannot be
explained exactly by this one-parameter model. A fitted multiplier is an
empirical effective miss cost, not a physical bandwidth result.

For whole-layer static, fit `t(K)=g_static+K*c_static` to D041 K30=37.1 and
K42=27.4 t/s. This two-anchor fit is saturated (zero residual degrees of
freedom), which is stated explicitly in the report. For each exact expert-bank
budget, choose the minimum integer K for which layers K..47 fit. Dynamic cache
versus trained top-frequency static is reported separately from comparison
against D041's whole-layer static placement.

Crossover searches use the frozen 0.5..12 GB grid with spacing 0.5 GB. A sign
change in `t_static-t_cache` yields a linearly interpolated budget plus the grid
bracket, and all reversals are retained. No crossing beyond the evaluated
range is invented. High-budget reversals arise in part because the 64-token
cold replay retains compulsory upload cost even with abundant cache; these
should not be extrapolated to long warmed generations. CPU predictions are
uncalibrated to actual CPU-share inference. The D041 screen itself is
indicative, not a frozen hardware benchmark; this is a frozen offline run.
