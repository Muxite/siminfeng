# Frozen offline cache replay

SLRU* is an unverified 50/50 reconstruction: upstream historical blob unavailable. Other policies are source-based. All throughput numbers are predictions.

Budgets are decimal GB of cache banks, excluding other model/runtime weights. Each evaluation trace begins cold at decode. Static ranks use prose prefill only. Pooled chat/code decode has equal token weight.

| GB | Policy | Upload hit % | CPU-share=1 hit % | Upload t/s physics | Upload t/s fitted | CPU t/s | Whole-layer static t/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | LRU | 52.77 | 35.12 | 43.85 | 18.93 | 31.77 | 26.82 |
| 2 | SLRU* | 51.83 | 35.12 | 43.23 | 18.63 | 31.77 | 26.82 |
| 2 | ClassLRU | 54.28 | 35.12 | 45.02 | 19.51 | 31.77 | 26.82 |
| 2 | Static | 12.30 | 11.94 | 26.23 | 10.70 | 26.28 | 26.82 |
| 4 | LRU | 71.49 | 59.18 | 63.93 | 29.57 | 41.13 | 30.02 |
| 4 | SLRU* | 72.43 | 59.18 | 65.45 | 30.43 | 41.13 | 30.02 |
| 4 | ClassLRU | 73.00 | 59.18 | 66.39 | 30.97 | 41.13 | 30.02 |
| 4 | Static | 29.01 | 28.73 | 31.51 | 13.07 | 31.30 | 30.02 |
| 6 | LRU | 82.77 | 73.90 | 88.48 | 44.82 | 48.85 | 35.03 |
| 6 | SLRU* | 82.87 | 73.90 | 88.82 | 45.05 | 48.85 | 35.03 |
| 6 | ClassLRU | 83.05 | 73.90 | 89.42 | 45.46 | 48.85 | 35.03 |
| 6 | Static | 41.76 | 41.46 | 37.25 | 15.74 | 36.63 | 35.03 |
| 8 | LRU | 86.50 | 84.23 | 101.66 | 54.27 | 55.25 | 40.70 |
| 8 | SLRU* | 86.60 | 84.23 | 102.09 | 54.59 | 55.25 | 40.70 |
| 8 | ClassLRU | 86.55 | 84.23 | 101.87 | 54.43 | 55.25 | 40.70 |
| 8 | Static | 51.64 | 51.30 | 43.35 | 18.69 | 42.25 | 40.70 |
| 10 | LRU | 87.08 | 87.08 | 104.17 | 56.19 | 56.19 | 50.51 |
| 10 | SLRU* | 87.08 | 87.08 | 104.17 | 56.19 | 56.19 | 50.51 |
| 10 | ClassLRU | 87.08 | 87.08 | 104.17 | 56.19 | 56.19 | 50.51 |
| 10 | Static | 58.49 | 58.43 | 48.84 | 21.44 | 47.62 | 50.51 |

## Calibration

D041 LRU chat anchors, with fixed GPU floor and H2D=28.7 GB/s. One nonnegative upload multiplier fitted in latency space; CPU forecasts use the same multiplier only for initial uploads.

| Cache MiB | Observed t/s | Physics t/s | Fitted t/s | Fit error t/s | Fit error ms | Leave-one-out t/s |
|---:|---:|---:|---:|---:|---:|---:|
| 2200 | 18.10 | 42.90 | 18.46 | +0.36 | -1.08 | 21.82 |
| 6000 | 51.30 | 87.94 | 44.45 | -6.85 | +3.00 | 43.69 |

Upload multiplier: 2.65208; fit RMSE 4.849 t/s, 2.257 ms/token.

## Crossover predictions

Piecewise linear interpolation of latency margins on the preregistered 0.5 GB grid. Whole-layer static K is discrete; crossings are bracketed model forecasts, not hardware measurements. Positive margin means cache wins. All transitions, including reversals, are saved in results.json.

| Policy | Miss mode | Against whole-layer static | Against trained top-frequency static |
|---|---|---|---|
| LRU | upload | 4.162 GB (cache wins, bracket [4.0, 4.5]); 10.879 GB (static wins, bracket [10.5, 11.0]) | 1.059 GB (cache wins, bracket [1.0, 1.5]) |
| LRU | CPU | cache wins at 0.5 GB; 10.879 GB (static wins, bracket [10.5, 11.0]) | wins throughout grid |
| SLRU* | upload | 3.925 GB (cache wins, bracket [3.5, 4.0]); 10.879 GB (static wins, bracket [10.5, 11.0]) | 1.075 GB (cache wins, bracket [1.0, 1.5]) |
| SLRU* | CPU | cache wins at 0.5 GB; 10.879 GB (static wins, bracket [10.5, 11.0]) | wins throughout grid |
| ClassLRU | upload | 3.819 GB (cache wins, bracket [3.5, 4.0]); 10.879 GB (static wins, bracket [10.5, 11.0]) | wins throughout grid |
| ClassLRU | CPU | cache wins at 0.5 GB; 10.879 GB (static wins, bracket [10.5, 11.0]) | wins throughout grid |

## Interpretation and limits

The calibration residual and leave-one-out predictions test whether a fixed miss-price model explains both D041 points. Cold 64-token replay omits D041 prefill warming and follows different greedy continuations. The fitted multiplier absorbs exposed copy overhead, synchronization, and model mismatch; it is not a measured PCIe bandwidth. A fit with large residuals does not validate absolute t/s or a precise crossover.

CPU-share=1 bypasses every miss once a group fills, so its resident set freezes. Its hit rates must not be replaced with upload-mode hit rates. Partial CPU-share can improve admission and may perform better; no share was tuned on evaluation. CPU cost is D039/T-QRT 0.757210 ms per fully CPU layer, scaled by missed bytes, plus 30 us per CPU-active layer. This is an aggregate CPU calibration, not a pure bandwidth measurement; double-counting some boundary cost is possible. GPU work, CPU computation and uploads are serialized in the forecast; overlap and scheduler overhead are unknown.

SLRU* lacks source verification and must not be cited as faithful #30637. LRU and ClassLRU share the pinned grouping/guard allocation for a controlled comparison. LRU/SLRU with CPU-share and Static with upload-on-miss are counterfactual policy/miss-mode combinations, not claims of upstream options. Static top-frequency residents never change, so upload misses are transient staging operations with no resident eviction. Whole-layer static is separately anchored to D041 K30/K42; its two-point fit has zero degrees of freedom and is not independent validation.

See cc/cache_policy/README.md for exact source lines, formulas, reproduction, and missing-source evidence. results.json includes per-domain and per-layer counts, allocations, selected static IDs, source/trace/protocol hashes and crossover margins.
