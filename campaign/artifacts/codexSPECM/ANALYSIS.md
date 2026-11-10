# T-SPECM — Expert-union speculation model

CPU-only analysis of local evidence. Writes are confined to `cc/spec_model/**` and `artifacts/codexSPECM/**`. No planner, engine, calibration or source trace was modified.

## Evidence and window definition

Qwen3-30B-A3B Q4_K_M: 48 layers, 128 experts/layer, top_k=8. Each domain has one 64-token decode continuation. Only records with phase `decode` enter the analysis; prefill is excluded. Every decode token must have all layers, exactly eight distinct in-range expert IDs and no duplicate token/layer. Records are ordered by sequence and position, and gaps/sequence boundaries split runs. For each layer and d=1..16, take every overlapping consecutive length-d window (65-d windows/domain), count the union of expert IDs, then average. Overlapping windows are correlated; these are descriptive curves, not independent confidence samples. Pooled curves weight all within-domain windows equally and never concatenate domains. Prose, chat and code are all descriptive here; none estimates a workload acceptance rate. Hashes and run lengths are recorded in `results.json`.

| d | naive d×8 | prose | chat | code | pooled |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8.000 | 8.000 | 8.000 | 8.000 |
| 2 | 16 | 12.375 | 12.988 | 12.360 | 12.575 |
| 3 | 24 | 16.034 | 17.022 | 15.734 | 16.263 |
| 4 | 32 | 19.131 | 20.427 | 18.668 | 19.409 |
| 5 | 40 | 21.972 | 23.480 | 21.199 | 22.217 |
| 6 | 48 | 24.535 | 26.233 | 23.454 | 24.741 |
| 7 | 56 | 26.805 | 28.693 | 25.481 | 26.993 |
| 8 | 64 | 28.876 | 31.021 | 27.330 | 29.076 |
| 9 | 72 | 30.773 | 33.156 | 29.077 | 31.002 |
| 10 | 80 | 32.444 | 35.157 | 30.662 | 32.755 |
| 11 | 88 | 34.039 | 36.971 | 32.159 | 34.390 |
| 12 | 96 | 35.505 | 38.628 | 33.574 | 35.902 |
| 13 | 104 | 36.735 | 40.077 | 34.896 | 37.236 |
| 14 | 112 | 37.921 | 41.356 | 36.162 | 38.480 |
| 15 | 120 | 39.049 | 42.531 | 37.353 | 39.644 |
| 16 | 128 | 40.029 | 43.668 | 38.452 | 40.717 |

Full layer curves: `union-per-layer.csv`; aggregate curves: `union-curves.csv`.

## Latency and accepted-token convention

Let U_l(d) be the layer union mean, B_t the actual full expert tensor bytes, and E=128. Static K offloads the first K expert layers. For each tensor in layer l, active bytes per step = B_t × U_l(d)/E = PL4 single-token bytes × U_l(d)/8. Sum CPU bytes by quantization type for l<K and GPU expert bytes for l≥K. All nonexpert weights are on GPU and read once per step, as dense weight reuse permits; they have no expert union. The CSV also includes a conservative literal whole-GPU scaling sensitivity, multiplying nonexpert bytes by mean(U)/8.

`T_us(K,d) = Σ_type CPU_bytes(type)/(BW_CPU(type)×1000) + GPU_bytes/(BW_GPU×1000) + K×handoff_us + draft_us + d×extra_us_per_token`

`accepted_drafts_per_step = a×d; predicted_tps = 1e6×a×d/T_us; speedup = a×d×T_us(K,1)/T_us(K,d)`

**a is the observed aggregate accepted/proposed draft fraction on engaged steps**, so the numerator is exactly expected accepted drafts when step length is fixed. Bonus/correction tokens and unengaged ordinary steps are excluded from the main grid. Thus d=1,a=1 is the ordinary PL4 baseline; d=1,a<1 counts only accepted drafts and is not an end-to-end decoder throughput prediction. If a instead means independent conditional prefix survival, expected accepted drafts are `Σ_(i=1..d) a^i`, not a×d. Substitute that numerator (or its implied aggregate fraction) rather than interpreting this grid as that different process. A bonus-token model needs explicit verification-token accounting; no free extra token is credited here.

Tables assume draft_us=0 and extra_us_per_token=0, once-only expert weight reads within a batch and no new compute/KV bottleneck. All d proposed positions cost traffic even when rejected. Actual rejected-path routing may differ from the observed accepted continuation, especially at a=0.5; this grid holds U(d) fixed as an approximation. Repeated expert execution, output projection arithmetic, attention, activation traffic, KV allocation, drafting and sampling overhead can lower performance. This is a bandwidth forecast, not measured speculative throughput.

PL4 constants: GPU 285.227 GB/s; CPU Q4_K/Q6_K 35.887 GB/s; handoff 136.641 µs/CPU layer. Q6_K shares Q4_K bandwidth as in PL4. The model retains each layer’s actual Q4_K/Q6_K tensor bytes. No additional 1.02 Q8-reuse multiplier is applied. PL4 has no other identifiable fixed per-step cost; inventing an intercept or adding a measured baseline residual would double-count or refit it.

| K | ordinary PL4 t/s | CPU ms | GPU ms | handoff ms |
| --- | --- | --- | --- | --- |
| 18 | 50.24 | 11.554 | 5.890 | 2.460 |
| 30 | 35.63 | 19.016 | 4.951 | 4.099 |
| 42 | 27.60 | 26.478 | 4.012 | 5.739 |
| 48 | 24.61 | 30.570 | 3.497 | 6.559 |

K=0 is omitted: weights alone are 17691.34 MiB, exceeding the recorded 16303 MiB GPU before KV/runtime buffers. K=18,30,42,48 have measured static precedents (D039). This does not establish their speculative batch buffer fit; the planner must check that separately.

## Predicted accepted-draft tokens/s

Each table gives the full requested K×d×a grid. All step components, speedups, whole-GPU scaling sensitivity and break-even acceptance are in `predictions.csv`.

### prose

| K | a | d=1 | d=2 | d=3 | d=4 | d=8 | d=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0.5 | 25.1 | 36.0 | 43.5 | 49.6 | 67.9 | 99.7 |
| 18 | 0.8 | 40.2 | 57.6 | 69.6 | 79.4 | 108.7 | 159.5 |
| 18 | 1.0 | 50.2 | 72.0 | 87.0 | 99.2 | 135.9 | 199.4 |
| 30 | 0.5 | 17.8 | 25.5 | 30.9 | 35.3 | 48.6 | 71.3 |
| 30 | 0.8 | 28.5 | 40.8 | 49.4 | 56.5 | 77.7 | 114.1 |
| 30 | 1.0 | 35.6 | 51.0 | 61.7 | 70.6 | 97.2 | 142.6 |
| 42 | 0.5 | 13.8 | 19.7 | 23.8 | 27.2 | 37.5 | 55.1 |
| 42 | 0.8 | 22.1 | 31.5 | 38.0 | 43.5 | 59.9 | 88.1 |
| 42 | 1.0 | 27.6 | 39.4 | 47.6 | 54.4 | 74.9 | 110.1 |
| 48 | 0.5 | 12.3 | 17.4 | 20.9 | 23.9 | 33.0 | 48.7 |
| 48 | 0.8 | 19.7 | 27.8 | 33.5 | 38.3 | 52.8 | 77.9 |
| 48 | 1.0 | 24.6 | 34.8 | 41.9 | 47.9 | 66.0 | 97.4 |

### chat

| K | a | d=1 | d=2 | d=3 | d=4 | d=8 | d=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0.5 | 25.1 | 34.9 | 41.9 | 47.6 | 65.2 | 94.5 |
| 18 | 0.8 | 40.2 | 55.9 | 67.0 | 76.2 | 104.4 | 151.2 |
| 18 | 1.0 | 50.2 | 69.9 | 83.7 | 95.2 | 130.5 | 189.0 |
| 30 | 0.5 | 17.8 | 24.5 | 29.3 | 33.4 | 45.7 | 66.2 |
| 30 | 0.8 | 28.5 | 39.2 | 46.9 | 53.4 | 73.2 | 105.9 |
| 30 | 1.0 | 35.6 | 49.0 | 58.6 | 66.7 | 91.4 | 132.3 |
| 42 | 0.5 | 13.8 | 18.9 | 22.5 | 25.6 | 35.0 | 50.7 |
| 42 | 0.8 | 22.1 | 30.2 | 36.0 | 40.9 | 56.0 | 81.1 |
| 42 | 1.0 | 27.6 | 37.7 | 45.0 | 51.2 | 70.0 | 101.4 |
| 48 | 0.5 | 12.3 | 16.7 | 19.9 | 22.6 | 31.0 | 45.0 |
| 48 | 0.8 | 19.7 | 26.8 | 31.9 | 36.2 | 49.6 | 72.1 |
| 48 | 1.0 | 24.6 | 33.5 | 39.9 | 45.3 | 62.0 | 90.1 |

### code

| K | a | d=1 | d=2 | d=3 | d=4 | d=8 | d=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0.5 | 25.1 | 35.5 | 43.1 | 49.3 | 69.1 | 100.3 |
| 18 | 0.8 | 40.2 | 56.8 | 69.0 | 78.9 | 110.6 | 160.4 |
| 18 | 1.0 | 50.2 | 71.0 | 86.3 | 98.6 | 138.2 | 200.5 |
| 30 | 0.5 | 17.8 | 25.3 | 30.9 | 35.4 | 50.1 | 73.0 |
| 30 | 0.8 | 28.5 | 40.4 | 49.4 | 56.6 | 80.1 | 116.8 |
| 30 | 1.0 | 35.6 | 50.5 | 61.7 | 70.8 | 100.2 | 146.1 |
| 42 | 0.5 | 13.8 | 19.6 | 24.0 | 27.6 | 39.1 | 57.1 |
| 42 | 0.8 | 22.1 | 31.4 | 38.4 | 44.1 | 62.6 | 91.3 |
| 42 | 1.0 | 27.6 | 39.2 | 48.0 | 55.1 | 78.2 | 114.2 |
| 48 | 0.5 | 12.3 | 17.4 | 21.3 | 24.5 | 34.8 | 50.7 |
| 48 | 0.8 | 19.7 | 27.9 | 34.1 | 39.2 | 55.6 | 81.1 |
| 48 | 1.0 | 24.6 | 34.9 | 42.6 | 49.0 | 69.6 | 101.4 |

### pooled

| K | a | d=1 | d=2 | d=3 | d=4 | d=8 | d=16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0.5 | 25.1 | 35.5 | 42.8 | 48.8 | 67.4 | 98.1 |
| 18 | 0.8 | 40.2 | 56.7 | 68.5 | 78.1 | 107.8 | 156.9 |
| 18 | 1.0 | 50.2 | 70.9 | 85.6 | 97.7 | 134.8 | 196.2 |
| 30 | 0.5 | 17.8 | 25.1 | 30.3 | 34.7 | 48.1 | 70.0 |
| 30 | 0.8 | 28.5 | 40.1 | 48.5 | 55.4 | 76.9 | 112.1 |
| 30 | 1.0 | 35.6 | 50.2 | 60.7 | 69.3 | 96.1 | 140.1 |
| 42 | 0.5 | 13.8 | 19.4 | 23.4 | 26.7 | 37.1 | 54.1 |
| 42 | 0.8 | 22.1 | 31.0 | 37.5 | 42.8 | 59.4 | 86.6 |
| 42 | 1.0 | 27.6 | 38.8 | 46.8 | 53.5 | 74.2 | 108.3 |
| 48 | 0.5 | 12.3 | 17.2 | 20.7 | 23.7 | 32.9 | 48.0 |
| 48 | 0.8 | 19.7 | 27.5 | 33.1 | 37.9 | 52.6 | 76.8 |
| 48 | 1.0 | 24.6 | 34.4 | 41.4 | 47.3 | 65.7 | 96.1 |

## D042 check: implied effective mean draft length

D042 §1 reports rewrite 35.6→85.2 t/s (rounded 2.4×), repeat 35.2→106.6 (rounded 3.0×), all proposed drafts accepted, K=30. The raw PL4 baseline is 35.63 t/s. Invert the a=1 curve. Between adjacent integer lengths, mix step times and token rewards to solve for a two-length mixture; do not linearly interpolate throughput. Relative inversion uses the rounded 2.4×/3.0×; absolute inversion uses 85.2/106.6 t/s directly.

| trace proxy | workload | relative implied d | absolute implied d |
| --- | --- | --- | --- |
| prose | rewrite | 6.12 | 6.08 |
| prose | repeat | 9.65 | 9.60 |
| chat | rewrite | 6.92 | 6.86 |
| chat | repeat | 11.04 | 10.98 |
| code | rewrite | 5.87 | 5.83 |
| code | repeat | 9.07 | 9.03 |
| pooled | rewrite | 6.28 | 6.24 |
| pooled | repeat | 9.88 | 9.83 |

This is a consistency check, not independent validation: these three routing continuations are not the D042 rewrite/repeat prompts, and D042 does not report the length distribution or step latencies. Acceptance ≈100% does not mean speculation engages on every step. Ordinary fallback steps or positive draft overhead would generally require longer engaged batches than these idealized effective lengths. A mean alone cannot identify E[U(D)] for a broad distribution. Do not infer observed draft lengths from these inversions. Capture per-step proposed/accepted lengths, target verification time, draft time and engagement alongside D042 timing counters to test the mechanism.

## Recommended planner `spec` lever

Use `ngram-mod` as a workload-gated candidate for copying, rewrite/refactor and repeat tasks. Require fit for the model, KV and chosen verification batch. Log draft_n and draft_n_accepted; estimate aggregate a by draft length and the fraction rho of output tokens produced during engaged steps. Open chat in D042 had zero drafts: select no speculative speedup when rho=0. Measure drafting and incremental per-token compute costs rather than assuming they vanish.

For each feasible d=2..16, compute the worst supplied domain speedup `s(d)=min_domain a(d)×d×T1/[T(d)+draft_us(d)+extra_us_per_token×d]`. Use `S(d)=1/[1-rho+rho/s(d)]` for a token-fraction engagement mixture. Enable only if the best S≥1.10 (an explicit engineering margin), and choose its d as draft_max. Recompute with observed a(d); a fixed acceptance fraction at every d is only the requested sensitivity grid. If the optimum reaches 16, it is the largest evaluated length, not a demonstrated global optimum. Do not extrapolate past 16 from these 64-token traces.

| K | a | enable | draft_max | worst-domain gain (rho=1, zero extra costs) |
| --- | --- | --- | --- | --- |
| 18 | 0.5 | True | 16 | 1.88× |
| 18 | 0.8 | True | 16 | 3.01× |
| 18 | 1.0 | True | 16 | 3.76× |
| 30 | 0.5 | True | 16 | 1.86× |
| 30 | 0.8 | True | 16 | 2.97× |
| 30 | 1.0 | True | 16 | 3.71× |
| 42 | 0.5 | True | 16 | 1.84× |
| 42 | 0.8 | True | 16 | 2.94× |
| 42 | 1.0 | True | 16 | 3.67× |
| 48 | 0.5 | True | 16 | 1.83× |
| 48 | 0.8 | True | 16 | 2.93× |
| 48 | 1.0 | True | 16 | 3.66× |

Treat this as an analytical candidate rule requiring an end-to-end check. Choose the smaller feasible length or disable if measured acceptance, engagement or added compute removes the margin. No existing planner files were changed; `cc/spec_model/model.py:recommend` provides the isolated rule.

## Reproduction and tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 cc/spec_model/run.py
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider --basetemp=artifacts/codexSPECM/pytest-tmp cc/spec_model > artifacts/codexSPECM/pytest.txt
```

All analysis uses only Python’s standard library. Tests cross-check d=1 against PL4, brute-force real trace unions, sequence/gap handling, type-aware byte accounting, rejection numerator semantics, inversion and lever gates.
