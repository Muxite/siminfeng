# D005 — 2-chunk PPL smoke gate calibration (E001, 5 A/A attempts)
- Evidence: experiments/E001-ppl-baseline/attempts/a1..a5/{result.json,stdout.txt}; ops in evidence/orchestrator.jsonl.
- Observed: PPL = 7.5325 in all 5 attempts (spread 0.0000%), each ~2.8 s wall, all owned containers removed, GPU lock uncontended.
- Reported statistical error of the metric itself: +/- 0.84642 (~11%) over 2 chunks.
- Interpretation: the gate is a DETERMINISM/regression tripwire for same-weights, same-precision, same-kernels candidates (a 1% relative change is ~100x the observed run noise but ~1/11 of the metric's own sampling error). It cannot establish equivalence of a candidate that changes numerics or routing; those need paired token-level comparison (logit/top-k/KL) and routing-ID equality.
- Frozen policy (provisional): candidate passes smoke if |PPL_c - 7.5325| / 7.5325 <= 1% AND same engine/weights/precision; any other change class must add the targeted check from the test catalog.
- Limit: determinism measured on one GPU state/driver/container; not across GPU clocks, other hardware, or thread counts (CPU-offload variants may differ).
- Would change: nondeterminism appearing under CPU offload or multi-sequence batching -> re-calibrate with A/A for each configuration.
