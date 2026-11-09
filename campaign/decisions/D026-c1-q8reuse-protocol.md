# D026 — C1 q8_reuse (computation-preserving engine change) — protocol frozen BEFORE results
Candidate: cc/q8_reuse/q8-activation-reuse.patch (sha256 9943dee4...), D011 hypothesis H-fuse-q8: reuse one quantize_q8_1 of an activation across consecutive consumers (q/k/v), removing ~2 quantize launches/layer.
Builds (same commit 31288c9c6, same Build CC v2, same base digests, +test-backend-ops): q8base-1 (unpatched), q8reuse-1 (patched; switch GGML_CUDA_Q8_REUSE=0 disables, anything else enables).
Primary comparison: SAME patched binary, reuse OFF vs ON (removes build-variance), lab/tests/c1q8.json, order artifacts/x2/c1-order.txt (2 ABBA cycles per model x {tg,gen}; ppl once each), host-load monitored, no concurrent builds/CPU inference.
Secondary: unpatched q8base-1 vs patched-OFF for gen text identity (patch must be inert when OFF).
Acceptance (frozen, from D011): decode (llama-bench tg128 avg_ts, median over samples) >= +2% for ON vs OFF on Q8_0; AND greedy text sha256 identical ON vs OFF; AND PPL identical (to printed precision) ON vs OFF; AND test-backend-ops MUL_MAT passes on CUDA. mixE4K results reported as a second track with the same rules. < +2% or any non-identity => rejected / inconclusive, reported as such.
Classification if accepted: full-engine code gain (computation-preserving), discrete RTX 5070 Ti only, this model/workload.
