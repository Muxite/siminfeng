# D022 — D020 completed + matched static layer offload vs engine LRU expert cache (x86-main, c2-resume)
Claim category: runtime-placement/offload trade-off between two EXISTING llama.cpp 31288c9c6 features. Not an engine-code gain.
Conditions: image llamacpp-b0 (sha256:cf383f82...), mixE4K (sha256 2e520c4a...; hashed by orchestrator op, lab spec now names mixE4K), chat prompt, -n 160, -c 4096, temp 0, seed 1, --no-warmup, -ngl 99; one GPU job; host-load monitor: no compiler active (artifacts/x2/*hostload.log; first monitor self-matched its own pattern, verified + fixed; Codex O's objects all predate the D020 window).
1. D020 rerun: job-8df3c7bb27 att-2 completed 10/10 under the T-M runner (2cf2925); st_prose_n18 flagged invalid_utf8 (the original crash site) and still scored (completion parser policy).
2. Matched job job-ad883810b3 (lab@2cf2925, 9 configs x 3 rounds, order alternated per round). Medians of decode t/s [3 samples]:
   static n24 742 MiB 128 | n18 902 MiB 152 | n12 1064 MiB 197 | n6 1226 MiB 323 | n0 1388 MiB 684
   moe-cache c128 870 MiB 54 | c256 998 MiB 99 | c384 1126 MiB 187 | c640 1382 MiB 458
   At every matched footprint static layer offload is faster (≈2.8x at ~880 MiB, ≈2x at ~1030, ≈1.7x at ~1180, 1.5x at ~1385). Output text identical (stdout sha256 equal) across all configs.
3. Interpretation (consistent with D019, not proven by it): LRU-with-admission pays H2D on every miss; layer offload pays none. Neither is expert-granular AND admission-free -> T-O static hot-expert candidate is the untested quadrant.
Limits: one prompt, decode only (prefill not compared), single host, discrete-only (PCIe). 3 samples/config.
