# D033 — Readahead wastes 57% of NVMe bytes but disabling it halves speed: the drive is IOPS-bound for small reads (2026-11-10, jetson-arm)
Patch cc/mmap_advice/patches/01-mmap-advice-random.patch (env LLAMA_MMAP_ADVICE=random: no FADV_SEQUENTIAL, no WILLNEED prefetch, MADV_RANDOM on the mapping), image jetson-up1108-ra (152 s build). llama-completion p1, -n 64, -lm mmap -ngl 99 -cmoe. experiments/jetson/OM/ra-20261110-0635, order base,rand,rand,base; switch verified active by log line (advice_log=1 only in rand arms).
| arm | prompt t/s | decode t/s | NVMe read/run | avg KiB per IO | output |
|---|---|---|---|---|---|
| base-a / base-b | 5.68 / 5.69 | 1.19 / 1.20 | 56.6 / 55.7 GiB | 93 / 95 | 7a1541… |
| random-a / random-b | 1.42 / 1.47 | 0.56 / 0.59 | 24.1 / 23.5 GiB | 4 / 4 | 7a1541… (identical) |
Interpretation:
- ~57% of the bytes the default reads are readahead the computation never uses ("H-RA" readahead pollution: confirmed as byte waste).
- But without readahead every fault is a 4 KiB random read; the drive does ~20 MiB/s at 4K qd1 (D031 fio) → speed halves. The waste was buying IO size. Both extremes are wrong.
- Target design (component J-OM7): MADV_RANDOM on expert tensors + router-informed exact prefetch: when a layer's top-k ids are known, madvise(MADV_WILLNEED) exactly those experts' gate/up/down byte ranges (≈1.9 MB/expert, contiguous slices) so the kernel issues large async reads for useful bytes only. Expected: ~24 GiB useful at expert-sized IO (fio 576K qd8 ≈ 536 MiB/s) ≈ 46 s IO vs 56 GiB at ~400 MiB/s ≈ 140 s → up to ~2x decode if IO dominates (prediction).
- Methodology: "zero-code switch" (--numa) was inert (D032 addendum); the real switch needed a log line to prove it applied.
