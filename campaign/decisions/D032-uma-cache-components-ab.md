# D032 — UMA cache components A/B: dedupe works but does not help; direct-fill never activated; cache does not reduce disk reads (2026-11-10, jetson-arm)
Image campaign/llamacpp:jetson-up1108-mc = upstream cf34d311 + cc/moe_cache_profile/patches/combined-01-02-03 (Build CC 519 s). llama-completion p1 prompt, -n 64, -lm mmap -ngl 99 -cmoe -lv 4. experiments/jetson/OM/mc-20261110-0602.
| arm | decode t/s | cache hit | NVMe read GiB/run | refault GiB | min MemAvail MiB | output sha |
|---|---|---|---|---|---|---|
| base-a / base-b | 1.14 / 1.20 | – | 58.1 / 56.7 | 37.2 / 35.9 | 2936 / 3097 | 7a1541… |
| cache 2048 | 0.97 | 64.0% | 60.5 | 15.9 | 953 | dc4b37… |
| cache 2048 + uma-dedupe | 0.89 | 64.0% (22,437 madvise, 14.4 GB, 0 fail) | 61.4 | 14.0 | 1035 | dc4b37… |
| cache 2048 + direct-fill | 0.88 | (fill never used) | 62.7 | 16.4 | 950 | dc4b37… |
| cache 2048 + direct-fill + cpu-share 0.5 | 1.12 | – | 57.6 | 15.3 | 1059 | b7b6e9… |
| cache 1024 + direct-fill | 0.93 | – | – | – | – | dc4b37… |
Findings:
1. **The D030 double-residency hypothesis is not supported**: removing the page-cache copy (dedupe, verified 0 failures) made decode slower, not faster.
2. **The GPU cache does not reduce NVMe traffic at all** (57–63 GiB per run in every arm, ≈2.8x model per run); it only moves hits from page cache to the bank and adds copies. On UMA the page cache already IS the expert cache. GPU util 8–13%: decode is I/O bound.
3. Direct-fill (J-OM6) disabled itself: "not … expert-contiguous or not 4 KiB aligned; mmap fallback" — slices are 4 KiB multiples but the GGUF tensor base offset is (likely) only 32 B aligned; O_DIRECT needs aligned file offsets → fix = read an aligned window and copy the slice (J-OM6b, only if still useful).
4. Cache arms change generated text vs base (GPU computes cached experts, CPU computes the rest in base) → any cache result needs a KL check before counting.
New hypothesis H-RA (testing now): llama.cpp mmap setup issues FADV_SEQUENTIAL + MADV_WILLNEED over the whole file (llama-mmap.cpp:486-525; prefetch=-1 at llama-model-loader.cpp:1442) — on random expert access this means large readahead pollution; `--numa distribute` switches to prefetch=0 + MADV_RANDOM (zero-code test, sweep_numa.sh).
## Addendum: `--numa distribute` test was INVALID (switch inert)
numa-20261110-0622 (base, numa, numa, base): eval 1.17 / 1.19 / 1.20 / 1.22, NVMe 57.0/56.0/55.9/55.4 GiB, identical output. But `is_numa` comes from ggml_backend_cpu_is_numa → ggml_is_numa() = n_nodes > 1 (ggml-cpu.c:734): on single-node Orin the flag does not change the mmap advice. H-RA remains untested → real env-gated patch cc/mmap_advice/patches/01-mmap-advice-random.patch (LLAMA_MMAP_ADVICE=random), pipeline3. Lesson for methodology: verify a "zero-code switch" actually reaches the code path (log line or counter) before reading its A/B.
