# Jetson (jetson-arm) experiment summary — campaign c1
All runs: image campaign/llamacpp:jetson-c1 (Build CC record cc/build_engine/results/jetson-c1/build-record.json, archive sha256 323c787c…), llama.cpp 31288c9c6 sm_87, nvpmodel MAXN_SUPER (queried), GPU devfreq governor nvhost_podgov 306–1020 MHz (not pinned). Model bf16 GGUF sha256 bb010fb6… (tensor_data_sha256 e1c1b4d5…, bf16-gguf-manifest.json); x86 file sha af76ade0… — file hashes differ; tensor equality vs x86 UNKNOWN until x86 runs cc/model_manifest/gguf_manifest.py.

| ID | Question | Result | Category |
|---|---|---|---|
| J-E001 | 2-chunk ppl (ctx512, -ngl 99) A/A x5 | 7.5479 ± 0.849 in 5/5 (spread 0). x86 E001: 7.5325 (rel +0.20%) | infrastructure/quality baseline; cross-host diff unexplained (kernels vs weights) |
| J-E002 | llama-bench bf16 pp512/tg128 r5 | pp 1149.9 t/s (samples 1020,1185,1179,1185,1181 — first = clock ramp); tg 50.2 ± 0.08 t/s | local baseline (not comparable to x86 raw rates) |
| J-E003 | Does mmap double DRAM residency on unified memory? (load-mode mmap vs none, 2 pairs, per-file page cache evicted before each) | Cached +2.5 GB (mmap) vs +1.8 GB (none), MemAvailable min drop 1.92 vs 2.02 GB: no loss of available memory from mmap; page cache reclaimable. tg r2 45–49 t/s (too noisy to compare). 4 earlier attempts failed: `-mmp` flag not in this build (recorded) | memory-fit observation, hypothesis not supported; CUDA/nvmap attribution in /proc/meminfo unresolved |
Telemetry for each: *.jsonl via cc/resource_observe.
| J-P001 | Where does decode time go? (nsys) | GPU-kernel bound, CUDA graphs on; bf16 matvec 81%; fused MoE gate/up MMVF ≈40 GB/s vs down MMF ≈68 GB/s | profile / hypothesis H-J1 |
| J-E004 | Screen H-J1 with existing switch GGML_CUDA_DISABLE_FUSION=1 (ABAB, tg128 r5 each) | steady 50.4 → 57.4 t/s (+14%), means 49.69/49.96 vs 56.69/56.51; ppl20 10.0094 vs 10.0096; 64-tok greedy byte-identical | runtime-configuration screen (all fusions off — attribution to gate/up pending patched build) |
