# D030 — Over-memory baselines on Jetson; upstream moe-cache is a net loss on UMA (2026-11-10, jetson-arm)
Model Qwen3.6-35B-A3B UD-Q4_K_XL 20.8 GiB on 7.4 GiB shared DRAM (D029). All runs via cc/om_run/om_run.sh (watchdog + io/proc/tegrastats telemetry). Category: runtime-placement / cache-policy observations (no code change of ours yet).
## Baselines, pinned engine (jetson-c1-p03, env unset; llama-bench -p 128 -n 64 -r 2) — experiments/jetson/OM/base-20261110-0448
| config | pp t/s | tg t/s | outcome |
|---|---|---|---|
| -lm mmap -ngl 99 -ncmoe 40 (all experts via mmap/NVMe, rest GPU) | 4.49 | 1.30 | ok |
| -lm mmap -ngl 99 -ncmoe 38 (2 layers' experts on GPU) | 4.67 | 1.21 | ok (no gain) |
| -lm mmap -ngl 0 (CPU only) | 4.09 | 1.09 | ok |
| -lm none -ngl 99 -ncmoe 40 | — | — | watchdog: MemAvailable 401 MiB at 11 s |
| **default (-lm auto) -ngl 99 -ncmoe 40** | — | — | **watchdog: MemAvailable 321 MiB at 7 s** — default cannot run a >RAM model on Orin (auto disables mmap on iGPU) |
| repeat of row 1 | — | — | watchdog tripped on zram swap-rate burst 286 MiB/s with 3.5 GiB available (false positive → swap-rate limit raised to 1000 MiB/s; rerun queued in cmp sweep) |
Earlier 6 attempts failed on llama-bench flags (-c unsupported) — recorded.
## Upstream cf34d311 moe-cache (llama-completion, 800-char wiki prompt, -n 64, -lm mmap -ngl 99 -cmoe) — up-20261110-0436 (clean) + upv-20261110-0519 (with -lv 4 counters; TIMING CONTAMINATED: a CPU-heavy replay simulation ran at nice 0 for part of it)
| config | decode t/s (clean / upv) | decode hit rate | uploaded |
|---|---|---|---|
| no cache | 1.18, 1.19 / 1.21, 1.16 | — | — |
| cache 512 MiB | – / 0.98 | 37.7% | 23.0 GB |
| cache 1024 MiB | 0.91 / – | – | – |
| cache 2048 MiB | 0.98 / 0.98 | **64.0%** | 13.8 GB |
| 2048 + cpu-share 0.5 | 1.13 / – | – | – |
| 2048 + cpu-share 1.0 | – / 0.87 | 36.2% | 2.0 GB |
| 1024 + cpu-share 0.5 | – / 1.02 | 54.7% | 8.9 GB |
Interpretation: the cache achieves real hit rates (64% at 2 GiB ≈ 10% of experts — consistent with replay-sim LRU locality) yet decode is 17–23% slower than no cache. On UMA every miss still page-faults from NVMe AND is then copied into a separate device bank (~215 MB/token of copies at 2 GiB) while the bank's DRAM is subtracted from the page cache that serves misses. On discrete GPUs the upload buys PCIe-free GPU compute; on unified memory it is pure overhead + double residency. CPU-share (misses computed on CPU in place) recovers most of the loss but not more.
## Design consequence (hypothesis to test)
For unified-memory targets: do not copy experts into a separate cache. Keep each byte once (page cache / pinned host memory), let the GPU read hot experts in place (zero-copy mapped memory; Orin: pageableMemoryAccess=0, so pages must be registered/pinned), and spend DRAM on hit rate. Candidates: J-OM5 dedupe (release page-cache copy after upload) and a zero-copy hot set (J-OM6). Pinned vs upstream same-tool comparison pending (cmp sweep).
