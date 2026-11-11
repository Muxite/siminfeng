# D031 — Upstream llama.cpp +7% over-memory decode; NVMe is a 570 MiB/s budget drive; first-order decode model (2026-11-10, jetson-arm)
## Engine version (dependency update, not our code) — experiments/jetson/OM/cmp-20261110-0534
llama-bench -p 128 -n 64 -r 2 -lm mmap -ngl 99 -ncmoe 40, Qwen3.6-35B UD-Q4_K_XL, order pin,up,up,pin:
- decode: pin 1.255 / 1.284 (mean 1.27), upstream cf34d311 1.365 / 1.353 (mean 1.36) → **+7%**; prefill pin 4.52/4.24, up 4.77/4.75 → ≈+9%. Caveat: runs partly overlapped a nice-19 replay simulation; pin arms show larger spread. Category: engine-version change (upstream moe/mmq/graph work between Oct 18 and Nov 8), local to Jetson.
## NVMe ceiling (fio on the model file, O_DIRECT, read-only) — experiments/jetson/OM/fio
rand4k qd1 19.6 MiB/s (181 µs) · rand4k qd32 74 MiB/s · rand128k qd8 475 · rand576k qd1 416 · rand576k qd8 536 · rand2m qd4 543 · seq1m qd8 573 MiB/s · 576k psync×4 535.
PCIe link is Gen3 x4 (8 GT/s, x4 — /sys current_link_*), so the limit is the drive: "X15 Lite SSD 128GB" (budget, small capacity). max_hw_sectors 128 KiB, read_ahead 128 KiB. The 825 MiB/s peak seen in D029 telemetry is likely a sampling-interval artefact (to verify).
## First-order model (prediction tool, validated loosely)
decode_tps ≈ disk_BW / (routed_expert_bytes_per_token × miss_rate). With 615 MB/token (layout) and 570 MiB/s: miss_rate 1.0 → ~1.0 t/s; measured 1.27–1.36 → implied miss ≈ 0.70–0.75 (page cache acting as a ~30% hit cache). Replay LRU at 1300–2048 expert slots (2.5–3.9 GB): miss 0.38–0.27 → predicted 2.4–3.4 t/s if the cache holds bytes ONCE and misses read at full drive bandwidth. Our UMA components (dedupe / direct-fill) target exactly that gap. Belady bound at 2048: miss 0.12 → ~7.8 t/s (prefetch headroom).
## Consequences
- For the 128 GB unified + 4 TB SSD target, SSD read bandwidth and access granularity are first-order design inputs; record drive model + fio matrix in every host inventory (added to NEEDS-MAP).
- Page-fault reads (≈20 KB) vs expert-sized reads (576 KB qd8 = 536 MiB/s) matter less on this drive than on a fast NVMe (both near the ceiling); on fast drives large aligned reads should matter more — x86 to measure.
