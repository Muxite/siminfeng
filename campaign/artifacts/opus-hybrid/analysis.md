## L0-23_t8_repack.json

layers 0..23 (24), threads 8, CPU weights CPU_REPACK, types q4_K,q4_K,q4_K, expert 884736 B (0.844 MiB), 352 decode positions x 1 pass(es), synth 200 x 1; UNSCORED/exploratory. All times: sum over the 24 layer(s) per decode step, us.

- correctness: all_checks_ok = **True**; endpoint rel-L2 vs fp64: gpu_ref median 1.16e-02 (max 1.65e-02), cpu_all median 1.83e-02 (max 3.48e-02)
- H2D admission of one expert: pinned median 37.9 us (p95 38.3) = 23.3 GB/s; pageable 90.2 us; bare GPU round trip (4 KB H2D + 4 KB D2H + sync) 6.1 us

### Controlled misses per layer (K=16 static slots, synthetic ids with exactly m misses per layer, real routing weights)

| m/layer | overlap med | p95 | serial med | serial GPU part | serial CPU part | overlap host launch+combine | cpu-only(m) | gpu-only(8-m) | overlap saving / min(gpu,cpu) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1227 | 1238 | 1228 | 1202 | 1 | 601 | 0 | 1203 | nan |
| 1 | 1844 | 2637 | 2176 | 1054 | 1095 | 665 | 1024 | 1042 | 0.32 |
| 2 | 2573 | 3756 | 2832 | 1017 | 1777 | 714 | 1909 | 949 | 0.25 |
| 3 | 3266 | 3529 | 3607 | 1028 | 2555 | 714 | 2512 | 910 | 0.33 |
| 4 | 4257 | 6290 | 4316 | 1007 | 3284 | 741 | 3249 | 897 | 0.06 |
| 5 | 5385 | 5947 | 5559 | 954 | 4547 | 858 | 4115 | 890 | 0.18 |
| 6 | 5354 | 6177 | 5590 | 746 | 4813 | 696 | 4969 | 641 | 0.32 |
| 7 | 6510 | 7862 | 6282 | 843 | 5400 | 836 | 5411 | 725 | -0.27 |
| 8 | 6645 | 7751 | 6224 | 1 | 6176 | 46 | 6151 | 0 | nan |

- CPU-only miss path (m = 1..8 per layer): t = 15.2 us/layer + **30.6 us per CPU expert** (least squares on medians)
- hybrid overlap total: 51.2 us/layer + 30.1 us per miss; serial: 72.8 us/layer + 26.2 us per miss
- predict_slots.py assumed 33.3 us per miss and no per-layer fixed cost

### Trace-driven (352 real decode positions; static = prefill-profile top-K, LRU cold start)

gpu_ref (all experts on GPU) median 1198 us, cpu_all median 6336 us.

| config | hit % | misses/step | med | p95 | pred serial (t0+33.3us*miss) | pred perfect overlap | measured/pred serial | admit between steps (mean) | check |
|---|---|---|---|---|---|---|---|---|---|
| gpu_ref | 100.0 | 0.0 | 1198 | 1208 | 1198 | 1198 | 1.00 | 0 | ok |
| cpu_all | 0.0 | 192.0 | 6336 | 8153 | 7590 | 6392 | 0.83 | 0 | ok |
| hybrid_static_overlap_K8 | 46.1 | 103.6 | 4265 | 4908 | 4694 | 3496 | 0.91 | 0 | ok |
| hybrid_static_serial_K8 | 46.1 | 103.6 | 4640 | 5349 | 4694 | 3496 | 0.99 | 0 | ok |
| hybrid_lru_overlap_K8 | 44.0 | 107.6 | 4519 | 6201 | 4877 | 3679 | 0.93 | 3886 | ok |
| upload_lru_K8 | 44.0 | 107.6 | 4958 | 6643 | 4877 | 3679 | 1.02 | 0 | ok |
| hybrid_static_overlap_K12 | 60.8 | 75.2 | 3501 | 4489 | 3795 | 2597 | 0.92 | 0 | ok |
| hybrid_static_serial_K12 | 60.8 | 75.2 | 3822 | 4669 | 3795 | 2597 | 1.01 | 0 | ok |
| hybrid_lru_overlap_K12 | 60.7 | 75.4 | 3795 | 5707 | 3595 | 2397 | 1.06 | 2786 | ok |
| upload_lru_K12 | 60.7 | 75.4 | 3609 | 5517 | 3595 | 2397 | 1.00 | 0 | ok |
| hybrid_static_overlap_K16 | 73.3 | 51.3 | 2808 | 3277 | 2996 | 1798 | 0.94 | 0 | ok |
| hybrid_static_serial_K16 | 73.3 | 51.3 | 3172 | 3694 | 2996 | 1798 | 1.06 | 0 | ok |
| hybrid_lru_overlap_K16 | 74.4 | 49.2 | 2895 | 4371 | 2729 | 1531 | 1.06 | 1853 | ok |
| upload_lru_K16 | 74.4 | 49.2 | 2685 | 4338 | 2729 | 1531 | 0.98 | 0 | ok |
| hybrid_static_overlap_K24 | 90.8 | 17.6 | 1597 | 1973 | 1797 | 1198 | 0.89 | 0 | ok |
| hybrid_static_serial_K24 | 90.8 | 17.6 | 1861 | 2542 | 1797 | 1198 | 1.04 | 0 | ok |
| hybrid_lru_overlap_K24 | 92.1 | 15.2 | 1621 | 2417 | 1597 | 1198 | 1.01 | 590 | ok |
| upload_lru_K24 | 92.1 | 15.2 | 1562 | 2362 | 1597 | 1198 | 0.98 | 0 | ok |
| hybrid_static_overlap_K32 | 100.0 | 0.0 | 1228 | 1251 | 1198 | 1198 | 1.03 | 0 | ok,bitwise==gpu_ref |
| hybrid_static_serial_K32 | 100.0 | 0.0 | 1228 | 1239 | 1198 | 1198 | 1.03 | 0 | ok,bitwise==gpu_ref |

## L12_t8_repack.json

layers 12..12 (1), threads 8, CPU weights CPU_REPACK, types q4_K,q4_K,q4_K, expert 884736 B (0.844 MiB), 352 decode positions x 2 pass(es), synth 300 x 2; UNSCORED/exploratory. All times: sum over the 1 layer(s) per decode step, us.

- correctness: all_checks_ok = **True**; endpoint rel-L2 vs fp64: gpu_ref median 1.16e-02 (max 1.36e-02), cpu_all median 1.85e-02 (max 2.44e-02)
- H2D admission of one expert: pinned median 38.2 us (p95 38.9) = 23.1 GB/s; pageable 93.8 us; bare GPU round trip (4 KB H2D + 4 KB D2H + sync) 6.1 us

### Controlled misses per layer (K=16 static slots, synthetic ids with exactly m misses per layer, real routing weights)

| m/layer | overlap med | p95 | serial med | serial GPU part | serial CPU part | overlap host launch+combine | cpu-only(m) | gpu-only(8-m) | overlap saving / min(gpu,cpu) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 38 | 39 | 38 | 37 | 0 | 24 | 0 | 43 | nan |
| 1 | 47 | 56 | 61 | 38 | 22 | 24 | 21 | 38 | 0.65 |
| 2 | 54 | 62 | 68 | 38 | 29 | 25 | 29 | 38 | 0.50 |
| 3 | 81 | 90 | 86 | 34 | 51 | 29 | 51 | 37 | 0.16 |
| 4 | 101 | 110 | 103 | 37 | 65 | 35 | 64 | 27 | 0.05 |
| 5 | 92 | 105 | 82 | 28 | 54 | 26 | 67 | 27 | -0.36 |
| 6 | 87 | 107 | 89 | 27 | 62 | 24 | 62 | 25 | 0.09 |
| 7 | 112 | 136 | 100 | 29 | 69 | 29 | 69 | 28 | -0.41 |
| 8 | 115 | 125 | 101 | 0 | 100 | 1 | 109 | 0 | nan |

- CPU-only miss path (m = 1..8 per layer): t = 13.4 us/layer + **10.2 us per CPU expert** (least squares on medians)
- hybrid overlap total: 44.5 us/layer + 9.2 us per miss; serial: 63.4 us/layer + 5.1 us per miss
- predict_slots.py assumed 33.3 us per miss and no per-layer fixed cost

### Trace-driven (352 real decode positions; static = prefill-profile top-K, LRU cold start)

gpu_ref (all experts on GPU) median 38 us, cpu_all median 78 us.

| config | hit % | misses/step | med | p95 | pred serial (t0+33.3us*miss) | pred perfect overlap | measured/pred serial | admit between steps (mean) | check |
|---|---|---|---|---|---|---|---|---|---|
| gpu_ref | 100.0 | 0.0 | 38 | 43 | 38 | 38 | 1.00 | 0 | ok |
| cpu_all | 0.0 | 8.0 | 78 | 106 | 304 | 266 | 0.26 | 0 | ok |
| hybrid_static_overlap_K8 | 44.2 | 4.5 | 85 | 124 | 204 | 166 | 0.42 | 0 | ok |
| hybrid_static_serial_K8 | 44.2 | 4.5 | 88 | 124 | 204 | 166 | 0.43 | 0 | ok |
| hybrid_lru_overlap_K8 | 47.0 | 4.2 | 102 | 147 | 171 | 133 | 0.60 | 148 | ok |
| upload_lru_K8 | 47.0 | 4.2 | 173 | 307 | 171 | 133 | 1.01 | 0 | ok |
| hybrid_static_overlap_K12 | 60.1 | 3.2 | 63 | 98 | 138 | 100 | 0.45 | 0 | ok |
| hybrid_static_serial_K12 | 60.1 | 3.2 | 74 | 104 | 138 | 100 | 0.54 | 0 | ok |
| hybrid_lru_overlap_K12 | 65.3 | 2.8 | 70 | 111 | 104 | 67 | 0.67 | 98 | ok |
| upload_lru_K12 | 65.3 | 2.8 | 107 | 242 | 104 | 67 | 1.03 | 0 | ok |
| hybrid_static_overlap_K16 | 77.9 | 1.8 | 61 | 86 | 104 | 67 | 0.58 | 0 | ok |
| hybrid_static_serial_K16 | 77.9 | 1.8 | 74 | 90 | 104 | 67 | 0.71 | 0 | ok |
| hybrid_lru_overlap_K16 | 79.5 | 1.6 | 53 | 89 | 71 | 38 | 0.74 | 59 | ok |
| upload_lru_K16 | 79.5 | 1.6 | 74 | 206 | 71 | 38 | 1.04 | 0 | ok |
| hybrid_static_overlap_K24 | 95.7 | 0.3 | 39 | 59 | 38 | 38 | 1.03 | 0 | ok |
| hybrid_static_serial_K24 | 95.7 | 0.3 | 39 | 67 | 38 | 38 | 1.03 | 0 | ok |
| hybrid_lru_overlap_K24 | 94.8 | 0.4 | 40 | 63 | 38 | 38 | 1.05 | 16 | ok |
| upload_lru_K24 | 94.8 | 0.4 | 38 | 103 | 38 | 38 | 1.00 | 0 | ok |
| hybrid_static_overlap_K32 | 100.0 | 0.0 | 39 | 41 | 38 | 38 | 1.02 | 0 | ok,bitwise==gpu_ref |
| hybrid_static_serial_K32 | 100.0 | 0.0 | 39 | 42 | 38 | 38 | 1.03 | 0 | ok,bitwise==gpu_ref |

