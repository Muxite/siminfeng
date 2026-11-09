# T-O completion checkpoint

Completed before the 40-minute checkpoint threshold (within the 60-minute limit).

- Source map: `artifacts/codexO/moe-cache-map.md`.
- Deliverables: `cc/moe_static_hot/{static-hot.patch,make_hot_set.py,test_make_hot_set.py,README.md}`.
- Isolated archive apply/check/reverse check passed; final applied source compiled and linked CPU-only with GCC 13.3.0 and max -j4. 22 pytest cases passed.
- Four real prose-prefill TRAIN hot sets (192/288/384/576 experts) and logical decode holdout reports retained under this directory. These are not engine performance measurements.
- Invariants: env unset retains LRU; init validates IDs/group capacity and fills all projections before use; static mode never emits slot lookups or admits/evicts after init; misses use unchanged selected-expert fallback; reference checkout clean.
- Important boundary: existing fallback uploads misses and computes on GPU; no CPU-miss dispatcher was added. Static hits require D2D staging. Total memory, synchronization and CUDA-graph behavior require GPU validation.
- Exact correctness and matched-memory experiment is in the README. No network, Docker, CUDA build or GPU run performed. All writes remained in the two assigned locations.

Patch SHA256: `0d91a155e727633270cf483f5f03e35d2d58c68b5e7088bea8ea7d5caa1f9880`
