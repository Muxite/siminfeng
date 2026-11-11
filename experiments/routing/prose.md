SIMULATION on a logical trace: no residency/transfer semantics; Granite fully fits GPU memory.

Cold caches; token-entry lookups; overall rates weighted by expert requests.

| Phase | Policy | 4 slots | 8 slots | 12 slots | 16 slots | 24 slots |
|---|---|---:|---:|---:|---:|---:|
| prefill | LRU | 20.37% | 47.95% | 59.45% | 70.90% | 89.87% |
| prefill | LFU | 18.68% | 39.18% | 55.64% | 68.77% | 90.26% |
| decode | LRU | 20.82% | 49.06% | 59.33% | 69.73% | 87.76% |
| decode | LFU | 20.06% | 44.10% | 59.71% | 72.16% | 89.34% |
