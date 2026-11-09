SIMULATION on a logical trace: no residency/transfer semantics; Granite fully fits GPU memory.

Cold caches; token-entry lookups; overall rates weighted by expert requests.

| Phase | Policy | 4 slots | 8 slots | 12 slots | 16 slots | 24 slots |
|---|---|---:|---:|---:|---:|---:|
| prefill | LRU | 18.91% | 43.76% | 55.59% | 68.23% | 88.13% |
| prefill | LFU | 19.70% | 37.67% | 54.05% | 69.82% | 87.68% |
| decode | LRU | 19.79% | 44.69% | 58.55% | 72.17% | 90.97% |
| decode | LFU | 19.10% | 34.48% | 52.06% | 67.50% | 92.20% |
