SIMULATION on a logical trace: no residency/transfer semantics; Granite fully fits GPU memory.

Cold caches; token-entry lookups; overall rates weighted by expert requests.

| Phase | Policy | 4 slots | 8 slots | 12 slots | 16 slots | 24 slots |
|---|---|---:|---:|---:|---:|---:|
| prefill | LRU | 16.26% | 37.29% | 59.10% | 79.48% | 95.51% |
| prefill | LFU | 19.03% | 40.04% | 58.37% | 73.18% | 93.60% |
| decode | LRU | 15.94% | 36.40% | 58.67% | 78.52% | 93.54% |
| decode | LFU | 22.52% | 48.57% | 65.84% | 80.94% | 94.17% |
