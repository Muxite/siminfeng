SIMULATION on a logical trace: no residency/transfer semantics; Granite fully fits GPU memory.

Cold caches; token-entry lookups; overall rates weighted by expert requests.

| Phase | Policy | 4 slots | 8 slots | 12 slots | 16 slots | 24 slots |
|---|---|---:|---:|---:|---:|---:|
| prefill | LRU | n/a | n/a | n/a | n/a | n/a |
| prefill | LFU | n/a | n/a | n/a | n/a | n/a |
| decode | LRU | 12.45% | 24.92% | 37.39% | 49.88% | 74.86% |
| decode | LFU | 12.44% | 25.06% | 37.51% | 50.04% | 74.88% |
