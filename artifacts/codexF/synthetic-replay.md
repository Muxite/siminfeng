SIMULATION on a logical trace: no residency/transfer semantics; Granite fully fits GPU memory.

Cold caches; token-entry lookups; overall rates weighted by expert requests.

| Phase | Policy | 4 slots | 8 slots | 12 slots | 16 slots | 24 slots |
|---|---|---:|---:|---:|---:|---:|
| prefill | LRU | 41.67% | 72.92% | 72.92% | 72.92% | 72.92% |
| prefill | LFU | 41.67% | 66.67% | 72.92% | 72.92% | 72.92% |
| decode | LRU | 47.22% | 82.64% | 82.64% | 82.64% | 82.64% |
| decode | LFU | 47.22% | 34.72% | 51.39% | 68.06% | 82.64% |
