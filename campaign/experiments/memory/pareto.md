# Memory / quality / speed Pareto report

Input: experiments/memory/configs.csv

GPU budget: **700 MiB** (default 700: rounded scaled analogue of 500 GB -> 128 GB on a 2.67 GB model; not a physical capacity prediction).

Pareto objectives: minimize GPU MiB, maximize TG tokens/s, minimize KLD. Only rows with all three objectives are eligible; exact ties all remain optimal. Budget fit is GPU MiB <= the budget, independent of quality availability. Global and budget frontiers are descriptive pooled comparisons, not transferable claims. Class frontiers compare rows only within their labelled class. Discrete-only, mixed and unclassified measurements do not establish transferable savings. No statistical significance is established; workloads, hardware, memory accounting and quality references must be comparable.

| Config | Description | Class | GPU MiB | Host MiB | TG t/s | PP t/s | KLD | Top1 % | PPL ratio | Fits budget | Pareto | Budget Pareto | Class Pareto | Missing quality | Missing objectives | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixE4K+cpu24 | mixE4K all experts on CPU | discrete-only | 576 | 411 | 128 | 7105 | 0.040017 | 90.471 | 1.02923 | yes | yes | yes | yes | none | none | job-36b7630119; attention+embeddings only on GPU; quality carried | experiments/memory/configs.csv:11 |
| q8+cpu24 | Q8_0 all experts on CPU | discrete-only | 576 | 406 | 81 | 4089 | 0.003389 | 97.039 | 1.00668 | yes | yes | yes | yes | none | none | job-36b7630119; quality carried | experiments/memory/configs.csv:13 |
| mixE4K+cpu18 | mixE4K 18 layers on CPU | discrete-only | 734 | 409 | 161 | 8744 | 0.040017 | 90.471 | 1.02923 | no | yes | no | yes | none | none | job-36b7630119; quality carried | experiments/memory/configs.csv:10 |
| mixE4K+cpu12 | mixE4K 12 layers on CPU | discrete-only | 894 | 408 | 212 | 11741 | 0.040017 | 90.471 | 1.02923 | no | yes | no | yes | none | none | job-36b7630119; quality carried | experiments/memory/configs.csv:9 |
| mixE2K | experts Q2_K / rest Q8_0 | transferable | 948 | NA | 841 | 35602 | 0.771232 | 58.804 | 2.09015 | no | yes | no | yes | none | none | quality collapse | experiments/memory/configs.csv:7 |
| mixE4K+cpu6 | mixE4K with 6 layers' experts on CPU (static) | discrete-only | 1056 | 408 | 336 | 18017 | 0.040017 | 90.471 | 1.02923 | no | yes | no | yes | none | none | job-36b7630119; quality carried from n0 (same weights; CPU/GPU kernels differ slightly, NOT re-measured) | experiments/memory/configs.csv:8 |
| mixE3K | experts Q3_K / rest Q8_0 | transferable | 1064 | NA | 733 | 38426 | 0.135039 | 81.804 | 1.10043 | no | yes | no | yes | none | none |  | experiments/memory/configs.csv:6 |
| q8+cpu12 | Q8_0 12 layers on CPU | discrete-only | 1184 | 403 | 146 | 7102 | 0.003389 | 97.039 | 1.00668 | no | yes | no | yes | none | none | job-36b7630119; quality carried | experiments/memory/configs.csv:12 |
| mixE4K | experts Q4_K / rest Q8_0 | transferable | 1218 | NA | 640 | 39111 | 0.040017 | 90.471 | 1.02923 | no | yes | no | yes | none | none | same memory as Q4_K_M with half the KLD | experiments/memory/configs.csv:5 |
| q4km | uniform Q4_K_M | transferable | 1222 | NA | 708 | 39498 | 0.077526 | 85.588 | 1.06034 | no | yes | no | yes | none | none |  | experiments/memory/configs.csv:4 |
| q8 | uniform Q8_0 | transferable | 1794 | NA | 783 | 41393 | 0.003389 | 97.039 | 1.00668 | no | yes | no | yes | none | none | job-aa57700ee5 | experiments/memory/configs.csv:3 |
| bf16 | Granite BF16 baseline | transferable | 2984 | NA | 564 | 17172 | 0 | 100 | 1 | no | yes | no | yes | none | none | t/s measured under CPU load from concurrent build: unscored; clean baseline tg 599 pp 17.7k (D008) | experiments/memory/configs.csv:2 |

```text
KLD (lower is better)
0.771232 +----------------------------------------------------------
     |             :    *                                       
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :                                            
     |             :      *                                     
     |           * :*  *  *  +                                  
     |           * :         *          *                      *
   0 +----------------------------------------------------------
      0 to 2984 GPU MiB (lower is better)
      : budget 700 MiB; * frontier; o other; + overlapping points
Points (table row: GPU MiB, KLD): 1: 576, 0.040017; 2: 576, 0.003389; 3: 734, 0.040017; 4: 894, 0.040017; 5: 948, 0.771232; 6: 1056, 0.040017; 7: 1064, 0.135039; 8: 1184, 0.003389; 9: 1218, 0.040017; 10: 1222, 0.077526; 11: 1794, 0.003389; 12: 2984, 0
```
