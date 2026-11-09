# Memory / quality / speed Pareto report

Input: /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv

GPU budget: **700 MiB** (default 700: rounded scaled analogue of 500 GB -> 128 GB on a 2.67 GB model; not a physical capacity prediction).

Pareto objectives: minimize GPU MiB, maximize TG tokens/s, minimize KLD. Only rows with all three objectives are eligible; exact ties all remain optimal. Budget fit is GPU MiB <= the budget, independent of quality availability. Global and budget frontiers are descriptive pooled comparisons, not transferable claims. Class frontiers compare rows only within their labelled class. Discrete-only, mixed and unclassified measurements do not establish transferable savings. No statistical significance is established; workloads, hardware, memory accounting and quality references must be comparable.

| Config | Description | Class | GPU MiB | Host MiB | TG t/s | PP t/s | KLD | Top1 % | PPL ratio | Fits budget | Pareto | Budget Pareto | Class Pareto | Missing quality | Missing objectives | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixE2K | experts Q2_K / rest Q8_0 | transferable | 948 | NA | 841 | 35602 | 0.771232 | 58.804 | 2.09015 | no | yes | no | yes | none | none | quality collapse | /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv:7 |
| mixE3K | experts Q3_K / rest Q8_0 | transferable | 1064 | NA | 733 | 38426 | 0.135039 | 81.804 | 1.10043 | no | yes | no | yes | none | none |  | /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv:6 |
| mixE4K | experts Q4_K / rest Q8_0 | transferable | 1218 | NA | 640 | 39111 | 0.040017 | 90.471 | 1.02923 | no | yes | no | yes | none | none | same memory as Q4_K_M with half the KLD | /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv:5 |
| q4km | uniform Q4_K_M | transferable | 1222 | NA | 708 | 39498 | 0.077526 | 85.588 | 1.06034 | no | yes | no | yes | none | none |  | /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv:4 |
| q8 | uniform Q8_0 | transferable | 1794 | NA | 783 | 41393 | 0.003389 | 97.039 | 1.00668 | no | yes | no | yes | none | none | job-aa57700ee5 | /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv:3 |
| bf16 | Granite BF16 baseline | transferable | 2984 | NA | 564 | 17172 | 0 | 100 | 1 | no | yes | no | yes | none | none | t/s measured under CPU load from concurrent build: unscored; clean baseline tg 599 pp 17.7k (D008) | /home/muk/projects/siminfeng/campaign/experiments/memory/configs.csv:2 |

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
     |             :         +                                  
     |             :                    *                      *
   0 +----------------------------------------------------------
      0 to 2984 GPU MiB (lower is better)
      : budget 700 MiB; * frontier; o other; + overlapping points
Points (table row: GPU MiB, KLD): 1: 948, 0.771232; 2: 1064, 0.135039; 3: 1218, 0.040017; 4: 1222, 0.077526; 5: 1794, 0.003389; 6: 2984, 0
```
