# Memory / quality / speed Pareto report

Input: /home/muk/projects/siminfeng/campaign/artifacts/codexI/pytest-tmp/test_cli_outputs_sorted_markdo0/configs.csv

GPU budget: **0 MiB** (default 700: rounded scaled analogue of 500 GB -> 128 GB on a 2.67 GB model; not a physical capacity prediction).

Pareto objectives: minimize GPU MiB, maximize TG tokens/s, minimize KLD. Only rows with all three objectives are eligible; exact ties all remain optimal. Budget fit is GPU MiB <= the budget, independent of quality availability. Global and budget frontiers are descriptive pooled comparisons, not transferable claims. Class frontiers compare rows only within their labelled class. Discrete-only, mixed and unclassified measurements do not establish transferable savings. No statistical significance is established; workloads, hardware, memory accounting and quality references must be comparable.

| Config | Description | Class | GPU MiB | Host MiB | TG t/s | PP t/s | KLD | Top1 % | PPL ratio | Fits budget | Pareto | Budget Pareto | Class Pareto | Missing quality | Missing objectives | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small\|x | synthetic | transferable | 0 | 100 | 20 | 50 | 0 | 98 | 1.02 | yes | yes | yes | yes | none | none |  | /home/muk/projects/siminfeng/campaign/artifacts/codexI/pytest-tmp/test_cli_outputs_sorted_markdo0/configs.csv:3 |
| large | synthetic | transferable | 900 | 100 | 20 | 50 | 0.1 | 98 | 1.02 | no | no | no | no | none | none |  | /home/muk/projects/siminfeng/campaign/artifacts/codexI/pytest-tmp/test_cli_outputs_sorted_markdo0/configs.csv:2 |
| unknown | synthetic | transferable | NA | 100 | 20 | 50 | 0.1 | 98 | 1.02 | unknown | ineligible | ineligible | ineligible | none | gpu_mib |  | /home/muk/projects/siminfeng/campaign/artifacts/codexI/pytest-tmp/test_cli_outputs_sorted_markdo0/configs.csv:4 |

```text
KLD (lower is better)
0.1 +----------------------------------------------------------
     |:                                                        o
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |:                                                         
     |*                                                         
   0 +----------------------------------------------------------
      0 to 900 GPU MiB (lower is better)
      : budget 0 MiB; * frontier; o other; + overlapping points
Points (table row: GPU MiB, KLD): 1: 0, 0; 2: 900, 0.1
```
