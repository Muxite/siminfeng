# J-ASK1

Status: completed

Page-cache protocol: carryover

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches | Cache start resident % | Cache end resident % |
|---|---|---|---:|---:|---|---|---:|---:|
| 1 | base | ok | 972.59 | 98.14 | cad62aa93e81 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_fill: stats calls=[1-9][0-9]*": 0} | 12.35931795051537 | 12.931976893981219 |
| 2 | fill | ok | 886.44 | 91.89 | cad62aa93e81 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: stats calls=[1-9][0-9]*": 1} | 12.931976893981219 | 14.76070466712697 |
| 3 | fill | ok | 812.6 | 90.56 | cad62aa93e81 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: stats calls=[1-9][0-9]*": 1} | 14.76070466712697 | 14.933425579990493 |
| 4 | base | ok | 865.27 | 98.39 | cad62aa93e81 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_fill: stats calls=[1-9][0-9]*": 0} | 14.933425579990493 | 15.291250408909368 |

Metric: eval_ms_per_token (lower is better)

| Arm | Mean | A/B | Gate |
|---|---:|---:|---|
| base | 918.9300000000001 | — | base |
| fill | 849.52 | 1.081704962802524 | null |

## Secondary metrics (descriptive; no verdict)

| Arm | ms/token | NVMe GiB/token (decode) | gpu_util med (%) | J/token | NVMe GiB/token (whole) | Prompt t/s | Eval t/s | CPU workers R | Main+workers D | Threads majflt/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 918.93 | 0.288565 | 13.25 | 6.49454 | 1.19513 | 0.66 | 1.095 | 0.187098 | 0.441885 | 5753.8 |
| fill | 849.52 | 0.201669 | 17.5 | 6.05683 | 1.10496 | 0.845 | 1.18 | 0.0615999 | 0.207956 | 697.031 |

A/A gap: 0.11678800343878211; drift: 0.11678800343878211; threshold: 0.23357600687756422

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["cad62aa93e81", "cad62aa93e81"], "fill": ["cad62aa93e81", "cad62aa93e81"]}}`

