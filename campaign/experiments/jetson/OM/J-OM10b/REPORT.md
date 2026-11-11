# J-OM10b

Status: completed

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches |
|---|---|---|---:|---:|---|---|
| 1 | base | ok | 1.15 | 143.78 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 2 | frd16 | ok | 1.29 | 139.35 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 3 | pg8b | ok | 1.42 | 134.37 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 4 | pg12b | ok | 1.3 | 139.76 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 5 | pg12b | ok | 1.27 | 170.87 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 6 | pg8b | ok | 1.42 | 140.68 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 7 | frd16 | ok | 1.28 | 140.5 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 8 | base | ok | 1.18 | 143.42 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_pregate: active": 0, "expert_pregate: stats": 0} |

Metric: eval_tps (higher is better)

| Arm | Mean | B/A | Gate |
|---|---:|---:|---|
| base | 1.165 | — | base |
| frd16 | 1.2850000000000001 | 1.1030042918454936 | improved |
| pg8b | 1.42 | 1.2188841201716738 | improved |
| pg12b | 1.2850000000000001 | 1.1030042918454936 | improved |

## Secondary metrics (descriptive; no verdict)

| Arm | NVMe GiB/token | Prompt t/s | Eval t/s |
|---|---:|---:|---:|
| base | 0.8915403759668744 | 5.699999999999999 | 1.165 |
| frd16 | 0.8098085494268508 | 5.67 | 1.2850000000000001 |
| pg8b | 0.852730221218533 | 5.720000000000001 | 1.42 |
| pg12b | 0.9253609672425285 | 5.675 | 1.2850000000000001 |

A/A gap: 0.02575107296137341; drift: 0.02575107296137341; threshold: 0.05150214592274682

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["7a1541792364", "7a1541792364"], "frd16": ["7a1541792364", "7a1541792364"], "pg8b": ["7a1541792364", "7a1541792364"], "pg12b": ["7a1541792364", "7a1541792364"]}}`

