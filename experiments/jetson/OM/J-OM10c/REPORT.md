# J-OM10c

Status: completed

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches |
|---|---|---|---:|---:|---|---|
| 1 | base | ok | 1.2 | 147.65 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 2 | pg8b | ok | 1.4 | 138.1 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 3 | pg8i8 | ok | 1.4 | 143.18 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 4 | pg8i12 | ok | 1.37 | 162.66 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 5 | pg10i8 | ok | 1.4 | 147.4 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 6 | pg10i8 | ok | 1.4 | 147.09 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 7 | pg8i12 | ok | 1.39 | 142.98 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 8 | pg8i8 | ok | 1.38 | 152.26 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 9 | pg8b | ok | 1.37 | 137.59 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 10 | base | ok | 1.18 | 148.17 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_pregate: active": 0, "expert_pregate: stats": 0} |

Metric: eval_tps (higher is better)

| Arm | Mean | B/A | Gate |
|---|---:|---:|---|
| base | 1.19 | — | base |
| pg8b | 1.385 | 1.1638655462184875 | improved |
| pg8i8 | 1.39 | 1.1680672268907564 | improved |
| pg8i12 | 1.38 | 1.1596638655462184 | improved |
| pg10i8 | 1.4 | 1.1764705882352942 | improved |

## Secondary metrics (descriptive; no verdict)

| Arm | NVMe GiB/token | Prompt t/s | Eval t/s |
|---|---:|---:|---:|
| base | 0.8733208671448722 | 5.295 | 1.19 |
| pg8b | 0.8556081983778212 | 5.58 | 1.385 |
| pg8i8 | 0.8659329490056114 | 5.525 | 1.39 |
| pg8i12 | 0.8674981858995225 | 5.675 | 1.38 |
| pg10i8 | 0.8606193179175967 | 5.285 | 1.4 |

A/A gap: 0.021660649819494445; drift: 0.016806722689075647; threshold: 0.04332129963898889

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["7a1541792364", "7a1541792364"], "pg8b": ["7a1541792364", "7a1541792364"], "pg8i8": ["7a1541792364", "7a1541792364"], "pg8i12": ["7a1541792364", "7a1541792364"], "pg10i8": ["7a1541792364", "7a1541792364"]}}`

