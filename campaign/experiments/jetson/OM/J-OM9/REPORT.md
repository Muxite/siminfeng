# J-OM9

Status: completed

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches |
|---|---|---|---:|---:|---|---|
| 1 | base | ok | 1.2 | 151.43 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_fill: diag": 0} |
| 2 | frd16 | ok | 1.28 | 147.14 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: diag": 0} |
| 3 | frd16d | ok | 1.17 | 145.93 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: diag": 4} |
| 4 | frd16p | ok | 1.2 | 155.82 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: diag": 4} |
| 5 | frd16p | ok | 1.2 | 168.87 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: diag": 4} |
| 6 | frd16d | ok | 1.19 | 141.07 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: diag": 4} |
| 7 | frd16 | ok | 1.31 | 149.42 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_fill: diag": 0} |
| 8 | base | ok | 1.19 | 169.55 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_fill: diag": 0} |

Metric: eval_tps (higher is better)

| Arm | Mean | B/A | Gate |
|---|---:|---:|---|
| base | 1.1949999999999998 | — | base |
| frd16 | 1.295 | 1.0836820083682008 | improved |
| frd16d | 1.18 | 0.98744769874477 | null |
| frd16p | 1.2 | 1.0041841004184102 | null |

## Secondary metrics (descriptive; no verdict)

| Arm | NVMe GiB/token | Prompt t/s | Eval t/s |
|---|---:|---:|---:|
| base | 0.8861809382362971 | 5.33 | 1.1949999999999998 |
| frd16 | 0.8106073046487475 | 5.195 | 1.295 |
| frd16d | 0.8104349166627913 | 5.7 | 1.18 |
| frd16p | 0.817302007523794 | 5.705 | 1.2 |

A/A gap: 0.02316602316602319; drift: 0.008368200836820092; threshold: 0.04633204633204638

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["7a1541792364", "7a1541792364"], "frd16": ["7a1541792364", "7a1541792364"], "frd16d": ["7a1541792364", "7a1541792364"], "frd16p": ["7a1541792364", "7a1541792364"]}}`

