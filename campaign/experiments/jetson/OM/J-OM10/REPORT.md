# J-OM10

Status: completed

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches |
|---|---|---|---:|---:|---|---|
| 1 | base | ok | 1.19 | 152.38 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 2 | frd16 | ok | 1.29 | 143.29 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 3 | pg8 | ok | 1.24 | 141.16 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 4 | pg16 | ok | 0.75 | 174.75 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 5 | pg16 | ok | 0.75 | 173.48 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 6 | pg8 | ok | 1.22 | 140.51 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 1, "expert_pregate: stats": 1} |
| 7 | frd16 | ok | 1.29 | 138.33 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1, "expert_pregate: active": 0, "expert_pregate: stats": 0} |
| 8 | base | ok | 1.18 | 148.09 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0, "expert_pregate: active": 0, "expert_pregate: stats": 0} |

Metric: eval_tps (higher is better)

| Arm | Mean | B/A | Gate |
|---|---:|---:|---|
| base | 1.185 | — | base |
| frd16 | 1.29 | 1.0886075949367089 | improved |
| pg8 | 1.23 | 1.0379746835443038 | improved |
| pg16 | 0.75 | 0.6329113924050632 | regressed |

## Secondary metrics (descriptive; no verdict)

| Arm | NVMe GiB/token | Prompt t/s | Eval t/s |
|---|---:|---:|---:|
| base | 0.885293445889912 | 5.6899999999999995 | 1.185 |
| frd16 | 0.8086656540159196 | 5.595 | 1.29 |
| pg8 | 0.8718466077532088 | 5.67 | 1.23 |
| pg16 | 1.2145396035815041 | 5.5649999999999995 | 0.75 |

A/A gap: 0.016260162601626032; drift: 0.008438818565400852; threshold: 0.032520325203252064

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["7a1541792364", "7a1541792364"], "frd16": ["7a1541792364", "7a1541792364"], "pg8": ["7a1541792364", "7a1541792364"], "pg16": ["7a1541792364", "7a1541792364"]}}`

