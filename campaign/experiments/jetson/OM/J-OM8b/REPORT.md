# J-OM8b

Status: completed

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches |
|---|---|---|---:|---:|---|---|
| 1 | base | ok | 1.18 | 141.32 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0} |
| 2 | fr | ok | 1.28 | 196.32 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 0} |
| 3 | frd | ok | 1.27 | 138.18 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 4 | frd16 | ok | 1.31 | 137.77 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 5 | frd16 | ok | 1.3 | 138.96 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 6 | frd | ok | 1.28 | 139.39 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 7 | fr | ok | 1.26 | 198.37 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 0} |
| 8 | base | ok | 1.18 | 143.86 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0} |
| 9 | base | ok | 1.19 | 141.83 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0} |
| 10 | fr | ok | 1.25 | 197.41 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 0} |
| 11 | frd | ok | 1.31 | 140.02 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 12 | frd16 | ok | 1.3 | 138.51 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 13 | frd16 | ok | 1.3 | 137.94 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 14 | frd | ok | 1.3 | 137.72 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 1} |
| 15 | fr | ok | 1.31 | 195.57 | 7a1541792364 | {"expert_fill: active": 1, "expert_fill: random advice applied at decode": 0} |
| 16 | base | ok | 1.2 | 143.93 | 7a1541792364 | {"expert_fill: active": 0, "expert_fill: random advice applied at decode": 0} |

Metric: eval_tps (higher is better)

| Arm | Mean | B/A | Gate |
|---|---:|---:|---|
| base | 1.1875 | — | base |
| fr | 1.275 | 1.0736842105263158 | null |
| frd | 1.29 | 1.0863157894736843 | null |
| frd16 | 1.3025 | 1.0968421052631578 | improved |

## Secondary metrics (descriptive; no verdict)

| Arm | NVMe GiB/token | Prompt t/s | Eval t/s |
|---|---:|---:|---:|
| base | 0.8870744856577071 | 5.4925 | 1.1875 |
| fr | 0.4338013331095378 | 1.46 | 1.275 |
| frd | 0.8086544218517485 | 5.5825 | 1.29 |
| frd16 | 0.8073108158414326 | 5.725 | 1.3025 |

A/A gap: 0.04705882352941181; drift: 0.01263157894736834; threshold: 0.09411764705882363

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["7a1541792364", "7a1541792364", "7a1541792364", "7a1541792364"], "fr": ["7a1541792364", "7a1541792364", "7a1541792364", "7a1541792364"], "frd": ["7a1541792364", "7a1541792364", "7a1541792364", "7a1541792364"], "frd16": ["7a1541792364", "7a1541792364", "7a1541792364", "7a1541792364"]}}`

