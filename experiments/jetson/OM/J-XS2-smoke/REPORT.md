# J-XS2-smoke

Status: INVALID

Page-cache protocol: carryover

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches | Cache start resident % | Cache end resident % |
|---|---|---|---:|---:|---|---|---:|---:|
| 1 | base | ok | 1290.3 | 177.45 | 7a1541792364 | {"expert_fill: active": 1, "expert_readiness: stats decode_tokens=[1-9]": 1, "join=1 phase_stats=1$": 0, "join_waits=[0-9]+": 2, "expert_phase: stats decode_tokens=[1-9]": 1} | 0.0 | 8.416174064313006 |

Metric: eval_ms_per_token (lower is better)

| Arm | Mean | A/B | Gate |
|---|---:|---:|---|
| base | 1290.3 | — | base |
| join | None | None | null |

## Secondary metrics (descriptive; no verdict)

| Arm | ms/token | NVMe GiB/token (decode) | gpu_util med (%) | J/token | NVMe GiB/token (whole) | Prompt t/s | Eval t/s | CPU workers R | Main+workers D | Threads majflt/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1290.3 | 0.520126 | 8 | 10.3215 | 1.17489 | 4.99 | 0.78 | 0.0843284 | 0.242404 | 637.873 |
| join | — | — | — | — | — | — | — | — | — | — |

A/A gap: 0.0; drift: None; threshold: 0.03

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["7a1541792364"], "join": []}}`

base: switch leaked (switch_lines present)
join: missing/nonpositive/nonfinite eval_ms_per_token
join: missing/nonpositive/nonfinite eval_ms_per_token
