# J-XS1-smoke

Status: INVALID

Page-cache protocol: carryover

| # | Arm | Status | Metric | Wall s | Output SHA | Switch matches | Cache start resident % | Cache end resident % |
|---|---|---|---:|---:|---|---|---:|---:|
| 1 | base | watchdog_tripped_killed | None | 222.46 | 2940aeea06d3 | {"expert_fill: active": 1, "expert_pregate: active k=8 ": 1, "expert_readiness: active": 1, "expert_readiness: stats decode_tokens=[1-9]": 0, "readiness=1 join=1$": 0, "join_waits=[0-9]+": 1} | 0.0 | 3.629355654433125 |

Metric: eval_ms_per_token (lower is better)

| Arm | Mean | A/B | Gate |
|---|---:|---:|---|
| base | None | — | base |
| join | None | None | null |

## Secondary metrics (descriptive; no verdict)

| Arm | ms/token | NVMe GiB/token (decode) | gpu_util med (%) | J/token | NVMe GiB/token (whole) | Prompt t/s | Eval t/s | CPU workers R | Main+workers D | Threads majflt/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | — | — | — | — | — | — | — | 0.0423233 | 0.159094 | — |
| join | — | — | — | — | — | — | — | — | — | — |

A/A gap: 0.0; drift: None; threshold: 0.03

Output class: identical

Output details: `{"class": "identical", "hashes": {"base": ["2940aeea06d3"], "join": []}}`

base: switch leaked (switch_lines present)
base: runner rc=0, status=watchdog_tripped_killed
base: missing/nonpositive/nonfinite eval_ms_per_token
base: unsuccessful run
join: missing/nonpositive/nonfinite eval_ms_per_token
base: missing/nonpositive/nonfinite eval_ms_per_token
base: unsuccessful run
join: missing/nonpositive/nonfinite eval_ms_per_token
