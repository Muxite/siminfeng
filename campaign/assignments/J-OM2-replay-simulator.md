# Assignment J-OM2 — generic expert-cache replay simulator (port pattern from Strata; parent over-memory track; worker Codex)
Base: branch jetson/c1. Allowed: new `cc/expert_replay/**` only. Python 3.10 stdlib (+numpy optional, fallback to pure python). No GPU/network.
Reference (MIT, read-only): ref/Strata/tools/bench_eviction.py, tools/replay_cache_policy.py, tools/make_profile.py, tools/make_profile_mix.py (hardcode NL,NE = 48,512). Our trace format: read cc/routing_trace/ (README + replay code) and support it as the primary input; also accept a simple JSONL `{"tok":i,"layer":l,"experts":[...],"phase":"prefill|decode"}`.
Deliver `expert_replay.py`:
- `simulate --trace T --layers L --experts E --capacity-experts C [--per-layer-capacity] --policy lru|lfu|lfu-decay:<half-life>|static-profile:<profile.json>|belady --warmup N --split train:test` → hit rate overall/per layer/decode-only, misses per token, bytes moved per token given `--expert-bytes B` (and optional per-layer bytes), and a predicted decode time per token = compute_s + misses*bytes/bw_bytes_s with `--bw-gbs` and `--compute-ms` inputs (label output "prediction").
- `profile --trace T --out profile.json [--weighting raw|per-request]` → ranked (layer,expert) list (state which weighting; Strata differs).
- `sweep` over capacities producing CSV.
- Belady (offline oracle) clearly labelled as reads-the-future upper bound.
- L/E from flags or trace metadata — never hardcoded; validate expert ids < E.
Tests: tiny synthetic traces with hand-computed hit counts for each policy; train/test split leakage check; belady ≥ every online policy; malformed/incomplete trace lines → counted + reported (Strata parser returned empty data on a truncated tail — we must report truncation explicitly).
Credit Strata (MIT) in README for the ideas; do not copy files wholesale. Time 40 min. Final ≤ 12 lines.
