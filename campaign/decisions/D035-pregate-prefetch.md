# D035 — Router prediction works; demand-first prefetch scheduling turns it into a gain (2026-11-11, jetson-arm)
`cc/mmap_advice/patches/06-pregate.patch` copies router weights from a read-only private file mapping into CPU-owned storage. The host callback uses the current normalized activation with the next layer's router to predict expert IDs, then queues asynchronous expert preads. It does not read a GPU weight pointer or change expert selection. Supported scope is qwen35moe without expert groups, LoRA or folded rotations.

| Experiment / arm | n | Mean decode t/s | B/A vs base | A/A gap | Gate |
|---|---:|---:|---:|---:|---|
| J-OM10 / base | 2 | 1.185 | — | 0.0163 | base |
| J-OM10 / frd16 | 2 | 1.29 | 1.09 | 0.0163 | improved |
| J-OM10 / pg8 | 2 | 1.23 | 1.04 | 0.0163 | improved |
| J-OM10 / pg16 | 2 | 0.75 | 0.633 | 0.0163 | regressed |
| J-OM10b / base | 2 | 1.165 | — | 0.0258 | base |
| J-OM10b / frd16 | 2 | 1.285 | 1.1 | 0.0258 | improved |
| J-OM10b / pg8b | 2 | 1.42 | 1.22 | 0.0258 | improved |
| J-OM10b / pg12b | 2 | 1.285 | 1.1 | 0.0258 | improved |
| J-OM10c / base | 2 | 1.19 | — | 0.0217 | base |
| J-OM10c / pg8b | 2 | 1.385 | 1.16 | 0.0217 | improved |
| J-OM10c / pg8i8 | 2 | 1.39 | 1.17 | 0.0217 | improved |
| J-OM10c / pg8i12 | 2 | 1.38 | 1.16 | 0.0217 | improved |
| J-OM10c / pg10i8 | 2 | 1.4 | 1.18 | 0.0217 | improved |

Sources (n counts the listed `inputs` observations; A/A is the report-wide maximum gap): `experiments/jetson/OM/J-OM10/verdict.json` and `experiments/jetson/OM/J-OM10/REPORT.md`, `experiments/jetson/OM/J-OM10b/verdict.json` and `experiments/jetson/OM/J-OM10b/REPORT.md`, `experiments/jetson/OM/J-OM10c/verdict.json` and `experiments/jetson/OM/J-OM10c/REPORT.md`.

Prediction and IO counters below are individual runs, not arm means. Cancelled bytes are queued speculative work retired before dispatch, not disk bytes actually read; async bytes are completed speculative reads and can overlap demand reads.

| Experiment / run | recall@k | Precision | Async bytes read | Cancelled bytes | Source |
|---|---:|---:|---:|---:|---|
| J-OM10 / 003-pg8 | 0.809320 | 0.809320 | 10231021568 | not instrumented | `experiments/jetson/OM/J-OM10/003-pg8/stderr.txt` (`expert_pregate: stats`) |
| J-OM10 / 004-pg16 | 0.955383 | 0.477691 | 28629131264 | not instrumented | `experiments/jetson/OM/J-OM10/004-pg16/stderr.txt` (`expert_pregate: stats`) |
| J-OM10b / 003-pg8b | 0.809320 | 0.809320 | 7779328000 | 4354129920 | `experiments/jetson/OM/J-OM10b/003-pg8b/stderr.txt` (`expert_pregate: stats`) |
| J-OM10b / 004-pg12b | 0.921093 | 0.614062 | 10992332800 | 14790897664 | `experiments/jetson/OM/J-OM10b/004-pg12b/stderr.txt` (`expert_pregate: stats`) |
| J-OM10c / 002-pg8b | 0.809320 | 0.809320 | 7764221952 | 4332380160 | `experiments/jetson/OM/J-OM10c/002-pg8b/stderr.txt` (`expert_pregate: stats`) |
| J-OM10c / 003-pg8i8 | 0.809320 | 0.809320 | 9136136192 | 1698582528 | `experiments/jetson/OM/J-OM10c/003-pg8i8/stderr.txt` (`expert_pregate: stats`) |
| J-OM10c / 004-pg8i12 | 0.809320 | 0.809320 | 9576804352 | 1114234880 | `experiments/jetson/OM/J-OM10c/004-pg8i12/stderr.txt` (`expert_pregate: stats`) |
| J-OM10c / 005-pg10i8 | 0.809320 | 0.809320 | 9014321152 | 1775513600 | `experiments/jetson/OM/J-OM10c/005-pg10i8/stderr.txt` (`expert_pregate: stats`) |

Interpretation:
- High recall alone did not win. J-OM10 pg8 loses to fill-only frd16; pg16 regresses even against base. The first hook queued next-layer IO before current-layer demand; stale queued work survived arrival, and async reads could occupy the whole worker pool. This created competing and duplicate IO (`06-pregate.patch`; `c2/worklog.jsonl`, J-OM10-ab result).
- `cc/mmap_advice/patches/07-pregate-sched.patch` changes the default order to demand first, then prediction/enqueue. Arrival cancels queued work for the current and older layers, and each new token retires the previous backlog. Demand reserves priority before residency queries, and an async occupancy cap leaves workers available. Already running preads finish, so duplicate IO remains possible. The callback never waits for async completion; teardown does.
- Keep pg8b as the best supported configuration: D034 fill settings plus `LLAMA_EXPERT_PREGATE=1 LLAMA_EXPERT_PREGATE_K=8 LLAMA_EXPERT_PREGATE_ORDER=after LLAMA_EXPERT_PREGATE_MAX_INFLIGHT=4` (`experiments/jetson/OM/J-OM10b/spec.json`, `J-OM10c/spec.json`). J-OM10b gives 1.42 t/s in both repeats, versus fill-only 1.29/1.28. J-OM10c reproduces an improved result, but raising occupancy gives no clear incremental win within the reported noise. Each gate compares against base, not against the other candidates.
- **Configuration correction:** J-OM10c's `pg10i8` requests K=10, but the parser in `06-pregate.patch` only accepts 12 or 16 and otherwise keeps 8; `07-pregate-sched.patch` does not change this. Its `005-pg10i8/summary.json` and stderr explicitly log `active k=8`. Its highest observed mean therefore does not establish a K=10 benefit; it is another effective K=8, occupancy=8 observation.
- Prefetch spends speculative IO and cancels substantial backlog (table); it is not a byte-minimization win over fill-only. J-OM10b NVMe GiB/token rises from frd16 0.8098085494268508 to pg8b 0.852730221218533 (`experiments/jetson/OM/J-OM10b/REPORT.md`).
- Output is identical across all arms, hash `7a1541792364` (reports/verdicts). The J-OM10b checkin in `c2/worklog.jsonl` passes the router host-copy, demand queue and no-write checks. This supports retaining the opt-in component on this workload; wider workload/model qualification remains open.

## Addendum (2026-11-11): patch 08 changes default prefetch behaviour
Checker (J-OM11 check-in, c2/worklog.jsonl) found that 08 is not "07 + counters": the hook now skips experts whose ranges are already resident (mincore-style probe on the decode path) and batches per expert instead of per file. J-OM11 arms therefore measure the 08 variant; compare them within J-OM11 only (vs its own base), not against J-OM10b/10c pg8b numbers.
