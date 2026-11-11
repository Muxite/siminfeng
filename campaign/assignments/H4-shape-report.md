# Assignment H4 — "shape of work" report generator (harness; host jetson-arm; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Purpose: the campaign must yield usable data about how inference-engine work is actually spent (who did what, how long, how often it failed and why, how much the GPU sat idle), not just results.
Inputs (read their formats from real files):
- `c2/worklog.jsonl` (events start/end/checkin per task; fields task, worker, wall_s, tokens, verdict, issues, rework_rounds, opus_fix).
- `evidence/jetson-arm/*.jsonl` run_op records (kind intent/started/outcome joined by `op`; fields agent, task, argv, elapsed_s, exit_code, status, ts). Ignore `claude-code.jsonl` hook events except to count tool calls per day (event PreToolUse, payload.tool_name).
- `lab/jobq/*.json` job records (class, state, start/end ns, concurrent_jobs) — may be absent.
- experiment result dirs: `experiments/jetson/OM/*/**/result.json` and abrun `verdict.json` files.
- CLOSEOUT.md section 2 tables are NOT inputs (hand-written).

## Deliverables
1. `c2/shape_report.py [--since ISO] [--out c2/SHAPE.md] [--json c2/shape.json]` (python3 stdlib):
   - Ops table by (agent, task-prefix class) with counts, total/median elapsed, failure rate; task-prefix class = mapping you define in a dict at top (J-OM*->over-memory, J-T*/H*->tooling, J-E*/J-P*->granite-experiments, J-BQ/build*->build, J-KS/KS*->kernel-screen, J-SPEC*->speculation, else other) — print unmapped task names so the map can be extended.
   - Failure taxonomy: classify non-ok outcomes by regex over argv/stderr path tail (read at most 4 KiB of each stderr file if present): flag_error (`invalid argument|unknown argument|error: invalid`), oom/watchdog, docker_start, build_fail, timeout, other. Counts + 3 example ops each.
   - Timeline: GPU-busy intervals = union of ops whose argv contains `om_run.sh|om_spec.py|llama-bench|kernel_screen|abrun.py|llama-perplexity` (and jobq gpu jobs); build intervals = argv containing `build_engine.py`. Report per day: gpu busy h, build h, overlap h (gpu AND build = contamination risk), and the longest idle gap between GPU ops while queued work existed (if jobq data available; otherwise just longest idle gap).
   - Delegation table from worklog: per task wall, tokens, check-in verdict, issues count, rework rounds, opus_fix; totals by worker.
   - Experiments: count of result.json by status; abrun verdicts by verdict value.
   - Write SHAPE.md (tables, <= 200 lines) and shape.json (all numbers).
2. `c2/test_shape_report.py` (pytest, < 30 s) on small synthetic fixtures in a tmp dir (env/args to point at fixture roots): interval union/overlap math, failure classification, worklog aggregation.
## Acceptance (checker runs)
- A1 `python3 -m pytest -q c2/test_shape_report.py` passes.
- A2 `python3 c2/shape_report.py --out /tmp/SHAPE.md --json /tmp/shape.json` exits 0 on the real repo; /tmp/SHAPE.md has the 5 sections; shape.json parses.
- A3 The real report shows the 2026-11-10 pipeline4 idle gap: longest GPU idle gap on 2026-11-10 >= 7 h (the known ~8.7 h deadlock 06:34-15:15). If it does not, explain why in the final message (that would mean the interval extraction is wrong).
Constraints: read-only on all inputs; no docker/GPU; do not commit. Time 45 min. Final message <= 10 lines incl. the top 5 numbers from the real report.
