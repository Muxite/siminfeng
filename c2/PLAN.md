# Campaign c2 (jetson-arm) — formalize the lab, then iterate the engine
Started 2026-11-11 by the Opus orchestrator. Branch jetson/c1 (continues). Inputs: CLOSEOUT.md sections 5.2 (missing tools), 6 (candidate CCs), 7 (open items).

## Roles
- Opus: plan, write assignments with acceptance criteria, fix project-level errors, decide.
- Codex (`codex exec -s workspace-write`): all bounded coding/experiment-scripting work.
- Haiku check-in agents: run each assignment's acceptance commands verbatim, report PASS/FAIL per criterion. No fixes.
- Every task gets a row in c2/worklog.jsonl (task, worker, start/end, wall, tokens, check-in verdict, rework rounds, failure class). This is the "shape of work" dataset.

## Phase H — harness (formalize the 5.2 gaps as tools)
| id | deliverable | replaces |
|---|---|---|
| H1 | `bin/jobq` file-backed job queue, PID waits, class locks (gpu/build exclusive), om_run status fix, telemetry host_id | pipeline2-6, build_queue.sh, pgrep loops |
| H2 | `cc/abrun`: declarative A/B spec -> preflight (flag probe) -> counterbalanced order -> om_run -> run_summary -> A/A-aware gate | inline Python in pipelines, perf_summary for OM runs |
| H3 | methods catalog `c2/METHODS.md` + repo skills (`.claude/skills/`) generated from what H1/H2 actually do | tribal knowledge in D-notes |

## Phase E — engine iteration (uses H1/H2)
| id | hypothesis | gate |
|---|---|---|
| J-OM8 | synchronous exact expert fill (mincore + parallel buffered pread of 1.9 MB slices, qd~8) before CPU MUL_MAT_ID turns fault-driven 4-95 KiB IO into large IO and beats `-lm mmap -cmoe` | decode t/s > base by > max(3%, 2x A/A gap), text identical, NVMe GiB/token recorded |
| E-SPEC2c | n-gram spec chat arm with chat template, 2 rounds | text identical, acceptance logged |
| E-KSq4 | kernel screen q4_K rerun (`--types q4_K`) | completes; closes CLOSEOUT 7 item |
