---
name: codex-delegation
description: Use when assigning bounded campaign work to a Codex worker and arranging independent acceptance check-in.
---

Work from /home/muk/projects/siminfeng/campaign.
Read c2/PLAN.md, c2/METHODS.md delegation protocol, c2/launch_codex.sh,
assignments/H1-jobq.md and assignments/J-OM8-expert-pread-fill.md.

1. Write assignments/TASK.md with these sections:
   - Read-first: repo, host, source pin and precise prerequisite paths.
   - Why: evidence-backed problem, hypothesis, discriminating result.
   - Deliverables: exact files/interfaces and output behavior.
   - Acceptance commands: A1…An, literal commands/cwd and expected results.
   - Constraints: allowed edits, no evidence/ref changes, GPU/build permissions,
     time budget and commit policy.
   - Final message: length limit, files, results, deviations and risks.
2. Keep independent assignments on disjoint files. Bound worker scope; do not
   delegate an undefined investigation with no observable acceptance criteria.
3. If GPU jobs may be queued, submit the launcher as cpu without opt-in overlap:
   ```sh
   bin/jobq submit --class cpu --name TASK --timeout 5400 -- \
     bash c2/launch_codex.sh TASK assignments/TASK.md
   ```
   Arrange a jobq worker and use job IDs for dependencies/wait/show/cancel.
   Launcher logs go to artifacts/codex/TASK.log and TASK.final.txt.
   Do not nest a synchronous wait for a gpu child under a running cpu parent:
   jobq excludes GPU while that parent runs. Split implementation from the later
   queued build/experiment. Use direct launch only when scheduling permits.
4. After completion, send a cheap Haiku checker this template:
   > Read assignments/TASK.md. Run each acceptance command A1…An verbatim in
   > its specified cwd. Report PASS/FAIL per criterion with outputs/test counts.
   > Spot-check referenced evidence, paths, scope and claimed activation/quality.
   > Do not fix files or run unassigned GPU/build work. Report deviations and
   > failures to the lead; acceptance cannot establish untested engine behavior.
5. Lead reviews findings, fixes project-level issues or assigns bounded rework,
   and repeats affected acceptance. Never treat a worker's final claim as check-in.
6. Preserve append-only c2/worklog.jsonl start/end events written by launcher:
   task, worker, assignment, ts, rc, wall_s, tokens. Add a checkin event with task,
   checker, verdict, issues, rework_rounds, failure_class and ts; record any lead
   fix attribution. Start/end timestamps plus wall and measured tokens are the
   shape-of-work dataset. Use unknown for absent usage; do not invent counts.
