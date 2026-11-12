# F09 — Demo-grade views: `factory story`, human `status`, `factory capsules`

Repo: `/home/muk/projects/jiuwenswarm` (branch `ai4r/iterative-cc-factory`, `.venv`). Factory home with real evidence: `~/.jiuwenswarm-factory` (READ ONLY for you). Exported bundles: `artifacts/factory/evidence/<task>/` (records/*.json, artifacts.json, INDEX.md). Read first: `jiuwenswarm/factory/{store,cli,controller,export}.py`, `jiuwenswarm/factory/ccs/*/cc.yaml`, `docs/architecture/factory/CONTROLLER.md`.

Goal: the 10-minute talk must make an audience say "this CC stuff is easy and elegant". Today `factory status` dumps raw JSON lists. Build three views that tell the story at a glance. Stdlib only (no rich/click). ANSI color only when stdout is a TTY and `NO_COLOR` is unset; `--plain` forces no color. Every line ≤ 100 columns. Unicode ✓ ✗ → · allowed.

## 1. `jiuwenswarm factory story TASK_OR_BUNDLE_DIR [--follow] [--plain] [--json]`
New module `jiuwenswarm/factory/story.py`. Source = the live store (task id) OR an exported bundle directory (path exists) — same renderer, so the talk can run it on the committed bundles with no factory home.

Build an ordered event list by merging (by created timestamp) decisions, observations, attempts (with history transitions), admissions/uploads and evaluations for the task. Render:

```
 factory · d4b71bcc · Qwen3.6-35B-A3B on Jetson Orin Nano 8 GB          ACCEPT · 3 turns · 11m46s
 ─────────────────────────────────────────────────────────────────────────────────────────────
 00:00  rev 1  decide     integrate factory.engine_component (patch 09) into owned checkout
 00:41    ✓    integrate  commit 690e4713  2 files +N −M
 01:02    ✗    inspect    CONFIG_REJECTED  expert_config: invalid LLAMA_EXPERT_PREGATE_K=48 (range 1..32)
 01:15  rev 2  replan     LLAMA_EXPERT_PREGATE_K 48 → 8          because obs abcedeb4
 ...      ✓    build      3216f109  124 s  → image f02f8a93  (durable job, survived turn end)
          ✓    admit      Manager · digest ✓ · target jetson-orin ✓
          ✓    lab        64 tokens · 745 ms/token · reaped
          ✓    contract   qwen36-greedy-v1  stdout 7a154179… == reference  · engagement 6/6
 11:46  rev 6  accept     <first 80 chars of reason>
```
Rules: times are relative to task creation (mm:ss). Derive every value from records — NO hard-coded ids/strings for this task (tests use synthetic stores). Decision lines show rev, action type, and the one-line summary of what changed vs the previous rev (diff of action/config keys when available; else first clause of `reason_for_change`). Failure observations (state failed / error code present) render ✗ in red with the error code + message; never hidden. Header: short task id (8 chars after `factory-`), model name from contract `model.path` basename (strip quant suffix sensibly, keep it generic), target, outcome, turns, wall clock from `budget_used`. Footer: one line `chain: component → commit → build → image → admission → lab → evaluation` with ✓/✗/· for each link found, plus `evidence: N records · M artifacts · K bytes`. Unknown record kinds → generic dim line, never crash. `--json` emits the event list. `--follow` (live store only) re-renders new events every 2 s until the task has an outcome (or Ctrl-C), printing only new lines (append-only; no screen clears) — this is what the presenter shows next to a live run.

## 2. Human `factory status`
Replace `format_status` in `cli.py`: compact sections — header (task, outcome, rev, budget used/limit), "Now" (latest decision action + reason, wrapped ≤ 100 cols), "Jobs" (active jobs: kind, state, age; or "idle"), "Evidence" (counts per kind + total bytes, NOT the full artifact list), "Last evaluation" (contract, correctness, execution_outcome, completeness, stdout sha prefix vs reference prefix, engagement x/y), readiness flags as ✓/✗. `--json` unchanged. `run`/`resume` final print uses the same formatter.

## 3. `jiuwenswarm factory capsules [--plain]`
Table of the CC folders in `jiuwenswarm/factory/ccs/`: id, version, binding kind (material-only vs native), capabilities (names), declared `changes` effects, files, and YAML line count; footer totals ("6 capsules · N capabilities · M lines of cc.yaml · 0 adapters"). Read the yaml with the same loader ccv2 uses (validate via ccv2 if it exposes a validator; show ✓ valid per row).

## 4. Tests `tests/factory/test_story.py`, `test_status_view.py`, `test_capsules_view.py`
Synthetic temp store with a failure→replan→success chain: story order, ✗ line present for the failure, replan line shows `48 → 8`-style diff, chain footer, `--plain` has no ESC bytes, every line ≤ 100 cols, bundle-dir source renders identically to store source for an exported synthetic task, unknown kinds tolerated. Status view: no artifact-id dump, counts correct. Capsules: 6 rows from the real folder, totals.

## 5. Docs
Add a "Demo views" section to `docs/architecture/factory/RUNBOOK.md` with the three commands and a pasted real output of `factory story artifacts/factory/evidence/factory-d4b71bcce337406b96f627d469bec092 --plain` and `factory capsules --plain`.

## Acceptance
- G1 `.venv/bin/python -m pytest tests/factory --no-cov -q -p no:cacheprovider` all pass.
- G2 `.venv/bin/jiuwenswarm factory story artifacts/factory/evidence/factory-d4b71bcce337406b96f627d469bec092 --plain` prints the real chain incl. the K=48 ✗ line and the accept line; all lines ≤ 100 cols; exit 0. Same for the attempt-1 bundle (ends inconclusive, ✗ lines visible).
- G3 `.venv/bin/jiuwenswarm factory status factory-d4b71bcce337406b96f627d469bec092` ≤ 40 lines, no raw artifact list.
- G4 `.venv/bin/jiuwenswarm factory capsules --plain` shows 6 rows.
- G5 No commits, no pushes, no builds/lab runs, no writes to `~/.jiuwenswarm-factory` or exported bundles.
Final message: files changed, the real G2 output, anything not done.
