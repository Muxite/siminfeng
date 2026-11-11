# Assignment H2b+H4b — per-arm switch checks in abrun; shape_report accuracy (harness rework; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Files you own: cc/abrun/*, c2/shape_report.py, c2/test_shape_report.py. Do NOT touch cc/abrun/specs/J-OM8b.json or any experiments/ dir (a GPU experiment may read them). Keep all existing tests passing.
## H2b (abrun)
1. Arms may declare `"switch_regex": [...]` (per arm, list = ALL must match >= 1 stderr line each). Spec-level `switch_regex` keeps its current meaning (OR) for backward compatibility. INVALID rules: an arm with `expect_switch:true` fails if any of its per-arm regexes has zero matches; an arm with `expect_switch:false` fails if ANY regex from any arm (spec-level or per-arm) matches its stderr (leak check).
2. Each run summary records `switch_matches: {regex: count}`; REPORT.md shows them per run.
3. Gate: add `--exclude RUN_ID` support to a new `cc/abrun/regate.py OUTDIR [--exclude 001-base ...] [--metric eval_tps]` that recomputes verdict from existing summary.json files and writes `verdict-regate.json` (+ prints it) with the excluded runs and reason field `--reason "..."` recorded. Never deletes data.
4. Gate also reports per-arm mean of `nvme_gib_per_token`, `prompt_tps`, `eval_tps` (secondary metrics table, no verdict) in REPORT.md and verdict.json.
## H4b (shape_report)
5. Flag-failure taxonomy: CLOSEOUT 5.2 lists 11 flag failures (J-E003 x4 `-mmp`, base-20261110-0435 x6 `-c` for llama-bench, feas-b1 x1 `-cmoe`). Find why only 5 classify as flag_error (inspect the actual stderr files of those ops; e.g. llama-bench prints `error: invalid parameter for argument: -c` or `usage:`; om_run stores stderr in the run dir not in run_op's stderr) and fix by reading om_run run-dir stderr.txt when argv contains om_run.sh (the OUTDIR is argv[1] of om_run.sh). Report the new count; >= 10 expected, else explain.
6. Map the 15 unmapped task names (J-FIO, J1, J2-*, J5-*, codex-*, sleep/true test ops -> 'test-noise') in the class map.
7. Worklog rows whose worker is missing: infer `opus` for event types contamination/result/failure; show them in a separate "Lead events" table, not the delegation table.
## Acceptance (checker runs)
- A1 `python3 -m pytest -q cc/abrun c2/test_shape_report.py` all pass; >= 4 new tests (per-arm ALL-match, leak check, regate exclude, flag taxonomy from om_run stderr fixture copied from a REAL run dir).
- A2 `python3 cc/abrun/regate.py experiments/jetson/OM/J-OM8 --exclude 001-base --reason "overlapped codex H3/H4" --no-write` prints verdict JSON (no file written with --no-write).
- A3 `python3 c2/shape_report.py --out /tmp/claude-1000/S.md --json /tmp/claude-1000/s.json` exit 0, flag_error >= 10 (or final message explains).
- A4 `python3 cc/abrun/abrun.py cc/abrun/specs/J-OM8b.json --dry-run` unchanged order (16 runs).
Constraints: no docker/GPU; no commits; no evidence edits. Time 45 min. Final <= 10 lines.
