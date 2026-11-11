# Assignment H1 — job queue + run-wrapper fixes (harness; host jetson-arm; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Read first: CLOSEOUT.md sections 5.2 and 5.6, cc/om_run/om_run.sh, cc/om_run/pipeline5.sh, bin/run_op (how ops are recorded), cc/resource_observe/resource_observe.py (sample/watchdog CLI).
Problem: experiment chains are hand-written bash pipelines that wait with `pgrep -f`; one self-matched and idled the GPU 8.7 h. Builds overlapped timed runs and contaminated results (D029-D031).

## Deliverables
1. `bin/jobq` (python3 stdlib only, single file, executable). State dir `lab/jobq/` (one JSON per job + append-only `events.jsonl`).
   - `jobq submit --class {gpu,build,cpu} [--after JOBID ...] [--name N] [--timeout S] -- CMD...` -> prints JOBID. Job runs through `bin/run_op --agent jobq --task <name>`.
   - `jobq worker [--once]`: runs queued jobs FIFO when dependencies succeeded; a job whose dependency failed becomes `skipped_dep_failed`. Exclusivity: at most one `gpu` OR `build` job at a time (they exclude each other because builds contaminate timings); `cpu` jobs may run alongside a build but NOT alongside a gpu job unless submitted with `--allow-during-gpu` (also record that flag in the job JSON). Lock via `fcntl.flock` on `lab/jobq/lock`.
   - `jobq list`, `jobq show JOBID`, `jobq wait JOBID [--timeout S]` (exit 0 if succeeded, 1 failed, 2 timeout), `jobq cancel JOBID` (SIGTERM the process group it started; never pattern-match process names).
   - Waiting uses stored PIDs (`os.kill(pid,0)`) and job JSON state; NO pgrep/pkill anywhere (grep the file to prove it).
   - Each job JSON records: argv, class, deps, state (queued/running/succeeded/failed/cancelled/skipped_dep_failed/timeout), pid, pgid, start/end, rc, and `concurrent_jobs` (ids of jobs running at any point during it) so contamination is auditable.
2. `cc/om_run/om_run.sh` fixes: result.json `status` must distinguish `watchdog_tripped_killed` (trip AND container exit 137/killed) from `watchdog_tripped_survived` (trip but exit 0); keep `ok`/`failed`/`oom_killed`. Add `"host_id"` from `~/.config/siminfeng/host_id` (fallback hostname) and `"image"`, `"env"`, `"argv"` fields.
3. `cc/resource_observe/resource_observe.py`: telemetry records' host id must use `~/.config/siminfeng/host_id` when present (CLOSEOUT 5.2 bullet). Keep existing tests green.
4. `bin/test_jobq.py` (pytest): uses a temp state dir (env `JOBQ_DIR`), covers: FIFO, dependency success/failure/skip, gpu-vs-build exclusion (two `sleep 1` jobs must not overlap: check timestamps), cpu-during-build allowed, cancel kills child process group, wait exit codes, timeout. Tests must not touch docker or GPU and must run in < 60 s. Commands in tests should run via a fake run_op (env `JOBQ_RUN_OP` override) so evidence streams are not polluted.
5. `bin/README-jobq.md` (<= 60 lines): usage, the exclusivity rule and why (D029/D031), migration example rewriting cc/om_run/pipeline6.sh as jobq submits.

## Acceptance (the check-in agent runs exactly these)
- A1 `cd campaign && python3 -m pytest -q bin/test_jobq.py` passes, < 60 s.
- A2 `python3 -m pytest -q cc/resource_observe` passes.
- A3 `grep -nE 'pgrep|pkill' bin/jobq` prints nothing.
- A4 `bash -n cc/om_run/om_run.sh` ok and `grep -n watchdog_tripped_survived cc/om_run/om_run.sh` matches.
- A5 smoke: `JOBQ_DIR=$(mktemp -d) bin/jobq submit --class cpu -- true` then `JOBQ_DIR=same bin/jobq worker --once` then `jobq wait` exits 0.
Constraints: no docker, no GPU, no builds, do not edit evidence/*.jsonl, do not commit. Time budget 60 min.
Final message <= 12 lines: files, test counts, any deviation from this spec and why.
