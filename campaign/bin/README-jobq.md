# Local job queue

Run from the campaign root. Python 3 standard library only; jobs use `bin/run_op`
with agent `jobq` and the submitted name as task. No worker starts automatically.

```sh
build=$(bin/jobq submit --class build --name engine --timeout 7200 -- python3 cc/build_engine/build_engine.py ...)
run=$(bin/jobq submit --class gpu --after "$build" --name timing --timeout 5400 -- python3 cc/om_spec/om_spec.py ...)
bin/jobq worker --once
bin/jobq wait "$run" --timeout 600
bin/jobq list
bin/jobq show "$run"
bin/jobq cancel "$run"
```

`worker` polls continuously; `worker --once` drains the queue until idle, including
jobs made ready by dependencies. Multiple workers share the same admission lock.
Eligible jobs are admitted FIFO; blocked jobs do not prevent compatible work.
A failed, cancelled, timed out, or skipped dependency causes `skipped_dep_failed`.
`wait` returns 0 for success, 1 for any other terminal state, 2 for wait timeout.
Wait timeout does not cancel work. Job timeout terminates its process groups.
Timeout and cancellation send TERM once, then wait for the command to exit.
`submit --grace SECONDS` sets the per-job delay before KILL (default 20 seconds).
A command that exits during grace retains its actual exit code; the job state
still records `timeout` or `cancelled`.

At most one GPU or build job runs at once. CPU jobs can overlap builds; CPU jobs
exclude GPU jobs unless submitted with `--allow-during-gpu`. This prevents the
build/timing contamination seen in D029/D031. The rule works in both directions.
Only commands admitted through this queue are covered by its exclusivity rules.

State defaults to `lab/jobq/`: one JSON per job, append-only `events.jsonl`, a
`lock`, and per-job logs. JSON records command PID/group, wrapper group, times,
exit status, dependencies, and symmetric `concurrent_jobs` overlap history.
A worker restart detects a dead job supervisor and fails the abandoned job.
`JOBQ_DIR` overrides state storage; `JOBQ_RUN_OP` selects a test wrapper.
Submit preserves the working directory and environment for execution, adding
`JOBQ_JOB_ID` for container ownership labels.

`bin/jobq reap-containers` inspects running/exited containers labelled `om=1`
or `acc=1`, asks `jobq show` about `jobq_job` owners, and prints Docker removal
commands for jobs that are absent or not running. Dry-run is the default;
`--yes` executes removals. Containers without a job label are retained.
`--dry-run` explicitly previews commands even if `--yes` is also present.
Docker discovery is read-only during a dry run and respects `DOCKER_CMD_PREFIX`.

Migration of `cc/om_run/pipeline6.sh` (run after the preceding pipeline's job):

```sh
O=experiments/jetson/SPEC2-$(date +%Y%m%d-%H%M)
IMG=campaign/llamacpp:jetson-up1108
prev=$pipeline5_job_id
for arm in 'base 1' 'ngram-mod 1' 'ngram-mod 2' 'base 2'; do
  set -- $arm
  prev=$(bin/jobq submit --class gpu --after "$prev" \
    --name "J-SPEC2-$1-r$2" --timeout 5400 -- \
    python3 cc/om_spec/om_spec.py "$O/$1-r$2" "$IMG" "$1" "$2")
done
bin/jobq worker --once
bin/jobq wait "$prev"
```
