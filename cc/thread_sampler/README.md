# Thread state sampling without perf or ptrace

Run on the same host/PID namespace as the engine:

```sh
python3 cc/thread_sampler/thread_sampler.py --pid 1234 --seconds 2 \
  --out threads.jsonl --summary threads.summary.json
DOCKER_CMD_PREFIX='sg docker -c' python3 cc/thread_sampler/thread_sampler.py \
  --container om-example --out threads.jsonl --summary threads.summary.json
python3 cc/thread_sampler/thread_sampler.py --summarize threads.jsonl \
  --stderr stderr.txt --summary threads.summary.json
```

Without `--seconds`, sampling stops when tasks exit (including unreaped zombies).
The default frequency is 500 Hz. `--eval-ms-total N --tokens N` provides a decode
window explicitly. `--stderr` uses the **last non-prompt** eval line instead.
Container lookup runs `docker inspect -f '{{.State.Pid}}' NAME`, wrapped as a
single shell command argument when `DOCKER_CMD_PREFIX` is set. The prefix must
operate locally; remote Docker PIDs are not local proc PIDs. There is no attach,
ptrace, perf, sudo, Docker exec, or capability requirement for proc reads.

`fixtures/host-proc-stat.json` contains actual own-process stat and comm captures
on satay, kernel 5.15.199-tegra, with capture labels and permission modes. Stat
was 0444; comm was 0644. Parsing uses the final `)` because comm can contain
spaces and parentheses. The sampling hot loop uses the same documented field
positions and is checked against integration samples.

JSONL schema version 1:

- Header: PID, requested Hz, monotonic origin, `clock_ticks`, column names.
- Each sample: `{"t":MONOTONIC_SECONDS,"tasks":[[tid,state,du,ds,dmin,dmaj,processor,comm],...]}`.
  Counters are deltas from that task's previous observation; `du` and `ds` are
  clock ticks, fault counts are events. First observations and recycled TIDs
  start at zero. Processor is stat field 39. Names are JSON-escaped strings;
  no binary data or compressed payloads are used.
- End: elapsed sampling seconds, sampler process CPU seconds, percentage of
  **one core**, sample count, and missed deadlines. Self-CPU measures collection
  and encoding, excluding offline aggregation. Use these recorded values to
  assess perturbation on each engine run.

Every discovered task is read each tick. The leader's thread count triggers
immediate directory discovery on changes; a 100 ms directory scan also catches
replacement TIDs with unchanged counts. Extremely brief tasks or same-count
replacement tasks can escape discovery. Proc reads and sample batches are not
atomic. Files remain open to reduce overhead. A delayed tick is recorded as a
missed deadline; the sampler does not fabricate catch-up samples.

Summary groups are `main` (leader), `cpu_workers` (llama-* inherited names and
OpenMP/ggml names), `expert_fill` (xfill-N), `cuda_driver` (cuda*/nv*), and
`others`. Identically named llama threads use host TID versus PID to distinguish
the leader. Linux truncates comm to 15 bytes. Group and per-TID summaries contain
R/D/S fractions (thread-sample denominator), all observed state counts, CPU
seconds, minor/major faults, and CPU/fault rates per decode token. The joint
`rd_histogram` maps `"R_COUNT,D_COUNT"` to the number of sample batches. Its time
order is available in the raw JSONL. Fractions are descriptive, not gate metrics.

The decode window ends at the **last thread sample before exit** and begins
`eval_ms_total / 1000` seconds earlier, following abrun's telemetry phase-window
method with the thread capture's own endpoint. Counter increments crossing the
left edge are prorated by interval overlap. State counts include only samples
inside the window. The first observation is a baseline, so initial process
faults/CPU and the final unsampled interval are not charged. The maximum observed
sample interval bounds endpoint sampling error; teardown lag has no finite known
bound. `phase_window.coverage_start` exposes a late attachment. Without eval
timing the summary explicitly labels itself `whole_capture`, and per-token
rates are null unless a token count was supplied.

For J-OM12, compare base/pg8 before interpreting pg8t4/pg8t6. Main/CPU-worker D
samples plus major faults/token suggest faults or other uninterruptible waits;
xfill D and faults locate fill-side IO. R includes both executing and runnable
threads: compare CPU seconds/token, R/D histogram concurrency, and elapsed
time rather than treating R as matmul time. High CPU with weak -t scaling could
include spin/barrier overhead. S can include CPU barriers, demand-fill waits,
and CUDA synchronization; this profiler alone cannot attribute those waits.
The roughly 480 ms remainder is a hypothesis to investigate, not an accounting
identity: fill and CPU/GPU work overlap, and summed thread CPU time is not wall
time. Use J-OM11 demand-wait stats, J-OM9's approximately 162 between-fill major
faults/token, and abrun GPU/IO telemetry alongside this capture.

`validation/overhead.json` records local 500 Hz measurements for one busy thread,
a process with 24 threads, and sampling the sampler's own process for one second.
Cost grows with task count and changes with CPU frequency and scheduler load;
the one-thread acceptance measurement does not establish an engine-wide budget.

Measured budget status on this host: the saved one-busy-thread run uses 1.43%
of one core with R=1.0; own-process sampling uses 2.18%. The 24-thread case
uses 23.34%, so the requested <3% budget is **not met for that task count**.
The implementation retains per-task 500 Hz reads and exposes actual overhead;
it does not silently lower sampling frequency to satisfy the budget. The
recorded numbers are measurements rather than a guarantee under other load.
