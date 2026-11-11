# Assignment J-OM12 — no-perf thread-state profiler + "where do the other 480 ms go" spec (worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. perf is not installed, no sudo, ptrace_scope restricts attach; container processes run as root. /proc/<pid>/task/<tid>/stat and /comm are world-readable (verify with a quick local test on your own process).
Evidence: experiments/jetson/OM/J-OM11 (REPORT.md + pg8 stderr `expert_pregate: stats`): Qwen3.6-35B-A3B over-memory, -ngl 99 -lm mmap -cmoe: 738 ms/token, of which demand_wait 256 ms; the remaining ~480 ms is unexplained (CPU expert matmul for ~615 MB/token should be ~20-60 ms at DRAM speed). J-OM9 (D034): majflt_between ~162/token.
## Deliverables
1. cc/thread_sampler/thread_sampler.py (stdlib): attach to a PID (or `--container NAME` → `docker inspect -f {{.State.Pid}}` via DOCKER_CMD_PREFIX), sample every task's stat at `--hz` (default 500) until the process exits; record per sample per tid: state (R/S/D/...), utime, stime, minflt, majflt, processor (field 39), comm. Output compact binary-free JSONL (one line per sample batch, delta-encoded counters) + summary.json:
   - per thread group (classify comm: main `llama-*`, OpenMP/ggml CPU workers, expert-fill workers (name them in 09 patch? see 3), CUDA/driver threads `cuda*`/`nv*`, others): fraction of samples in R, D, S; CPU seconds; majflt; per-token rates given a decode window (reuse cc/abrun/run_summary.py phase-window method: decode window = last eval_ms_total before exit).
   - timeline of "how many threads in R / D" (histogram) in decode window.
   Overhead budget: <3% of one core at 500 Hz (measure and report self-CPU).
2. om_run.sh integration: optional `OM_THREADS=1` starts the sampler on the container after start, writes OUTDIR/threads.jsonl + threads.summary.json; must not change behaviour when unset. abrun spec field `"threads": true` sets it for all runs; run_summary adds threads.* headline numbers to secondary table (R-fraction of CPU workers, D-fraction of main+workers, majflt/token).
3. Small engine patch 09-thread-names.patch on 01-08 (combined-01..09): name expert-fill worker threads `xfill-N` (pthread_setname_np) so the sampler can separate them; no behaviour change. COMPILE-CHECK 09.
4. cc/abrun/specs/J-OM12.json: image campaign/llamacpp:jetson-up1108-om12, threads true, metric eval_ms_per_token lower; arms base (no env), pg8 (as J-OM11 pg8), pg8t4 (pg8 + `-t 4`), pg8t6 (pg8 + `-t 6`, explicit), order base pg8 pg8t4 pg8t6 pg8t6 pg8t4 pg8 base. (Note: default -t is 6 on this host; argv per arm must be supported by abrun — if per-arm argv is not supported, add `extra_argv` per arm to abrun with tests.)
5. Tests: sampler parsing on captured /proc stat lines from THIS host (capture real lines from your own process into a fixture file; label them), classification, decode-window aggregation, overhead measurement test (sample own process 1 s).
## Acceptance (checker runs)
- A1 pytest cc/thread_sampler cc/abrun pass; make -C cc/expert_fill test pass.
- A2 `thread_sampler.py --pid $$ --seconds 2` on a busy local python loop reports R-fraction > 0.8 for that thread and self-CPU < 3%.
- A3 fresh cf34d311: 01..09 sequential == combined-01..09 (patch -F0), COMPILE-CHECK 09 exit 0, header byte-equal.
- A4 abrun dry-run J-OM12 = 8 runs in order, OM_THREADS=1 set, per-arm -t present.
- A5 om_run unchanged when OM_THREADS unset (diff the generated docker command with/without).
Constraints: no docker/GPU/full build; no commits. Time 75 min. Final <= 12 lines.
