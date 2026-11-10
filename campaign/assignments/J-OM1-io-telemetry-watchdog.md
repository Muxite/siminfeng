# Assignment J-OM1 — I/O + fault telemetry backend and safety watchdog (host jetson-arm; parent over-memory track; worker Codex)
Base: branch jetson/c1. Allowed: `cc/resource_observe/**` only. Python 3.10 stdlib. No sudo/installs/GPU/docker.
Why: over-memory runs stream expert weights from NVMe via mmap page cache; we must measure faults and I/O and never drive the 7.4 GiB shared-DRAM host into OOM/swap thrash.
1. New backend `io` in resource_observe.py (same `resource-observe/0` contract, units, unavailable-with-reason):
   - /proc/vmstat: pgmajfault, pgfault, pgpgin, pgpgout, pswpin, pswpout, workingset_refault_file (as cumulative counters + per-interval deltas/rates).
   - /sys/block/<dev>/stat for a configurable device (default: the block device holding a given path, resolve via /sys/dev/block/MAJ:MIN of `os.stat(path).st_dev`, walk to parent disk): read sectors→MiB, read ios, io_ticks → rate MiB/s and %util per interval.
   - per-PID (`--pid`, or `--pid-from-cgroup PATH` reading cgroup.procs and summing): /proc/<pid>/stat majflt/minflt, /proc/<pid>/io read_bytes (permission → unavailable with reason), /proc/<pid>/status VmRSS/RssFile/RssAnon/VmSwap.
   - cgroup v2 (`--cgroup PATH`): memory.current, memory.stat (anon, file, file_mapped, active_file, inactive_file, workingset_refault_file, pgmajfault), io.stat rbytes, memory.events (oom, oom_kill).
   - Scopes: system / device / process / cgroup — never sum across scopes.
2. `watchdog` subcommand: `resource_observe.py watchdog --min-available-mib 900 --max-swap-used-mib 512 --max-swap-rate-mibs 20 --interval-ms 250 [--kill-cgroup PATH | --kill-pid P | --kill-cmd 'docker kill NAME'] --out events.jsonl`: samples MemAvailable/swap; when a limit is crossed for 2 consecutive samples, emits a `watchdog_trip` record (which limit, values), then performs the configured kill action ONCE (SIGTERM, then SIGKILL after 3 s for pid/cgroup), records the action result, and exits 3. Exit 0 when the watched pid/cgroup ends normally. Never kills anything not explicitly configured.
3. Tests (pytest, fixture files for vmstat/diskstats/proc/cgroup, fake clock): delta/rate math incl. counter reset, device resolution from path via fixture sysfs, permission-denied → unavailable, watchdog trips only after 2 consecutive samples, single kill action, exit codes.
4. One real 5 s live `sample --backends io --path ~/models/campaign` on this host to fixtures/live-io-sample.jsonl.
Time 40 min. Final ≤ 12 lines: files, pytest line, live capability, gaps.
