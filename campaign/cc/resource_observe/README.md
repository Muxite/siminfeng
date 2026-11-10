# resource_observe (J-T2)
Stdlib-only Python 3.10 sampler emitting JSONL under one contract, with platform backends.

## Usage
    resource_observe.py capabilities
    resource_observe.py sample --backends auto|tegrastats,proc,sysfs_tegra,nvml [--interval-ms 500] [--duration-s N] [--pid P] [--cgroup PATH] [--nvpmodel-mode STR] --out F.jsonl
    resource_observe.py summarize F.jsonl [--interval-ms 500]   # per-metric min/max/mean/high_water + coverage (expected vs received, gaps > 2x interval)
Tests: `python3 -m pytest` (no GPU/root; fixtures in `fixtures/`, paths injectable).

## Record
`{"schema":"resource-observe/0","ts","mono_s","host_id","backend","seq","metrics":[{name,value,unit,scope,source}],"unavailable":[{name,backend,reason}],"parse_errors":[...]}`
Missing data is listed in `unavailable` with a reason, never as 0. Malformed tegrastats lines yield a record with `parse_errors` carrying `line_sha256_16` (no crash). Units: MiB, MHz, percent, mW, celsius, load.
`tegrastats` emits one record per line it produces (the interval is set on the child); other backends emit one per tick.

## Attribution scopes (do NOT sum across scopes)
- `system-shared-dram` (tegrastats RAM): whole-system DRAM shared by CPU and GPU on Jetson.
- `system` (/proc/meminfo): same physical pages viewed system-wide; `process` (VmRSS/VmHWM) and `cgroup` (memory.current/peak/max) are subsets of the same pages. Adding them to tegrastats RAM double counts.
- cgroup metrics carry `accounting_note`: cgroup v2 may not include nvmap/driver GPU allocations on Tegra, so it can under-report GPU memory. Unlimited `memory.max` is reported unavailable. memory.peak may be missing on older kernels (unavailable). This host's cgroup controllers file lists memory.
- `device-discrete` (nvml): dedicated VRAM, x86 only. On Orin nvidia-smi exists but returns [N/A]; backend reports unavailable.
- sysfs_tegra: GPU devfreq (MHz), thermal zones (celsius, unreadable zones listed unavailable), nvpmodel mode (`nvpmodel -q`, query only, or `--nvpmodel-mode`).

## Known gaps
No NvMap fields on this kernel's meminfo (reported unavailable); no iomem/debugfs nvmap per-client attribution (needs root); tegrastats gets its own interval, summarize infers interval from median spacing if not given; nvml parsed from CSV fixtures only on this host (no discrete GPU).

## I/O telemetry and watchdog (J-OM1)

    python3 resource_observe.py sample --backends io --path ~/models/campaign --duration-s 5 --out io.jsonl
    python3 resource_observe.py sample --backends io --device nvme0n1 --pid 123 --cgroup /sys/fs/cgroup/example --out io.jsonl
    python3 resource_observe.py sample --backends io --pid-from-cgroup /sys/fs/cgroup/example --out io.jsonl
    python3 resource_observe.py watchdog --min-available-mib 900 --max-swap-used-mib 512 --max-swap-rate-mibs 20 --interval-ms 250 --kill-pid 123 --out events.jsonl

`io` reports vmstat counters in `system`, whole-disk counters in `device`,
PID data in `process`, and cgroup v2 data in `cgroup`. Never add these scopes.
`--path` defaults to the current directory and resolves the filesystem device
through `/sys/dev/block/MAJ:MIN`, walking partitions to the parent disk.
`--device` explicitly selects a disk name (or `/dev/NAME`). Overlay/network
filesystems without a block-device mapping report unavailable with a reason.
Disk read sectors use the kernel's fixed 512-byte sector unit.

Cumulative counters have `_delta` and `_rate` companions. First observations,
resets, missing samples, and nonpositive time intervals have unavailable interval
values; counters resume from a fresh baseline. Units: faults/refaults/I/Os/events
are `count`, pgpgin/out are `KiB`, swap-in/out are `pages`, read data and memory
are `MiB`, disk io_ticks are `ms`; rates append `/s`. `device_util` is
`100 * delta(io_ticks) / interval_ms` in `percent`. Device utilization is the
kernel busy-time measure, not an estimate of parallel NVMe queue saturation.
`pid_read_bytes` retains the kernel field name but is converted to MiB.
PID fields include minflt/majflt, read_bytes, VmRSS, RssFile, RssAnon, VmSwap.
`--pid` and `--pid-from-cgroup` are mutually exclusive. The latter sums only
PIDs in that file; incomplete per-field process sums are suppressed, and PID
membership/start-time changes invalidate interval baselines. Cgroup `io.stat`
read bytes remain separate per major:minor device. Cgroup memory accounting
may omit Tegra driver allocations.

The watchdog samples MemAvailable and SwapTotal minus SwapFree, and calculates
combined swap-in/out MiB/s using the host page size and monotonic elapsed time.
A particular limit must be crossed on two consecutive samples (equal to the
limit is safe). It flushes `watchdog_sample`, `watchdog_trip`, and
`watchdog_action` JSONL events using the same schema envelope. Targets are
mutually exclusive: `--kill-pid`, `--kill-cgroup`, or `--kill-cmd`.
With no target it records the trip without sending signals. PID/cgroup targets
receive SIGTERM, a three-second grace period, then SIGKILL for remaining
processes. Cgroup targeting includes descendants; PID start times are checked
before each signal to avoid targeting a reused PID. Commands are parsed with
`shlex.split` and run once without a shell (15-second timeout); for example
`--kill-cmd 'docker kill NAME'`. The watchdog does not otherwise invoke Docker.

Exit codes: 3 after a trip, including a failed kill action; 0 if the watched PID
ends or the cgroup becomes empty; 2 for unreadable/invalid monitoring data.
Command-only and untargeted watches run until a trip or interruption because
there is no process lifetime to observe. Monitoring errors stop the watchdog
without killing anything. No live destructive watchdog validation was run;
signals and command execution are covered with mocks and a fake clock.
The five-second live system/device capture is `fixtures/live-io-sample.jsonl`.
