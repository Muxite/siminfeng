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
