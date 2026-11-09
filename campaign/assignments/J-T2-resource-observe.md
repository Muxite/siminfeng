# Assignment J-T2 — portable resource-observation CC (host jetson-arm; parent campaign c1; owner Opus-jetson; worker Codex)

Base: branch `jetson/c1` (from f0e676b). Python 3.10 stdlib only (+ pytest for tests). Write ONLY under `campaign/cc/resource_observe/`. Do not touch other campaign files, nvpmodel/clocks, drivers, or install anything.

## Question / output
A small library + CLI `resource_observe.py` that samples resource state on either host and emits JSONL records under ONE common contract with explicit platform backends.

Backends (auto-detected, each can be unavailable with a reason):
- `tegrastats` (Jetson): spawn `tegrastats --interval <ms>` and parse lines. Fixture: `fixtures/tegrastats-idle-orin-nano-r36.5.txt`. Fields: RAM used/total MB (SHARED CPU+GPU DRAM — attribution scope `system-shared-dram`), lfb, SWAP used/total/cached, per-CPU load%@MHz, GR3D_FREQ % (and @MHz if present), EMC_FREQ if present, temperatures (named), power rails (instant/avg mW, rail names vary by board — parse generically).
- `proc` (both): /proc/meminfo (MemTotal, MemAvailable, Cached, SwapTotal/Free, plus NvMap* if present), /proc/loadavg, optional per-PID RSS/VmHWM from /proc/<pid>/status (attribution `process`), cgroup v2 memory.current/memory.peak/memory.max for a given cgroup path (attribution `cgroup`; note cgroup does NOT necessarily account nvmap/driver GPU allocations on Tegra — mark `accounting_note`).
- `sysfs_tegra` (Jetson): GPU devfreq cur/min/max freq (`/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/`), thermal zones (`/sys/class/thermal/thermal_zone*/{type,temp}`), nvpmodel mode string passed in or read via `nvpmodel -q` (query only).
- `nvml` (x86): `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits` and per-process `--query-compute-apps`. Attribution `device-discrete`. On Orin this must report unavailable (nvidia-smi absent or no memory fields) — never mandatory.

## Contract (each sample record)
`{"schema":"resource-observe/0","ts":ISO8601 UTC,"mono_s":float,"host_id":str,"backend":str,"seq":int,"metrics":[{"name","value","unit","scope","source"}...],"unavailable":[{"name","backend","reason"}],"parse_errors":[...]}`
Rules: no false zeros (missing → `unavailable` with reason); units explicit (MiB, MHz, percent, mW, celsius); never sum tegrastats RAM with process RSS or cgroup (different scopes of the same physical pages); a malformed line yields a record with `parse_errors` and the raw line hash, not a crash.
CLI: `resource_observe.py sample --backends auto|tegrastats,proc,... [--interval-ms 500] [--duration-s N] [--pid P] [--cgroup PATH] --out FILE.jsonl` and `resource_observe.py summarize FILE.jsonl` (per-metric min/max/mean/high-water + coverage: expected vs received samples, gaps > 2x interval).
Also `capabilities` subcommand: prints which backends/metrics are available on this host with reasons.

## Tests (`test_resource_observe.py`, pytest, no GPU, no root)
Parse the real fixture; malformed tegrastats lines (truncated, missing RAM, unknown rail names, extra fields, empty); missing /proc fields via fixture files; nvml CSV fixture incl. `[N/A]` values; summarize coverage gap detection; no-double-count assertion. Allow injecting fixture roots/paths so tests don't depend on the host.

## Limits / checkpoint
Time 30 min. Run the tests and ONE real 3-second `sample --backends auto` on this host (it is a Jetson Orin Nano) writing to `cc/resource_observe/fixtures/live-sample.jsonl`. Add `README.md` (agent-facing: usage, contract, per-backend attribution limits). Final message: files, test result line, live capability output, known gaps.
