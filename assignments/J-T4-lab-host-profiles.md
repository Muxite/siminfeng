# Assignment J-T4 — lab: per-host capability profiles, platform admission, portable GPU request/telemetry (host jetson-arm; parent c1; owner Opus-jetson; worker Codex)

Base: branch `jetson/c1`. File: `bin/lab` (275 lines, x86-written; Python 3.10 here). Read `MANAGER-DESIGN.md` §admission if present, `decisions/D022-*.md`, `cc/resource_observe/README.md`.
Allowed files: `bin/lab`, new `lab/hosts/*.json`, new `lab/test_lab_hosts.py`. Do NOT change existing job/attempt state semantics, result fields, or `lab/jobs/*` history (x86 evidence). x86 behaviour must be byte-for-byte the same command lines when the host profile is `x86-main` (default when no host profile is configured).

## Observed x86-only assumptions to remove (verified by Opus)
1. `--gpus all` → on Jetson the GPU request is `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility` (D022).
2. `nvidia-smi memory.used` polling: on Orin nvidia-smi exists but prints `[N/A]` → `int()` crashes the runner mid-attempt. Make the GPU sampler a host-profile choice: `nvml` (x86, unchanged) or `tegrastats` (Jetson: shared DRAM RAM used MB via `cc/resource_observe/resource_observe.py` parser functions imported or a `tegrastats --interval 250` child; record scope `system-shared-dram`, never label it GPU memory). Unparseable values → field null + reason, never crash, never 0.
3. `gpu_state()` likewise (check what it calls) → backend-aware with `unavailable` reasons.
4. docker invocation: add env `DOCKER_CMD_PREFIX` support (this session needs `sg docker -c` with a single quoted command string — implement by shlex-joining argv when the prefix is set).

## Host profile + admission (new)
- `lab/hosts/<host_id>.json`: `{host_id, platform: "linux/arm64"|"linux/amd64", gpu: {request: "gpus-all"|"nvidia-runtime-env", sampler: "nvml"|"tegrastats", memory_domain: "discrete-vram"|"shared-dram", arch: "sm_87"|"sm_120"}, l4t_major (Jetson only), resources_max: {cpus, memory}, capabilities: [..]}`. Host id comes from `~/.config/siminfeng/host_id` (as in `bin/run_op` evdir), default `x86-main`. Write `lab/hosts/jetson-arm.json` (6 CPUs, memory max 5g for test containers, l4t_major 36, sm_87) and `lab/hosts/x86-main.json` (8 cpus/16g, sm_120, matching current defaults).
- Image records (`lab/images/*.json`) may carry `platform` and `target_requirements` (Build CC schema 2: platform, l4t_major, cuda_arch, runtime). `lab image add` gains optional `--platform` and `--requirements <build-record.json>`.
- `lab submit` performs admission against the current host and records a decision file `admission.json` in the job dir with `decision: accepted|rejected|needs_information`, `reasons[]`, host profile sha256: reject on platform mismatch, cuda_arch mismatch, l4t_major mismatch, resources above host max; `needs_information` when the image record lacks platform/requirements or the model/data path is missing. Rejected/needs_information jobs are still written (job.json + admission.json, read-only) and `lab run` refuses them (exit code 10 with reason). Legacy images with no platform on x86-main stay accepted with reason "legacy image record: platform unverified" (do not break x86).
- `lab status` shows admission decision.

## Tests (`lab/test_lab_hosts.py`, pytest, no docker/GPU): profile loading/default, argv generation for both GPU request modes (+ DOCKER_CMD_PREFIX quoting), tegrastats/nvml sampler parse incl. `[N/A]` → null, admission accept/reject/needs_information matrix, x86 argv unchanged vs a golden list built from the current code. Monkeypatch subprocess. Run with `python3 -m pytest -q lab/test_lab_hosts.py`.

## Limits
Time 45 min. Do not run docker, the GPU, or `lab run`. Final message ≤ 20 lines: files, pytest line, exact commands Opus should use to (a) add the Jetson image from `cc/build_engine/results/jetson-c1/build-record.json`, (b) submit an accepted Jetson job and an amd64 job that must be rejected; open risks.
