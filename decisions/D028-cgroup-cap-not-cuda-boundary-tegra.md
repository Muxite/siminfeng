# D028 — Container memory cap is NOT a reliable bound on CUDA allocations on Tegra (2026-11-10, jetson-arm)
- Probe: cc/smoke/cuda_alloc_probe.cu (cudaMalloc / cudaHostAlloc-mapped in 250 MiB chunks, each touched by a kernel), ubuntu:22.04 + nvidia runtime, `--memory 1g --memory-swap 1g`.
- Results (ops J-ISO1 in evidence/jetson-arm/opus-jetson.jsonl):
  - 2 runs (device, host-mapped): killed with exit 137 after 1000 / 750 MiB (first attempts after the D022 env fix).
  - control 600 MiB device: OK.
  - 5 later runs, 1500 MiB device under the same 1 GiB cap: ALL completed (exit 0, OOMKilled=false); container cgroup memory.current peaked at 629–638 MiB while 1500 MiB was allocated and touched; memory.events empty.
- Interpretation: cudaMalloc/nvmap pages on Orin are largely not charged to the allocating container's memory cgroup (only ~0.6 GiB, likely context/runtime/host-side buffers, is). The two early kills are unexplained (possibly charging of first-touch pages under conditions not reproduced) → the cap neither reliably limits nor reliably passes GPU allocations.
- Decision: never describe a Docker `--memory` limit as a GPU or shared-DRAM budget on Jetson (same rule as the campaign prompt for x86 VRAM). Budget field in admission must be `guarantee: none-for-device-allocations` on Tegra. Over-memory experiments use the PHYSICAL limit (20.8 GiB model on 7.4 GiB DRAM) + the J-OM1 watchdog for host safety; a cooperative cap (cc/gpu_budget cudaMemGetInfo/cudaMalloc interposer) is the remaining candidate for an artificial budget — unqualified on Jetson.
- Open: reproduce the early kills (order effects? page-cache charging of the bind-mounted binary? first container start?) — low priority.
