# D004 — Profiler qualification (host RmProfilingAdminOnly=1, perf_event_paranoid=4; GPU RTX 5070 Ti sm_120)
Image: campaign/cuda-tools:c1 (FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 + cmake, git, python3, cuda-nsight-systems-12-8 = nsys 2024.6.2; ncu 2025.1.1). Build: builds/cuda-tools/Dockerfile.
Evidence: artifacts/nsight-qual/, ops in evidence/orchestrator.jsonl (descs "nvcc build vecadd", "ncu WITHOUT...", "ncu WITH...", "nsys ...").
| Capability | Result |
|---|---|
| nvcc -arch=sm_120 build + run on GPU | works |
| ncu, no extra caps | FAILS ERR_NVGPUCTRPERM |
| ncu, container --cap-add SYS_ADMIN | works (SpeedOfLight section OK) — narrowest workable, no host change |
| nsys GPU/CUDA/NVTX/OS-runtime trace, no caps | works (cuda_gpu_kern_sum stats OK) |
| nsys CPU sampling / ctx-switch | NOT qualified (perf_event_paranoid=4); run with --sample=none --cpuctxsw=none; default invocation did generate a report but CPU sampling coverage unverified |
Host-wide NVreg change (approved by user) therefore NOT applied/needed so far. No reboot or driver reload.
Observation: first kernel launch 23.8 ms vs ~28 us steady-state (managed-memory page faults) -> profiles of short runs must discard/flag warmup.
Limit: this qualifies tools on a toy kernel only; MoE kernels, replay overhead, and report size for the real engine are still unmeasured.
