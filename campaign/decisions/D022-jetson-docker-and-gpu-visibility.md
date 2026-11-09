# D022 — Jetson container path qualified; GPU visibility must be explicit (2026-11-09, jetson-arm)
- Owner action (user-authorized): docker.io 29.1.3, ccache 4.5.1, `nvidia-ctk runtime configure --runtime=docker` (wrote /etc/docker/daemon.json with `nvidia` runtime), muk added to group docker (current session uses `sg docker -c`). No JetPack/driver/nvpmodel change.
- Test: host-compiled sm_87 CUDA smoke (`cc/smoke/cuda_smoke.cu`, nvcc 12.6, cudart static) bind-mounted read-only into `ubuntu:22.04@sha256:076e7bf8…`, `--network none`.
  - `--runtime nvidia` alone: exit 1, "CUDA driver version is insufficient for CUDA runtime version" (misleading: driver libs simply not injected).
  - `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`: PASS, Orin CC 8.7, integrated=1, 0/1024 mismatches.
- Consequence for the job/launch contract: on Jetson the GPU request is the env pair + runtime, not `--gpus`; a missing request yields a CUDA error that looks like a driver mismatch. Admission/launch must set it explicitly and the result classifier must not label this "driver incompatible". l4t base images set these env vars by default; generic arm64 images do not.
- Own error noted: an earlier check piped through `tail`, which masked the exit status; exit codes must come from run_op records, not pipelines.
