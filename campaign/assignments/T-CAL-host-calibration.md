# T-CAL — Host calibration microbenchmark CC
Question: what are this host's decode-relevant constants (GPU weight-read BW, CPU expert BW per quant type, PCIe H2D BW, per-layer GPU<->CPU handoff), measured in about 60 s, so cc/moe_planner/decode_model.py (T-PL4) can be fitted per host instead of from global constants?
Base: branch x86/c2-resume HEAD. Read cc/moe_planner/{decode_model.py,fit_decode.py,PLANNER.md}, decisions/D039, D041, cc/resource_observe/CONTRACT.md.
Write ONLY: cc/host_calibrate/** and artifacts/codexCAL/**.
Deliver:
1. cc/host_calibrate/calibrate.py. It emits a COMMAND PLAN (JSON list of docker argv) and, separately, PARSES the outputs into host-calibration/1 JSON (constants + raw samples + uncertainty + host identity from nvidia-smi/lscpu text). Probes use the release engine image (`--image RUN_REF`):
   (a) `test-backend-ops perf -o MUL_MAT_ID` and `-o MUL_MAT` on CUDA and CPU backends (check the pinned source ref/llama.cpp/tests/test-backend-ops.cpp for perf-mode flags and output format; parse GB/s);
   (b) PCIe H2D via a tiny CUDA program built in the campaign/cuda-tools:c1 image (provide the .cu source + build argv; pinned-memory cudaMemcpyAsync sweep of 1–256 MiB);
   (c) handoff: `llama-bench -p 0 -n 64` at two `--n-cpu-moe` values on a supplied model, with handoff solved from the difference minus the CPU-byte term (document the algebra).
2. cc/host_calibrate/apply.py: host-calibration/1 → constants override consumed by decode_model.py (add a minimal `--host-calibration` hook in the planner ONLY if unavoidable; otherwise output a file the planner already reads).
3. Tests (pytest) built on REAL output samples. Copy actual test-backend-ops perf output format from the source's printf statements; do not invent formats. Mark any parser that only has synthetic input as UNVERIFIED in README.
4. README with exact commands for Opus to run on the GPU (you have no Docker/GPU).
Invariants: no host changes, no sudo, nothing that fills GPU/host memory (cap allocations to ≤2 GiB), each probe ≤30 s timeout.
Final message: files, test count, which parsers are verified against real formats vs unverified, the command plan Opus must run.
