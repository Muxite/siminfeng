# Jetson → x86 requests, round 2 (2026-11-10) — branch jetson/c1 @ fddac2f+
Status seen from Jetson: origin/main still f0e676b; no x86 branch published. Please publish a feature branch (e.g. `x86/c1`) even if partial — mark incomplete state explicitly. Each request: purpose, exact steps, expected output files, what it decides. Ordered by value. Do not pool raw t/s across hosts.

## R1 — Does the MoE gate/up fusion rule transfer to sm_120? (decides: is D025/D027 a portable component or Orin-only)
1. `git fetch origin jetson/c1`; take `campaign/cc/moe_dispatch_sm87/patches/03-unfuse-ids-by-type.patch` (no cc guard; env `GGML_CUDA_J_UNFUSE_IDS_TYPES`).
2. Build with your Build CC: `--patch .../03-unfuse-ids-by-type.patch` on 31288c9c6 (amd64 target).
3. For each model bf16 / mixE4K / Q8_0 (you already have them): same image, arms base vs `-e GGML_CUDA_J_UNFUSE_IDS_TYPES=<bf16|q4_K|q8_0>`, order base,on,on,base, `llama-bench -p 0 -n 128 -r 5 -ngl 99`, report steady mean (samples 2–5).
4. Quality per model: `llama-perplexity -c 128 --chunks 8 -b 128 -ub 1 --kl-divergence-base <bf16 base ub1 logits> --kl-divergence` (base arm and on arm).
5. Optional cheap screen first: `test-backend-ops perf -o MUL_MAT_ID` (build tests target) for bf16/q4_K/q8_0 at n=1.
Output: `experiments/x86/X-E-unfuse/{*.json,kl-*.txt,SUMMARY.md}`. Orin reference: bf16 +19.3%, q4_K +23.6% (identical outputs), q8_0 −6.2%.

## R2 — Tensor-content identity of the bf16 GGUF (decides: are cross-host quality differences weights or kernels)
`PYTHONPATH=campaign/ref/llama.cpp/gguf-py python3 campaign/cc/model_manifest/gguf_manifest.py ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf > experiments/x86/bf16-gguf-manifest.json`. Jetson tensor_data_sha256 = e1c1b4d5f2f4ae1967bb48c61694b45c6b169263fc67a6136dc6977f3dceec3d. Jetson ppl2 7.5479 vs x86 7.5325.

## R3 — Finish your own open items before new work (from START-HERE): rerun `lab/tests/static1.json` with new attempt IDs (D020); compile+qualify cc/gpu_budget (T-L) with the coverage table; reconcile the budget-shim writer. Jetson finding for that shim: llama.cpp on Tegra sizes memory via `ggml_backend_cuda_get_available_uma_memory` (reads system memory) — please state whether the shim intercepts that path or only cudaMemGetInfo.

## R4 — Over-memory on discrete GPU with the SAME big model (decides: unified vs discrete applicability of placement policies)
Model: `unsloth/Qwen3.6-35B-A3B-GGUF` @ a483e9e6cbd595906af30beda3187c2663a1118c, file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (20.8 GiB; > 16 GB VRAM, < 48 GB RAM). Baselines: `-lm mmap -ngl 99 -cmoe`, `-ncmoe N` sweep (N chosen so VRAM fits), `--moe-cache-mib` sweep (llama-cli/server only; llama-bench lacks the flag). Pin `-c 4096`. Record VRAM peak, host RSS, decode/prefill t/s, and the same routing trace capture as Jetson (cc/routing_trace adapted to qwen35moe — Jetson will publish the adaptation). Note: on x86, `-lm auto` keeps mmap (dGPU); on Jetson it disables it.

## R5 — Envelope/identity decisions you own
(a) Docker containerd store: image `.Id` = manifest digest; record `local_image_digest`, `config_digest` (from archive), `registry_digest` (after push only). (b) Binaries are not bit-reproducible across Build CC attempts (D026): adopt record+functional identity, consider `-ffile-prefix-map`/`SOURCE_DATE_EPOCH`. (c) Merge candidates: bin/run_op (non-UTF-8 echo fix), bin/cc_hook (per-host evidence), cc/build_engine (jetson profile + ccache keying), cc/resource_observe (+io/watchdog), lab host profiles (bin/lab), cc/expert_replay. Review `handoff/manager-jetson-findings.md`.

## R6 — Things only x86 can answer cheaply
- nvidia-smi/NVML fixture: run `cc/resource_observe/resource_observe.py capabilities` and a 5 s `sample --backends auto` → commit as fixtures (the nvml backend is fixture-tested only).
- `lab/test_lab_hosts.py` + one real `bin/lab run` of an old x86 job to confirm no regression from host profiles (reviewer found no regression by reading; not executed on x86).
