# D021 — GPU budget shim (T-L) qualified on the RTX 5070 Ti; coverage on the real engine measured (x86-main, c2-resume)
Claim category: infrastructure/fixture validation. NOT a hard device-memory boundary.

Evidence: evidence/claude-code.jsonl task X3 (build, GPU test, engine probe); artifacts/x3-shim/cuda/{libgpubudget.so,test_budget,test-budget.jsonl}; artifacts/x3-shim/probe/nocap.jsonl.
Source: commit 712fcd5 (Codex L final output; it finished AFTER the c1 snapshot, so the c1 tree held an earlier partial version).

1. Build in campaign/cuda-tools:c1 succeeded; real CUDA test (budget 250 MiB) 39/39 PASS: runtime malloc/async/pitch/3D/managed, driver cuMemAlloc_v2/cuMemAllocAsync, meminfo virtualization, deny-leaves-accounting-unchanged.
2. Engine probe (Build CC image b0 = sha256:cf383f82..., llama-completion, mixE4K, -ngl 99, default ctx, no cap, logging only): exactly 3 intercepted cudaMalloc: weights 778.7 MiB, KV 6144.0 MiB (default n_ctx = model max), compute 168.0 MiB; all freed at exit. Output correct ("Paris").
3. NOT seen: ggml CUDA pool temporaries (VMM pool, cuMemCreate — uncovered by design), CUDA context/cuBLAS workspace, driver-internal allocations. Build CC NO_VMM variant (T-N) needed before temporaries are counted.

Guarantee we can state: cooperative per-process cap on intercepted allocation APIs + virtual cudaMemGetInfo. It is bypassable (VMM, static cudart, cuGetProcAddress, other processes). Hard restriction: INCOMPLETE.
Design consequence: D018's ~700 MiB scaled budget is below the weights alone (779 MiB mixE4K); any budget experiment must declare n_ctx (KV dominates at default ctx) and count KV+compute, else the "budget" is meaningless. The budget should be expressed as {weights, kv@ctx, compute, uncounted-overhead measured by nvidia-smi delta}.

Separate gap found while resuming D020: job-8df3c7bb27's spec hashes the bf16 GGUF but every test loads model_file=mixE4K from the same directory -> the model actually used was not hash-verified by the lab. Lab fix: hash every model_file a test references (per-test input identity).
