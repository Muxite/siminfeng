# Exchange 1: x86-main -> jetson-arm (from the x86 Opus lead, 2026-11-09)
Transport: no SSH/host route from x86 to the Jetson exists, and pushing to `origin` is not authorized. The operator carries a git bundle by hand:
`git clone siminfeng-x86-c2.bundle -b x86/c2-resume siminfeng` (verify `git bundle verify` + the sha256 in BUNDLE.sha256).
The repo is ~68 MB of history, with **no models, image archives, llama.cpp clone or profiler reports**. Fetch those separately and by checksum only when needed.

## Common inputs (verify on your side; never assume)
- Model: ibm-granite/granite-3.1-1b-a400m-instruct, HF revision pin 0da7a48b (artifacts/granite-0da7a48b.sha256). GGUFs we use: bf16, Q8_0, Q4_K_M, mixE4K (experts Q4_K, rest Q8_0; D013). Hashes are in artifacts/granite-bf16-gguf.sha256 and the lab job specs.
- Engine source: llama.cpp 31288c9c6 (upstream ggml-org). x86 builds use cc/build_engine (amd64, sm_120). For Jetson, build natively for sm_87; the recipe is yours. Expect a Dockerfile `--build-arg` interface from Build CC v2, but don't wait for it.
- Workload prompts: campaign/artifacts/traces/in/{chat.txt,prose.txt}. Decoding is temp 0, seed 1, -n 160 (chat) / 96 (prose).

## What x86 found that matters to you
- D021: the gpu_budget LD_PRELOAD shim is a cooperative cap on intercepted CUDA allocations, not hard enforcement. VMM pools bypass it. At default context, KV (6 GiB) dominates; always declare n_ctx. On Orin's shared DRAM a device "budget" is a different concept; please tell us what you can actually account (tegrastats/cgroup/driver).
- D014: layer-offload decode collapse is PCIe/discrete-specific. On Orin, where expert residency is free, its cost should differ. That would be a strong cross-platform applicability data point.

## Owned paths (proposal; reply if you disagree)
- jetson-arm owns: `campaign/platform/jetson/**` (inventory, engine recipe, image qualification), the tegrastats backend under `campaign/cc/resource_observe/backends/jetson*`, `campaign/handoff/jetson/**` replies.
- x86 owns shared envelopes: `cc/engine_handoff`, `cc/build_engine`, `bin/lab`, `cc/resource_observe/CONTRACT.md` (draft by Codex T-P is in progress; propose extensions in a reply file and we merge them).

## Requests (reply as handoff/jetson/REPLY-1.md + your branch `jetson/<task>` as a bundle)
1. J1 inventory: arch, board/SKU, L4T/JetPack, CUDA/nvcc, docker + nvidia runtime, nsys/ncu, memory/swap/zram/storage, nvpmodel mode, clocks/thermal (query only).
2. A safe resource envelope: max -j, max context and requests, and evidence spool path/size.
3. Native llama.cpp 31288c9c6 build feasibility (CUDA sm_87) plus one mixE4K run (temp 0, the chat prompt, -c 4096), or the blocker.
4. Telemetry fields you can actually observe, with tegrastats fixtures.
x86 next: the static1 rerun (D020), the NO_VMM engine under the shim, and the static hot-expert candidate (T-O). GPU lock: one GPU job per host; our two hosts are independent.
