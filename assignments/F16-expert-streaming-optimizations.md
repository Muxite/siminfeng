# F16 — expert streaming: selection trace + offline simulator, populate backend, hot-expert pinning, temporal predictor

Repo: `/home/muk/projects/xs-dev/llama-opt` (private clone; branch `xs-opt` at `6e9d96cf4` = llama.cpp `cf34d3113` + expert-streaming patches 01–08). Work ONLY here and in `/home/muk/projects/xs-dev/opt-artifacts/`. No Docker/GPU/network. Another worker is building in `/home/muk/projects/xs-dev/llama.cpp` concurrently: use `-j2` for builds and do not touch that clone. No pushes.

Read first: `/tmp/claude-1000/-home-muk-projects-siminfeng/f788ee04-950c-45f8-ae6d-aa6e956451d3/scratchpad/expert-streaming-sourcemap.md`, then `src/llama-expert-fill.h`, `src/llama-context.cpp` (`expert_prefetch_cb`, env validation in patch 08, `expert_config: effective` line), `src/llama-model.cpp` (`expert_file_info`).

Target model facts (Qwen3.6-35B-A3B on Satay): 40 MoE layers, 256 experts, 8 routed per token per layer, ~1.83 MiB per expert per layer (gate+up+down), experts 18.3 GiB total, 7.4 GiB RAM shared with GPU, batch-1 greedy decode. Every patch must be env-gated, default OFF, register its env names in the strict `LLAMA_EXPERT_*` validator (unknown names abort), add its fields to the `expert_config: effective` line, never write weights, never change which experts compute uses, and keep stdout byte-identical.

Each patch is its OWN branch from `6e9d96cf4` (so each can be a separate factory component) unless a dependency is stated:

## P12 `selection_trace` (observability; branch `p12-trace`)
`LLAMA_EXPERT_TRACE=<path>`: append one binary or CSV record per (token_idx, layer, phase prompt|decode, selected expert ids[8], predicted ids[K] if pregate active) from the host copy of `ffn_moe_topk` the callback already reads. Buffered, written at teardown or every N tokens; overhead must be negligible (no syscalls on the hot path per layer). Print `expert_trace: wrote records=%llu path=%s`.
Plus `opt-artifacts/expert_trace_sim.py` (stdlib only, Python 3.10): reads the trace; reports per-layer and overall (a) predictor recall@K for: router-pregate (from recorded predictions), previous-token same-layer, frequency top-K, union variants; (b) cache simulation of expert residency under a byte budget (e.g. 2,3,4,5 GiB) for LRU, LFU, LRU+frequency (ARC-like), Belady (optimal, upper bound) — hit rate and bytes that must be read per decode token; (c) temporal reuse distance histogram. Tests: `opt-artifacts/test_expert_trace_sim.py` with a synthetic trace. This is the offline lab that lets an agent evaluate predictors/caches without the GPU.

## P13 `fill_populate_backend` (performance; branch `p13-populate`)
`LLAMA_EXPERT_FILL_BACKEND=pread|populate` (default pread = today). `populate`: fill workers call `madvise(addr, len, MADV_POPULATE_READ)` on the mmap'd range chunks (page-aligned) instead of pread into private scratch, so pages enter the page cache AND get mapped (no scratch copy, no later minor faults). Fall back to pread per chunk on EINVAL/ENOSYS (log once). Applies to demand fill and async pregate reads. Count `populate_calls/populate_bytes/populate_fallbacks` in the fill stats line (append at end).

## P14 `expert_pin` (performance/residency policy; branch `p14-pin`)
`LLAMA_EXPERT_PIN_MIB=<0..8192>` (0 off), `LLAMA_EXPERT_PIN_WARMUP=<tokens>` (default 8): after warmup decode tokens, rank (layer,expert) by running selection frequency, and `mlock` the top ranges up to the MiB budget (re-rank every `LLAMA_EXPERT_PIN_EVERY` tokens, default 32; munlock demoted). If mlock fails (RLIMIT_MEMLOCK), fall back to `MADV_WILLNEED` + record `pin_mode=willneed`. Stats line `expert_pin: stats pinned_mib= pinned_experts= repins= mlock_failures= mode=`. Note: in Docker the memlock limit may be small; say what happens.

## P15 `pregate_temporal` (performance/predictor; branch `p15-temporal`)
`LLAMA_EXPERT_PREGATE_MODE=router|temporal|hybrid` (default router = today). temporal: predict layer L+1's experts for the next token = experts selected at L+1 on the previous token (issued when layer L of the current token arrives). hybrid: router top-K ∪ previous-token set, ranked by router score then recency, capped at `LLAMA_EXPERT_PREGATE_K`. Extend pregate stats with recall per source (`recall_router= recall_temporal=`) appended at end.

## Checks (record into `opt-artifacts/F16-check.md`)
For each branch: `git format-patch -1` → `opt-artifacts/1N-<name>.patch`; verify each applies alone onto `6e9d96cf4` (`git apply --check` in a throwaway worktree). CPU-only build per branch into separate build dirs (`cmake -B build-<name> -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release` then `cmake --build build-<name> -j2 --target llama-completion`; ccache if available). Functional run on `/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-Q8_0.gguf`: `llama-completion -m <gguf> -p "The capital of France is" -n 32 -c 512 --temp 0 -s 1 -no-cnv -cmoe -lm mmap -t 4` with base env `LLAMA_EXPERT_FILL=1 LLAMA_EXPERT_FILL_RANDOM=decode LLAMA_EXPERT_FILL_MAX_TOKENS=1 LLAMA_EXPERT_PREGATE=1 LLAMA_EXPERT_PREGATE_K=8` with and without the patch's switch: stdout sha256 identical, stats lines printed. For P12 also run the simulator on the produced trace and paste its summary. Note if the small model is fully cached (counters ~0) — fine.

## Acceptance (run verbatim)
- B1 four branches exist, each one commit on `6e9d96cf4`; four patch files in opt-artifacts.
- B2 `python3 /home/muk/projects/xs-dev/opt-artifacts/test_expert_trace_sim.py` passes.
- B3 F16-check.md shows identical stdout sha per branch with switch on/off, and the stats lines.
Final message: per patch — mechanism, counters, risks, what would make it win or lose on Satay (be specific about bandwidth/memory), and anything you could not complete.
