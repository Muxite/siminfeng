# D042 — External-research probes, round 1 (INDICATIVE): n-gram speculation, perf, bpftrace, research intake

Status: INDICATIVE. These are single-round screens, no protocol was frozen, and the host was quiet. Date: 2026-11-10. Engine: release-1 (`sha256:5b8a6a22…`).
Research intake: `research/intake-1.json` and `research/INTAKE.md`, written by a Sonnet agent.
- Channels that worked: paper-gatherer, `gh`, and the HF API.
- Semantic Scholar rate-limited abstract lookups (HTTP 429), and the GitHub MCP failed to connect.

## 1. N-gram self-speculation (`llama-server --spec-type`), Qwen3-30B-A3B Q4_K_M, K=30, op T-SPEC1/1b
Settings: 256 tokens, temperature 0, single slot. Scripts: `artifacts/spec1/probe*.py`.

| Prompt | base t/s | ngram-mod t/s | ngram-map-k t/s | drafts accepted | Text identical |
|---|---:|---:|---:|---|---|
| code rewrite (`/no_think`) | 35.6 | **85.2 (2.4×)** | 60.2 | 225/225 | yes |
| list repeat (`/no_think`) | 35.2 | **106.6 (3.0×)** | 36.4 | 229/229 | yes |
| open chat | 36.9 | 36.7 | 34.8 | 0 drafts | yes |

- **Engagement lesson.** Run T-SPEC1 used prompts without `/no_think`. Qwen3 then began with free-form reasoning, so no drafts were made, and the probe looked like "no effect".
  - Always log `timings.draft_n` and `draft_n_accepted`. Without them, the absence of a gain can't be told apart from speculation never running.
- **Result: lossless, and gated by workload.** The factory should enable `ngram-mod` for copy-heavy work (editing, RAG quoting, code refactors). It costs nothing on chat.
- **Why it suits offloaded MoE.** Verifying a batch of drafted tokens reads the CPU-resident experts once per batch instead of once per token. That cuts the dominant per-token CPU byte cost (see §3).
- The Granite all-GPU repetitive prompt reached 1787 t/s, about 2.2× its baseline.

## 2. perf (host change: `perf_event_paranoid` 4→1, runtime only; system-wide counting runs through `sudo perf`)
- Zen 2 on kernel 6.8 exposes **no `amd_df` / UMC PMU**, so true DRAM bandwidth can't be counted.
- The proxy, L1 data-cache fills from DRAM × 64 B, gave 14.9 GB/s during Qwen all-CPU-expert decode (24.4 t/s).
  - It undercounts because it misses fills made only by the L2 prefetcher.
  - IPC was 0.58 across 16 threads.
- Conclusion: **CPU bandwidth must come from a microbenchmark** (T-CAL), not counters. perf remains useful for IPC and cycles.

## 3. bpftrace uprobes on the CPU backend (sudo; target `/proc/<container pid>/root/app/libggml-cpu.so`)
Settings: Qwen K=48, all experts on the CPU, 5 s window. Output: `artifacts/bpf1/bpf.txt`.
- 5626 calls to `ggml_graph_compute`, which is **48.9 per token at 23.1 t/s**. That is one CPU split per layer.
- CPU split time was **4.45 s of 5 s (89% of wall time)**, with a median per-split time of 0.5–1 ms (mean 0.79 ms).
  - This independently confirms D039's 0.757 ms per CPU layer.
  - The remaining 11% is GPU work, handoff and sampling.
- **Pitfalls:**
  - Use `docker inspect -f '{{.State.Pid}}'`. `pgrep` matches the host docker client.
  - The CPU MoE matmul function is a templated static (`ggml_compute_forward_mul_mat_id_tiled_one_expert<…>`). Probe the C-exported `ggml_compute_forward_mul_mat_id_tiled` or `ggml_graph_compute` instead.
- This is a zero-rebuild, low-overhead per-layer timer, and it should become a CC (`bpf_split_timer`).

## 4. Intake priorities (see intake-1.json)
1. Rebase onto upstream moe-cache changes after the pin. PR #30966 CPU-share targets our 4 GB loss in D041.
2. Rebase onto ClassLRU, #31233 and #31853.
3. Hit-rate regression issue #31421: check the hit rate at batch 1.
4. MTP with Qwen3.6-35B-A3B-MTP GGUF (about 21 GB download).
5. REAP-pruned Qwen3-30B (15B).
6. S2-MoE (arXiv 2608.15018, llama.cpp-based code).
7. Speculation and expert-bytes model.
8. CUDA fusion PRs.

## 5. Host calibration (T-CAL) first real run — NEGATIVE, fixed in T-CAL2
- The CPU probe first failed with rc 127. The engine binary links `libcuda` even with `-b CPU`, so every engine-image probe needs `--gpus`. That is fixed in calibrate.py.
- Constants after the rerun:
  - GPU: 1384 GB/s. **Impossible**; the peak is about 896 GB/s.
  - CPU: Q4_K 97, Q8_0 147 GB/s. **Impossible**; the DDR4 peak is about 50 GB/s.
  - H2D: 28.7 GB/s. Plausible for PCIe 4.0.
  - Handoff: 199 µs. Plausible; PL4's fitted value was 137 µs.
- Cause: test-backend-ops perf reuses one weight buffer, about 33 MB on the GPU and about 7 MB on the CPU. Those fit the 48 MB GPU L2 and 32 MB CPU L3.
- **Methodology rule:** every bandwidth probe must (a) use a working set of at least 8× the last-level cache and (b) pass a physical-ceiling plausibility guard. T-CAL2 adds both.
