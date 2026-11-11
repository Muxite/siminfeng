# Assignment J-OM10 — pre-gated async prefetch of layer L+1 experts (component; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Builds on 01-05 (cc/mmap_advice/patches/combined-01-02-03-04-05.patch; cc/expert_fill/expert_fill.h single source of truth). Engine tree ref/llama.cpp-up1108 @ cf34d311 — never modify in place; work in /tmp copy.
## Evidence
J-OM9 (experiments/jetson/OM/J-OM9, `expert_fill: diag` lines): per decode token ~461 ms synchronous fill IO (164 MB at ~364 MB/s) + ~394 ms everything else, strictly serial. Faults are minor (162 majflt/token). J-OM7 (CLOSEOUT 2.4 om7 row) showed naive next-layer prediction (reuse layer L ids for L+1) has 3% precision. Hypothesis: predicting layer L+1's experts by applying layer L+1's router to layer L's MoE input (pre-gating; known from MoE offloading literature to give high next-layer recall) gives enough lead time to overlap most fill IO with compute.
## Design (verify each assumption in the code; state where it is false)
- At the existing CPU observer hook for layer L's expert MUL_MAT_ID (patch 02/05), under `-cmoe` the MoE input activations for layer L are on host (src1 of the CPU MUL_MAT_ID, or the tensor feeding it). Read it (decode ubatch only, n_tokens == 1).
- Router of layer L+1 = `ffn_gate_inp` of layer L+1 (check type: f32/bf16/f16/q8? and whether the arch applies a norm, bias `exp_probs_b`, softmax/sigmoid, grouped top-k — read build_moe_ffn and the arch's graph builder for the model in use: Qwen3.6-35B-A3B GGUF at ~/models/campaign/qwen36-35b-a3b/model.gguf, arch name from its metadata via gguf reader or `cc/model_manifest`). The router weight lives on GPU but is also in the mmap'd file: obtain a host pointer through the same file-offset machinery as patch 03 (it is small: ~2 MB/layer) — do NOT read it via the GPU.
- Compute logits = W_{L+1} · x_L on the CPU (dequantize if needed; single vector), top-K_pred (env `LLAMA_EXPERT_PREGATE_K`, default 8; also accept 12/16). Ignore approximation error (no residual/attention update) — that is the point of measuring recall.
- Enqueue an ASYNC fill of those experts' gate/up/down slices in layer L+1 on the existing fill pool (separate low-priority queue; never block the hook on it). The synchronous fill at layer L+1 stays as-is (mincore skips what the prefetch already made resident; if a range is in flight, duplicating the pread is acceptable — say what happens).
- Counters: predictions, recall@K (fraction of actual L+1 selected experts that were in the predicted set — compute at the L+1 hook), precision, async bytes read, async reads still in flight when L+1 hook arrives, sync-fill bytes at L+1 with vs without prediction. Log `expert_pregate: active k=..` at start, `expert_pregate: stats ...` at exit. Diag-free hot path when `LLAMA_EXPERT_PREGATE` unset (no extra syscalls/allocs).
- Last layer: no L+1; first decode layer: nothing to predict from (fine).
## Deliverables
1. `cc/mmap_advice/patches/06-pregate.patch` (on 01-05) + `combined-01-02-03-04-05-06.patch`; header updates (single-source rule), `make -C cc/expert_fill test` passing (add a unit test for the async queue: async fill completes, a later sync fill of the same ranges reads 0 bytes).
2. COMPILE-CHECK.txt section 06.
3. `cc/abrun/specs/J-OM10.json` image `campaign/llamacpp:jetson-up1108-om10`, argv as J-OM9 spec. Arms (no DIAG anywhere): frd16 (as J-OM9), pg8 (frd16 + LLAMA_EXPERT_PREGATE=1 K=8), pg16 (K=16), base. Explicit order: base frd16 pg8 pg16 pg16 pg8 frd16 base. Per-arm switch regex: pg arms require `expert_pregate: active` and `expert_pregate: stats`.
4. README section in cc/expert_fill/README.md: mechanism, the router math for this arch, expected effect arithmetic (if recall r and prefetch fully overlapped, sync IO per token drops ~r×; bound by compute ~394 ms/token), risks (page-cache pressure from wrong predictions evicting useful experts; CPU contention between prefetch threads and expert matmul threads — 6 cores).
## Acceptance (checker runs)
- A1 `make -C cc/expert_fill test` PASS.
- A2 fresh cf34d311: sequential 01..06 == combined-01..06 (diff -r empty).
- A3 COMPILE-CHECK 06 all exit 0.
- A4 dry-run J-OM10 = 8 runs in order.
- A5 header single-source diff empty.
- A6 checker quotes: (i) router math incl. any bias/norm handling, (ii) where x_L is read from and that it is host memory, (iii) that the hook never waits on async work.
Constraints: no docker/GPU/full build; no commits; no evidence edits. Time 90 min. Final <= 12 lines: arch name, router type & gating func, where x_L comes from, risks.
