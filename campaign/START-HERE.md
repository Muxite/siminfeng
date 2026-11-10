# Campaign c1 — start here (wrap-up snapshot, 2026-11-09)
Goal (user, latest): methodology + tools for fitting a very large MoE (~500 GB weights -> 128 GB unified memory + 4 TB SSD) at similar accuracy and acceptable tokens/s. Proxy: ibm-granite/granite-3.1-1b-a400m-instruct on llama.cpp (commit 31288c9c6), RTX 5070 Ti (discrete). Every result is labelled transferable / discrete-only / prediction (D012).

## What is here
- `decisions/D001..D020` — chronological decisions with evidence (read D012-D020 first).
- `LAB-FEASIBILITY.md`, `MANAGER-DESIGN.md`, `cc-inventory.csv` — Lab Manager findings and the CC inventory (observed vs inferred).
- `bin/` — `run_op` (recorded commands), `lab` (image/job/attempt/cancel/reconcile/register/image-load/chaos), `perf_summary`, `index_evidence`, `cost_report`, `cuda_compile_check`.
- `cc/` — Build CC (`build_engine`), `engine_handoff`, `routing_trace` (patch + replay), `memory_pareto`, `hybrid_expert` (Opus harness), `q8_reuse` (UNTESTED speed patch), `gpu_budget` (memory-restriction shim, may be incomplete), tests (`pytest` in each dir).
- `evidence/*.jsonl` — raw per-agent streams (intent/started/outcome); `index/ops.jsonl` merged; `lab/jobs/*` job/attempt state + results; `experiments/` summaries; `claims/paper-claims.md` (15 source-supported claims, none tested yet).
Not committed (reproducible/bulky): llama.cpp clone (`ref/`), image archives, nsys/ncu reports, base logits, traces (regenerate with the lab tests), Codex scratch trees. Models live in `~/models/campaign` (Granite pin 0da7a48b; GGUFs bf16/Q8_0/Q4_K_M/mixE4K/3K/2K).

## Findings to trust (with caveats in the D-notes)
1. Experts are 90.4% of weight bytes; mixed precision (experts Q4_K, rest Q8_0) halves Q4_K_M's quality loss at equal memory (D013).
2. Static layer offload (`--n-cpu-moe`) cuts GPU MiB linearly but decode collapses (discrete-only, D014). Page-cache-as-expert-cache has a sharp cliff when RAM < working set (D015, contaminated by concurrent compiles).
3. Granite routing skew is real vs a uniform control (D017), mild, weaker pooled across workloads.
4. Hybrid hit/miss dispatch is bitwise-exact; the serial cost model holds (33 us/miss); perfect-overlap upper bound NOT supported (D019).
5. Engine-native `--moe-cache-mib` (LRU, upload-on-miss) is slower than no cache until ~60% coverage (job-d50c052f94). The matched static-offload comparison in the same tool did not finish (D020) — rerun `lab/tests/static1.json`.
6. Build CC: failures recorded (3 failed attempts), ccache makes patched rebuilds ~10-20 s; image identity bug found by independent observation (D016); real handoff `accepted` for the corrected record (experiments in `cc/engine_handoff/example-real5`).
7. Lab core bugs found by testing recovery per phase (D006, D020); duplicate delivery, sandbox kill, retry-as-new-attempt, stale/conflicting result rejection all demonstrated (LAB-FEASIBILITY.md, MANAGER-DESIGN.md).

## Unfinished (honest list)
Hard GPU-memory restriction shim (T-L) not compiled/tested by me; Strata concepts not yet applied end-to-end (only studied + CPU tests, and the single-layer harness); static hot-set expert placement (managed-memory idea, D018) not started; no end-to-end speed/memory result beats the baselines in a way I would call an engine win; q8_reuse patch not compiled; Opus/Codex outputs only partly reviewed; Claude-side token cost not captured.

## Host state to restore (I changed these with your approval, D002)
Stopped: container `euaaa-llm-gpu-1` (`docker start euaaa-llm-gpu-1`) and user unit Sunshine (`systemctl --user start app-dev.lizardbyte.app.Sunshine.service`). GPU lock removed (D007). No host-wide driver/profiling change was made. Campaign containers are labelled `campaign=c1` (none should be running).

## Resume
`pip`-free: python3 stdlib + PyYAML/pytest. Re-create the lab: `bin/lab image add llamacpp-31288c9 ghcr.io/ggml-org/llama.cpp@sha256:395d01bb47eb23d5e967129c6ebd2b5969365401621e16dd3db33e50fa7462e3`; build an engine: `python3 cc/build_engine/build_engine.py --source <llama.cpp> --commit 31288c9c6 [--patch P] [--extra-target T] --tag X --out builds/results/Y` (needs BASE_RUNTIME digest, see cc/build_engine/README.md).

## Jetson lane (jetson-arm, branch jetson/c1) — CLOSED OUT 2026-11-10
**Read `CLOSEOUT.md` first** (full experiment/attempt ledger, predictions vs observations, CC candidates, resume steps). Then `handoff/jetson-checkpoint-1.md`, `handoff/jetson/REPLY-1.md`, `research/x86-c2-digest.md`.
Decision numbering: Jetson D021–D033 collide with x86 D021–D048 (proposal: rename ours J-D0xx; not yet agreed).

Strongest findings (Jetson-local, Orin Nano sm_87):
- **Full-engine gain:** per-type MoE gate/up unfuse (`cc/moe_dispatch_sm87` patch 03, `GGML_CUDA_J_UNFUSE_IDS_TYPES=bf16,q4_K`): bf16 +19.3% (KLD 2.7e-4), Q4_K-expert mixE4K +23.6% with identical outputs, Q8_0 must stay fused (−6.2%) (D025/D027). x86 applicability untested (request R1).
- **Dependency gain:** upstream llama.cpp cf34d311 decodes +7% over the pin on the over-memory model (D031).
- **Over-memory (Qwen3.6-35B-A3B 22.4 GB on 7.4 GiB):** best is the baseline `-lm mmap -ngl 99 -cmoe` at ~1.2–1.36 t/s, bound by a ~570 MiB/s budget NVMe. Default `-lm auto` cannot run (D030). Nothing we built beat the baseline: upstream GPU expert cache −17..−23% (D030/D032), dedupe/direct-fill no help (D032), whole-file MADV_RANDOM −50% (D033), router-informed WILLNEED prefetch −38% (J-OM7; bytes −57% but IO stays 4–5 KiB). The drive is IOPS-bound; any next loader must issue large aligned expert reads with lead time.
- **Over-memory win (decoding strategy, workload-dependent):** upstream n-gram self-speculation (`--spec-type ngram-mod`, llama-server) on copy-heavy prompts: +63% (rewrite) / +97% (repeat), text identical, NVMe bytes −27..−35% (SPEC2-20261110-1634). Chat arm invalid (1-token output; needs chat template).
- **Simulations:** page-cache LRU ≈ 62% hit at 12.7% capacity (Belady 80%); static cross-workload hot sets ≈ 17% (avoid). Window-union replay: speculative decoding gives no IO gain on natural text (J-SPEC1).
- **Tool lesson:** `test-backend-ops` single-op kernel screen did NOT predict the D027 graph-fusion effect (KS-20261110-1620).

Jetson host state: docker.io + nvidia runtime installed by the user (D022); buildx user plugin; venv `~/venvs/convert`; models in `~/models/campaign` (see CLOSEOUT §1); images `campaign/llamacpp:jetson-*` (local only). **No services were stopped on the Jetson; nothing to restore.** No clocks/power changes (MAXN_SUPER queried only). Docker in old sessions: `sg docker -c`.
