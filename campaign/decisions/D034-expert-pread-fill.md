# D034 — Expert pread fill improves decode; synchronous fill IO remains the bottleneck (2026-11-11, jetson-arm)
Patch chain `cc/mmap_advice/patches/03-expert-fill.patch`, `04-fill-phase.patch`, `05-fill-diag.patch`. Fill the selected experts with aligned, residency-aware parallel preads into scratch buffers before the existing mmap consumer; keep model weights and compute unchanged. Phase gating preserves prefill readahead and applies random advice only at decode. Same-image counterbalanced arms, llama-completion p1, mmap and CPU experts; exact argv/env are in each experiment's `spec.json`.

| Experiment / arm | n | Mean decode t/s | B/A vs base | A/A gap | Gate |
|---|---:|---:|---:|---:|---|
| J-OM8 / base | 2 | 1.185 | — | 0.0928 | base |
| J-OM8 / fill | 2 | 1.33 | 1.12 | 0.0928 | null |
| J-OM8 / fill_random | 2 | 1.39 | 1.17 | 0.0928 | null |
| J-OM8b / base | 4 | 1.1875 | — | 0.0471 | base |
| J-OM8b / fr | 4 | 1.275 | 1.07 | 0.0471 | null |
| J-OM8b / frd | 4 | 1.29 | 1.09 | 0.0471 | null |
| J-OM8b / frd16 | 4 | 1.3025 | 1.1 | 0.0471 | improved |
| J-OM9 / base | 2 | 1.195 | — | 0.0232 | base |
| J-OM9 / frd16 | 2 | 1.295 | 1.08 | 0.0232 | improved |
| J-OM9 / frd16d | 2 | 1.18 | 0.987 | 0.0232 | null |
| J-OM9 / frd16p | 2 | 1.2 | 1 | 0.0232 | null |

Sources (n counts the listed `inputs` observations; A/A is the report-wide maximum gap): `experiments/jetson/OM/J-OM8/verdict.json` and `experiments/jetson/OM/J-OM8/REPORT.md`, `experiments/jetson/OM/J-OM8b/verdict.json` and `experiments/jetson/OM/J-OM8b/REPORT.md`, `experiments/jetson/OM/J-OM9/verdict.json` and `experiments/jetson/OM/J-OM9/REPORT.md`.

Interpretation:
- J-OM8's apparent gains remain null under its A/A gate. J-OM8b establishes frd16 as improved, and J-OM9 reproduces it. Use `LLAMA_EXPERT_FILL=1 LLAMA_EXPERT_FILL_RANDOM=decode LLAMA_EXPERT_FILL_MAX_TOKENS=1 LLAMA_EXPERT_FILL_THREADS=16 LLAMA_EXPERT_FILL_CHUNK_KIB=2048`, with diagnostics and populate off (`experiments/jetson/OM/J-OM8b/spec.json`). Full-time random advice (`fr`) saves bytes but harms prefill: prompt mean 1.46 t/s versus frd16 5.725 t/s (`J-OM8b/REPORT.md`).
- The serial dependency is the synchronous per-layer fill barrier, despite parallel workers inside each fill. It does not overlap IO with compute. The diagnostic run spends 29038848380 ns in fill and 24817519341 ns between fills (`experiments/jetson/OM/J-OM9/003-frd16d/stderr.txt`, `expert_fill: diag tokens=63`). The lead's interpretation is synchronous fill IO at 364 MB/s (`c2/worklog.jsonl`, J-OM9-ab result).
- `MADV_POPULATE_READ` removes many later faults but pays their cost up front: `majflt_between` falls from 10208 to 454; populate costs 9953651850 ns, with 59031 calls (`experiments/jetson/OM/J-OM9/003-frd16d/stderr.txt` and `004-frd16p/stderr.txt`). Its mean is effectively baseline, and its gate is null: net zero end-to-end gain, not a solution to the synchronous fill bottleneck. Diagnostic instrumentation itself slows the fill arm; do not interpret fault counts as free instrumentation.
- `majflt_between` uses RUSAGE_SELF and includes GPU-driver thread faults; sparse diagnostic lines cannot localize every nonresident event (`c2/worklog.jsonl`, J-OM9 checkin). Residency/fault counts alone do not establish eviction as the dominant cost.
- Every listed output is identical, hash `7a1541792364` (the cited reports/verdicts). These are measured results for this prompt/model/drive, not a cross-workload quality qualification.
