# c2 status — Jetson Orin Nano 8GB (unified memory), 2026-11-12

Host: Orin Nano 8GB Super, 7.4 GiB shared DRAM, NVMe ~570 MiB/s, llama.cpp cf34d311 + our patches (cc/mmap_advice/patches 03-09).
Goal set by the owner: best-accuracy model on 8 GB at decode >= 2 tok/s (min), 5 (goal), 10 (stretch); prefer models/quants with published benchmarks.

## Results so far
| What | Result | Source |
|---|---|---|
| Qwen3.5-9B UD-Q4_K_XL (5.56 GiB, fits DRAM), `-ngl 99 -fa on -lm mmap -c 4096` | **10.79 tok/s decode** (92.7 ms/token), prompt 41 tok/s, load 29 s — single smoke run, gated runs pending | c2/worklog.jsonl `smoke-qwen35-9b` |
| Same, `-ngl 99` without mmap | fails: cudaMalloc 5.1 GiB OOM | worklog `ladder-fit-qwen9` failure |
| Qwen3.6-35B-A3B UD-Q4_K_XL (20.8 GiB, over-memory) baseline `-lm mmap -cmoe` | 883 ms/token (1.13 tok/s) | experiments/jetson/OM/J-OM11 |
| + expert pread fill + pre-gated next-layer prefetch (patch 08, K=8) | 738 ms/token (1.36 tok/s), 1.20x, output identical | J-OM11; D034, D035 |
| Earlier best (patch 07, pg8b) | 1.42 tok/s vs 1.17 base (+22%) | J-OM10b |
| Lookahead 2 layers | worse (783-843 ms) | J-OM11 |
| + 4 CPU threads (pg8t4), om12 image | **684.7 ms/token (1.46 tok/s), 1.26x vs base**, output identical; 6 threads 704.8 (1.22x) | experiments/jetson/OM/J-OM12 |
| **J-OM13** stock llama.cpp vs factory-grown engine (run 4), 4 reps each, ABBA | stock 841.5 ms/token (1.19 tok/s); factory 711.4 (1.41 tok/s, **1.18x**); factory + `-t 4` 659.3 (1.52 tok/s, **1.28x**); all 12 outputs identical; J/token 5.65 -> 4.80 | experiments/jetson/OM/J-OM13 |
| Where the 35B token goes (pg8) | ~256 ms waiting on SSD fill, ~480 ms unexplained (not expert matmul) → J-OM12 thread profiler | J-OM11 stderr stats |

Published scores (thinking mode, model cards): Qwen3.5-9B MMLU-Pro 82.5 / GPQA-D 81.7; Qwen3.6-35B-A3B 85.2 / 86.0; Gemma-4-E4B 69.4 / 58.6 (research/model-candidates-20261111.md).

Ladder (non-thinking MMLU-Pro subset, experiments/jetson/LADDER/): Qwen3.6-35B-A3B 62.9% (n=70, 95% CI 51–73); Qwen3.5-9B 45.7% (n=140, provisional) at 10.6 tok/s full-GPU kvq8-ub128; Gemma-4-E4B 40.7% (n=140, provisional) at 17.8 tok/s.

J-OM12 thread sampler: with pre-gate on, main+worker D-state drops 0.43 → 0.005 and major faults/token 5170 → 494; the remaining gap is CPU-side, not page-fault waits.

CC inference factory (Satay handoff 2026-11-11): `~/projects/jiuwenswarm` branch `ai4r/iterative-cc-factory` (Stellven/jiuwenswarm). The agent integrated patch 09, survived a real config rejection, built om12 (`campaign/llamacpp:jetson-up1108-om12` = factory image f02f8a93), output byte-identical to om11. Report: artifacts/factory/COMPLETION-REPORT.md in that repo.

## In progress
- Model ladder (cc/ladder): remaining KLD for kvq8; Qwen3.6 UD-IQ3_S needs disk.
- Wave 2 downloads: Qwen3.6-35B-A3B UD-IQ3_S (12.7 GiB) / IQ2_M, Qwen3.5-4B Q8_0 + Q4 (KLD pair).

## Tooling (how to use)
- Queue: `bin/jobq` (README-jobq.md) — gpu/build/cpu exclusion, deps, private env.
- A/B: `cc/abrun` — counterbalanced arms, A/A+drift gate on ms/token, switch-value checks.
- Accuracy: `cc/acc` — MMLU-Pro/ARC/GSM8K subsets, llama-server logprob MCQ, KLD, McNemar.
- Ladder: `cc/ladder` — per model fit/speed/acc/kld/report.
- Methods/skills: c2/METHODS.md, c2/skills/*; known measurement gaps: c2/MEASUREMENT-GAPS.md; shape of work: c2/SHAPE.md.
- Decisions: decisions/D034-D036; ledger: CLOSEOUT.md (Addendum c2).

## Known issues
- ladder resume treats a failed fit as complete if argv changed (fix queued); preflight rejection should fail fast.
- thread_sampler costs ~23% of one core at 500 Hz with 24 threads → run at 100 Hz.
- Qwen3.5 output may contain reasoning text despite `--reasoning off` (check `no_letter` in MCQ results).
