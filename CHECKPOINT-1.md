# Checkpoint 1 (stage 1 in progress) — 2026-11-09
Done: workspace + run_op (tested pass/fail/kill; D001); host qualified (GPU 16303 MiB, 31 MiB used after approved stops; D002); Granite pinned @0da7a48b (config verified: GraniteMoE, 24L, 32 experts, top-8, hidden 1024, BF16; hashes artifacts/granite-0da7a48b.sha256); llama.cpp image digest sha256:395d01bb... (build 11723, commit 31288c9c6, no nvcc/nsys/ncu); D003 goal addition (reverse-engineer Strata methodology; other engines as inputs).
Active: codex-A (Strata map), codex-B (claims+tests) running; downloads: llama.cpp source clone, nvidia/cuda:12.8.1-devel pull.
Stopped by approval (restore cmds in D002): euaaa-llm-gpu-1, Sunshine user unit.
Unresolved: Nsight perms (RmProfilingAdminOnly=1; try container CAP_SYS_ADMIN first); T-C llama.cpp comparison agent not yet launched (needs clone); GGUF conversion not done; Claude Code hooks + codex native-stream registration not set up (gap).
Next decision: after devel image lands, test nsys/ncu in a CAP_SYS_ADMIN container; convert BF16 GGUF; run first lifecycle experiment.
