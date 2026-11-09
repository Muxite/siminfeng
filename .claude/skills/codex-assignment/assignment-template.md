# Assignment T-<ID> (campaign <cN>, x86-main; owner Claude Opus; worker Codex) — <one-line goal, in correctness/engineering terms>
Context: <pinned base: llama.cpp commit 31288c9c6 in ref/llama.cpp (read-only); relevant decisions D0xx; prior artifacts to reuse (paths)>.
HOST NOTE: <a GPU timing job is running -> pure Python only, no compiles / light CPU;  OR  CPU free, use nice -n 10 and -j4>.
Goal: <the single question or deliverable; what "done" means>.
Deliver:
 1. <file path 1> — <behaviour, CLI, schema; "never load full tensors" style invariants>
 2. <file path 2> ...
 3. <evidence artifact(s) under artifacts/codex<ID>/ : logs, JSON, before/after>
Tests: <pytest on REAL records (name the real files: lab/jobs/..., builds/results/..., ~/models/campaign/gguf/*.gguf) PLUS synthetic edge cases. Real-record test is mandatory.>
Invariants: <no behaviour change for default path; deterministic; stdlib/numpy only; read-only inputs...>
Allowed writes: <cc/<name>/**, artifacts/codex<ID>/**>  (nothing else). <N> min, no network, no Docker, no GPU, nice, -j4.
Final message: <files; test result; key numbers; root cause in 3 sentences; open risks/gaps>.
