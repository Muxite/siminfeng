# Assignment J-OM8b — phase-aware expert advice + fill tuning knobs (component; host jetson-arm; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Builds on J-OM8 (assignments/J-OM8-expert-pread-fill.md; cc/expert_fill/; cc/mmap_advice/patches/03-expert-fill.patch). Engine tree ref/llama.cpp-up1108 @ cf34d311 — do not modify in place; work in a /tmp copy.
## Evidence (experiments/jetson/OM/J-OM8/, REPORT.md + */summary.json + `expert_fill: stats` lines in */stderr.txt)
base eval 1.13/1.24 t/s, fill 1.33/1.33, fill_random (MADV_RANDOM on experts from load) 1.37/1.41; output identical. But fill_random prefill collapsed 5.7 -> 1.4 t/s (prompt faults become 4-9 KiB IOs). Fill: 72% of requested expert bytes already resident; 11 GB read in ~29 s per run (~380 MB/s, drive ceiling ~536 MiB/s at 576K qd8).
## Deliverables
1. `cc/mmap_advice/patches/04-fill-phase.patch` (applies on top of 01+02+03) and `combined-01-02-03-04.patch`:
   a. `LLAMA_EXPERT_FILL_RANDOM=decode`: keep default advice during prompt processing; issue MADV_RANDOM on the expert ranges once, at the first ubatch with n_tokens == 1 (log `expert_fill: random advice applied at decode` once). Values `1` (from load, existing) and `0` unchanged.
   b. Fill only for decode-like ubatches when `LLAMA_EXPERT_FILL_MIN_TOKENS`/`LLAMA_EXPERT_FILL_MAX_TOKENS` say so: env `LLAMA_EXPERT_FILL_MAX_TOKENS` (default 0 = no limit): skip fill when ubatch n_tokens > value (prefill then relies on readahead).
   c. Split counters by phase: stats line gains `prompt_` and `decode_` prefixed copies of calls/bytes_read/fill_wall_ns (decode = n_tokens==1).
   d. If the header changes, update cc/expert_fill/expert_fill.h too (single source of truth rule from J-OM8 A5 still holds against the final header in combined-01-02-03-04) and keep `make -C cc/expert_fill test` passing.
2. COMPILE-CHECK: append to cc/expert_fill/COMPILE-CHECK.txt a section for 04 (g++ -fsyntax-only of every touched .cpp in the 01-04 tree, exit codes).
3. Spec `cc/abrun/specs/J-OM8b.json` (image `campaign/llamacpp:jetson-up1108-om8b`; same argv as cc/abrun/specs/J-OM8.json incl. `-lv 4`): arms
   base (no fill), fr (FILL=1 RANDOM=1, threads 8 chunk 1024), frd (FILL=1 RANDOM=decode MAX_TOKENS=1, threads 8 chunk 1024), frd16 (as frd with threads 16 chunk 2048). Order explicit, 2 rounds, counterbalanced: base fr frd frd16 frd16 frd fr base base fr frd frd16 frd16 frd fr base (use `explicit_order` if `rounds` cannot express it; check cc/abrun/abrun.py). switch_regex must include `expert_fill: active`; for frd arms also require `random advice applied at decode` (if abrun supports only one regex list per spec, list both and confirm the INVALID logic still works: base must match none, fill arms must match >= 1).
## Acceptance (checker runs)
- A1 `make -C cc/expert_fill test` PASS.
- A2 fresh cf34d311 extract: sequential 01,02,03,04 == combined-01-02-03-04 (diff -r empty); `git apply --check` ok.
- A3 COMPILE-CHECK 04 section: all exit 0.
- A4 `python3 cc/abrun/abrun.py cc/abrun/specs/J-OM8b.json --dry-run` lists 16 runs in the order above.
- A5 grep in 04 patch: `random advice applied at decode`, `LLAMA_EXPERT_FILL_MAX_TOKENS`, `decode_bytes_read`.
Constraints: no docker/GPU/full build; no commits; no evidence edits. Time 60 min. Final <= 10 lines with hook location for the decode transition.
