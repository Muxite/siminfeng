# Assignment J-OM11 — pregate: strict config, effective-config line, timeliness counters, lookahead 2 (worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Builds on 01-07 (cc/mmap_advice/patches/combined-01-02-03-04-05-06-07.patch; header single source cc/expert_fill/expert_fill.h copied into the patch as src/llama-expert-fill.h). Engine tree ref/llama.cpp-up1108 @ cf34d311: never modify in place; work in a scratch copy.
Read: decisions/D034*, D035*, c2/MEASUREMENT-GAPS.md §3-4, assignments/J-OM10b-pregate-scheduling.md.
## Problems
- J-OM10c arm pg10i8 asked K=10 and silently ran K=8 (parser accepts only 12/16). Every env var of 03-07 must be parsed strictly.
- We cannot tell whether correct predictions arrive in time.
## Deliverables: 08-pregate-observe.patch (on 01-07) + combined-01..08.patch
1. Strict env parsing for ALL LLAMA_EXPERT_FILL_* and LLAMA_EXPERT_PREGATE_* vars: integers accept any value in a documented range (K 1..32, MAX_INFLIGHT 1..64, threads 1..64, chunk 64..65536 KiB...); invalid/out-of-range → print `expert_config: invalid NAME=VALUE (range ...)` and abort (std::abort / GGML_ABORT) at init. Unknown LLAMA_EXPERT_* vars → abort too (typo protection).
2. One line at init: `expert_config: effective fill=.. random=.. max_tokens=.. threads=.. chunk_kib=.. pregate=.. k=.. order=.. max_inflight=.. lookahead=..` (all parsed values, including defaults).
3. Timeliness counters in the stats line (at exit): for each predicted (layer, expert) at its layer's arrival, classify its async job as done / inflight / queued(cancelled) / not-enqueued(already resident at predict time) → `pred_done`, `pred_inflight`, `pred_queued`, `pred_resident_at_predict`; and for correct predictions only (expert actually selected): `hit_done`, `hit_inflight`, `hit_queued`, `hit_resident`. Also `demand_wait_ns_total` (wall time in demand fill, all layers) and `demand_wait_ns_per_token`.
4. `LLAMA_EXPERT_PREGATE_LOOKAHEAD` = 1 (default, current behaviour) or 2: at layer L, also predict layer L+2 using x_L and router of L+2 (enqueue after the L+1 jobs, lower priority: L+1 jobs first in queue). Cancel rule extends naturally (arrival of layer M cancels queued jobs for layers <= M). Stats per lookahead distance: recall@k for d=1 and d=2 separately.
5. Unit tests in cc/expert_fill (strict parser table-driven incl. K=10 accepted, K=0/33/"abc" rejected; queue ordering L+1 before L+2; classification counters with a fake residency oracle). `make -C cc/expert_fill test` passes.
6. COMPILE-CHECK 08 section (g++ -fsyntax-only on touched files of a freshly patched cf34d311 tree, as for 07).
7. cc/abrun/specs/J-OM11.json: image campaign/llamacpp:jetson-up1108-om11, argv as J-OM10c, metric eval_ms_per_token direction lower if gate supports it else eval_tps. Arms: base (no env), pg8 (FILL frd16 env + PREGATE K=8 ORDER=after MAX_INFLIGHT=4), pg10 (K=10), pg8la2 (K=8 LOOKAHEAD=2), pg12la2 (K=12 LOOKAHEAD=2). Per-arm switch regex must match the effective line VALUES (e.g. `expert_config: effective .*k=10 `) — ALL-match. Order explicit, counterbalanced, 2 reps/arm (10 runs): base pg8 pg10 pg8la2 pg12la2 pg12la2 pg8la2 pg10 pg8 base.
## Acceptance (checker runs)
- A1 make test PASS incl. named strict-parser and lookahead-order tests.
- A2 fresh cf34d311: sequential 01..08 == combined-01..08 (diff -r empty), patch -F0.
- A3 COMPILE-CHECK 08 exit 0 for all touched files.
- A4 abrun dry-run J-OM11 = 10 runs in that order; each arm's regex includes its k value.
- A5 header byte-equal to patch copy.
- A6 checker quotes abort path for K=0 and the effective-config print site.
Constraints: no docker/GPU/full build; no commits; no evidence edits. Time 75 min. Final <= 10 lines.
