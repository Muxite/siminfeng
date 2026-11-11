# Assignment D-c2-ledger — consolidate c2 engine results into decisions + CLOSEOUT addendum (worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Docs only. Runs after J-OM10c (gpu) finishes; include it.
## Sources (read, never edit)
experiments/jetson/OM/{J-OM8,J-OM8b,J-OM9,J-OM10,J-OM10b,J-OM10c}/ (REPORT.md, verdict.json, */summary.json, stderr `expert_fill:`/`expert_pregate: stats` lines); experiments/jetson/KS-q4K-20261111-r3/q4_K/results.md; c2/worklog.jsonl (checkin events); cc/mmap_advice/patches/0[3-7]*.patch; cc/kernel_screen/kernel_screen.py git diff (case-folding bug: q4_K lower-cased to q4_k → wrong ggml type; fixture had been fabricated with q4_k).
## Deliverables
1. decisions/D034-expert-pread-fill.md (J-OM8, 8b, 9): what, numbers with n and A/A, the chosen config (FILL=1 RANDOM=decode MAX_TOKENS=1 threads 16 chunk 2048), why POPULATE_READ is net 0, bottleneck = serial synchronous fill IO. Follow format of decisions/D033-*.md.
2. decisions/D035-pregate-prefetch.md (J-OM10, 10b, 10c): router-from-host prediction recall@k, why first version lost (order/stale/occupancy), 07 fixes, J-OM10b/10c numbers, best config, cost (cancelled bytes), output identity.
3. decisions/D036-ks-q4k-and-screen-bug.md: KS-q4K r1 (case bug) / r2 (env not exported to jobq) / r3 result (micro-screen does not predict D027 graph-level unfuse); the CLOSEOUT text that attributed earlier KS results must be marked as affected where it is.
4. CLOSEOUT.md: append a section "## Addendum c2 (2026-11-11)" — table of new results (rows like existing results table), harness components added (bin/jobq, cc/abrun, c2/shape_report.py, c2/METHODS.md, c2/skills), and a corrections list. Do not rewrite existing sections; only append, plus one-line "(see Addendum c2)" markers next to KS rows that the bug affects.
5. Regenerate c2/SHAPE.md and c2/shape.json with `python3 c2/shape_report.py` (its usual args; read its --help).
Every number must be copied from a source file; cite the path next to each table.
## Acceptance (checker runs)
- A1 every numeric value in D034/D035/D036 tables found verbatim (or rounded to 2-3 sig figs) in a cited source file — checker samples 8.
- A2 `git diff CLOSEOUT.md` shows only additions (no '-' lines except none).
- A3 shape_report regenerated; `python3 -m pytest c2/test_shape_report.py -q` passes.
- A4 no edits outside decisions/, CLOSEOUT.md, c2/SHAPE.md, c2/shape.json.
Constraints: no docker/GPU; no commits. Time 40 min. Final <= 10 lines.
