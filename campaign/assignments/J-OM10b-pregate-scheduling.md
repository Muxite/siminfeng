# Assignment J-OM10b — fix pre-gate prefetch scheduling (component rework; worker Codex)
Repo root: /home/muk/projects/siminfeng/campaign. Builds on 01-06 (cc/mmap_advice/patches/combined-01-02-03-04-05-06.patch, cc/expert_fill/expert_fill.h single source). Engine tree ref/llama.cpp-up1108 @ cf34d311, never modify in place.
## Evidence (experiments/jetson/OM/J-OM10: REPORT.md, `expert_pregate: stats` in */stderr.txt)
Predictor works: recall@8 0.81. But decode pg8 1.23 vs frd16 (no prefetch) 1.29 t/s; pg16 0.75. pg8 run: async_bytes_read 10.3 GB + sync 3.4 GB (frd16 reads ~10.5 GB total) → ~30% more IO; pending_reads 3725, inflight_at_arrival 571.
Lead's diagnosis from 06-pregate.patch (verify, then fix):
 (a) In the CPU observer path, `expert_pregate_hook(il, ...)` (prefetch for L+1) is called BEFORE `expert_fill_selected(il, ...)` (demand fill for L) → prefetch reads compete with the demand fill on the drive and on the workers.
 (b) When layer L+1 arrives, async jobs for L+1 that have NOT started are still queued; the sync fill reads the same non-resident pages, and the stale async jobs later read them again (or evict useful pages).
 (c) Async jobs may occupy all fill workers, delaying demand jobs that arrive while they run.
## Deliverables
1. `cc/mmap_advice/patches/07-pregate-sched.patch` (on 01-06) + `combined-01-02-03-04-05-06-07.patch`:
   - Order: demand fill for L first, then enqueue the L+1 prediction (`LLAMA_EXPERT_PREGATE_ORDER=after` default; `before` keeps old behaviour for A/B).
   - At arrival of layer L+1 (before its demand fill), cancel not-yet-started async jobs belonging to layer L+1's batch (and any older layer's), count `cancelled_jobs` / `cancelled_bytes`. In-flight jobs finish.
   - Async concurrency cap `LLAMA_EXPERT_PREGATE_MAX_INFLIGHT` (default 4 workers may run async jobs at once; demand jobs always preferred).
   - Stats add: cancelled_jobs, cancelled_bytes, async_wait_ns_in_demand (time demand fill spent while any async job of the same layer was in flight, if cheap to measure; else omit and say so).
   - Header single-source rule; `make -C cc/expert_fill test` passes, add unit tests for cancel-before-start and the in-flight cap.
2. COMPILE-CHECK 07 section.
3. `cc/abrun/specs/J-OM10b.json` image `campaign/llamacpp:jetson-up1108-om10b`, argv as J-OM10. Arms: base; frd16 (as J-OM10); pg8b (PREGATE=1 K=8, ORDER=after, MAX_INFLIGHT=4); pg12b (K=12, same). Order explicit: base frd16 pg8b pg12b pg12b pg8b frd16 base. Per-arm switch regexes as in J-OM10 for pg arms.
## Acceptance (checker runs)
- A1 `make -C cc/expert_fill test` PASS (incl. new cancel/cap tests by name).
- A2 fresh cf34d311: sequential 01..07 == combined-01..07 (diff -r empty).
- A3 COMPILE-CHECK 07 all exit 0.
- A4 dry-run J-OM10b = 8 runs in order.
- A5 header single-source diff empty.
- A6 checker quotes the call order in the observer (demand fill before pregate enqueue when ORDER=after) and the cancel site.
Constraints: no docker/GPU/full build; no commits; no evidence edits. Time 60 min. Final <= 10 lines.
