---
name: kernel-profile
description: Use when investigating decode kernel costs with Nsight Systems or screening a kernel-local patch before a full engine experiment.
---

Work from /home/muk/projects/siminfeng.
Read c2/METHODS.md profiling methods, cc/kernel_screen/README.md,
experiments/jetson/J-P001-nsys-decode/FINDINGS.md,
decisions/D025-jetson-moe-gateup-mmf-decode-gain.md and
decisions/D027-unfuse-by-type-q4k-gain.md. Use evidence commands for this host.

1. State whether the hypothesis concerns a kernel implementation or a graph
   transformation. Record source/patch/toolchain/image, model precision and
   decode shape. Queue profiling/screen execution as gpu; builds as build.
2. Capture a small matched nsys decode workload, retaining report and commands.
   Export kernel names, call counts and summed time for base/candidate. Verify
   export rows; stale SQLite can need --force-export=true to avoid empty CSV.
   Compare kernel inventories to ensure work was not silently skipped.
3. Use traced time only to identify mechanisms/hypotheses. J-P001 found kernel
   rather than launch overhead; J-P002 linked MMVF replacement to MMF calls.
   Tracing perturbs execution; full-engine speed requires clean untraced A/B.
   ncu counters are admin-blocked here; do not change host permissions/settings.
4. For a kernel-local change, prepare test-backend-ops via Build CC with
   --with-tests --extra-target test-backend-ops and the appropriate perf preset
   patch combined with the candidate. Keep ref trees read-only.
5. Run cc/kernel_screen/kernel_screen.py with the built image/binary, Granite
   or Qwen preset, exact types/ops, A/A2/B env arms, repeats and fresh output.
   Follow CLI help/README schema; use `--types q4_K` here (case sensitive),
   not the older uppercase example. Inspect emitted params, nonempty matches,
   separate layouts and counts. Assert activation separately where available.
6. Accept a screen only above its A/A noise with correct cases and complete
   results. It can reject local regressions or prioritize candidates; it cannot
   establish model quality, memory savings or end-to-end speed.
7. KS-20261110-1620 failed to predict D027: graph-fusion selection is absent
   from the measured single-op cases. Use an actual fused graph/model A/B for
   fusion/launch scheduling claims; never extrapolate the standalone ratio.
8. Follow engine-ab-experiment procedure for untraced counterbalanced runs and
   ub1 KL quality, then report component versus full-engine claim (D012).
   Keep platform/type limits: Orin bf16/q4_K gains do not transfer to q8_0 or x86.
