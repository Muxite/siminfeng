---
name: over-memory-run
description: Use when running or assessing inference with weights larger than Jetson physical DRAM, including loader and cache comparisons.
---

Work from /home/muk/projects/siminfeng.
Read c2/METHODS.md over-memory/IO methods, cc/om_run/om_run.sh,
cc/resource_observe/README.md and decisions/D028 through D033 as relevant.

1. Verify model hash/layout, image/build record and exact binary help. This host
   has 7.4 GiB UMA plus about 3.8 GiB zram; Docker --memory is not a CUDA budget.
2. Use a fair explicit baseline: completion `-lm mmap -ngl 99 -cmoe -c 4096`;
   bench uses `-ncmoe 40` and does not support -c/-cmoe here. Keep prompt, context,
   seed and token counts matched. Report auto-default failure separately (D030).
3. Prefer an abrun spec (cc/abrun/README.md), activation assertion and ABBA or
   ABCCBA, submitted with --submit. For a single diagnostic run use jobq gpu:
   ```sh
   bin/jobq submit --class gpu --name OM_ID --timeout 2400 -- \
     bash cc/om_run/om_run.sh FRESH_OUT IMAGE "" llama-completion \
     -m /om/model.gguf -f /om/prompts/p1.txt -n 64 -c 4096 \
     --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap -cmoe
   ```
   Set OM_MODELDIR before submission if changing the host model mount.
   ENV_FLAGS is Docker flags such as `-e KEY=value`, not shell assignments.
4. Inspect effective defaults in code, not the stale header: OM_MIN_AVAIL_MIB=600,
   OM_MAX_SWAP_MIB=2560, OM_MAX_SWAP_RATE=1000; watchdog checks every 250 ms and
   needs two consecutive crossings. Record overrides and why. Do not disable
   protection to force a fit. OM_MAX_S only bounds sampling; job timeout bounds run.
5. Require result.json status ok, no unexplained watchdog trip, activation/stats,
   telemetry coverage and no competing build/heavy CPU work. Check host_id,
   image, env and argv. Distinguish trip-killed, trip-survived, OOM and failure;
   cancel only the owned job/container. D029 validates kill quoting live.
6. Read summary.json (or cc/abrun/run_summary.py): eval_tps, nvme_gib,
   nvme_gib_per_token, avg_kib_per_io, mem_available_min_mib, swap_max_mib,
   gpu_freq_median; inspect refaults, swap and IO windows in raw telemetry.
   Whole-run bytes include loading/prefill; do not label them decode-only.
7. Apply A/A/drift and decode-quality gates; report identical/differs/lossy and
   memory domain. Smaller IO volume/higher cache hit rate alone is not a win:
   D033's 4 KiB reads halved speed. Use D031's first-order model as prediction,
   not validation; peaks above the fio ceiling may be sampling artefacts.
