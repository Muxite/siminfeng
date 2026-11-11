---
name: engine-ab-experiment
description: Use when planning, running, or assessing a counterbalanced inference-engine A/B experiment in this lab.
---

Work from /home/muk/projects/siminfeng/campaign.
Read c2/METHODS.md methods and cc/abrun/README.md; use the latter's actual schema.

1. State the hypothesis, host/model/type/batch scope, quality budget and D012 claim
   class before running. Verify model and build identities; fix prompt, seed,
   offload, context and workload. Prefer one image with env-only arms.
2. Copy cc/abrun/specs/example-om7-replay.json to a new spec, replacing id,
   images, argv, switches and fresh output destination. Keep mandatory `base`.
   Choose ABBA for two arms, ABCCBA for three, and enough repeated rounds.
   Env values must be whitespace-free strings. Set candidate expect_switch=true
   and switch_regex to an actual activation log; base must remain inactive.
3. Select eval_tps for completion or tg_steady for warmed benchmark throughput.
   Keep raw samples; do not quietly substitute avg_ts for steady throughput.
4. Inspect the planned commands:
   `python3 cc/abrun/abrun.py SPEC.json --dry-run`
   This does not preflight. Probe exact image/tool capabilities with
   `python3 cc/abrun/preflight.py IMAGE TOOL -- FLAGS...` when preparing argv;
   execution automatically preflights again (cached help; beware retagged images).
5. Submit the whole experiment:
   `python3 cc/abrun/abrun.py SPEC.json --submit`
   Use bin/jobq show/wait on its job ID; arrange `bin/jobq worker --once` if no
   worker exists. Never run a queued parent that submits/waits on its own GPU job.
   Use dependencies for prerequisite builds; no build or heavy CPU overlap.
6. Read numbered summary.json files, verdict.json and REPORT.md. Check status,
   activation, missing metrics/reasons, output hashes, concurrent_jobs and external
   load. Label contaminated timings UNSCORED and retain qualified counters only.
7. Apply cc/abrun/gate.py's exact A/A-gap + drift threshold, then the separate
   decode-path ub1 KL/top-1/PPL and greedy-output gate from c2/METHODS.md.
   Identical hashes alone do not prove numerical equivalence; classify differing
   and intentionally lossy outputs explicitly. INVALID or absent replication is
   not a win. Report memory domain/footprint, quality and speed together.
8. Classify full-engine gain, quantization/offload trade-off, component,
   simulation, negative/null, or dependency update; cite decisions/D012 and
   evidence, with tested scope and transfer limits. Do not promote profiler or
   replay numbers to untraced engine performance.
