# Shape of work

Generated 2026-11-11T14:48:08.273579+00:00; since all available evidence.
UTC; busy means matching command lifetime, not measured GPU utilization. Inputs are read-only.
Incomplete ops have no inferred end and are excluded from intervals. Running GPU jobs end at this snapshot.
Elapsed totals use whole ops selected by end/start >= since; timeline intervals are clipped to since.
Experiment time uses embedded timestamp, otherwise file mtime. All detail is retained in JSON.

## 1. Operations

Failure rate denominator: completed outcomes; pending ops remain in count. Failures include om_run result.json even when the wrapper exits 0.

| Agent | Class | Ops | Completed | Elapsed s | Median s | Failure rate |
| --- | --- | --- | --- | --- | --- | --- |
| abrun | over-memory | 56 | 56 | 8591.611 | 145.3595 | 0.0% |
| jobq | build | 5 | 5 | 1187.668 | 220.943 | 0.0% |
| jobq | kernel-screen | 3 | 3 | 736.374 | 88.203 | 66.7% |
| jobq | over-memory | 6 | 6 | 8617.225 | 1241.2015000000001 | 0.0% |
| jobq | test-noise | 5 | 5 | 6.469 | 0.097 | 0.0% |
| jobq | tooling | 6 | 5 | 1806.775 | 309.985 | 0.0% |
| opus-jetson | build | 19 | 18 | 15437.661 | 175.3155 | 16.7% |
| opus-jetson | granite-experiments | 108 | 108 | 1519.263 | 11.1525 | 12.0% |
| opus-jetson | isolation | 8 | 8 | 8.871 | 1.347 | 50.0% |
| opus-jetson | kernel-screen | 3 | 3 | 800.267 | 333.759 | 33.3% |
| opus-jetson | model-preparation | 4 | 4 | 762.919 | 74.568 | 0.0% |
| opus-jetson | over-memory | 63 | 63 | 15013.338 | 158.105 | 19.0% |
| opus-jetson | quality | 4 | 4 | 77.565 | 12.6385 | 50.0% |
| opus-jetson | speculation | 8 | 8 | 1409.438 | 148.6705 | 0.0% |
| opus-jetson | storage | 8 | 8 | 162.427 | 20.287 | 0.0% |
| opus-jetson | tooling | 11 | 11 | 4813.234 | 188.151 | 27.3% |

Unmapped task names: none.

| Tool calls UTC day | Count | By tool |
| --- | --- | --- |
| 2026-11-09 | 83 | Agent: 2, AskUserQuestion: 1, Bash: 71, Edit: 1, ExitPlanMode: 1, Read: 1, ToolSearch: 2, Write: 4 |
| 2026-11-10 | 509 | Agent: 10, Artifact: 3, Bash: 469, Edit: 5, Read: 1, SendUserFile: 1, TaskStop: 1, ToolSearch: 2, WebFetch: 2, WebSearch: 3, Write: 12 |
| 2026-11-11 | 177 | Agent: 7, Bash: 167, Read: 1, TaskStop: 1, Write: 1 |

## 2. Failure taxonomy

First matching regex wins; stderr reads are bounded to a 4 KiB tail per stderr file, including om_run OUTDIR/stderr.txt.

| Category | Count | Example ops (up to 3) |
| --- | --- | --- |
| flag_error | 11 | op-0751d4f0786d, op-29b6ae427db2, op-817420cc765b |
| oom/watchdog | 5 | op-34f484ff0b44, op-4c65e2050f3b, op-bf2ecfd02e23 |
| docker_start | 1 | op-42b9e5a28469 |
| build_fail | 2 | op-5f72a82807c0, op-fdb40d139c4c |
| timeout | 2 | op-5ce5f9a963cb, op-f7dc5536fc7e |
| other | 19 | op-0c736ec49231, op-1a37540ab664, op-2b6791aa3ad9 |

## 3. Timeline

Overlap is contamination risk. Idle gaps are bounded by GPU ops and split at UTC midnight.
Queue idle uses observed submitted-to-start intervals for all jobs (including dependency waits); — means no queue coverage that day.

| UTC day | GPU busy h | Build h | Overlap h | Longest idle h | Queued idle h | Idle bounds UTC |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-09 | 0.3106 | 0.4884 | 0.0523 | 0.4388 | — | 2026-11-09T23:33:40.399418+00:00 to 2026-11-10T00:00:00+00:00 |
| 2026-11-10 | 3.3511 | 3.468 | 0.0935 | 9.0915 | — | 2026-11-10T06:50:12.416213+00:00 to 2026-11-10T15:55:41.657565+00:00 |
| 2026-11-11 | 2.5988 | 0.3302 | 0.0 | 7.1222 | 0.2642 | 2026-11-11T07:12:59.025591+00:00 to 2026-11-11T14:20:18.874311+00:00 |

## 4. Delegation

Wall/tokens sum end events; latest check-in verdict, cumulative issues, maximum rework rounds, any opus fix.

| Task | Worker | Wall s | Tokens | Verdict | Issues | Rework | Opus fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-c2-ledger | codex | 0 | 0 | — | 0 | 0 | False |
| H1 | codex | 269.0 | 33944.0 | PASS 5/5 | 1 | 0 | True |
| H2 | codex | 395.0 | 55172.0 | PASS 4/4 | 1 | 0 | False |
| H2b-H4b | codex | 266.0 | 60549.0 | — | 0 | 0 | False |
| H3 | codex | 313.0 | 97187.0 | PASS 4/4 | 1 | 0 | False |
| H4 | codex | 262.0 | 52067.0 | PASS 3/3 | 3 | 0 | False |
| J-OM10 | codex | 694.0 | 96058.0 | — | 0 | 0 | False |
| J-OM10b | codex | 310.0 | 55325.0 | PASS | 0 | 0 | False |
| J-OM8 | codex | 397.0 | 70621.0 | PASS 5/5 | 2 | 0 | False |
| J-OM8b | codex | 185.0 | 82843.0 | PASS 5/5 | 1 | 0 | False |
| J-OM9 | codex | 352.0 | 57244.0 | PASS 6/6 | 2 | 0 | False |

| Worker | Tasks | Wall s | Tokens | Issues | Rework | Opus fixes |
| --- | --- | --- | --- | --- | --- | --- |
| codex | 11 | 3443.0 | 661010.0 | 11 | 0 | 1 |

### Lead events

Missing workers on contamination/result/failure events are inferred as opus.

| UTC time | Task | Worker | Event | Detail |
| --- | --- | --- | --- | --- |
| 2026-11-11T04:20Z | J-OM8-ab | opus | contamination | run 001-base (start 04:14:29Z) overlapped Codex H3/H4 (ended 04:16:23Z/04:16:47Z), launched outside jobq. Treat 001-base timing as suspect; compare with 006-base. |
| 2026-11-11T04:45Z | J-OM8-ab | opus | result | base 1.13*/1.24, fill 1.33/1.33, fill_random 1.37/1.41 eval t/s; output identical; gate null (A/A 9.3% from contaminated 001-base). fill_random prefill 5.7->1.4. fill stats: 72% of requested expert bytes resident, 11 GB read in ~29 s (~380 MB/s). |
| 2026-11-11T04:55Z | KS-q4K | opus | failure | kernel_screen lower()-cased ggml type names (q4_k vs real q4_K); test fixture was fabricated with the same wrong case so tests passed; CLOSEOUT misattributed to pipeline. Fixed by opus (canonical type map + fixture). Lesson: fixtures must be captured from real tool output. |
| 2026-11-11T06:10Z | J-OM8b-ab | opus | result | 16 runs clean. base 1.1875 (A/A 1.7%), fr 1.275, frd 1.29, frd16 1.3025 (+9.7%, gate improved, threshold 9.4%); prefill kept 5.6-5.7 for frd*, fr 1.46; output identical x16. decode: 764 ms/token, fill 427 ms (164 MB/token @ ~384 MB/s, 230 KB/syscall); ~900 major faults/token remain during decode (telemetry, system-wide, approx window). |
| 2026-11-11T06:20Z | KS-q4K-r2 | opus | failure | submitted from shell without DOCKER_CMD_PREFIX; jobq replays submit-time env -> docker permission denied. Resubmitted as r3 with env in argv. Candidate jobq feature: --require-env K. |
| 2026-11-11T07:30Z | KS-q4K-r3 | opus | result | q4_K kernel screen ran (108 runs): A/B 0.91-1.06 within A/A 0.94-1.18; end-to-end D027 mixE4K +23.6%. Third type confirming micro-screen cannot predict graph-level unfuse. Closes CLOSEOUT 7 item 1. |
| 2026-11-11T07:30Z | J-OM9-ab | opus | result | frd16 reproduced +8.4% (gate improved, A/A 2.3%) on new image. diag overhead ~9%. per token (diag): fill 461 ms, between 394 ms; majflt_between 162/token, pre_use_nonresident 45/token -> faults are minor. POPULATE_READ removes faults (10208->454) but costs ~10 s/run: net 0. Bottleneck = synchronous fill IO (364 MB/s) not overlapped with compute. |
| 2026-11-11T08:40Z | J-OM10-ab | opus | result | pregate predictor recall@8 0.81 (vs 3% naive J-OM7); recall@16 0.955 precision 0.48. But pg8 1.23 < frd16 1.29 (gate improved vs base only), pg16 0.75 regressed. Opus code read: pregate hook runs BEFORE layer L sync fill (L+1 prefetch competes with L demand reads), not-started async jobs never cancelled at L+1 arrival (pending_reads ~3.7k -> duplicate IO), async can occupy all 16 workers. |
| 2026-11-11T09:40Z | J-OM10b-ab | opus | result | pg8b (pregate K=8, demand-first, cancel-at-arrival, async cap 4) 1.42/1.42 t/s vs base 1.15/1.18 (+21.9%), frd16 1.29/1.28 (+10.5% over fill). gate improved, A/A 2.6%, output identical x8. pg12b 1.30/1.27. pg8b: async 7.8 GB, cancelled 4.35 GB, sync 5.4 GB. |

## 5. Experiments

Abrun counts comparison verdicts (or a legacy top-level verdict); unrelated handoff verdicts are excluded.

| Kind | Status / verdict | Count |
| --- | --- | --- |
| result_status | failed | 6 |
| result_status | ok | 99 |
| result_status | watchdog_killed | 5 |
| abrun_verdict | improved | 11 |
| abrun_verdict | null | 6 |
| abrun_verdict | regressed | 1 |

Totals: {"abrun_count": 6, "abrun_verdict_count": 18, "build_h": 4.28660471022129, "completed_ops": 315, "failed_ops": 40, "gpu_busy_h": 6.260478064417839, "ops": 317, "overlap_h": 0.1457733991410997, "result_count": 110}
Input warnings: 0 (details in JSON).
