# Campaign START-HERE — c2-resume state (2026-11-10, x86-main lane, Claude Opus lead)
Read sections 1-3 and 6 first (about 10 minutes). The unedited c1 wrap-up text is at the end under "c1 snapshot (historical)"; where it disagrees with this page, this page wins (e.g. q8_reuse is no longer "UNTESTED", the goal wording changed).

## 0. Latest: D041–D048 (2026-11-10 overnight). Read this first.
The progress ledger with every table is at https://claude.ai/artifact/8Uqsnssn6R6KY2EtAyTiyD (private to the owner).

| ID | Verdict | Category | One line |
|---|---|---|---|
| D041 | indicative | placement | upstream `--moe-cache-mib` +38% decode at 8 GB on Qwen3-30B, −34% at 4 GB; ncu: 222 GB/s, 40% of kernel time in tiny kernels |
| D042 | indicative | spec + tools | ngram-mod 2.4–3.0× lossless on copy-heavy prompts; bpftrace 0.79 ms/CPU layer; research intake |
| D043 | protocol | — | overnight E1–E4 frozen; A1 (download A/A), A2 (post-hoc decode-path KLD) |
| D044 | **speed rule MET** | placement | cache + CPU-share 0.9 beats static at 4/6/8/10 GB (+30 to +76%), with 3–6% less VRAM; frozen absolute KLD gate miscalibrated (the reference fails its own batch-shape floor of 0.0058 / 96.5%), all placements within that floor |
| D045 | ACCEPTED (static) | spec | ngram on K30: 2.32× / 3.01×, text identical, chat +1%; stacks with the cache (3.9× over static); cache output depends on cache history |
| D045 | trade-off | representation | REAP-15B: 6.6× decode all-GPU, code PPL +8%, wikitext PPL ×2.84; needs a domain-matched task eval |
| D046 | ACCEPTED (copy), chat equivalent (via D048 X2) | spec | MTP on Qwen3.6-35B: +23–42% copy (text identical), +11–14% chat |
| D047 | protocol | — | overnight-2 X1–X8 frozen; A1 reruns (X8 contaminated by a swap episode; X7 flag fix) |
| D048 | results | placement/spec/load | cache transfers to Qwen3.6 (+21 to +42%); MTP beats the same VRAM spent on layers (+9.8% chat); ub 2048 cuts cache TTFT 2.9×; 4 threads beat 8, SMT −50%; cache + MTP stack (+52% chat vs static); cold load 1.48 GB/s |

**Factory rules adopted (discrete GPU, measured on two models):**
1. Expert cache with CPU-share is the default placement. Share 0 if the predicted hit rate is ≥ about 90%, otherwise 0.9.
2. ub sets the TTFT vs VRAM trade: ub 1024–2048 removes the cache's TTFT penalty.
3. MTP n_max 3 when the model ships MTP heads. n-gram for copy-heavy workloads.
4. Gate quality with distributional checks against the reference's own batch-shape floor, never absolute text identity.
5. Threads ≤ physical cores, autotuned. Never SMT.
6. mmap. Budget cold load as GB ÷ disk GB/s.

**Host state:** `euaaa-llm-gpu-1` and Sunshine were restored after the GPU measurements (D002). `kernel.perf_event_paranoid` is still 1 (runtime only; restore with `sudo sysctl -w kernel.perf_event_paranoid=4`). Pushing to `origin x86/c2-resume` is authorized; `main` is not.

## 1. Goal now
User goal: a **MoE inference-engine software factory**: input = model + system (host profile, memory budget) -> output = an engine (build + model representation + runtime flags) that uses **much less memory at about the same accuracy**. Load time, TTFT and tok/s may change (report them); KV-cache changes are allowed. Also in scope: (a) **Lab Manager design** (`MANAGER-DESIGN.md`, `LAB-FEASIBILITY.md`; deterministic coordination, our `bin/lab` is a TEMPORARY reference adapter, not the other team's lab agent), (b) the **CC portfolio** (`cc-inventory.csv`, builds/tools/methodologies with observed-vs-inferred status), (c) a **Friday 5-8 minute demo** (`cc/demo/DEMO.md`, `cc/demo/demo.sh`). Every result carries a claim category (computation-preserving engine gain | representation trade-off | runtime-placement trade-off | failed candidate | infrastructure/methodology | prediction) and a D012 transfer class (transferable | discrete-only | prediction). Proxies: Granite-3.1-1b-a400m (D001-D037; experts 0.84 MiB) and Qwen3-30B-A3B Q4_K_M (D038/D039; experts 2.72 MiB, does not fit the 16 GB GPU). Engine pin: llama.cpp `31288c9c6` (full `31288c9c66e59377f7788d920678a565bb89dd37`).

## 2. Lanes, hosts, branch
- Branch `x86/c2-resume` (HEAD bb6a65e at time of writing, 30 commits). `main` untouched. Pushing this branch to `origin` is authorized (2026-11-10); pushing `main` is not.
- x86-main lane (this host): RTX 5070 Ti (discrete, sm_120), amd64, Docker (containerd store), one GPU job at a time by procedure only (no admission lock, D007). Models in `~/models/campaign/{gguf,qwen3-30b-a3b,data,tools}`; llama.cpp clone `ref/` and image archives are NOT in git.
- jetson-arm lane: separate host/lead. No SSH route. Exchange = operator-carried git bundle of `x86/c2-resume` (`git bundle verify` + sha256 in BUNDLE.sha256; ~68 MB history, no models/images/profiler reports) plus `handoff/jetson/EXCHANGE-1.md`. Jetson replies as `handoff/jetson/REPLY-1.md` + bundle of `jetson/<task>`. **No reply exists yet** (only EXCHANGE-1.md is in `handoff/jetson/`). Ownership: jetson owns `platform/jetson/**`, `cc/resource_observe/backends/jetson*`, `handoff/jetson/**` replies; x86 owns `cc/engine_handoff`, `cc/build_engine`, `bin/lab`, `cc/resource_observe/CONTRACT.md`.
- Working tree is dirty by design (many untracked `artifacts/codex*`; modified `evidence/claude-code.jsonl`, `artifacts/x2/loadmon.pid`, and `cc/routing_trace/routing-trace.patch` which T-QRT is editing). Do not `git add -A` blindly.

## 3. Results D020-D040 (verdict | category | numbers; all x86-main RTX 5070 Ti unless noted)
| ID | Verdict | Category | One line |
|---|---|---|---|
| D020 | fixed | lab bug | runner died on non-UTF-8 output; raw bytes in, decode at parser (T-M); rerun = att-2 |
| D021 | qualified (limited) | infra | gpu_budget shim 39/39 CUDA tests; cooperative cap only, NOT a hard VRAM limit; default ctx KV 6144 MiB dominates |
| D022 | measured | placement trade-off | static `--n-cpu-moe` beats engine LRU cache at every matched footprint: 2.8x at ~880 MiB, 1.5x at ~1385 (n12 197 vs c384 187 t/s, n24 128 vs c128 54); same text |
| D023 | qualified (limited) | infra | NO_VMM engine under shim: counted 1078.9 MiB vs 1388 nvidia-smi (~310 MiB uncounted); cap 900 + `-ngl 99` -> clean abort; auto-fit under cap too conservative |
| D024 | REJECTED | failed candidate | static hot-expert patch never engages (text sha = CPU path 76dc9422, speed ~off); untimed |
| D025 | demonstrated | infra | full chain build -> job -> proposal -> admission -> run -> result; x86 accepted, jetson template rejected |
| D026/D029 | ACCEPTED | computation-preserving engine gain | q8_reuse: Q8_0 tg128 791.3 -> 807.6 = +2.06%; mixE4K +1.78% (sub-threshold); text/PPL identical; MUL_MAT 2445/2445 |
| D027/D030 | REJECTED | failed candidate | hybrid expert split slower than static at equal GPU MiB: -16% (n18), -19% (n12), -32% (n6); CUDA unit-test mismatch still open (D027 amend. 1) |
| D028/D033 | ACCEPTED (q8_0 KV) / REJECTED (q4_0) | representation trade-off + engine defect fix | stock quantized KV aborts (hash set 2065 > 2053 slots); `kv_quant_fix` -> q8_0 KV @32k: -720 MiB, PPL 7.5392 vs 7.5385, decode -10.5%; q4_0 -1104 MiB but PPL +7.9% (fails 1% rule); FA alone: -984 MiB, +10% decode (D028, PPL equivalence of FA not measured) |
| D031 | diagnosis | methodology | nsys: per-layer CPU<->GPU handoff dominates at batch 1 (static n12 5.35 ms vs split 7.38 ms); Granite is a "small-expert regime" proxy |
| D034 | validated in range | prediction scored | planner at 900 MiB/ctx4096: GPU MiB +0.1..+0.3%, decode +2.6..+7.2% optimistic (LOO bound 16%); KLD 0.040 |
| D035 | imatrix ACCEPTED; pap12_q3 **RETRACTED (D040)**; others rejected | representation trade-off | mix4_imat KLD 0.040 -> 0.027 (-33%, same bytes); pap12_q3 KLD 0.073 (+9.8% in a download-contaminated window, provisional); pap q2 KLD 0.19-0.36 unacceptable; pap24_q3 KLD 0.117 fails 0.08 |
| D036 | demonstrated | component finding | q8base vs q8reuse builds differ in exactly one .so (libggml-cuda); swapping it reproduces +2.0% |
| D037 | release-1 correctness-qualified | infra + component qualification | stack q8_reuse + kv_quant_fix (patch sha 2e340d15, tree 954f7c74, order-independent); factory run #2 reproduces release-1 byte-identically (20 files); run #1 with `--max-kld 0.08` found 0 plans (fail closed) |
| D038 | protocol (frozen before results) | - | Qwen frontier + pre-registered planner/orchestrator predictions; (b) release speed, (c) PAP rerun |
| D039 | measured | runtime-placement trade-off | Qwen3-30B-A3B decode t/s [GPU MiB]: K8 93.5 [15684], K18 54.6 [12246], K30 36.6 [8162], K48 24.4 [1798]: **10x less GPU memory for 3.8x slower**, weights unchanged; CPU experts ~30 GB/s (Granite ~26); K18+FA+q8 KV 12066 MiB, 53.9 t/s. Orchestrator bytes/bandwidth prediction OK; **planner extrapolation FAILED ~7x pessimistic** (K18 8.1 vs 54.6) |
| D040 | release-1 speed QUALIFIED; pap12_q3 REJECTED | engine gain confirmed / representation | release-1 Q8_0 tg128 793.2 -> 810.4 = +2.17%; PAP quiet-host rerun pap12_q3_n12 +1.5% (< +5% rule); pap24_q3 +11.8% but KLD 0.117 fails. Q3_K CPU unpack cost: fewer bits != proportionally faster |
Pending (no verdict): T-PL4 (bandwidth decode model) and T-QRT (Qwen routing skew), both Codex, running now (section 6). Release-1 engine = the only accepted full-engine gain (+2.1%); memory wins come from placement/representation/KV, not engine code.

## 4. Factory map (stage -> component -> status)
| Stage | Component (path under `campaign/`) | Status |
|---|---|---|
| inspect | `cc/model_inspect/inspect_model.py` (T-MI, header-only, 35 tests; `--trace` adds routing coverage) | done |
| system | `cc/resource_observe/` (resource-sample/0; x86 nvidia-smi + Linux; Jetson/tegrastats backend NOT demonstrated), host profiles `cc/engine_handoff/hosts/` | x86 done |
| plan | `cc/moe_planner/plan.py` (+ quality ledger, T-PL3, 64 measured entries, fail-closed; docs `cc/moe_planner/PLANNER.md`) | works in calibrated Granite range; extrapolation to Qwen failed (D039) -> T-PL4 |
| recipe | `cc/factory/factory.py` recipe stage; models `cc/pap/make_pap.sh` (imatrix accepted; PAP proposals pending/rejected) | done |
| components | `cc/factory/components.json` registry; `cc/components/` (versioning, `patch_stack.py`, `abi_diff.py`, `release-stack-1.patch`) | accepted: q8_reuse, kv_quant_fix, imatrix_quant, flash-attn, gpu_budget (infra only); rejected: hybrid_split, moe_static_hot; pap: still "pending" in registry (stale, see section 8) |
| build | `cc/build_engine/build_engine.py` (Build CC v2: cmake defines, host-side linkage, config digest vs `run_ref`) | done; release-1 built |
| measure | `bin/lab` (submit/run/status/cancel/reconcile/chaos/register/image-load), tests `lab/tests/*.json`, `cc/nsys_decode`, `cc/routing_trace` | done; factory only SUBMITS, does not start GPU jobs |
| report | `cc/factory/factory.py report` (prediction vs measured, exact test-name joins) | done |
| handoff | `cc/engine_handoff` (facts/validate/envelopes/observe-archive); job envelopes T-S | offline; no remote receiver exercised |
| demo | `cc/demo/demo.sh`, `cc/demo/inspect.py` (T-DEMO1 live rehearsal passed once; Codex rerun artifacts in `artifacts/codexDEMO1/`) | ready, needs idle GPU |

## 5. How to run
- Factory dry run (no Docker/GPU): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 python -B cc/factory/factory.py run --model ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf --host x86-main --budget-gpu-mib 900 --ctx 4096 --dry-run --run-dir artifacts/codexFAC/factory/real-dry-run`. Real run: drop `--dry-run` (authorizes Build CC; needs Docker + pinned CUDA bases; use a FRESH `--run-dir`, e.g. `artifacts/codexFAC/run-granite-900b` exists = run #2). Stages: inspect, system, plan, recipe, build, measure, report; `--stages inspect,system,plan` stops early; `--max-kld`, `--include-pending`, `--objective`, `--allow-extrapolation`. Offline score of a finished job: `python -B cc/factory/factory.py report --predictions artifacts/codexFAC/report-predictions.json --result lab/jobs/job-ad883810b3/att-1/result.json --out artifacts/codexFAC/report-real.json`. Docs: `cc/factory/README.md`. Tests: `python -B -m pytest -q -p no:cacheprovider cc/factory/test_factory.py --basetemp=artifacts/codexFAC/pytest-tmp`.
- release-1 image: `builds/results/release-1/build-record.json` -> `.image.run_ref` = `sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e` (= docker-reported ID = OCI index digest; execute with this); `.image.config_digest` = `sha256:d6d073145e272a2c47afe69dc20e2e0a66deb9830203c236b8a6579ccc0e493f` (identity/admission key, NOT accepted by `docker run`). Tag `campaign/llamacpp:release1`; lab image name `llamacpp-release1` (`lab/images/llamacpp-release1.json`). Register: `bin/lab image <op> <name> <run_ref>`.
- Lab: `bin/lab submit --image llamacpp-release1 --model <model-dir> --tests <name,name> --key <unique-key> [--data D --model-sha256 H]` -> prints job id; `bin/lab run <job>`; `bin/lab status <job>`; a rerun is a NEW attempt (att-2), never a relabel. Wrap every external command in `bin/run_op --agent <a> --task <T> ...` (evidence goes to `evidence/*.jsonl`). Tests are data in `lab/tests/*.json` (qwen1, c1q8, pap1, plan1, kv1, hsplit1, ...); the job embeds immutable copies.
- Demo: `DEMO_DRY=1 bash cc/demo/demo.sh` (print only), `DEMO_PAUSE=1 bash cc/demo/demo.sh` (rehearsal), `bash cc/demo/demo.sh` (live; idle GPU, image `llamacpp-novmm4`, mixE4K, `artifacts/traces/in/chat.txt`). Do not claim remote/Jetson execution, hard VRAM cap, or accuracy/performance superiority (see DEMO.md).
- Skills (`.claude/skills/*/SKILL.md`): `codex-assignment` (write/launch/review bounded Codex tasks; Codex has no Docker/GPU), `decision-note` (D0xx notes + claim category + accept/reject), `engine-candidate` (patch -> Build CC -> engagement -> correctness -> only then GPU timing; run_ref, test-backend-ops, text identity), `frozen-experiment` (pre-registered A/B on GPU lab: alternating order, host-load monitor, aggregate, verdict), `model-variant` (GGUF quantization/imatrix/PAP, KLD gate).
- Environment: python3 stdlib + PyYAML/pytest (+ jsonschema for `cc/components` tests). Re-create lab image if missing: see "Resume" in the historical section.

## 6. Open work / next bounded tasks (in this order)
1. **Collect T-PL4** (`assignments/T-PL4-bandwidth-decode-model.md`, Codex, writes `cc/moe_planner/**`, `artifacts/codexPL4/`): bytes/bandwidth decode model, per-quant-type CPU screen `cc/moe_planner/cpu_type_screen.py`, leave-one-model-out errors, rescoring of the pre-registered Qwen scenarios (12000/8000/4000 MiB). Review, then decision note.
2. **Collect T-QRT** (`assignments/T-QRT-qwen-routing-skew.md`, Codex, writes `cc/routing_trace/**`, `artifacts/codexQRT/`): Qwen3-30B-A3B routing skew -> train/eval hot-set coverage -> predicted crossover vs the static frontier. **Hybrid expert split for Qwen only if T-QRT predicts a crossover** (D031 argues large experts amortize handoff; D030 shows it lost on Granite); otherwise record "not justified".
3. **Quiet-host CPU screen rerun**: T-PL4 and T-QRT ran CPU jobs concurrently, so PL4 CPU constants are indicative only (gaps.md). Rerun `python3 cc/moe_planner/cpu_type_screen.py` on a quiet host (check `pgrep -af 'codex exec'`, load < 1) before adopting constants.
4. **Planner extrapolation fix**: switch the decode model to bytes / host bandwidth (BW_cpu per type ~26-30 GB/s, t_handoff per CPU layer) instead of Granite-fitted per-layer time (D039); re-run the three pre-registered Qwen scenarios; add D034 Granite points (+5% optimism) to calibration.
5. **Jetson**: send the bundle + `handoff/jetson/EXCHANGE-1.md`; requested items are 1 J1 inventory, 2 resource envelope (max -j, ctx, spool), 3 native sm_87 build + one mixE4K run, 4 tegrastats fields + fixtures (the file lists four numbered requests; no J5). Cross-platform check: layer-offload collapse (D014) is PCIe-specific and should differ on shared DRAM.
6. Other candidates: apply PAP only after per-type CPU screen (D040 rule); long-context combined plan (1317 MiB ctx32768 q8 KV is the smallest feasible per PLANNER.md; 1200 MiB infeasible); engagement + KLD on Qwen with imatrix; commit decision notes D041+ for T-PL4/T-QRT.

## 7. Host state to restore, active resources
- D002 changes (approved): container `euaaa-llm-gpu-1` stopped (`docker start euaaa-llm-gpu-1`; currently "Exited (137)") and user unit Sunshine stopped (`systemctl --user start app-dev.lizardbyte.app.Sunshine.service`; currently inactive). GPU lock removed (D007). No host-wide driver/profiling change.
- Check before any timing window: `docker ps --filter label=campaign=c1` (none expected), `pgrep -af 'codex exec'` (at time of writing: T-PL4 pid 1860892 and T-QRT pid 1860893, started under `bin/run_op`), `uptime` (load 3.9 with those running), no compile/download. Timed windows run only on a quiet host; otherwise label "untimed".
- Qwen mmap needs ~18.6 GB of ~46 GB host RAM: one such process at a time; Qwen runs need container memory 40g.

## 8. Known gaps
- `decisions/gaps.md`: G-001 Claude Code project hooks installed mid-session, no live `claude_hook` records seen (only selftest); verify after restart. Delegation friction: Codex T-KVB refused by provider for crash/gdb wording (op-9a911bb7e729); phrase engine-debug briefs as correctness bugs; log refusals as failed attempts. Concurrent CPU screens (PL4 + QRT) -> indicative only.
- MANAGER-DESIGN open: admission/queueing and capacity model, leases/heartbeats (not used), runner timeouts, disk/capture budgets, retention/GC, remote receiver states (received/accepted/rejected/needs-information) never exchanged with a real lab agent, host-loss failure domain (evidence on same host), hard device-memory limit not demonstrated. New requirements R-A..R-E (pinned CC interface contracts, observed-vs-declared admission, load contamination, memory as first-class measurement, delegation verification).
- Technical: hybrid_split CUDA unit-test mismatch unresolved (D027); PL-quality for combined placements inherited by rule, not measured; imatrix calibration only on wikitext; FA PPL equivalence unmeasured (D028); no combined release-1 + hybrid qualification; Jetson acceptance/backends, remote transport and crash-during-chain not demonstrated; Claude-side token cost not captured.
- Doc inconsistencies found while writing this page (not fixed; I edited only this file): (1) `cc/factory/README.md` calls kv_quant_fix "pending" but `components.json` and D037 say accepted; (2) `components.json` `pap` scope still says "pending quiet-host rerun" though D040 rejected pap12_q3; (3) `MANAGER-DESIGN.md` c2 items 2, 5, 10 say input identity "not run against a live job", T-N3 "not confirmed", job envelopes "not demonstrated", all superseded by D025; (4) `cc/components/COMPONENTS.md` strict ABI diff says base -> reuse libggml-cuda is INCOMPATIBLE (weak std::_Hashtable exports) while D036 empirically shows the swap works: keep both, the swap claim is empirical for this exact pair only; (5) `decisions/D035` still reads "PROVISIONALLY ACCEPTED" for pap12_q3, superseded by D040.

---
## c1 snapshot (historical)
# Campaign c1 — start here (wrap-up snapshot, 2026-11-09)
Goal (user, latest): methodology + tools for fitting a very large MoE (~500 GB weights -> 128 GB unified memory + 4 TB SSD) at similar accuracy and acceptable tokens/s. Proxy: ibm-granite/granite-3.1-1b-a400m-instruct on llama.cpp (commit 31288c9c6), RTX 5070 Ti (discrete). Every result is labelled transferable / discrete-only / prediction (D012).

## What is here
- `decisions/D001..D020` — chronological decisions with evidence (read D012-D020 first).
- `LAB-FEASIBILITY.md`, `MANAGER-DESIGN.md`, `cc-inventory.csv` — Lab Manager findings and the CC inventory (observed vs inferred).
- `bin/` — `run_op` (recorded commands), `lab` (image/job/attempt/cancel/reconcile/register/image-load/chaos), `perf_summary`, `index_evidence`, `cost_report`, `cuda_compile_check`.
- `cc/` — Build CC (`build_engine`), `engine_handoff`, `routing_trace` (patch + replay), `memory_pareto`, `hybrid_expert` (Opus harness), `q8_reuse` (UNTESTED speed patch), `gpu_budget` (memory-restriction shim, may be incomplete), tests (`pytest` in each dir).
- `evidence/*.jsonl` — raw per-agent streams (intent/started/outcome); `index/ops.jsonl` merged; `lab/jobs/*` job/attempt state + results; `experiments/` summaries; `claims/paper-claims.md` (15 source-supported claims, none tested yet).
Not committed (reproducible/bulky): llama.cpp clone (`ref/`), image archives, nsys/ncu reports, base logits, traces (regenerate with the lab tests), Codex scratch trees. Models live in `~/models/campaign` (Granite pin 0da7a48b; GGUFs bf16/Q8_0/Q4_K_M/mixE4K/3K/2K).

## Findings to trust (with caveats in the D-notes)
1. Experts are 90.4% of weight bytes; mixed precision (experts Q4_K, rest Q8_0) halves Q4_K_M's quality loss at equal memory (D013).
2. Static layer offload (`--n-cpu-moe`) cuts GPU MiB linearly but decode collapses (discrete-only, D014). Page-cache-as-expert-cache has a sharp cliff when RAM < working set (D015, contaminated by concurrent compiles).
3. Granite routing skew is real vs a uniform control (D017), mild, weaker pooled across workloads.
4. Hybrid hit/miss dispatch is bitwise-exact; the serial cost model holds (33 us/miss); perfect-overlap upper bound NOT supported (D019).
5. Engine-native `--moe-cache-mib` (LRU, upload-on-miss) is slower than no cache until ~60% coverage (job-d50c052f94). The matched static-offload comparison in the same tool did not finish (D020) — rerun `lab/tests/static1.json`.
6. Build CC: failures recorded (3 failed attempts), ccache makes patched rebuilds ~10-20 s; image identity bug found by independent observation (D016); real handoff `accepted` for the corrected record (experiments in `cc/engine_handoff/example-real5`).
7. Lab core bugs found by testing recovery per phase (D006, D020); duplicate delivery, sandbox kill, retry-as-new-attempt, stale/conflicting result rejection all demonstrated (LAB-FEASIBILITY.md, MANAGER-DESIGN.md).

## Unfinished (honest list)
Hard GPU-memory restriction shim (T-L) not compiled/tested by me; Strata concepts not yet applied end-to-end (only studied + CPU tests, and the single-layer harness); static hot-set expert placement (managed-memory idea, D018) not started; no end-to-end speed/memory result beats the baselines in a way I would call an engine win; q8_reuse patch not compiled; Opus/Codex outputs only partly reviewed; Claude-side token cost not captured.

## Host state to restore (I changed these with your approval, D002)
Stopped: container `euaaa-llm-gpu-1` (`docker start euaaa-llm-gpu-1`) and user unit Sunshine (`systemctl --user start app-dev.lizardbyte.app.Sunshine.service`). GPU lock removed (D007). No host-wide driver/profiling change was made. Campaign containers are labelled `campaign=c1` (none should be running).

## Resume
`pip`-free: python3 stdlib + PyYAML/pytest. Re-create the lab: `bin/lab image add llamacpp-31288c9 ghcr.io/ggml-org/llama.cpp@sha256:395d01bb47eb23d5e967129c6ebd2b5969365401621e16dd3db33e50fa7462e3`; build an engine: `python3 cc/build_engine/build_engine.py --source <llama.cpp> --commit 31288c9c6 [--patch P] [--extra-target T] --tag X --out builds/results/Y` (needs BASE_RUNTIME digest, see cc/build_engine/README.md).
