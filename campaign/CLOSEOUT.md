# Campaign c1 / c2-resume — CLOSEOUT (x86-main lane), 2026-11-10

Lead: Claude Opus (orchestrator). Delegates: Codex (bounded tasks T-*), Sonnet (T-SYN*, T-SKILL), one Opus sub-harness (T-J).
Host: Ryzen 7 3700X (8C/16T, DDR4 dual channel), 46 GB RAM, RTX 5070 Ti 16 GB (sm_120, PCIe 4.0), Linux 6.8, Docker (containerd store).
This file is the authoritative closeout. `START-HERE.md` section 0 points here. Where they disagree, this file wins.

---
## 0. State at closeout

| Item | Value |
|---|---|
| Branch | `x86/c2-resume` (push authorized). `main` untouched, never pushed. |
| Commit | The closeout commit that adds this file. Its parent is `1df60b5`. The pushed hash is in the closeout report and in `git log origin/x86/c2-resume -1`. |
| Other machine | `origin/jetson/c1` (Jetson lane, head `01fa56c`, 2026-11-10 06:51 UTC) exists and was **not merged, rebased or overwritten**. It contains its own D029–D032 (over-memory on Orin) and `research/strata-*`. The Jetson lane uses its own D-numbering, so its D029–D032 are **not** x86's D029–D032. |
| Campaign processes | **None running.** No `codex exec`, no `run.py` or `run2.py`, no chain*.sh, no campaign containers (`docker ps -a` has no `campaign=c1` label and no llamacpp-* names). The Codex `app-server` daemon (pid 691713, up 4.5 days) is the user's general Codex service, not a campaign process, and was left alone. |
| GPU at closeout | **Not used by the campaign.** `nvidia-smi` shows `euaaa-llm-gpu-1` llama-server (Qwen3.5-9B, 7592 MiB) and Sunshine (229 MiB). Both are **restored owner services** (D002 policy), not campaign work. |
| Services | `euaaa-llm-gpu-1` is Up (restored after the overnight GPU work). The Sunshine user unit is `active`. Nothing else was stopped by the campaign. |
| Host settings | `kernel.perf_event_paranoid=1` (runtime only, set with user authorization for D042 perf/bpftrace; it reverts on reboot). To restore now: `sudo sysctl -w kernel.perf_event_paranoid=4`. **Not restored by the campaign:** it needs sudo, and the change was authorized. Bash permission rules were never changed. |
| Disk / RAM | Disk: 60 GB free of 914 GB (94% used). RAM: 46 GB. Swap: about 6.5 GB in use during the night, which caused the X8 contamination (D047 A1). |
| Working tree after commit | Clean, except for ignored large binaries (see section 7) and Codex scratch trees (gitignored). |

### Privacy incident (must read)
- The user's email address was present in **already-pushed** history of `x86/c2-resume`, in two files:
  - `campaign/evidence/claude-code.jsonl` (first pushed in `712fcd5`, still present through `1df60b5`);
  - `campaign/artifacts/codexPOL/runop.log`.
- In both files it appears inside captured `git` command output (a commit-author line).
- The closeout commit removes it from the working tree in both files, and also from `artifacts/op-27e568329e65/stdout`. It is replaced with `[redacted-email]`.
- **Removing it from history needs a history rewrite plus a force-push. The campaign rules forbid that, so it was NOT done.** The repo `Muxite/siminfeng` is public. Owner action options: make the repo private, or rewrite the branch with `git filter-repo` and force-push under your own authority.
- **Root cause:** the Claude Code evidence hook logs every Bash command and its output verbatim. Any command whose output prints a commit author lands in evidence.
- **Lab Manager requirement:** redact at capture time, not at commit time. See section 5, R-F.

---
## 1. Identities

### Engines (Docker images; execute by `run_ref`, admit by `config_digest`, see D016)
| Name | run_ref (docker-reported id = OCI index) | config digest | llama.cpp commit | Patch (sha256) | Qualification |
|---|---|---|---|---|---|
| release-1 | `sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e` | `sha256:d6d073145e272a2c47afe69dc20e2e0a66deb9830203c236b8a6579ccc0e493f` | `31288c9c66e59377f7788d920678a565bb89dd37` | `cc/components/release-stack-1.patch` `2e340d159bc51d9918bfdf41327f989478daa52cd2dab3e876a8fe722d663031` (q8_reuse + kv_quant_fix) | Correctness (D037): test-backend-ops, text and PPL. Speed (D040): +2.17% Q8_0 tg128. Factory run #2 rebuilt it byte-identically (20 files). |
| up0 | `sha256:97d9e5f947486ebe52fa6c5a0633f49c40f73176d76e5734c4234b1906edf9f8` | `sha256:e32fd64154e0d599213426c18a2deec7990c8584bfeb15aa1a2ee8bcd93c2f79` | upstream `3b4d6ab3f79651935fba1dd93732f29022a7f5f5` | none | Built rc=0 (`artifacts/dl2/build_up.log`). Text gate only (`artifacts/up1gate/U0-*`). |
| up1 | `sha256:00454e957a5a859275606f811bb9fa7e85b9855b157c9125703ba535f2977e51` | `sha256:dd6b28d3e542cf6ebb5f4781f7e1b5239a97dbcfb231c899d7302016131ed018` | upstream `3b4d6ab3f…` | `cc/components/release-stack-2-up3b4d6ab.patch` `516e21606a20ade71e1210f3f8cccd70ac111b6dfbf96aafd538fca885c8e71b` (release-stack-1 rebased) | **Text gate only.** U1 == U0 byte-identical on Granite Q8_0 and mixE4K. U1 == R1 on Q8_0. U1 ≠ R1 on mixE4K (different upstream base). **No test-backend-ops run and no speed requalification of q8_reuse on up1.** It was used as the engine for D043–D048 runtime-placement and speculation results. |
| older | b0-baseline*, q8base/q8reuse-1/2, hsplit-1, kvfix-1, novmm-1..4, shot-1/2, rt1 | see `builds/results/<name>/build-record.json` | 31288c9c6 | various | per D-notes (D016–D037) |

The build recipe is `cc/build_engine/build_engine.py` (Build CC v2) with `cc/build_engine/README.md`; the up0/up1 recipe is `artifacts/dl2/build_up.sh`. Image tars are not in git (section 7).

### Models and data (sha256; also in `ARTIFACT-MANIFEST.tsv`)
| Model | sha256 | Bytes | Notes |
|---|---|---|---|
| Granite-3.1-1B-A400M bf16 / Q8_0 / Q4_K_M / mixE4K / mixE3K / mixE2K | see manifest (`af76ade0…`, `e92cd1e6…`, `8b6bf885…`, `2e520c4a…`, `f2e687d3…`, `29d38ece…`) | 0.5–2.7 GB | Proxy 1. HF pin `0da7a48b`. The mix* variants were made locally (`experiments/memory`, D013). |
| Qwen3-30B-A3B Q4_K_M | `9f1a24700a339b09…` (= HF etag) | 18,556,686,912 | Proxy 2. 48 layers × 128 experts, top-8, 2.72 MiB/expert. |
| REAP-15B (Qwen3-30B-A3B pruned) Q4_K_M | `2f586d1482df2aea…` | 9,754,938,912 | E3. |
| Qwen3.6-35B-A3B UD-Q4_K_M (MTP head) | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | 22,663,387,424 | qwen35moe architecture: 41 blocks (30 DeltaNet + 11 attention), 256 experts × 1.81 MiB, top-8 + shared, nextn MTP head. |
| wikitext-2 test; code PPL text (`data/codeppl/code.raw`, from llama-model.cpp) | see manifest | | Quality sets. |
| Prompts | `artifacts/traces/in/{chat,code,chat36}.txt` plus the D045 rewrite/repeat prompts in the runner | | chat36 = ChatML with an empty think block. |

---
## 2. Every experiment and attempt

The format is: ID, verdict, claim category, settings, then evidence. Raw records live in:
- `lab/jobs/<job>/att-N/` — lab jobs, 35 jobs;
- `artifacts/night/results.jsonl` (395 rows) and `artifacts/night/logs/` (208 logs) — D043–D048;
- `evidence/*.jsonl` — every command, through `bin/run_op`;
- `index/ops.jsonl`;
- `artifacts/op-*/` — run_op stdout/stderr.

**Claim categories:**

| Code | Category |
|---|---|
| FE | full-engine code gain (our code, computation-preserving) |
| UP | upstream engine feature used as a runtime lever |
| PL | runtime-placement trade-off |
| RP | representation trade-off (numerics or the model change) |
| CB | component benchmark or harness |
| SIM | simulation or prediction |
| INF | infrastructure or methodology |
| FAIL | failed or inconclusive |

### 2a. Granite proxy era (D001–D037; llama.cpp 31288c9c6, Granite 1B-A400M)
| ID | Verdict / cat | Settings → result | Evidence / gaps |
|---|---|---|---|
| D001 | INF bug | run_op left an orphan child when SIGKILLed | fixed with process-group handling |
| D002 | host | GPU freed by stopping `euaaa-llm-gpu-1` and Sunshine, with user approval | restored at session ends (see section 0) |
| D003 | goal | reverse-engineer Strata's methodology | `research/`, `assignments/T-A` |
| D004 | INF | profiler qualification (RmProfilingAdminOnly=1, paranoid=4) | ncu needs per-container SYS_ADMIN |
| D005 | INF | 2-chunk PPL smoke gate from 5 A/A attempts | E001 |
| D006 | INF bug | reconcile overwrote a completed result; found by a recovery test | fixed |
| D007 | INF decision | GPU admission lock removed (user); one job at a time by procedure | — |
| D008 | CB | decode profile, Granite BF16 (nsys, build 11723) | node tracing inflated gaps 3× (perturbation check) |
| D009 / D010 | RP | representation screen; paired KL vs BF16 (20×512): Q8_0 +0.67% PPL, top-1 97.0% | — |
| D011 | CB | ncu bandwidth screen; Q8_0 profile → q8_reuse candidate selected | — |
| D012 | goal | objective = GPU memory at similar accuracy; transfer labels | — |
| D013 | RP | mixed precision (experts Q4_K, rest Q8_0) beats uniform Q4_K_M at equal memory: KLD 0.040 vs 0.078; 2-bit collapses | job-aa57700ee5 |
| D014 | SIM | slot-cache prediction from routing traces + static calibration | prediction only |
| D015 | PL (contaminated) | SSD-tier emulation through a container memory limit: page-cache cliff | concurrent compiles; indicative |
| D016 | INF bug | Build CC recorded the index digest as the config id | run by run_ref |
| D017 | measurement | Granite routing skew is real vs a uniform control, but mild | — |
| D018 | directive | apply Strata concepts plus a memory restriction | — |
| D019 | CB | hybrid hit/miss harness (Opus T-J): bitwise-exact, 33 µs/miss serial model holds | synthetic activations; UNSCORED |
| D020 | INF bug | lab crashed on non-UTF-8 output → bytes-safe (T-M); att-2 | job-8df3c7bb27 |
| D021 | INF (limited) | gpu_budget LD_PRELOAD shim: 39/39 CUDA tests; cooperative, NOT a hard cap | `artifacts/x3-shim` |
| D022 | PL | static `--n-cpu-moe` beats the engine LRU cache at every matched footprint on Granite (2.8× at ~880 MiB) | **reversed on Qwen (D041/D044)** |
| D023 | INF (limited) | NO_VMM engine under shim: about 310 MiB uncounted | — |
| D024 | FAIL | static hot-expert patch never engaged (text = CPU path); untimed | engagement-check rule born here |
| D025 | INF | first full chain build → job → proposal → admission → run → result | — |
| D026 / D029 | **FE ACCEPTED** | q8_reuse: Q8_0 tg128 791.3 → 807.6 t/s (+2.06%); text and PPL identical; MUL_MAT 2445/2445 | builds q8base-2 / q8reuse-2 |
| D027 / D030 | FAIL | hybrid expert split −16/−19/−32% vs static at equal MiB | CUDA unit-test mismatch still open |
| D028 / D033 | RP + engine defect fix | stock quantized KV aborts (hash set 2065 > 2053); kv_quant_fix → q8_0 KV −720 MiB at 32k, PPL 7.5392 vs 7.5385, decode −10.5%; q4_0 KV rejected (+7.9% PPL); FA −984 MiB | FA PPL equivalence not measured |
| D031 | CB diagnosis | nsys: per-layer handoff dominates batch-1 decode (5.35 vs 7.38 ms) | proxy-scale bias |
| D032 / D034 | SIM scored | planner at 900 MiB: GPU MiB ±0.3%; decode 2.6–7.2% optimistic | — |
| D035 / D040 | RP | imatrix −33% KLD at equal bytes (ACCEPTED default); PAP pap12_q3 provisional +9.8% → **RETRACTED** after a quiet rerun (+1.5%) | download-contaminated window |
| D036 | CB | builds differ in one .so (libggml-cuda); swapping it carries +2.0% | this exact pair only; the strict ABI diff says incompatible |
| D037 | INF + qualification | release-1 = q8_reuse + kv_quant_fix, correctness-qualified; factory run #2 byte-identical | `artifacts/codexFAC/run-granite-900b` (image tar local only) |

### 2b. Qwen era (D038–D048)
| ID | Verdict / cat | Settings → numbers | Evidence / gaps |
|---|---|---|---|
| D038 / D039 | PL measured; SIM failed | Qwen3-30B K = n-cpu-moe layers, t/s [GPU MiB]: K8 93.5 [15684], K18 54.6 [12246], K30 36.6 [8162], K48 24.4 [1798]. **10× less VRAM for 3.8× slower.** CPU experts about 30 GB/s; each CPU layer adds 0.757 ms. Planner extrapolated from Granite: **7× pessimistic** | job-9f50eae42a 27/27 clean |
| D040 | FE confirmed | release-1 Q8_0 +2.17% | — |
| T-PL4 | SIM | bytes/bandwidth decode model: Granite→Qwen 7.9% MAPE, Qwen→Granite 52.8%; constants indicative (concurrent CPU jobs) | committed 7e3155d, **"review pending"**, no D-note |
| T-QRT | SIM | Qwen routing traces (`artifacts/codexQRT/{prose,code}.jsonl`); hybrid split predicted **no-go** (needs about 54 µs handoff, measured 136–199 µs) | committed 7e3155d, no D-note |
| D041 | UP indicative | upstream `--moe-cache-mib` +38% at 8 GB, −34% at 4 GB; ncu 222 GB/s, 40% of kernel time in tiny kernels | superseded by D044 |
| D042 | UP indicative + tools | ngram-mod 2.4–3.0× lossless on copy prompts; bpftrace 0.79 ms per CPU layer (89% of decode); perf has no DRAM counters on Zen 2 | — |
| D043 | protocol | E1–E4 frozen; A1 (download A/A, curl SIGSTOP control: 36.46 vs 36.53 → no contamination); A2 post-hoc decode-path KLD | runner `artifacts/night/run.py` |
| D044 (E1) | **PL / UP: speed rule MET** | Qwen3-30B, engines rel1/up1, ctx 4096, greedy, 128 tok, 2 rounds ABBA. Median decode chat/code: static S2 26.4/26.3 → C3 34.2/35.3 (4 GB); 31.4/28.9 → 48.4/48.1 (6 GB); 36.5/34.4 → 64.3/56.8 (8 GB); 43.8/43.3 → 71.9/71.1 (C2, 10 GB). VRAM 3–6% below static. TTFT is flat at about 1.1 s chat / 5.2 s code (+8% to +78% vs static) | Quality: frozen absolute gate (KLD ≤ 0.01, same-top ≥ 97%) **miscalibrated**. The reference fails its own batch-shape floor (0.0058 / 96.47%). All placements fall within 0.0055–0.0063 / 96.4–97.3% |
| D045 (E2) | UP ACCEPTED on static | ngram on K30: 2.32× rewrite, 3.01× repeat, chat +1%, text identical | — |
| D045 (E2 cache) | FAIL as frozen → explained | cache + ngram 1.53× / 1.68× (3.9× over static on repeat). Chat text differed with 0 drafts. Cause: **cache-history dependence** (a fresh server gives identical text with and without spec) | addendum 4fab1e9 |
| D045 (E3) | RP trade-off | REAP-15B: all-GPU 242.7/214.5 t/s at 10.2 GB; K12 72.2/68.3 at 8.07 GB; wikitext PPL 25.05 vs 8.82 (×2.84); code PPL 2.736 vs 2.534 (+8%) | **no task-level eval** |
| D046 (E4) | UP ACCEPTED (copy); chat equivalent | MTP n_max 2/3 on Qwen3.6: copy +23–42%, text identical. Chat +11–14% with acceptance about 65–70%; text differs → equivalent via X2. Costs about +0.88 GB | first chat arm invalid (EOS at token 1) → ChatML rerun (A3) |
| D047 | protocol | X1–X8 frozen; A1: X8 rerun (swap 07:21–07:26), X7 `--no-mmap` rejected → `-lm none` | runner `artifacts/night/run2.py` |
| D048 X1 | UP | MTP vs the same VRAM spent on layers (K30+mtp3 vs K28): chat +9.8%, rewrite +35%, repeat +25% | — |
| D048 X2 | method | verify-batch numerics: ub1 0.005876 / 97.108% vs ub4 0.005769 / 97.157% → equivalent | — |
| D048 X3 | PL / UP | cache transfers to Qwen3.6. At 7.66 GB: static 46.3/47.3 → C3 65.6/61.2. At 4.84 GB: 40.9/40.8 → 49.3/51.5. C2 loses at the low budget | — |
| D048 X4 | PL | CPU-share sweep: 0.9 best at 8 GB chat, 0.6 best on code, 0 best at 10 GB; 0.3 worse than both ends | — |
| D048 X5 | host | threads: 4 best (static 38.5, C3 66.5); 8 is −2–3%; SMT 16 about −50% | — |
| D048 X6 | PL | ub 2048: cache prompt time 5111 → 1784 ms for +0.97 GB. Cache at ub1024: 8263 MiB / 2989 ms / 57.9 t/s, vs static default: 8203 MiB / 3510 ms / 36.4 t/s | — |
| D048 X7 | load | mmap warm 6.3 s, cold 18.8 s (1.48 GB/s); `-lm none` 16 s warm / 26 s cold; mlock no change | `--no-mmap` arm invalid (flag rejected) |
| D048 X8 | UP + PL | cache + mtp3: chat 70.6 / rewrite 73.1 / repeat 95.1 t/s at 7867 MiB. Cache alone 66.3/58.7/73.2. Static K30+mtp3 52.5. **+52% chat vs plain static** | first X8 contaminated by swap → rerun (A1) |

### 2c. Other delegated work (Codex T-*, all reviewed against real records unless noted)
Each assignment brief is `assignments/T-*.md`, its evidence is `evidence/codex-*.jsonl`, and its outputs are in `artifacts/codex*/` and `cc/*`.
- **Delivered and used:**
  - Factory and build: T-D (Build CC), T-N/N2/N3 (cmake defines, identity, ELF linkage), T-FAC (factory pipeline), T-COMP (component versioning, patch_stack, abi_diff; `artifacts/codexCOMP/abi-diff.json`).
  - Lab and handoff: T-E/T-K (handoff + observe-archive), T-M (raw bytes), T-P (resource_observe), T-Q (lab input identity), T-S (job envelopes).
  - Planning and model: T-MI (model_inspect), T-PL/PL2/PL3 (planner, quality ledger), T-PAP.
  - Profiling and calibration: T-PROF (nsys diagnosis), T-PROFW (profiler wrappers), T-CAL/CAL2 (host calibration), T-VER (verdict tool).
  - Simulations: T-POL (cache_policy replay), T-SPECM (spec expert-union).
  - Demo: T-DEMO1.
- **Delivered but not adopted:** T-O (static hot; D024 never engaged), T-R (hybrid split; D030 rejected), T-L (gpu_budget; cooperative only).
- **Refused by provider:** T-KVB. Its brief used crash/gdb wording (op-9a911bb7e729). Re-briefed as T-KVB2, which delivered.
- **Review pending, no D-note:** T-PL4, T-QRT.
- **Research and mapping:** T-A (Strata map), T-B (claims tests), T-C (llama.cpp compare), T-G (Strata CPU tests), T-F (routing trace), T-H (q8 activation reuse → D029), T-I (memory pareto).
- **Sonnet:** T-SYN1, T-SYN2 (doc syntheses), T-SKILL (5 project skills).

### 2d. Test suites at closeout
Run 2026-11-10 on CPU only, with `pytest -q`. Code was not changed to make them pass.

**Pass:**
- build_engine 113, cache_policy 15, components 31, engine_handoff 90, host_calibrate 25, hybrid_split 8, memory_pareto 17
- model_inspect 35, moe_planner 64, moe_static_hot 22, nsys_decode 3, profilers 29, resource_observe 26, routing_trace 29
- spec_model 30, verdict 26 (+9 subtests), `bin/` 48

**FAIL: `cc/factory` 3 failed / 5 passed** (`test_real_full_dry_run_resume_and_output_integrity`, `test_component_gates_and_environment`, `test_pending_patch_bundle_and_pap_proposal`).
- The tests still expect the pre-D037 registry, so they assert `['q8_reuse']` while the registry also accepts `kv_quant_fix` and `imatrix_quant`. They are stale expectations, not code regressions. **Fix not done** (closeout froze new work).
- The factory tests must use `--basetemp` under an assignment root (`artifacts/codexFAC/pytest-tmp`). Otherwise the factory refuses with "write outside assignment roots", which is by design.
- **GPU-path tests (test-backend-ops, CUDA units) were not rerun at closeout.** Their last results are in D029/D033/D037. hybrid_split's CUDA unit mismatch (D027) remains open.

---
## 3. Predictions vs observations

There were 31 pre-written predictions:

| Outcome | Count |
|---|---|
| Confirmed | 15 |
| Magnitude off or biased | 4 |
| Reversed or wrong | 5 |
| Silent failure (non-engagement, wrong measurement or stall) | 7 |

The full table is in the ledger artifact (section "Prediction scorecard") and in the D-notes.

| Prediction | Predicted → measured | Outcome |
|---|---|---|
| Planner GPU MiB in range | → ±0.1–0.3% | confirmed |
| Qwen decode = bytes ÷ 26 GB/s (K18, K48) | 50–60, 20–25 → 54.6, 24.4 | confirmed |
| Serial hybrid model | 2996 → 2808 µs | confirmed |
| Q8_0 near-lossless | → KLD 0.0034 | confirmed |
| q8_0 KV same PPL | ≤1% → +0.01% | confirmed |
| .so swap carries the gain | +2% → +2.0% | confirmed |
| T-POL replay at about 2 GB | 18.9/31.8 → 18.4/34.2 | confirmed |
| SPECM baseline | 35.6 → 35.6–36.5 | confirmed |
| X1 K28 vs K30 | +4–5% → +4.1% | confirmed |
| X1 MTP vs layers | +7% → +9.8% | confirmed |
| X2 verify-shape | Δ0.0001 KLD | confirmed |
| X3 cache transfers, smaller gain | → +21–42% | confirmed |
| X4 share falls with budget | → 0 best at 10 GB | confirmed |
| X6 ub cuts TTFT | → 2.9× | confirmed |
| X8 stacking | ≥5% → +6.5% / +34% | confirmed |
| q8_reuse gain | +3–4.5% → +2.1% | magnitude ≈2× off |
| Planner decode in range | → +2.6–7.2% optimistic | biased |
| PL4 Qwen→Granite | → 52.8% MAPE | magnitude off |
| PAP speed from the BW model | +15% → +1.5% | 10× off |
| Planner Granite→Qwen | 8.1 → 54.6 t/s | wrong (−85%) |
| Hybrid split from harness | ≈330 → 125 t/s (−16% vs static) | reversed |
| "Static beats cache" carried to Qwen | → cache +38% | reversed |
| PAP provisional | +9.8% → +1.5% quiet | reversed |
| Cache loses at 4 GB (D041) | → +30% with CPU-share | reversed by a new lever |
| 8 threads best (X5) | → 4 best, SMT −50% | reversed |
| Static-hot engages | → never ran | silent |
| ngram first run | → 0 drafts (thinking mode) | silent |
| Granite moe-cache 128 MiB | → 0% hit | silent |
| CAL1 bandwidth | → 1384 GB/s (cache-resident) | silent wrong measurement |
| CAL2 parse | → container banner broke it | fixture mismatch |
| Overnight chain | → 75 min deadlock (pgrep self-match) | silent stall |

**Base rates:**
- Same-model, in-range predictions are almost always within 10%.
- Proxy→target transfers were reversed or off by more than 2× about half the time.
- First-run tooling against real output needed a fix about 1 time in 2.

### Distinguishing the kinds of wins (do not mix these up)
- **Full-engine code gains (our code, computation-preserving):** only **q8_reuse, +2.1%** (Granite Q8_0, D029/D040).
  - `kv_quant_fix` is an engine **defect fix**. It enables q8_0 KV, which is a representation trade-off.
  - Neither was requalified on Qwen or on up1 for speed.
- **Upstream features used as levers (not our engineering), placement or runtime trade-offs:**
  - expert cache + CPU-share (D044, X3, X4);
  - ub (X6);
  - threads (X5);
  - mmap (X7).
- **Speculation, also upstream:** ngram (D045) and MTP (D046, X1, X8).
  - Text-identical on copy prompts.
  - Distributionally equivalent on chat (X2); text is not identical.
- **Representation trade-offs:**
  - mixE4K (D013);
  - imatrix (D035, accepted);
  - q8_0 KV (D033);
  - REAP pruning (E3; a model change with ×2.84 wikitext PPL);
  - PAP (rejected).
- **Component benchmarks:**
  - harness D019;
  - .so swap D036;
  - T-CAL2 (GPU 766, CPU 30.8, H2D 28.7 GB/s, handoff 52.5 µs residual);
  - bpftrace 0.79 ms per layer;
  - ncu 222 GB/s.
- **Simulations and predictions:** D014, planner (D034, D039), T-PL4, T-QRT, T-POL, T-SPECM. None of these is an engine result.
- **Headline memory result:** Qwen3-30B runs at 1.8 GB VRAM (vs 15.7 GB) with identical weights, 3.8× slower (D039). The cache + CPU-share + MTP stack then recovers +52% to +100% over static at about 8 GB (X8). **All of it is runtime configuration of upstream llama.cpp plus our release patches; the speed-ups are not our engine code.**

---
## 4. Lab Manager and CC portfolio discovery

Sources: `MANAGER-DESIGN.md`, `LAB-FEASIBILITY.md`, `cc-inventory.csv`, `TOOLKIT.md` (have/gap/try catalog and Strata gap audit G1–G18), `decisions/gaps.md`.

### 4a. Tools, commands and skills used
- **Recording and execution:** `bin/run_op` wraps every command (intent/started/outcome go to `evidence/*.jsonl`). `bin/lab` (submit/run/status/cancel/reconcile/chaos/register/image/image-load/assess) is a **temporary reference adapter**, not the Lab Manager.
- **Evidence tools:** `bin/index_evidence`, `bin/cost_report`, `bin/perf_summary`, `bin/cuda_compile_check`.
- **Overnight runners:** `artifacts/night/run.py` and `run2.py`, plus chain*.sh. These are scripted rather than lab-run because they must sample peak VRAM per arm, a deviation recorded in D043.
- **Engine binaries used:** llama.cpp `llama-completion`, `llama-server`, `llama-perplexity --kl-divergence`, `llama-bench`, `test-backend-ops`, `llama-quantize`, `llama-imatrix`.
- **Profilers:** nsys 2024.6.2, ncu 2025.1.1 (needs container SYS_ADMIN), bpftrace (needed paranoid=1), perf (no DRAM counters on Zen 2), nvidia-smi polling at 0.5 s.
- **Project skills** (`.claude/skills/`): codex-assignment, decision-note, engine-candidate, frozen-experiment, model-variant.
- **External:** Codex CLI (`codex exec`), the claude.ai Artifact ledger, HF downloads with curl.

### 4b. Missing tools (built ad hoc or by hand)
| Missing tool | Cost of not having it / workaround |
|---|---|
| Hash-pinned model fetch (HF etag → sha256 → manifest) with quiet-host-aware throttling | Downloads contaminated D035/D040, and needed a SIGSTOP A/A in D043 A1 |
| Host-quiet gate v2 | v1 checks only process names. It missed a swap episode (X8), which needs swap-in/out and major-fault rates |
| A workload pack with cache-history control | Cache output and speed depend on prior prompts (D045). A fresh server per arm is the current workaround |
| A per-model prompt-template engagement check | Qwen3 thinking mode gave 0 ngram drafts. A raw prompt on Qwen3.6 hit EOS at token 1 |
| A verify-shape and batch-shape quality floor tool | Built ad hoc in D043 A2 and X2. Needs to be folded into `cc/verdict` |
| A peak-VRAM sampler that is part of the lab | Lives inside `run.py` |
| A runtime autotuner (threads/ub/share/spec) | Measured by hand in X4–X6 |
| Codex brief linter (provider-refusal wording) | — |
| Self-safe process matching | `pkill -f` killed its own shell twice. A stale waiter deadlocked for 75 min |

### 4c. Repeated manual work (automation candidates)
- Restoring services and the GPU around timing windows (D002), done by hand every session.
- Calibrating cache MiB per budget so that VRAM ≤ static + 3%. `run2.py` automates it per run; it should be a planner function.
- Writing D-notes from results.jsonl. The decision-note skill helps; the aggregation is still hand-written per experiment.
- Rebuilding the ledger HTML by hand from D-notes. It should be generated from results.jsonl and D-notes.
- Scrubbing private data before push (twice).
- Reviewing Codex deliverables against real records.

### 4d. Build and dependency problems
- **Image identity:** buildx under the containerd store reports the index digest as `.Id`. `docker run` rejects the bare config digest (exit 125). Rule: execute by run_ref, admit by config digest (D016, D023).
- **Build entitlements:** the b0 attempt 1 failed on bake fs entitlements.
- **Upstream drift:**
  - `--no-mmap` was replaced by `-lm none|mmap|mmap+mlock`.
  - The speculative flags changed (`--spec-type draft-mtp`, `--spec-draft-n-max`).
  - release-stack-1 needed a rebase to `3b4d6ab` (release-stack-2).
- **ABI:** the strict ABI diff says base → reuse libggml-cuda is INCOMPATIBLE (weak std::_Hashtable exports), yet the empirical swap works (D036). Keep both facts.
- **CPU probes in engine images need `--gpus`:** libcuda is linked even for CPU paths.
- **ccache:** patched rebuilds take about 10–20 s. A cold build takes minutes and pollutes timing windows.

### 4e. Permissions and settings needed
- **GPU sharing:** the owner's GPU services must be stopped by approval (D002) and restored afterwards.
- **Profiling:**
  - `perf_event_paranoid` lowered for perf and bpftrace (runtime, authorized);
  - ncu needs `--cap-add SYS_ADMIN` per container;
  - host `RmProfilingAdminOnly=1`.
- **Push policy:** only the assigned branch; never `main`; no force-push; no permission-rule changes.
- **Container memory:** Qwen runs need a 40g container memory limit. Only one 18 GB-mmap process at a time on a 46 GB host.

### 4f. Platform, memory, profiling, sandbox, networking, scheduling, logging and recovery issues
- **Platform:**
  - PCIe-discrete results (layer-offload collapse, cache vs static) are labelled discrete-only (D012).
  - The Jetson lane found the upstream moe-cache **loses on UMA** (origin/jetson/c1 D030).
  - Never exchanged with x86: there is no SSH route, and the bundle has not been carried.
- **Memory:**
  - The gpu_budget shim is cooperative only; there is no hard VRAM cap (D021, D023).
  - About 310 MiB of VRAM is uncounted under NO_VMM.
  - Default-ctx KV (6144 MiB) dominates small budgets.
  - Host swap at 6.5 GB slowed runs by 2× (X8).
- **Profiling:**
  - Node tracing inflates gaps 3×; ncu serialises kernels.
  - CAL1 measured cache-resident bandwidth (1384 GB/s), so plausibility guards are now mandatory.
  - The handoff estimate varies 4× by method (52–199 µs).
- **Sandbox:**
  - Codex has no Docker or GPU, so all GPU work is the orchestrator's.
  - Codex's writes are confined to assignment roots, which the factory enforces.
  - The provider refused a debug-worded brief.
- **Networking:**
  - HF downloads run at about 1.4 MB/s (6 h for 22 GB).
  - The GitHub MCP server failed auth at closeout; plain git over SSH works.
- **Scheduling:**
  - One GPU job at a time by procedure, since there is no lock (D007).
  - Overnight chains are serial shell scripts with completion-marker waiters. Fragile: self-matching pgrep and old tracebacks firing a waiter early.
- **Logging:**
  - Raw evidence is verbose (`claude-code.jsonl` is 8 MB) and captured private data. Redaction is needed at capture.
  - `results.jsonl` is append-only per arm, which is good.
- **Recovery:**
  - A reconcile/overwrite bug (D006) and an orphan child (D001) were found by kill-per-phase tests.
  - A rerun is always a new attempt (att-N), never a relabel.

---
## 5. Candidate CCs (portfolio for the Lab Manager)

| Type | CC | Concrete example from this campaign | Integration requirements | Upstream reuse | Avoid |
|---|---|---|---|---|---|
| Build | **Build CC v2** (`cc/build_engine`) | release-1, up0, up1; byte-identical factory rebuild (D037) | inputs: commit + patch sha + base digests; outputs: build-record with run_ref and config_digest, linkage, cmake defines | llama.cpp CMake, buildx | treating the index digest as a config id; building during timing windows |
| Build | **Patch stack / component registry** (`cc/components`, `cc/factory/components.json`) | release-stack-1 → -2 rebase; order-independence check | per-component status (accepted/pending/rejected) with D-note citation; rebase gate = text identity + test-backend-ops | upstream PR stream | registry drift from D-notes (the 3 stale factory tests) |
| Profiling/test | **Engagement gate** | D024 never engaged; ngram 0 drafts; Granite cache 0% hits | per-mechanism counter or log-line assertion before any timing; `draft_n > 0`, cache stats present | engine stderr counters | timing anything not yet shown to run |
| Profiling/test | **Quality gate v2** (`cc/verdict` + decode-path KLD) | D044 A2 floor 0.0058 / 96.5%; X2 equivalence | reference logits per model (`.kl`, about 0.6–1 GB, local); `-ub 1` decode path; gate = within the reference's own batch-shape floor | `llama-perplexity --kl-divergence` | absolute text identity or fixed 97% gates; unpaired PPL |
| Profiling/test | **Frozen A/B runner with peak VRAM** (`run.py` / `run2.py` → lab) | E1–E4, X1–X8 | ABBA order, medians, quiet gate per arm, VRAM polling, rc check per arm, results.jsonl | — | timing during downloads, compiles, Codex runs or swap |
| Profiling/test | **Host calibration** (`cc/host_calibrate`) | GPU 766, CPU 30.8, H2D 28.7 GB/s | rotating buffers + plausibility guards; JSON-lines-only parsing | STREAM-like kernels, `test-backend-ops perf` | cache-resident buffers (CAL1) |
| Profiling/test | **Profiler wrappers** (`cc/profilers`, `cc/nsys_decode`) | 222 GB/s ncu roofline; nsys handoff diagnosis; bpftrace split timer | perturbation check vs unprofiled; capability probe (paranoid, SYS_ADMIN) | nsys, ncu, bpftrace | trusting profiled wall time |
| Tool | **Model inspector** (`cc/model_inspect`) | Qwen3-30B, Granite stats | **gap: hybrid ssm/DeltaNet layers (qwen35moe, G17)** and MTP heads | gguf-py | assuming all blocks are attention+MoE |
| Tool | **Planner levers** (`cc/moe_planner` + `cc/cache_policy` + `cc/spec_model`) | adopted rules 1–6 (START-HERE section 0) | bytes/BW decode model + cache hit predictor + share rule + ub/TTFT model + spec gain model; label extrapolations | — | Granite-fitted per-layer time (failed 7×) |
| Tool | **Routing trace + replay** (`cc/routing_trace`) | T-QRT traces; T-POL 2 GB predictions within 3–8% | patched engine build (rt1) or a generic cb_eval trace | llama.cpp eval callback | policy fine-tuning (LRU/SLRU/ClassLRU within 2 points) |
| Tool | **Handoff / envelopes** (`cc/engine_handoff`) | x86 accepted, Jetson template rejected (D025) | real remote receiver states | — | claiming remote execution |
| Tool | **Hash-pinned fetcher** (missing) | 3 Qwen downloads verified by hand | HF etag → sha256 → manifest; throttle and pause during timing | huggingface_hub | unpinned `latest` |
| Methodology | pre-register → A/A → ABBA → verdict | D026–D048 | D-note skill; amendments dated before results | — | relabelling a rerun |
| Methodology | claim categories + transfer labels | D012; section 3 | every result carries a category | — | calling upstream levers "engine improvements" |
| Methodology | research intake → smallest probe → indicative → frozen → planner lever | ngram, MTP, CPU-share | — | — | skipping the indicative stage |
| Component | **q8_reuse** (accepted, Granite Q8_0) | +2.1% | requalify on up1 and Qwen | — | — |
| Component | **kv_quant_fix** (accepted) | q8_0 KV −720 MiB at 32k | check whether upstream fixed it at 3b4d6ab | — | q4_0 KV (quality) |
| Component | **gpu_budget shim** (infra only) | D021 | — | — | calling it a hard cap |
| Component (rejected) | hybrid_split, moe_static_hot, PAP q2/q3 | D030, D024, D040 | keep as documented failures | — | reviving without a new mechanism (handoff floor < 54 µs needed) |

**New Lab Manager requirements found at closeout:**
- R-F: redact private data at evidence capture.
- R-G: the quiet-host gate includes swap and major faults.
- R-H: a per-arm return-code and flag-validity check (`--no-mmap`).
- R-I: process matching by PID or cgroup, never by self-matching patterns.
- R-J: large-artifact store with checksums (no approved store exists; see section 7).

These add to R-A..R-E in MANAGER-DESIGN.

---
## 6. Retained materials (in git)
- **Scripts and runners:** `bin/*`; `artifacts/night/{run.py,run2.py,chain*.sh,chain*.log}`; `artifacts/dl2/{fetch.sh,build_up.sh,*.log}`.
- **Patches:**
  - `cc/components/*.patch` (sha above);
  - per-candidate patches in `cc/{q8_reuse,kv_quant_fix,hybrid_split,moe_static_hot,routing_trace}`.
- **Build recipes and records:** `cc/build_engine/`, `builds/results/*/build-record.json` (now including up0/up1 build records and linkage files).
- **Manifests:** `ARTIFACT-MANIFEST.tsv` (new), `lab/images/*.json`, `cc/factory/components.json`, `cc-inventory.csv`.
- **Datasets and workloads:** `artifacts/traces/in/*`; routing traces `artifacts/codexQRT/{prose,code}.jsonl` (30 MB); lab test definitions `lab/tests/*.json`; wikitext and code text (local, hashed in the manifest).
- **Raw logs and results:** `artifacts/night/results.jsonl` + `logs/`; `lab/jobs/*`; `evidence/*.jsonl` (including the new codex-CAL2/POL/PROFW/SPECM streams); `artifacts/op-*`; `artifacts/up1gate/*`; `artifacts/bpf1/out.txt`; `artifacts/codexCAL2/gpu/*` (stdout and status of calibration runs).
- **Failed attempts and unfinished work:**
  - D024 and D030 artifacts;
  - `artifacts/codexR/debug` (hybrid split debug dumps; the .bin tensors are local only);
  - the T-KVB refusal (`evidence/codex-KVB.jsonl`);
  - gpu_budget;
  - T-PL4/T-QRT (review pending);
  - the stale factory tests.
- **Ledger artifact:** https://claude.ai/artifact/8Uqsnssn6R6KY2EtAyTiyD (v6, closeout state; sharing reported as "anyone with the link", change it from the page's Share menu if unwanted).

## 7. Large artifacts NOT in git (local only; checksums in `ARTIFACT-MANIFEST.tsv`)
**No approved remote artifact store exists for this campaign.** MANAGER-DESIGN's evidence store is the local `lab/export/`. Nothing was uploaded anywhere. All of the following are **untransferred** and live only on the x86 host under `~/projects/siminfeng/campaign/` or `~/models/campaign/`.

| Artifact | Bytes | How to regenerate / retrieve |
|---|---|---|
| `artifacts/codexFAC/run-granite-900b/builds/f7a30a23ee88e34a/engine-image.docker.tar` (release-1 rebuild) | 2,210,023,936 | `cc/factory/factory.py run …` (START-HERE section 5) or `build_engine.py` with release-stack-1 on 31288c9c6. The image is also in the local Docker store as `campaign/llamacpp:release1`. |
| `artifacts/night/kl/qwen-k18.kl`, `q36-k18.kl` (KLD reference logits) | 619,931,604 / 1,013,178,324 | `llama-perplexity --kl-divergence-base` at K18 on the same 8×512 wikitext chunks (`run.py` `phase_kld` / `run2.py` x2). |
| `artifacts/x7-comp/{base,reuse,swap}/*.so*` (D036 component libs) | about 257 MB | Extract from builds q8base-2 / q8reuse-2 (`docker cp`). |
| `artifacts/codexR/debug/*.bin` (hybrid split tensor dumps) | about 50 MB | T-R debug harness. |
| `artifacts/bpf1/libggml-cpu.so.0.27.0`; `artifacts/codexCAL2/gpu/{bw,bw_cpu,h2d}` (compiled probes) | small | rebuild from `cc/host_calibrate` sources; extract from the up1 image |
| Models (Granite set, Qwen3-30B, REAP-15B, Qwen3.6) | about 60 GB | HF downloads; verify against the manifest sha256. Qwen GGUF shas come from verified `.sha256` sidecars. |
| Docker images release-1, up0, up1 and older | — | local Docker store; rebuild per section 1 |
| llama.cpp clone `ref/`, nsys/ncu reports, Codex scratch trees | — | gitignored, reproducible |

## 8. Resumption instructions
1. `git fetch && git checkout x86/c2-resume`. Read this file, then `START-HERE.md` section 0, then `TOOLKIT.md` "Build next".
2. Check `ARTIFACT-MANIFEST.tsv` against local files: `awk -F'\t' 'NR>1' campaign/ARTIFACT-MANIFEST.tsv | while IFS=$'\t' read s b p k r; do f="${p/#\~/$HOME}"; [ -f "$f" ] || echo "MISSING $p"; done`
3. Before any GPU timing:
   - get owner approval to stop `euaaa-llm-gpu-1` and Sunshine (D002);
   - quiet check: no `codex exec`, compile or download; **swap delta 0**; load < 1;
   - one Qwen-sized process at a time.
4. **Next bounded tasks, in order:**
   1. Fix the 3 stale factory tests (registry expectations).
   2. Review T-PL4/T-QRT → D049.
   3. Planner levers: share rule, ub/TTFT, spec, threads, load model.
   4. Quality gate v2 in `cc/verdict` (batch-shape floor).
   5. Quiet gate v2 (swap and faults).
   6. Requalify q8_reuse and kv_quant_fix on up1 (test-backend-ops + speed).
   7. inspect_model support for hybrid ssm layers (G17).
   8. Handoff-floor microbench (G16).
   9. REAP task-level eval.
   10. Workload pack with cache-history control.
5. **Jetson:**
   - Read `origin/jetson/c1`, which has its own D029–D032, the over-memory on Orin findings and "upstream moe-cache loses on UMA".
   - Reconcile in a merge branch. **Never force or rebase over it.**
   - x86's EXCHANGE-1 bundle has still not been confirmed delivered.
6. **Demo:** `DEMO_DRY=1 bash cc/demo/demo.sh` first. A live run needs the idle GPU (step 3).
7. **Privacy:** decide whether to rewrite history (section 0) before sharing the repo further.
