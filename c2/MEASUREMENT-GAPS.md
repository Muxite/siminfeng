# Measurement gaps — what we could not measure, or measured badly (2026-11-11)

Sources: CLOSEOUT.md §5/§7 + Addendum c2, c2/SHAPE.md, D034–D036, J-OM10c run dirs (summary.json, telemetry.jsonl, stderr.txt).
Severity: **H** = can change a conclusion; M = limits interpretation; L = bookkeeping.

## 1. Collected but never reported (cheap fixes; data already on disk)
| Gap | Evidence | Sev | Fix |
|---|---|---|---|
| `gpu_freq_median` is null in **56/56** abrun summaries | summary.json `reasons` lists it every run; tegrastats on this L4T gives GR3D % only | M | read `/sys/class/devfreq/*gpu*/cur_freq` in resource_observe |
| GPU utilisation sampled but not summarised | J-OM10c/003 telemetry: gpu_util median 4 %, max 99 % → decode is IO-bound, GPU idle | M | add gpu_util median/p90 to summary + secondary table |
| Power sampled but no energy metric | VDD_IN median ~6.06 W; no J/token anywhere | M | J/token = ∫VDD_IN over eval window / tokens |
| NVMe busy (`device_util`, io_ticks) not summarised | collected per 0.5 s sample | M | add device_util during eval window |
| t/s printed to 2 decimals and used by the gate | `1.40 tokens per second` vs `716.54 ms per token`; at 1.4 t/s one step = 0.7 %, a third of the A/A gap | **H** for close arms | gate on ms/token (eval_ms / runs) instead of rounded t/s |

## 2. Window/resolution problems
| Gap | Evidence | Sev |
|---|---|---|
| NVMe GiB/token divides whole-run bytes (load + prefill + decode) by decode tokens | load 60 s of ~145 s wall (J-OM10c/003 stderr) | **H** for IO conclusions — need per-phase byte split using eval timestamps |
| Telemetry is 0.5 s sampling vs ~715 ms/token | tegrastats dt median 0.50 s | M — cannot see per-token phases (fill vs compute vs idle) |
| vmstat / block counters are system-wide | include docker, codex, other procs | M — fine under jobq exclusivity, not otherwise |
| Peak MiB/s values are sampling artefacts | 825 MiB/s > drive ceiling 570 (D031) | L |
| Per-token time breakdown exists only with the diag patch (J-OM9), which costs ~9 % | D034 | M — need a cheap always-on counter (ns in fill wait per token) |

## 3. Prefetch internals we cannot see (blocks next tuning step)
- **Timeliness**: `hits` counts correct predictions, not "page resident before demand". `async_wait_ns_in_demand` was omitted in 07. We do not know how many of the 81 % correct predictions actually arrived in time. **H** — this is the metric that decides K, inflight and L+2.
- Wasted IO: cancelled_bytes is counted (4.35 GB in pg8b), but bytes read async and then evicted before use are not counted.
- Drive queue depth / latency distribution: no iostat/blktrace (no sudo); only avg KiB/IO.
- Fill bandwidth ~380 MB/s vs drive 570 MiB/s: the cause of the gap (syscall size, QD, page-cache insertion cost) is unmeasured.

## 4. Config verification (the switch is on, but is it the value you asked for?)
- **J-OM10c pg10i8 requested K=10 but ran K=8** (parser accepts only 12/16, silently falls back). The switch regex proved "pregate active", not "k=10". Same class as earlier: `--numa` inert (D032), direct-fill never activated (D032), OR-regex in J-OM8b. **H** — arm names are not trustworthy.
- Fix: every env-gated component logs one `effective config` line with all parsed values; abrun asserts each arm's expected values (not just a regex on "active"); unknown/invalid values → hard fail, not fallback.

## 5. Things we cannot measure on this host / with these permissions
| Gap | Cause | Effect |
|---|---|---|
| GPU kernel counters (ncu) | RmProfilingAdminOnly=1 | no kernel-level attribution on Orin |
| CPU profiles (perf) | not installed, paranoid=2, no sudo | CPU-side fill/syscall cost unprofiled |
| Fixed clocks | jetson_clocks/nvpmodel not run (no sudo) | GPU 306–1020 MHz floats; mitigated by counterbalancing only |
| Cold-cache control | no `drop_caches` (no sudo); grep finds none in abrun/om_run | page-cache state at run start uncontrolled (partly self-flushing since 20.8 GiB > 7.4 GiB RAM) |
| Target scale (≈500 GB on 128 GB) | host is 7.4 GiB + 128 GB drive | proxy ratio 2.8× vs target ~3.9×; no validated extrapolation; one drive model |

## 6. Statistical / workload coverage
- n = 2 runs per arm (4 in J-OM8b); A/A 1.6–2.6 %, gate threshold ~4–5 %. Gate compares arms to base only. **Neighbouring arms (pg8b/pg8i8/pg8i12/"pg10i8": 1.37–1.40) are indistinguishable** — no occupancy conclusion is possible.
- One prompt, 64 decode tokens, greedy, batch 1, `-c 4096`. No TTFT, long-context, multi-turn, batch >1, or chat-template workloads (SPEC2 chat arm invalid). Routing (and so prefetch recall) is workload-dependent (replay: LRU 0.54–0.69 by workload).
- Prefill run-to-run spread is large (4.92–5.73 t/s inside one arm in J-OM10c); prefill is not gated.
- Quality = output-hash identity on 64 tokens only (fine for lossless IO changes; not enough for any lossy change).

## 7. Bookkeeping / provenance
- Model checksums: Qwen sha not re-verified locally; Granite safetensors and KL base sha not recorded on Jetson.
- Trace token counts not recorded; replay ↔ engine slot/MiB mapping not audited.
- SHAPE: "GPU busy" = command lifetime, not utilisation; 19 failures classified `other`; Haiku check-in tokens/cost not tracked (only Codex).
- x86 cross-host transfer (sm_120 for D025/D027) still unanswered.

## Priority fix list (proposed H5 / J-OM11)
1. Gate on ms/token; per-phase NVMe bytes (decode window only). — harness, cheap
2. Summarise gpu_util, device_util, J/token, GPU freq from sysfs. — harness, cheap
3. Effective-config log line + per-arm value assertions; invalid env → fail. — engine + abrun
4. Prefetch timeliness counters: prefetched-and-resident-at-demand, demand wait ns per token, async bytes evicted unused. — engine
5. Workload matrix (≥3 prompts incl. chat template, 256 tokens) and n ≥ 3 for arms < 5 % apart. — spec
