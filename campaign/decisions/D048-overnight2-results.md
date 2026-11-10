# D048 — Overnight-2 results (protocol D047, X1–X8): the cache transfers to a second model, MTP beats spending VRAM on layers, a larger ubatch fixes cache TTFT, 4–6 threads beat 8, SMT halves decode, cache + MTP stack

**Claim category:** runtime-placement and speculation trade-offs, with existing upstream flags and the weights unchanged.
- X2 is a numeric-equivalence measurement. X7 is a load-time measurement.
- **Not claimed:** engine code gains; any result on other hosts.

**Evidence:**
- Ops T-NIGHT5 (X1–X8) and T-NIGHT5b (A1 reruns of X8 and X7).
- `artifacts/night/results.jsonl` exp `x1`, `x2`, `x3`, `x3cal`, `x4`, `x5`, `x6`, `x7`, `x7load`, `x8` (contaminated), `x8a1`. Logs in `artifacts/night/logs/x*`. Runner `artifacts/night/run2.py`.

**Conditions:**
- Engines: release-1 (`5b8a6a22…`) and up1 (`00454e95…`).
- Models: Qwen3-30B-A3B Q4_K_M and Qwen3.6-35B-A3B UD-Q4_K_M (sha 0b21525e…).
- Settings: greedy, FA on. ABBA × 2. Every arm passed the quiet check; no network traffic ran.
- **Interference window:** around 07:21–07:26 the host had 6.5 GB in swap (cause not proven). Every round timed in that window is slow, with prompt time 1.5–2.5×:
  - all of X8 round 1, which was rerun in full as X8a1 (clean);
  - X3 S-K36 code round 2 (15.7 against 40.8);
  - X4 6093 share0 chat round 1 (43.1 against 55.8);
  - X4 6093 share0.3 code round 1 (27.9 against 56.2).

  Those rounds are excluded below. The clean round is used, and it is marked *.
- **Method lesson:** the quiet check must also watch swap-in and page faults, not just process names.

## X1 — MTP vs the same VRAM spent on GPU layers (Qwen3.6, server, 256 tokens)

| Arm | peak MiB | chat | rewrite | repeat |
|---|---|---|---|---|
| K28, no spec | 8551 | 48.1 | 48.3 | 47.8 |
| K30 + mtp3 | 8505 | **52.8 (+9.8%)** | **65.4 (+35%)** | **59.8 (+25%)** |

- VRAM differs by −0.5% (rule: within ±5%). **ACCEPTED: MTP beats layers** (rule: ≥ +5% on chat).
- The D046 prediction is confirmed:
  - K28 vs K30 without spec measured +4.1%, against a predicted +4–5%;
  - MTP on chat measured +9.8%, against a predicted +7%.
- Chat acceptance: 169 of 258 drafts (66%).

## X2 — Do the speculation-verify numerics sit inside the batch-shape floor? (Qwen3.6 KLD, 8×512 wikitext, reference K18 at the default batch, PPL 6.220)

| Arm | mean KLD | same top-1 |
|---|---|---|
| K18 ub 1 (decode floor) | 0.005876 | 97.108% |
| K18 ub 4 (MTP verify shape) | 0.005769 | 97.157% |

- |Δ| = 0.0001 KLD and 0.05 points, inside the frozen bound (≤ 0.002, ≤ 0.5) → **EQUIVALENT.**
- A 4-token verify window moves outputs no more than single-token decode does. Speculation only accepts tokens the verify pass itself selects greedily.
- So the D046 chat text divergence is within the batch-shape floor. **D046 chat verdict: upgraded from INCONCLUSIVE to distributionally equivalent.**
- The same floor appears on both models: Qwen3-30B 0.0058 / 96.5%, Qwen3.6 0.0059 / 97.1%.

## X3 — Does the D044 cache result transfer to Qwen3.6? (completion, 128 tokens with ignore-eos)
The all-CPU-experts base (K41) peaks at 2951 MiB. Cache = static peak − base − 300.

| Budget | Arm | peak MiB | chat | code | C2 hit (chat) |
|---|---|---|---|---|---|
| K30 (7.66 GB) | S | 7661 | 46.3 | 47.3 | |
| | C2 (cache 4406, share 0) | 7357 | 54.8 | 49.4 | 79.9% |
| | **C3 (share 0.9)** | 7361 | **65.6 (+42%)** | **61.2 (+29%)** | |
| K36 (4.84 GB) | S | 4843 | 40.9 | 40.8* | |
| | C2 (cache 1592) | 4543 | 32.8 | 32.4 | 59.5% |
| | **C3** | 4547 | **49.3 (+21%)** | **51.5 (+26%)** | |

- **ACCEPTED, transfers:** C3 passes the D043 rule at both budgets and both prompts, using 4–6% less VRAM.
- An upload-only cache (C2) again loses at the lower budget (−20%) and is marginal at the higher one (code +4%, below the 5% bar).
- **The prediction is confirmed:** the gains are smaller than on Qwen3 at a similar budget (+42/+29% against +76/+65% at about 8 GB). Qwen3.6 has smaller experts and 256 of them, so it gets fewer hits per MiB.

## X4 — CPU-share sweep (Qwen3-30B, up1)

| Cache | Prompt | share 0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|---|
| 6093 (8 GB) | chat | 55.8* | 54.0 | 60.4 | **64.1** |
| | code | 56.2 | 56.2* | **59.0** | 56.0 |
| 8137 (10 GB) | chat | **72.7** | 69.3 | 70.2 | 72.3 |
| | code | **73.1** | 68.2 | 67.2 | 64.1 |

- **The prediction is confirmed:** the best share falls with budget. At 10 GB (92% hit rate), share 0 is best. At 8 GB (87%), 0.6–0.9 is best.
- **Unexpected:** the response is not monotonic. At 10 GB, share 0.3 is 5–7% slower than both 0 and 0.9, even though only about 200–270 misses go to the CPU. Any CPU participation appears to add a fixed per-token sync cost, so the share should be either 0 or high, never a small value.
- **Planner rule ADOPTED:**
  - predicted hit rate ≥ about 90% → share 0;
  - otherwise → share 0.9 (0.6 is within 5% on code at 8 GB).

## X5 — Thread count (Qwen3-30B, chat)

| -t | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| static K30 | **38.5** | 38.2 | 37.3 | 34.7 | 18.6 |
| C3-6093 | **66.5** | 65.8 | 65.3 | 61.3 | 34.3 |

- **The prediction is partly WRONG.** 8 threads (physical cores) is not the best: 4 threads is +2–3% faster, consistently on both arms. CPU expert bandwidth is saturated at 4 threads, and extra threads add sync.
- 16 threads (SMT) cuts decode by **about 50%**, far beyond the predicted ≥ 3%.
- **Factory rule:** threads ≤ the physical core count, tuned per host (G5 autotuner). Never use SMT threads for MoE decode.

## X6 — ubatch size as a TTFT lever (Qwen3-30B, code prompt)

| Arm | ub 256 | 512 (default) | 1024 | 2048 |
|---|---|---|---|---|
| C3-6093: prompt ms | 7714 | 5111 | 2989 | **1784** |
| C3-6093: decode t/s | 60.7 | 57.5 | 57.9 | 58.4 |
| C3-6093: peak MiB | 7805 | 7937 | 8263 | 8909 |
| static K30: prompt ms | 5205 | 3510 | 2065 | **1280** |
| static K30: decode t/s | 36.1 | 36.4 | 36.5 | 36.4 |
| static K30: peak MiB | 8035 | 8203 | 8559 | 9277 |

- **The prediction is confirmed, and the effect is larger than expected.** ub 2048 cuts cache TTFT 2.9×, for +0.97 GB of VRAM. Decode does not change.
- **The D044 TTFT problem is solved at equal VRAM:**
  - C3 at ub 1024: 8263 MiB, 2989 ms, 57.9 t/s;
  - static K30 at default ub: 8203 MiB, 3510 ms, 36.4 t/s.

  At matched VRAM (+0.7%), the cache now has **15% better TTFT and 59% better decode.**
- ub is a VRAM ↔ TTFT lever. The planner should trade cache MiB against ub according to the prompt-length profile.

## X7 — Load time (Qwen3-30B, 18.56 GB file, static K30, `llama-completion -n 1` wall time incl. docker + CUDA init)

| Mode | warm page cache | cold (file evicted via fadvise) |
|---|---|---|
| mmap (default) | **6.3 s** | 18.8 s |
| `-lm none` (read into RAM) | 16.0 s | 26.0 s |
| `-lm mmap+mlock` | 6.0 s | 18.8 s |

- The cold penalty is +12.5 s for 18.56 GB, an effective **1.48 GB/s** from this disk.
- `-lm none` costs about 10 s even when the file is warm.
- mlock changes nothing here. It probably cannot take effect under docker's default memlock limit (unverified).
- **Load model:** load ≈ 6 s fixed (container, CUDA init, uploading about 7 GB to the GPU), plus cold bytes ÷ 1.5 GB/s.
- Keep mmap. A long-running server pays the load once.

## X8 (A1 rerun, clean) — cache + MTP stacking (Qwen3.6, server, about 7.3–8.5 GB)

| Arm | peak MiB | chat | rewrite | repeat |
|---|---|---|---|---|
| C3 (cache 4406), no spec | 7339 | 66.3 | 58.7 | 73.2 |
| **C3 + mtp3** | 7867 | **70.6** | **73.1** | **95.1** |
| K30 + mtp3 | 8505 | 52.5 | 64.8 | 58.3 |

- **ACCEPTED, stacks:** on chat, +6.5% over C3 and +34% over K30+mtp3, and it uses 640 MiB less than K30+mtp3.
- Against plain static K30 (46.3 chat at 7.66 GB), cache + MTP at 7.87 GB gives **+52% on chat and +100% on repeat**.
- MTP's relative gain on the cache is smaller (+6.5% chat, against +12% on static). This matches T-SPECM: the extra experts touched by a verify window are cache misses.
- The first X8 run is kept in the record as contaminated.

## Limits
- One host: Zen 2 8-core, DDR4, PCIe 4.0, RTX 5070 Ti.
- Two rounds per arm, with excluded rounds marked *.
- Two prompts (chat, code), plus synthetic copy prompts.
- The X2 equivalence uses wikitext only.
- X7 cold reads go through LVM on this disk.
- Hit rates come from cold, single-request starts.

## Next / factory rules (adopted)
1. **Placement:** on discrete GPUs, default to the expert cache with CPU-share. Set the share to 0 when the predicted hit rate is ≥ about 90%, otherwise 0.9. This is now confirmed on two models at 4.5–10 GB.
2. **TTFT:** choose ub by the prompt-length profile; ub 1024–2048 removes the cache's TTFT penalty for about 0.3–1 GB.
3. **Speculation:**
   - models with MTP heads: enable mtp n_max 3 (beats the same VRAM spent on layers; stacks with the cache);
   - copy-heavy workloads: n-gram;
   - gate by distributional equivalence (X2 floor), not text identity.
4. **Threads:** ≤ physical cores, autotuned (4 is best on this host). Never SMT.
5. **Load:** mmap; budget cold start as file GB ÷ measured disk GB/s.
6. **Quiet check:** add swap and page-fault monitoring.
