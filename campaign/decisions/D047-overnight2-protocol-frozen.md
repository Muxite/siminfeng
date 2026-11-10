# D047 — Overnight-2 protocol (FROZEN before results), 2026-11-10: follow-ups to D044–D046

The user approved all of these overnight experiments ("overnight experiments all approved").

**Runner:** `artifacts/night/run2.py`, reusing the D043 runner helpers.
- Quiet-host check before every arm. The download is finished, so no network traffic runs during timing.
- Every peak VRAM figure comes from nvidia-smi, polled every 0.5 s.
- Order is ABBA × 2 unless stated otherwise. Medians are reported.
- Engines are the same images as D043 (release-1, up1).

**Prompts:**
- Qwen3-30B: `chat.txt` and `code.txt`.
- Qwen3.6:
  - chat: `chat36.txt` (ChatML with an empty think block, from D046 A3);
  - code: `code.txt` with `--ignore-eos`, so every run has a fixed 128 tokens;
  - spec prompts (rewrite/repeat): as in D043.

**Engagement:**
- Every arm must report n > 0 and t/s > 0. Otherwise it is invalid.
- Cache arms must log cache stats.
- Spec arms must report draft_n > 0 on the copy prompts.

## X1 — MTP vs spending the same VRAM on GPU layers (Qwen3.6, up1, llama-server, 256 tokens)
- **Arms:** K28-none, against K30-mtp3. MTP costs about 0.9 GB and one layer about 0.46 GB, so these match in VRAM.
- **Prompts:** chat (ChatML), rewrite, repeat.
- **Accept "MTP beats layers"** if K30-mtp3 decode is ≥ +5% over K28-none on chat, with peak VRAM within ±5%.
- **Predicted (D046):** MTP wins on chat by about +7%, and by +20–35% on copy prompts.

## X2 — Is the speculation verify numerics within the batch-shape floor? (Qwen3.6 KLD)
- **Method:** `llama-perplexity` on 8×512 wikitext chunks. The reference is K18 at the default batch, saved. Arms: K18 at `-ub 1` (decode floor) and K18 at `-ub 4` (the shape of an MTP verify window, 1 + 3 drafts).
- **Equivalence claim:** the ub-4 arm is within the ub-1 arm's distance from the reference: |Δ mean KLD| ≤ 0.002 and |Δ same-top| ≤ 0.5 points.
- **Why this suffices:** speculation accepts only tokens that the verify pass itself would pick greedily. The only remaining numeric change is batch shape.

## X3 — Does the D044 cache result transfer to a second model? (Qwen3.6, completion, 128 tokens, ignore-eos)
- **Budgets:** the static K30 peak (about 7.6 GB) and the static K36 peak (about 4.8 GB).
- **Cache sizing:** the cache is sized so that its peak is ≤ the static peak. It equals the static peak, minus the measured all-CPU-experts base (K41), minus 300 MiB.
- **Arms:** S (static, up1); C2 (cache, share 0); C3 (cache, share 0.9).
- **Accept "transfers"** under the D043 E1 rule: C3 ≥ +5% over S on both prompts, with peak ≤ S + 3%.
- **Predicted:** a gain smaller than Qwen3's at the same budget, because Qwen3.6 has smaller experts and 256 of them, so the hit rate per MiB is lower.

## X4 — CPU-share sweep (Qwen3-30B, up1, completion)
- **Settings:** cache 6093 (8 GB) and 8137 (10 GB). Share 0, 0.3, 0.6 and 0.9. Chat and code prompts.
- **Predicted (D044):** the best share falls as the budget rises. At 10 GB, the best share is ≤ 0.3.
- **Planner rule adopted if:** the best share at 10 GB is ≤ 0.6, and within 3% of the best at 8 GB.

## X5 — Thread sweep (Qwen3-30B, chat prompt)
- **Arms:** static K30 (release-1) and C3-6093 (up1), each at `-t` 4, 6, 8, 12 and 16.
- **Predicted:** 8 threads (one per physical core) is best. 16 (SMT) is worse by ≥ 3%.

## X6 — TTFT lever for cache placements (Qwen3-30B, code prompt, the longer one)
- **Arms:** C3-6093 and static K30, each at `-ub` 256, 512 (default), 1024 and 2048, with `-b 2048`.
- **Predicted:** a larger ub cuts cache prompt time, since misses are streamed per ubatch. Decode is unchanged.
- **Report:** prompt ms and decode t/s.

## X7 — Load time (Qwen3-30B, static K30, release-1)
- **Metric:** wall time of `llama-completion -n 1`, three repetitions per condition.
- **Conditions:**
  - warm page cache, mmap (default);
  - cold for the model file only (`posix_fadvise(DONTNEED)` on that one file, no host-wide flush);
  - warm, `--no-mmap`;
  - cold, `--no-mmap`.
- **Report:** load seconds and effective GB/s. No acceptance rule; this is a measurement for the planner.

## X8 — Cache + MTP stacking (Qwen3.6, up1, llama-server, 256 tokens)
- **Arms:** C3 at the X3 8 GB budget with spec none; C3 at that budget with mtp3; K30-mtp3.
- **Prompts:** chat (ChatML), rewrite, repeat. One server per arm, prompts run in a fixed order, which is reported.
- **Accept "stacks"** if C3+mtp3 ≥ +5% over both C3-none and K30-mtp3 on chat. VRAM is reported, since MTP adds about 0.9 GB.

## Not allowed
- Pushing `main`.
- Host changes beyond the D042 perf setting.
- Restarting `euaaa-llm-gpu-1` or Sunshine during measurements. Both are restored after X1–X8 finish.
