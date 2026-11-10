# D043 — Overnight protocol (FROZEN before results), 2026-11-10

Purpose:
- Turn the INDICATIVE results in D041 (expert cache) and D042 (n-gram speculation) into frozen ones.
- Test the upstream rebase, which brings CPU-share and ClassLRU.
- Run the first probes of REAP pruning and MTP.

Runner: `artifacts/night/run.py`, recorded through `bin/run_op`. It is a scripted runner rather than `bin/lab` because it has to sample peak VRAM per arm; that deviation is recorded here.
- Quiet-host rule: no compiles, downloads or Codex processes may run during timed arms. The runner checks this before each arm and records it. If the check fails, the arm waits up to 10 minutes and is then marked `untimed`.
- Settings: greedy decoding, seed 1, `-fa on`, ctx 4096, 128 generated tokens.
- Prompts: `artifacts/traces/in/chat.txt` and `code.txt`.
- Peak VRAM comes from nvidia-smi, polled every 0.5 s.

## E1 — Expert cache vs static at matched budgets (Qwen3-30B-A3B Q4_K_M)
Budgets: B ≈ 4, 6, 8 and 10 GB.

| Arm | Engine | Configuration |
|---|---|---|
| S1 | release-1 | static `--n-cpu-moe K_B` |
| C1 | release-1 | `--n-cpu-moe 48 --moe-cache-mib (B − base)` |
| S2 | up1 | static `K_B` |
| C2 | up1 | cache, cpu-share 0 |
| C3 | up1 | cache, cpu-share 0.9 |

- K_B for 4, 6, 8 and 10 GB is 42, 36, 30 and 24. The cache size is set so that peak VRAM is at most the S1 peak + 3%.
- Order: arms are alternated ABBA across 2 rounds, and each arm is one process per prompt.
- **Accept "cache beats static at B"** only if both conditions hold:
  - median decode ≥ +5% over the same engine's static arm, on both prompts;
  - peak VRAM ≤ static + 3%.
- TTFT (prompt eval ms) is reported, never hidden.
- Engagement: every cache arm must log cache activity. If no cache stats appear in stderr, the arm is invalid.
- Quality: all arms use the same weights, so this measures numeric drift only.
  - Measure KLD against S1 at K=18 with `llama-perplexity --kl-divergence` on 8 chunks of 512 from the existing KL text set.
  - Gate: mean KLD ≤ 0.01, and top-1 agreement ≥ 97%.

## E2 — N-gram speculation on top of the best placement (Qwen3-30B)
- Arms: {static K30, the best E1 cache arm at about 8 GB} × {spec none, `ngram-mod`}.
- Prompts: code rewrite and list repeat (`/no_think`), plus chat. 256 tokens, 2 rounds ABBA.
- Accept only if all three hold:
  - text is identical to spec-none;
  - copy-heavy prompts reach ≥ 1.5×;
  - chat is within −3%.
- `draft_n` must be greater than 0 on the copy-heavy prompts (engagement).

## E3 — REAP-pruned Qwen3-30B (15B total, Q4_K_M, 9.75 GB)
- Arms: REAP at K=0 (all-GPU, if it fits in 16 GB) and at the K needed to fit 8 GB, against full Qwen at the same budgets (best E1 arm).
- Quality: wikitext PPL for both models on the same 8×512 chunks, plus greedy text examples.
- Pruning changes the model, so the D035 KLD gate doesn't apply. This is a **trade-off** result, never "lossless".
- Report the PPL ratio, decode t/s and VRAM.

## E4 — MTP speculative decoding (Qwen3.6-35B-A3B UD-Q4_K_M with MTP heads; engine up0/up1)
- Arms: spec none against `--spec-type draft-mtp` with `--spec-draft-n-max` 2 and 3, at K chosen for about 12 GB and about 8 GB.
- Report acceptance, decode t/s, TTFT and VRAM.
- Accept only if text is identical to spec-none at temp 0 and decode ≥ +10%.

## Not allowed tonight
- Pushing `main`.
- Changing Bash permission rules or the host beyond the D042 perf setting.
- Restarting `euaaa-llm-gpu-1` or Sunshine during measurements. Both are restored at session end, per D002.

## Amendment A1 (made before any E1–E4 result exists)
- HF downloads run at about 1.4 MB/s and take about 6 h. A throttled `curl` (nice 19) is allowed during timed arms. Compiles and Codex processes are still not allowed.
- Interference control: the runner times the S1-K30 chat arm 3× with the download running, then 3× with it paused (SIGSTOP/SIGCONT on the curl process). If the medians differ by more than 2%, every arm timed during the download is marked `contaminated` and rerun after the download finishes.
