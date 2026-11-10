# D044 — Expert cache + CPU-share beats static offload at 4/6/8/10 GB on Qwen3-30B-A3B (frozen speed rule MET at all budgets; absolute KLD gate MISCALIBRATED, equivalent to its own floor)

**Claim category:** runtime-placement trade-off. This uses existing upstream flags (`--moe-cache-mib`, `--moe-cache-cpu-share`) with the weights unchanged.
- **Not claimed:** an engine code improvement; any result on other GPUs, CPUs or models; TTFT gains (TTFT gets worse).

**Evidence:**
- Ops T-NIGHT1 (chain1) and T-NIGHT2 (chain2) in `evidence/orchestrator.jsonl`.
- `artifacts/night/results.jsonl` (exp `aa`, `e1`, `kld`, `kld2`) and the raw stdout/stderr in `artifacts/night/logs/`.
- Runner `artifacts/night/run.py`. Protocol D043 plus amendments A1 (before results) and A2 (after results).

**Conditions:**
- Engines:
  - release-1: run_ref `sha256:5b8a6a22…`, pin 31288c9c6.
  - up1: run_ref `sha256:00454e95…`, upstream 3b4d6ab3f + release-stack-2 (sha 516e21606a20); passed text identity and 2517/2517 MUL_MAT.
- Model: Qwen3-30B-A3B Q4_K_M (sha256 9f1a2470…, D038).
- Settings: `llama-completion`, greedy, seed 1, `-fa on`, ctx 4096, 128 tokens, prompts `chat.txt` and `code.txt`.
- Peak VRAM from nvidia-smi every 0.5 s.
- Order: ABBA × 2 rounds. 80 timed E1 runs, all rc 0. Every arm was `quiet=true` (no compiler or Codex). A throttled curl ran throughout, as allowed by A1.
- Host: RTX 5070 Ti 16 GB, Ryzen 7 3700X, DDR4 dual channel, PCIe 4.0.

**Protocol (D043 E1, frozen):**
- Accept "cache beats static at B" only if both hold:
  - median decode is ≥ +5% over the same engine's static arm on both prompts;
  - peak VRAM is ≤ static + 3%.
- Quality gate: mean KLD ≤ 0.01 and same-top ≥ 97% against S1 K18, on 8×512 wikitext chunks.

**A/A interference (A1):** S1-K30 chat measured 36.46 t/s with the download running against 36.53 paused (n=3 each), a 0.2% difference against the 2% limit. **No arm is contaminated.**

## Numbers
Median decode t/s, chat / code, n=2 each.

| B | K | S1 rel1 | C1 rel1 | S2 up1 | C2 up1 share 0 | C3 up1 share 0.9 | cache MiB | C1 hit (chat) | peak MiB S / C3 |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 42 | 26.5 / 26.9 | 17.3 / 20.8 | 26.4 / 26.3 | 18.4 / 20.6 | **34.2 / 35.3** | 2005 | 50.1% | 4105 / 3837 |
| 6 | 36 | 30.8 / 30.8 | 28.9 / 32.6 | 31.4 / 28.9 | 32.2 / 34.0 | **48.4 / 48.1** | 4051 | 72.9% | 6151 / 5887 |
| 8 | 30 | 36.1 / 36.0 | 51.1 / 58.3 | 36.5 / 34.4 | 54.8 / 55.7 | **64.3** / 56.8 | 6093 | 86.7% | 8193 / 7931 |
| 10 | 24 | 43.5 / 42.0 | 70.1 / 71.8 | 43.8 / 43.3 | **71.9 / 71.1** | 71.9 / 64.2 | 8137 | 91.7% | 10237 / 9973 |

- **Within-arm spread** across the two rounds is ≤ 3% for most arms. C3 code shows 9% (33.6–36.9 at 4 GB, 61.4–67.0 at 10 GB).
- **CPU-share engagement** (from the logs): C3 computed 22044 of 25242 misses on the CPU at 4 GB (chat) and 1093 of 4218 at 10 GB. C2 logged "0 on the CPU".
- **Prompt eval ms** (TTFT proxy), median:
  - Cache arms are flat with budget: about 1100 (chat) and 5200 (code).
  - Static improves with budget: 1007 → 657 (chat) and 4781 → 2935 (code).
  - The cache TTFT penalty is therefore +8% at 4 GB, rising to +66% (chat) and +78% (code) at 10 GB.

**Quality**, KLD against the S1 K18 reference saved at the default batch of 2048:

| Arm | batch | mean KLD | same top-1 |
|---|---|---|---|
| S1 K42 (frozen run) | 2048 | ≈0 (parser dropped the negative value) | 100.000% |
| C1 6000 (frozen run) | 2048 | ≈0 | 100.000% |
| C3 6000 (frozen run) | 2048 | 0.005365 | 96.716 ± 0.395% |
| S1 K18 = **reference config itself** (A2) | ub 1 | 0.005822 | 96.471% |
| S1 / S2 K42 (A2) | ub 1 | 0.006265 | 96.373% |
| C1 / C2 6000 (A2) | ub 1 | 0.005494 | 97.304% |
| C3 6000 (A2) | ub 1 | 0.005486 | 96.912% |
| C3 2005 (A2) | ub 1 | 0.005637 | 97.157% |

## Verdict against the frozen rule
- **Speed + VRAM rule:**
  - C3 against S2 **MEETS** it at 4 GB (+30 / +34%), 6 GB (+54 / +66%), 8 GB (+76 / +65%) and 10 GB (+64 / +48%), with VRAM 2.6–6.5% below static.
  - C1 and C2 meet it only at 8 and 10 GB (+42–71%). At 4 GB they FAIL (−22 to −35%), and at 6 GB they FAIL on the chat prompt (C1 −6%, C2 +3%).
- **Quality gate, applied literally:**
  - The frozen batch-2048 run gives C3-6000 same-top 96.72%, below 97% → **FAIL**.
  - Under A2's decode path, C1, C2 and C3-2005 pass, and C3-6000 fails by 0.09 points.
  - The **reference configuration fails against itself** (96.47%) when only the batch shape changes, and so does the static K42 offload the factory already ships (96.37%).
  - The gate threshold therefore sits inside batch-shape noise and cannot separate placements. **Gate verdict: inconclusive / miscalibrated, not a pass.**
- **Post-hoc equivalence** (labelled as such): every placement falls in KLD 0.0055–0.0063 and same-top 96.4–97.3%, while the reference's own batch-shape floor is 0.0058 / 96.5%. The cache arms drift no more than static offload.

## Interpretation
1. **CPU-share is the missing lever behind D041's 4 GB loss.**
   - When misses are computed where static offload would have computed them anyway (on the CPU), while hits come from VRAM, the cache wins even at a 50% hit rate.
   - An upload-only cache wins only above roughly 70–75% hit rate. That matches D041 and T-POL.
2. **CPU-share costs speed at high hit rates.** At 10 GB (92% hit), code decode is 64.2 against 71.1 for share 0, because computing a few misses on the CPU costs more than uploading them.
3. **Prompt processing does not use the cache.** The stats cover only ubatch ≤ 8, so the TTFT penalty grows as static gains GPU layers.
4. **T-POL's offline replay predicted the 2 GB cell** at 18.9 (upload) and 31.8 (CPU-share); measured 18.4 and 34.2. Replay plus calibration is good enough for the planner.

## Limits
- One host: discrete PCIe 4.0 GPU, Zen 2 AVX2, and DDR4 at about 30.8 GB/s (CAL2). One model, at batch-1 decode.
- Two prompts of 128 tokens. Only 2 rounds per arm, with a spread of 3–9%. Every effect here (≥ 30%) is far larger than that spread, but the 10 GB C2/C3 difference (−10% on code only) is near it.
- KLD covers only 8×512 wikitext chunks, and the reference is K18 at batch 2048, not bf16. No task-level eval was run.
- Hit rates come from a cold start within one request. Multi-turn and long-context behaviour is not measured (gap G3).

## Next
- Add the planner lever: cache with CPU-share by default, and lower the share as the predicted hit rate exceeds about 90% (sweep share 0.3/0.6 at 8–10 GB).
- Add a TTFT model and a "TTFT-optimised" option (static, or a hybrid of static GPU layers plus the cache).
- Re-gate quality against the reference's own batch-shape floor in the next protocol. Add a bf16-free task check.
- Run a frozen 3-round follow-up on the 10 GB share sweep.
- **Factory rules:**
  - On discrete GPUs, use the expert cache with CPU-share for decode at every budget tested (4–10 GB), with less VRAM than static.
  - Always report TTFT next to decode.
  - KLD for a placement must run on the decode path (`-ub 1`). At large batches, CPU-resident experts are computed on the GPU.
- **Method lesson:** an absolute quality threshold must first be checked against the reference configuration's own run-to-run and batch-shape floor. A gate that the reference fails cannot judge candidates.
