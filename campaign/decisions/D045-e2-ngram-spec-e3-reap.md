# D045 — E2: n-gram self-speculation ACCEPTED on static K30 (2.3× / 3.0×, text identical) and STACKS with the cache (3.9× over static on repeat), but the cache arm FAILS text identity on chat. E3: REAP-15B is a domain trade-off (code PPL +8%, wikitext ×2.84) at 6.6× decode

**Claim category:**
- E2 is a runtime trade-off: `--spec-type ngram-mod` is an existing upstream feature, workload-gated, with no change to the weights.
- E3 is a representation trade-off: a pruned model.
- **Not claimed:** lossless REAP; a general-purpose spec speed-up (chat gains nothing); task-level code quality from PPL.

**Evidence:**
- Op T-NIGHT2 (chain2) for E2 and E3.
- T-NIGHT3 (wikitext PPL) and T-NIGHT3b (code PPL).
- `artifacts/night/results.jsonl` exp `e2`, `e3`, `e3ppl`, `e3codeppl`, and the logs in `artifacts/night/logs/`. Protocol D043 E2/E3.

**Conditions:**
- E2:
  - Engines: `llama-server` on release-1 (K30) and up1 (cache arm: `--n-cpu-moe 48 --moe-cache-mib 6093 --moe-cache-cpu-share 0.9`, the best 8 GB E1 arm from D044).
  - Generation: 256 tokens, greedy, ABBA × 2.
  - Prompts: rewrite (`/no_think`, rename a variable in 3 kB of code), repeat (a 60-item list), and chat.
  - One server per arm. All three prompts run in sequence on that server.
- E3:
  - Models: 12bitmisfit REAP-15B Q4_K_M (9.75 GB, sha256 checked against the HF oid) against full Qwen3-30B-A3B Q4_K_M.
  - Engine: release-1, `llama-completion`, 128 tokens, ABBA × 2.
  - PPL on 8×512 wikitext-2 test chunks and on 16×512 chunks of `llama-model.cpp` source, used as a code corpus (local, upstream 3b4d6ab).
- Every arm ran with the host quiet; a throttled download ran, as allowed by A1.

## E2 numbers
Median decode t/s, n=2. Each pair varies by ≤ 4%, except cache-none chat (54.9–59.1).

| Placement | Prompt | spec none | ngram-mod | × | draft_n | text identical |
|---|---|---|---|---|---|---|
| static K30 (8.47 GB) | rewrite | 35.9 | 83.4 | **2.32** | 225 | yes (8ae79ff5) |
| | repeat | 35.5 | 106.8 | **3.01** | 229 | yes (39d706f1) |
| | chat | 35.4 | 35.8 | +1.1% | 0 | yes (920a5220) |
| cache C3 (8.26 GB) | rewrite | 65.7 | 100.7 | **1.53** | 225 | yes (8ae79ff5) |
| | repeat | 81.8 | 137.7 | **1.68** | 229 | yes (39d706f1) |
| | chat | 57.0 | 65.8 | +15% | 0 | **no** (43c59811 vs 67923475) |

**Verdict against the frozen rule** (all three must hold: text identical, copy-heavy ≥ 1.5×, chat within −3%; draft_n > 0 on copy-heavy prompts):
- **Static K30: ACCEPTED.** All conditions hold.
- **Cache C3: REJECTED as frozen.** Speed holds on every prompt (1.53× and 1.68× on copy-heavy, chat +15%), but chat text differs.
  - Chat made 0 drafts, so speculation never ran on that prompt. The text difference does not come from verification.
  - The most likely cause is cache history. Each server runs rewrite → repeat → chat in sequence. Speculative verify batches leave different experts resident. With CPU-share, residency decides which misses are computed on the CPU and which on the GPU, so the arithmetic, and therefore the text, depends on what ran earlier.
  - The same config gave the same text in both rounds, so the outcome is deterministic given the history.
  - The +15% chat gain is probably a warmer cache, not speculation. This is unverified.
  - This is G11 again: when CPU-share is on, the output depends on the cache's history.

**Stacking result:**
- Cache + ngram reaches 137.7 t/s on repeat, against 35.5 for static without speculation: **3.9×** at about the same 8.3–8.5 GB. Rewrite is **2.8×**.
- Speculation gains less on the cache arm (1.5–1.7×) than on static (2.3–3.0×). Verify windows touch more distinct experts (T-SPECM: about 19 per layer for 4 drafts), and on the cache those extra experts are misses. On static offload the CPU-layer cost is shared across the window.

## E3 numbers

| Model / placement | Peak MiB | Decode chat / code t/s | Prompt ms chat / code | wikitext PPL (8×512) | code PPL (16×512) |
|---|---|---|---|---|---|
| full K30 | 8193 | 36.7 / 35.2 | 769 / 3502 | 8.820 (K18 ref) | 2.534 ± 0.097 |
| full, cache C3 8 GB (D044) | 7931 | 64.3 / 56.8 | 1100 / 5228 | (same weights) | (same weights) |
| REAP K12 | 8069 | 72.2 / 68.3 | 366 / 1190 | — | — |
| REAP K0 (all GPU) | 10213 | **242.7 / 214.5** | 200 / 421 | **25.05 (×2.84)** | 2.736 ± 0.103 (+8.0%) |

**Verdict:** a trade-off, never lossless (D043).
- REAP gives 6.6× decode and 3–8× faster TTFT with everything on the GPU, using 10.2 GB.
- At about 8 GB it gives 2× over static and +12–20% over the best cache arm, with much better TTFT.
- The cost is a large loss on general text (wikitext PPL ×2.84) against a modest loss on code (+8%; the two runs overlap within about 1σ of each other's errors).
- This matches Strata's note that its REAP "Coder" model is weaker outside code.

## Limits
- One host and one model family. Two rounds per arm.
- The spec prompts are synthetic copy tasks. The rewrite prompt is a best case.
- The code PPL corpus is llama.cpp source, which may be in either model's pretraining data. PPL is not task accuracy: no SWE/HumanEval-style eval was run.
- The REAP greedy text was not inspected for coherence.
- The E2 chat text mismatch is not yet diagnosed (rerun with chat first, or one prompt per server).

## Next
- Diagnose the cache text drift: same arm, prompt order permuted, one prompt per server.
- Add a workload-gated spec lever to the planner: enable for copy-heavy workloads only, and give it a lower expected gain on cache placements.
- Add a REAP-style pruned variant as a "domain" lever, gated on a task-level eval (lm-eval-harness or a code benchmark) before the factory may offer it.
- E4 (MTP) once the Qwen3.6 download lands.

**Factory rules:**
- N-gram speculation is free on chat and gives 2–3× on copy-heavy text with static placement, or 1.5–1.7× on cache placement. It stacks with the cache.
- With CPU-share on, greedy output depends on cache history. Use KLD and agreement checks, not text identity, to gate anything that runs on top of it.
- Pruned models need a domain-matched quality check. Wikitext alone would reject REAP-15B, and code PPL alone would accept it.

**Method lesson:** a multi-prompt server shares state across prompts. Order and history are experimental variables.

## Addendum: diagnosis of the E2 cache text mismatch (op T-E2DIAG, exp `e2diag`, run after D045)
Setup: one server per run, chat prompt only, ABBA × 2.

| Arm | text | t/s r1, r2 |
|---|---|---|
| C3 none | ee080114 | 77.1, 72.6 |
| C3 ngram | ee080114 | 71.5, 76.1 |
| C2 none | df2e7b05 | 68.8, 68.4 |
| C2 ngram | df2e7b05 | 68.1, 67.7 |

- On a fresh server, text is **identical with and without ngram** for both share values. The E2 mismatch is **cache-history dependence, not speculation**. The chat text after rewrite → repeat (43c59811 or 67923475) differs from the fresh-server text (ee080114).
- Decode depends on history too. C3 chat runs at about 74 t/s on a fresh server against 57–66 after the two code-heavy prompts, because the earlier prompts leave the wrong experts resident.
- Under the corrected (post-hoc) reading, the cache + ngram arm meets the E2 criteria. As frozen, the literal verdict stays REJECTED.
- **Factory rule:** benchmark cache placements per workload with a fresh or warmed-for-that-workload cache, and treat multi-tenant or mixed workloads as a separate condition. The planner's hit-rate prediction must come from the target workload's traces.
