# D046 — E4: MTP speculative decoding on Qwen3.6-35B-A3B ACCEPTED for copy-heavy prompts (+23–42%, text identical); chat +11–14% but text differs → INCONCLUSIVE under the frozen rule

**Claim category:** runtime trade-off. This is an existing upstream `--spec-type draft-mtp`, run on the model's own MTP (nextn) layer, and it costs +0.8–0.9 GB of VRAM.
- **Not claimed:** lossless output on chat; results for any other model; that the MTP VRAM is better spent than on GPU layers (a predicted comparison, below).

**Evidence:**
- Op T-NIGHT4 (chain3 E4: rewrite/repeat/chat) and op T-E4CHAT (chat rerun, amendment A3).
- `artifacts/night/results.jsonl` exp `e4` and `e4chat`; server logs `artifacts/night/logs/e4*-server.log`.

**Conditions:**
- Model: unsloth Qwen3.6-35B-A3B-MTP UD-Q4_K_M, 22,663,387,424 B, sha256 0b21525e…, matching the HF LFS oid.
  - Architecture `qwen35moe`: 41 blocks, 30 Gated-DeltaNet (ssm) layers + 11 full-attention layers, and one `nextn` MTP head.
  - 256 experts with top-8 routing, plus a shared expert. Experts are 1.81 MiB each (Q4_K gate/up, Q5_K down), against 2.72 MiB for Qwen3-30B.
- Engine: up1, `llama-server`, `-fa on -c 8192 -np 1`, 256 tokens, greedy, ABBA × 2. Every arm ran with the host quiet; the download finished before E4 started.
- **Amendment A3 (post-hoc, engagement):** the frozen chat arm sent the raw chat prompt to `/completion`. Qwen3.6 emitted EOS at token 1 (0 tokens, sha of the empty string), so the arm is invalid. It was rerun with a ChatML prompt plus an empty `<think></think>` block (op T-E4CHAT).

## Numbers
Median t/s, n=2. Pairs are within 3%. Text "=" means identical to spec-none.

| K (peak MiB none / mtp3) | Prompt | none | mtp n_max 2 | mtp n_max 3 | text | accepted / drafted (mtp2, mtp3) |
|---|---|---|---|---|---|---|
| K18 (13189 / 14073) | rewrite | 63.9 | 86.8 (+36%) | **90.7 (+42%)** | = | drafted 170, 191 |
| | repeat | 64.9 | 81.2 (+25%) | **86.5 (+33%)** | = | drafted 170, 191 |
| | chat (A3) | 64.9 | 72.2 (+11%) | 72.7 (+12%) | **≠** | 148/212 (70%), 167/259 (64%) |
| K30 (7623 / 8505) | rewrite | 45.8 | 59.8 (+31%) | **64.1 (+40%)** | = | drafted 170, 191 |
| | repeat | 46.2 | 57.0 (+23%) | **58.6 (+27%)** | = | drafted 170, 191 |
| | chat (A3) | 46.2 | 51.3 (+11%) | 51.9 (+12%) | **≠** | 145/220 (66%), 169/258 (66%) |

- MTP VRAM cost: +820 MiB (n_max 2) and +882 MiB (n_max 3).
- The chat text is stable within each config across rounds, but differs between none, mtp2 and mtp3. It also differs between K18-none and K30-none, so placement alone changes it too.

**Verdict against the frozen rule** (text identical to spec-none at temp 0 AND decode ≥ +10%):
- rewrite and repeat: **ACCEPTED** at both K, for n_max 2 and 3. The text is identical and the gain is +23% to +42%.
- chat: the speed condition is met (+11% to +14%), but the text condition FAILS → **INCONCLUSIVE**.
  - A verify batch of 3–4 tokens runs different kernels than single-token decode. A2 (D044) showed batch shape alone moves outputs by as much as placement does, so this divergence is the expected class of numeric difference.
  - No logprob/KLD-level check was run for MTP, so equivalence is **unproven**.

## Interpretation
- **MTP helps every prompt type, unlike n-gram speculation** (D045: 0 drafts on chat).
  - About 65–70% of drafts are accepted on chat.
  - On copy-heavy text, n-gram speculation is stronger (2.3–3.0× on Qwen3 K30 against MTP's +27–40%).
  - The two could be combined; that is not tested.
- **Is the VRAM worth it? (Predicted, not measured.)** One CPU layer on Qwen3.6 costs about 464 MiB of VRAM ((13189 − 7623) / 12). The +882 MiB for MTP would therefore buy about 1.9 more GPU layers. On the K-curve, that is roughly +4–5% decode, against +12% (chat) and +27–40% (copy) for MTP. So MTP is the better use of about 0.9 GB here.
- **Qwen3.6 decodes faster than Qwen3-30B at similar VRAM** (46 t/s at 7.6 GB against 36 at 8.2 GB), despite having more parameters. Its experts are smaller (1.81 MiB against 2.72), so each token moves fewer bytes, and only 11 of its layers keep a KV cache.
- **G17 confirmed:** `inspect_model` reports the architecture but has no state/KV accounting for the 30 DeltaNet layers.

## Limits
- One host and one model. Two rounds per arm.
- Synthetic copy prompts, and one chat prompt.
- MTP acceptance on the copy prompts was not recorded: the server returned `draft_n` only on the first pass, and the accepted count came back null.
- No logprob-equivalence test was run for chat. The matched-VRAM comparison against more GPU layers is predicted, not measured.

## Next
- Equivalence test for speculation: per-token top-1/KLD between spec-none and MTP on the chat prompt, against the batch-shape floor (D044 A2).
- Combine MTP with n-gram, and MTP with the expert cache plus CPU-share.
- Measure the matched-VRAM comparison (K28 none vs K30 mtp3).
- Extend `inspect_model` to hybrid layers (ssm state bytes and the attention-layer count).

**Factory rules:**
- For models that ship MTP heads, MTP speculation is a lever worth +11–14% on chat and +23–42% on copy-heavy text, for about 0.9 GB of VRAM.
- Prefer n-gram speculation for copy-heavy workloads; prefer MTP for general chat.
- Gate it with distributional equivalence, not text identity.

**Method lesson:** prompts must be formatted for each model. A raw prompt that works on one model can end at token 1 on another, so engagement (n > 0) must be checked per arm.

## Addendum (D048 X2)
The verify-shaped batch (ub 4) gives KLD 0.005769 and same-top 97.157%, against the decode floor (ub 1) of 0.005876 and 97.108%. That is inside the D047 equivalence bound. **The chat verdict is upgraded from INCONCLUSIVE to distributionally equivalent**: the MTP chat text divergence is batch-shape numerics within the floor. X1 also confirms that MTP beats spending the same VRAM on GPU layers (+9.8% chat).
