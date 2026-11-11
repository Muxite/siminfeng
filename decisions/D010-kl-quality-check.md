# D010 — Paired token-level quality check (KL vs BF16, 20 chunks x 512, wikitext-2 test)
Evidence: artifacts/kl/bf16.kl (base logits), ops "KL compare: ..." in evidence/orchestrator.jsonl.
Q8_0:   PPL ratio 1.0067 +/-0.0015 (+0.67%); mean KLD 0.00339; 99% KLD 0.027; max 0.24; corr(lnPPL) 99.92%.
Q4_K_M: PPL ratio 1.0603 +/-0.0070 (+6.0%); mean KLD 0.0775.
Interpretation: Q8_0 is a small but statistically detectable quality loss (paired PPL ratio is >4 sigma above 1); it is NOT bit-equal. It passes the provisional 1% PPL tripwire but must be reported as "representation change with measured paired drift", never as computation-preserving. Q4_K_M fails the 1% tripwire.
Method note: unpaired PPL stderr (0.37) hides this; paired ratio stderr (0.0015) resolves it => methodology CC candidate "paired quality comparison" (llama-perplexity --kl-divergence-base).
Same top-1 token: Q8_0 97.04% +/-0.24; Q4_K_M 85.59% +/-0.49 (artifacts of the KL ops).
