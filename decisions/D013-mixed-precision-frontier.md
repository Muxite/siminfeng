# D013 — Representation Pareto for GPU memory (job-aa57700ee5; BF16 baseline logits artifacts/kl/bf16.kl, 20x512 wikitext-2)
Class: TRANSFERABLE (accuracy + byte accounting); t/s columns were taken with the engine build compiling in parallel => unscored.
- Experts are 90.4% of all weight bytes (Q8_0 GGUF: 1283.5 of 1420.5 MB) => the bulk-compression lever is expert precision; attention/embeddings (~10%) can stay high precision cheaply.
- Fixed overhead ~300 MiB GPU (CUDA context + KV + compute buffers; file size vs peak delta across variants) does NOT shrink with weights: scaled budget (0.26 x model) must be evaluated on weights+KV+buffers, and at this scale the floor is ~650 MiB of weights even at 2 bits.
- mixE4K vs uniform Q4_K_M at the same memory (1218 vs 1222 MiB): KLD 0.040 vs 0.078, top-1 90.5% vs 85.6%, PPL +2.9% vs +6.0%. Hypothesis (untested): Q4_K_M also quantizes attention tensors at 4-bit, which are small but sensitive.
- 3-bit experts: KLD 0.135 / top-1 81.8%; 2-bit experts collapse (KLD 0.77, PPL x2.09).
- Unanswered: sensitivity per layer/expert (routing flips under quantization noise), frequency-based per-expert precision (needs routing trace), imatrix-calibrated low-bit quantization (not tried; likely helps 3-bit), generation-level / task quality beyond wikitext KL.
