# Paper Claim Register (upstream claims, NOT local facts)

Read date: 2026-11-09. Every claim below is what a paper says under ITS conditions. Nothing here has been tested on our proxy
(RTX 5070 Ti 16 GB + 48 GB RAM + NVMe, granite-3.1-1b-a400m, 24 layers x 32 experts, top-8, llama.cpp). Status of every claim: source-supported.

How read: arXiv abs page (version/date from listing) + PDF text extracted locally with pdftotext from the arXiv PDF of the stated version.
Section/figure numbers are taken from the extracted text; where I could not pin a number I say "location approximate".
Papers were read selectively (grep + targeted passages), not cover-to-cover.

Target = unified memory, ~500 GB weights -> 128 GB + 4 TB SSD, single/low-batch interactive decode.
Verdicts: reproduce | adapt mechanism | inspiration only | not applicable.

Key framing caveat: nearly all offloading papers below assume DISCRETE memory (PCIe is the bottleneck). On unified memory there is
no CPU->GPU copy; the tiers are RAM (high BW) and SSD (low BW). Latency numbers do not transfer; hit-rate / accuracy / predictability
properties (model properties) can transfer.

---------------------------------------------------------------------------------------------------

## P01 Fiddler - CPU compute vs weight transfer for missed experts
- Paper: Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models, arXiv 2402.07033, read v3 (1 May 2025; ICLR 2025). https://arxiv.org/abs/2402.07033
- Location: Sec 1 / Sec 3.1-3.3 (latency model, Algorithm 1), Appendix A + Fig 7 (microbenchmark), Sec 4.2 Fig 4 (result).
- Claim: Expert-layer CPU latency grows ~linearly with number of input tokens; GPU latency ~constant but needs weight transfer when expert not resident;
  transfer time "independent of the batch size". For small inputs it is faster to run a missed expert on the CPU than copy its weights. Activation copy is <1% of latency.
  Result: 1.26x avg speedup over best baseline (llama.cpp) in single-request end-to-end tok/s; 1.30x on long prefill (TTFT); 11.57x beam search (abstract numbers).
- Conditions: Mixtral-8x7B 16-bit (>90 GB); Env1 Quadro RTX 6000 24 GB + Xeon 6126 48c, PCIe3 x16 (32 GB/s), 56/256 experts on GPU; Env2 RTX 6000 Ada 48 GB + Xeon 8480+ 112c, PCIe4 (64 GB/s), 125/256 on GPU.
  ShareGPT, input 32-256, output 64-512, SINGLE request (batch 1; beam widths 4-16 for the 11.57x). Baselines DeepSpeed-MII, Mixtral-Offloading, llama.cpp b2956. 16-bit weights in host DRAM (no SSD).
- Applicability: target = inspiration only (unified memory has no transfer; "CPU vs GPU for a miss" becomes "compute from RAM vs stream from SSD", where SSD read-then-compute vs partial compute is a different tradeoff).
  Proxy = adapt mechanism (we can measure CPU-expert vs PCIe-copy crossover; granite experts are tiny).
- Hypothesis: For one decode token on granite, computing a non-resident expert on the CPU (from RAM) is faster than PCIe-copying it to the GPU and computing there, for all experts sizes in this model.
- Experiment: microbench per-expert CPU matvec (llama.cpp CPU backend, Q4_K) vs cudaMemcpy of one expert tensor slice + GPU matvec, token count 1,8,64. Refuted if copy+GPU <= CPU latency at 1 token.
  Transfer to unified target: ratio does NOT transfer (no copy); only the batch-dependence shape (CPU linear, GPU flat) is portable.
- Status: source-supported.

## P02 MoE-Infinity - skew is per-request, vanishes across requests (batch 1)
- Paper: MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache, arXiv 2401.14361, read v3 (12 Mar 2025). https://arxiv.org/abs/2401.14361
- Location: Sec 1 Contribution 1; Sec 4 (Fig 2 expert reuse counts; text "Analyzing over 1000 sequences"); Table 2 / Fig 3-4 (K-means groups, Markov chain).
- Claim (quote): "With a batch size of one ... memory available on a consumer-grade GPU is often sufficient to store the most frequently used experts during decoding";
  "experts exhibit skewed reuse patterns when continuously decoding tokens within a single request... However, this reuse pattern is only present at the request level;
  after processing multiple requests, the skew disappears, and all experts tend to be activated uniformly." Hence global-frequency caching fails; per-request activation traces (EAMs) are needed.
- Conditions: DeepSeek-V2-Lite and others (Mixtral, Switch, NLLB, Arctic); traces from BigBench/FLAN/MMLU; batch size 1; no GPU/SSD numbers for this specific claim (analysis).
- Applicability: target = adapt mechanism (the property is model/workload-level, memory-tier independent; reuse within a request is exactly what a 128 GB cache + SSD needs). Proxy = reproduce on granite traces.
- Hypothesis: On our granite traces, per-request frequency (or recency) residency achieves a higher hit rate than a single global static top-k set chosen from pooled traces at equal budget (16/32), and the global distribution is closer to uniform than per-request.
- Experiment (offline, no GPU): replay existing routing traces; compute per-layer expert-usage entropy pooled vs per-request; hit rate of (a) global static top-16 (b) per-request LFU/LRU (c) random static (expected 50%). Refuted if (a) >= (b) within noise or pooled entropy ~ per-request entropy.
  Transfers to unified target: qualitatively yes IF the large model's routing resembles granite (fine-grained experts); magnitude is model specific.
- Status: source-supported.

## P03 MoE-Infinity - trace-guided cache+prefetch latency
- Paper: same as P02 (v3). Location: Table 1 (p1), Fig 7 (Sec 5.1), Fig 9 (EAMC capacity), Table 3 (workload shift).
- Claim: DeepSeek-V2-Lite TPOT (time per output token) on one A5000 24 GB: MoE-Infinity 155 ms vs vLLM 485, DeepSpeed 737, Mixtral-Offloading 1250, Ollama (llama.cpp) 2590 ms (avg TPOT) = 3.1x-16.7x.
  Small trace store suffices (EAMC capacity ~3% of requests; Fig 9). Mixtral is their worst case (836 ms). Long context shrinks GPU buffer for experts: latency degrades as context grows 2^12 -> 2^17 (Fig 8).
- Conditions: single RTX A5000 24 GB, PCIe4 (24 GB/s measured, "32 GB/s" nominal), FULL MODEL RESIDENT IN HOST DRAM (32 GB host for DeepSeek, 128 GB Mixtral, 1 TB Arctic); no SSD tier in the main experiments; batch size 1; LongBench/BigBench/FLAN/MMLU.
  Baselines were run in offload configurations; the 16.7x is against Ollama/llama.cpp (CPU compute for offloaded layers).
- Applicability: target = inspiration only (their win is hiding PCIe fetch; our tiers differ and their SSD path is not evaluated). Proxy = adapt mechanism (the cache policy; latency numbers not comparable).
- Hypothesis: Trace-based prefetch+replacement raises hit rate above plain LRU by >=5 points on granite at 16/32 residency.
- Experiment: extend the P02 replay with an EAM-like predictor (match current request prefix to stored traces) vs LRU/LFU. Refuted if gain <5 points or only on one domain.
  Transfer: hit-rate delta transfers weakly; tok/s does not.
- Status: source-supported.

## P04 FlexGen - SSD tier is viable only for throughput; latency-insensitive batched
- Paper: FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU, arXiv 2303.06865, read v2 (12 Jun 2023). https://arxiv.org/abs/2303.06865
- Location: Sec 1 bullets; Fig 1 (latency-throughput); Table 1 (hardware); Sec 6.1 Table 2; Appendix A.4 (Table 24 SSD speeds).
- Claim: OPT-175B on one 16 GB T4 reaches 1 tok/s with effective batch size 144 (4608 tokens generated) at ~4000 s block latency (4-bit, weights held in CPU RAM, no disk);
  without compression 69x over baselines at batch up to 256 / 12000 s latency. "The batch size of 1 did not result in a noticeably better latency" (Appx) - i.e. batch-1 latency is poor.
- Conditions: T4 16 GB, 208 GB DRAM, 1.5 TB cloud NVMe SSD (read ~2 GB/s), OPT-30B/175B, prompt 512, output 32, THROUGHPUT/batched, dense model, latency-insensitive.
  Authors state unified-memory architectures "could be more friendly to our approach" (Appx, no experiment).
- Applicability: target = not applicable for the throughput result (we need single-request tok/s). Inspiration only for the lesson: when weights spill to SSD, decode rate is bounded by SSD BW / bytes touched per token unless batch amortizes. Proxy = inspiration only.
- Hypothesis: In the SSD-bound regime, single-request decode tok/s is approximately SSD_BW / (bytes of non-resident experts touched per token), within 30%.
- Experiment: container memory-limit run (known collapse regime) logging read bytes/token via iostat/perf vs measured tok/s; fio sequential and 4k-random expert-sized read BW. Refuted if model error >30% (page-cache readahead/granularity dominates).
  Transfer: the bandwidth-bound form transfers (constants change: target SSD BW, expert size).
- Status: source-supported.

## P05 FlexGen - 4-bit group-wise quantization of weights+KV to avoid disk
- Paper: same as P04 (v2). Location: Sec 5 (group-wise quantization), Table 5 (accuracy), Sec 6.1 (effective batch 144 + compression fits CPU memory).
- Claim: Group-wise 4-bit weights and KV cache, no retraining/calibration: OPT-175B Lambada acc 0.758 (FP16) -> 0.756 (4-bit); WikiText ppl 10.82 -> 10.94; OPT-30B ppl 12.72 -> 12.90.
  Compression's throughput benefit comes mainly from fitting in CPU memory and avoiding SSD swapping (112x vs baselines with compression at prompt 512).
- Conditions: dense OPT models (not MoE), group-wise asymmetric min/max (not k-quants), perplexity + Lambada only (no instruction-following or KLD), batch throughput setting.
- Applicability: target = adapt mechanism (reducing bytes moved is the lever when below working-set RAM). Proxy = reproduce the principle (we already measured Q4 vs mixed precision quality; the untested part is tok/s vs bytes under SSD/page-cache pressure).
- Hypothesis: At fixed (insufficient) RAM, decode tok/s scales with 1/(expert bytes): Q4 experts give >=1.5x tok/s over Q8 experts in the page-cache-collapse regime.
- Experiment: same container limit, run Q8 vs Q4 vs Q3 expert GGUFs, record tok/s and read MB/token. Refuted if speedup <1.3x or does not track bytes ratio.
  Transfer: yes in direction; magnitude depends on the target SSD vs RAM ratio and cache hit rates.
- Status: source-supported.

## P06 PowerInfer - hot/cold split from power-law activation
- Paper: PowerInfer: Fast LLM Serving with a Consumer-grade GPU, arXiv 2312.12456, read v2 (12 Dec 2024; SOSP 2024). https://arxiv.org/abs/2312.12456
- Location: Sec 3.1 Insight-1 / Fig 4 (CDF), footnote 3; Table 1 (sparsity); Sec 8.2, Fig 10 / Fig 14 (batch scaling).
- Claim: Neuron activation follows a power law: a minority of "hot" neurons carries ~80% of activations (OPT-30B ~26% of neurons; LLaMA2-70B ReGLU ~43%; SwiGLU-70B ~69% per Sec 3.1 and footnote 3; figures partially garbled in extraction, check Fig 4 before citing).
  Preloading hot neurons on GPU and computing cold on CPU: avg 8.32 tok/s, up to 16.06 tok/s, up to 11.69x over llama.cpp on RTX 4090 (24 GB); GPU handles ~70% of neuron load (vs 20% llama.cpp). Speedup shrinks as batch grows (Fig 14); input-heavy/short-output cases only 1.07x-4x.
- Conditions: ReLU-family sparse models (OPT, Falcon-ReLU-40B, ReGLU LLaMA2-70B); RTX 4090 24GB (PC-High) and 2080Ti 11GB (PC-Low); batch size 1 default; Alpaca/ChatGPT prompts 8-128 tokens; FP16 and INT4; trained activation predictors; best gains need high-sparsity ReLU models.
- Applicability: target = inspiration only (neuron-level sparsity of dense models, not expert routing; SwiGLU models gain much less). Proxy = inspiration only; the analog is expert-level skew (see P12 which suggests modern fine-grained MoEs are flatter).
- Hypothesis: The top 25% of granite experts per layer account for >50% of routed token-expert assignments in our traces (i.e. a real power law exists at expert granularity).
- Experiment: compute per-layer expert usage CDF from existing traces (also per domain). Refuted if top-8/32 experts cover <40% (near-uniform = 25%).
  Transfer: skew property is model-specific; the target model's own router stats must be measured.
- Status: source-supported.

## P07 vLLM / PagedAttention - KV cache fragmentation
- Paper: Efficient Memory Management for LLM Serving with PagedAttention, arXiv 2309.06180, read v1 (12 Sep 2023; SOSP 2023). https://arxiv.org/abs/2309.06180
- Location: Sec 3 / Fig 2 ("only 20.4%-38.2% of the KV cache memory is used to store actual token states" in existing systems); Sec 6 Fig 12-14 (throughput).
- Claim: Existing systems waste 62-80% of KV memory through reservation/fragmentation; PagedAttention (block-table paging) gets near-zero waste, enabling larger batches: 2-4x throughput at same latency vs FasterTransformer/Orca (more at longer sequences, larger models, complex sampling).
- Conditions: A100 40GB (OPT-13B), 4xA100 (66B), 8xA100-80GB (175B); ShareGPT/Alpaca; many concurrent requests (batch-serving throughput); dense OPT/LLaMA.
- Applicability: target = not applicable for the 2-4x (we are single-request, no KV fragmentation problem). Inspiration only for the paging abstraction and for budgeting: KV competes with experts for fast memory (see P03 Fig 8). Proxy = not applicable.
- Hypothesis: At our intended context lengths, KV cache for the target model is <10% of the 128 GB budget, so KV management is not a first-order lever.
- Experiment: compute KV bytes = 2*layers*kv_heads*head_dim*ctx*bytes for target config (needs target model config - unknown to me) and measure granite KV size/tok/s vs ctx 2k/8k/32k. Refuted if KV >10% at the target context.
  Transfer: arithmetic transfers once target config is known; granite measurement does not.
- Status: source-supported.

## P08 MxMoE - bits by block sensitivity x expert frequency; linear-block granularity
- Paper: MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design, arXiv 2505.05799, read v1 (9 May 2025; ICML 2025 format). https://arxiv.org/abs/2505.05799
- Location: Sec 3.1 + Fig 1a/1b (sensitivity and activation-frequency heterogeneity, DeepSeekV2-Lite layer 11); Sec 5.4 Table 3 (granularity); Sec 4.2 Eq 7 (ILP).
- Claim: Sensitivity varies across experts and across linear blocks inside an expert (e.g. Down proj of Expert 40 needs more bits than Gate proj); expert activation frequencies within one MoE block vary by >10x (Fig 1b).
  Allocating bitwidth per linear block beats per-expert allocation at 5-bit w-a: DeepSeek-V2-Lite PPL 6.11 vs 6.32, avg acc 69.01 vs 67.88; Qwen1.5-MoE PPL 6.95 vs 6.98 (Table 3; the gain is small on some models).
- Conditions: DeepSeek-V2-Lite (29.3 GB), Qwen1.5-MoE (26.7), Qwen2-MoE (106.9), Mixtral-8x7B (92.9); RTX 4090; calibration-based sensitivity statistics; WikiText2 PPL + 7 zero-shot tasks (not KLD, no long-form generation); GPU-resident kernels (no offload).
- Applicability: target = adapt mechanism (accuracy-per-byte allocation is hardware independent; frequency-weighting also tells which experts deserve precision vs residency). Proxy = adapt (stock llama.cpp quantizes per-tensor and each layer's experts are one 3D tensor, so per-expert precision is not available without a custom quantizer; per-layer and per-matrix-type (gate/up/down) precision IS available).
- Hypothesis: At equal total bytes, giving ffn_down (and/or high-frequency layers) more bits than gate/up lowers KLD vs uniform Q4 on granite.
- Experiment: llama-quantize with tensor-type overrides (down at Q6/Q5, gate/up at Q3/Q4, matched bytes) vs uniform Q4_K_M and our 4/8 mix; measure KLD vs bf16. Refuted if KLD not lower than the existing mixed result (0.040) at equal GB.
  Transfer: accuracy conclusions likely transfer to larger MoEs only partly (sensitivity patterns are model-specific); speed claims (kernel) do not.
- Status: source-supported.

## P09 MxMoE - very low bit (2.25-bit) MoE accuracy vs GPTQ
- Paper: same as P08 (v1). Location: Sec 5.2 Table 1; abstract.
- Claim: At 2.25 average bits (weight-only, group 128 asym + Hadamard): WikiText2 PPL MxMoE vs GPTQ*: DeepSeekV2-Lite 7.01 vs 8.49 (fp16 5.92); Qwen1.5-MoE 8.79 vs 11.19 (fp16 6.79; gap 2.40 = the abstract's "2.4 lower"); Qwen2-MoE 7.57 vs 7.98 (fp16 5.84); Mixtral 5.63 vs 5.69 (fp16 3.88).
  At 3.25-bit: MxMoE better than GPTQ except Qwen2-MoE (6.18 vs 6.11). Abstract also: up to 3.4x speedup over FP16 and up to 29.4% over uniform quant at equal accuracy (5-bit w-a; kernel throughput, Fig 5).
- Conditions: GPU-resident, RTX 4090, PPL gap numbers are for weight-only; the 3.4x and 29.4% are batched MoE-block kernel throughput (512 tokens memory-bound; 8192 compute-bound), not end-to-end single-token decode.
- Applicability: target = adapt mechanism for depth: indicates ~2-3 bit experts are feasible with allocation but degrade (PPL +1.7 to +2.0 on DeepSeek-Lite even with best method). Proxy = adapt (llama.cpp IQ2/IQ3 + imatrix is the closest available).
- Hypothesis: On granite, expert tensors at IQ3_XXS/IQ2_S with imatrix and Q8 attention remain within KLD ~2x of the Q4 mixed configuration while cutting expert bytes by 25-40%.
- Experiment: quantize expert tensors at Q3_K/IQ3/IQ2 with imatrix, keep attention Q8; measure KLD and size. Refuted if KLD >4x the Q4-mixed result at <=30% byte saving.
  Transfer: small 1B model is far more fragile than large MoEs; accuracy at depth likely OVER-penalizes vs target. Treat as lower bound on target quality, not as estimate.
- Status: source-supported.

## P10 HOBBIT - substitute low-precision experts on cache miss
- Paper: HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference, arXiv 2411.01433, read v2 (6 Nov 2024). https://arxiv.org/abs/2411.01433
- Location: Sec 2.2 / Fig 3b (replace vs skip), Sec 3.2 (Expert Scorer, token-level), Table 3 (accuracy), Sec 5.2 Fig 14 (speed).
- Claim: Replacing a required-but-missing, low-importance expert with a low-precision copy (importance from gate score x magnitude per token) preserves accuracy: "when fewer than 20% of experts are quantized, performance declines by no more than 1%";
  fp16->int4 loading is up to 4x faster. Table 3 (Mixtral-8x7B GSM8K): FP16 0.52, FP16+Int4 mix 0.51, Int8 0.52, Int8+Int2 mix 0.51; Phi-MoE 0.83 / 0.82 / 0.83 / 0.83.
  Speed (decode, batch 1): vs MoE-Infinity 3.64x (Mixtral) and 9.93x (Phi-MoE) on Jetson AGX Orin; 13.0x / 18.9x vs llama.cpp on Jetson ("up to 9.93x" in abstract is vs SOTA offloading).
- Conditions: Mixtral-8x7B (45B, 87 GB, top-2 of 8) and Phi-MoE (42B, top-2 of 16); RTX 4090 24 GB + 256 GB DRAM (PCIe 32 GB/s, FP16) and Jetson AGX Orin 32 GB UNIFIED memory with SSD ~3 GB/s (INT8 base, INT4/INT2 variants); batch 1; accuracy only GSM8K + TruthfulQA, on one 2-model set. Built on llama.cpp.
- Applicability: target = adapt mechanism - closest paper to the target (unified memory, SSD, batch 1, llama.cpp). Caveat: top-2 experts means each expert is individually important; granite is top-8 of 32 (less per-expert importance, possibly MORE tolerant). Proxy = adapt.
- Hypothesis: On granite, replacing the lowest-gate-weight 20% of selected experts per token with a Q2/Q3 copy raises KLD by less than half the gap between uniform Q4 and Q8-experts.
- Experiment: offline emulation with two GGUF variants of experts (needs per-expert switching in a patched path, or simulate in PyTorch/HF by swapping dequantized weights for the lowest-scoring experts) and measure KLD/task accuracy. Refuted if KLD rise exceeds that threshold or if gate-score is a poor importance proxy (rank-correlation with per-expert output-norm < 0.5).
  Transfer to unified target: accuracy side yes; speed benefit depends on how often misses go to SSD (4x smaller transfer = ~4x faster miss).
- Status: source-supported.

## P11 HOBBIT - next-layer expert prediction from current hidden state; page-fault collapse on unified memory
- Paper: same as P10 (v2). Location: Sec 3.3 / Fig 7 (cosine similarity, prediction accuracy), Sec 5.2 (Jetson discussion).
- Claim: Feeding layer i's gate input to layers i+1... gates predicts next-layer experts with 96% average accuracy (accuracy decays for 2-3 layers ahead; adaptive prefetch depth); high prediction accuracy lets prefetch overlap load with compute.
  Observed on unified-memory Jetson: llama.cpp mmap "can result in severe page faults when there is insufficient CPU memory to store the model weights"; MoE-Infinity also struggled there due to slow SSD reads.
- Conditions: Mixtral-8x7B (top-2 of 8, 32 layers) and Phi-MoE; batch 1; accuracy definition = predicted set matches actual (random baseline for top-2/8 is 25%).
- Applicability: target = adapt mechanism (lookahead is hardware-independent; essential for hiding SSD latency). Proxy = reproduce: granite has 24 layers, top-8/32 (random baseline 25%).
- Hypothesis: Applying layer i+1's router to layer i's input hidden state recovers >=80% of the true top-8 set at i+1 on granite, and >=60% two layers ahead.
- Experiment: forward-hook run on a few hundred tokens (HF transformers, small model); compute recall@8 for lookahead 1,2,3 vs actual routing, by domain. Refuted if recall@8 < 60% at lookahead 1.
  Transfer: recall property depends on depth/width/residual structure; measure again on target. Prefetch benefit also depends on SSD latency vs per-layer compute time (target layers are much heavier -> more time to hide).
- Status: source-supported.

## P12 ProMoE - modern decoder-only MoEs have near-uniform expert access, LRU caching is weak
- Paper: ProMoE: Fast MoE-based LLM Serving using Proactive Caching, arXiv 2410.22134, read v3 (1 Sep 2025). https://arxiv.org/abs/2410.22134
- Location: Sec 3 (motivation), Fig 1 and Fig 4 (blocking time vs cache rate), Fig 5 (cumulative expert access frequency and LRU hit rate).
- Claim: Switch-Transformer and NLLB (encoder-decoder) show power-law expert access, giving high LRU/static hit rates; "modern decoder-only MoE models exhibit a more uniform access pattern" so caching is much less effective.
  DS-1 (DeepSeek-V2-Lite): caching 50% of experts leaves 60.4% of decode time blocked on missing experts (82.7% in prefill; llama.cpp 79.0% decode); fetching experts takes 58.1 ms = 85.6% of a decode step at 23 GB/s (2.67 GiB experts/token FP16).
- Conditions: RTX 4090 24 GB, i9-14900K, 128 GB DRAM, PCIe4 32 GB/s; DS-1/DS-2/QW-1 FP16, QW-2/Mixtral INT4; ShareGPT; batch 1 default (also 1-4); hit rates from LRU on ShareGPT.
- Applicability: target = adapt mechanism (decides whether residency policies can work at all; QUALIFIES the "70% hit rate at 50% residency" local result). Proxy = reproduce (we have traces).
- Hypothesis: Our measured ~70% hit rate at 16/32 is materially above the random-static baseline (50%), and this advantage shrinks (<10 points) when traces from many unrelated prompts are mixed (domain dilution).
- Experiment: from existing traces compute random-static, global-LFU, and LRU hit rate at 16/32 for single-domain vs mixed-domain streams; also cold-start vs warm. Refuted if LRU/LFU advantage over random >=15 points even on mixed streams.
  Transfer: tells us whether to expect residency to work; target model's routing must be measured (the 500 GB model is likely closer to DeepSeek-class fine-grained routing, which ProMoE says is flat).
- Status: source-supported.

## P13 ProMoE - learned predictor + chunked/priority prefetch
- Paper: same as P12 (v3). Location: Sec 4 (GoodPred = Accuracy x FetchRate, Fig 8), Sec 5 (prefetcher, Algorithm 1-2), Sec 7.2 Fig 10-11.
- Claim: A small learned predictor on intermediate hidden states with stride prefetching achieves avg 84.7% accuracy (token-based predictor 58.3%; skip-based predictor as low as 66.9% on QW-2). Chunked, preemptible prefetch with reordered execution gives avg 2.20x (up to 3.21x) prefill and 2.07x (up to 5.02x) decode vs existing offloading
  (vs static and LRU baselines decode avg 1.47x and 1.31x in transformers; llama.cpp codebase decode 1.34x avg per intro).
- Conditions: as P12; DRAM-resident offload (no SSD); PCIe-bound; predictors trained offline on traces per model.
- Applicability: target = adapt mechanism (learned predictor + chunk-level scheduling are what hides SSD latency). Proxy = adapt (compare to P11's training-free predictor).
- Hypothesis: A small learned linear predictor trained on granite traces beats the training-free lookahead of P11 by >=10 points recall@8.
- Experiment: same hooks as P11; fit logistic regression on (hidden state at layer i -> experts at i+k) with a train/test split by prompt. Refuted if gain over training-free lookahead <5 points or fails to generalize across domains.
  Transfer: training cost and generalization for the large target are unknown; it needs traces from the target.
- Status: source-supported.

## P14 Pre-gated MoE - router lookahead built into the model (requires retraining)
- Paper: Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable MoE Inference, arXiv 2308.12066, read v3 (27 Apr 2024; ISCA 2024 format). https://arxiv.org/abs/2308.12066
- Location: Sec 1 (contribution bullets and results sentence), Sec 4 (pre-gate function, Fig 5-6), Sec 6 Fig 10-12 (latency, throughput, memory).
- Claim: Pre-gate in block N selects experts for block N+1, removing the select-then-execute dependency so CPU->GPU migration overlaps compute; only +23% latency over an oracle GPU-only baseline and 4.2x lower peak GPU memory, while matching model quality (ROUGE/ExactMatch after fine-tuning).
- Conditions: Switch-Transformer (encoder-decoder, top-1, 64/128/256 experts; Switch-Large 128 experts), A100 80GB, PCIe4, host DRAM / SSD described as tiers; pre-gate functions are trained during task fine-tuning (model modified, not a drop-in); batch-1 oriented but dated model family.
- Applicability: target = not applicable as-is (we cannot retrain a 500 GB model; changes architecture). Inspiration only: shows the ceiling for lookahead accuracy when training is allowed. Proxy = not applicable.
- Hypothesis: Retrofit pre-gating is unnecessary: training-free lookahead (P11) recall is within 10 points of what a trained predictor (P13) achieves on granite.
- Experiment: reuse P11/P13 results; no extra run. Refuted by P13 gain >10 points.
  Transfer: same as P11/P13.
- Status: source-supported (numbers from intro text and abstract; Fig 10-12 not examined).

## P15 MoE-Lightning - roofline policy for batch offload; CPU attention
- Paper: MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs, arXiv 2411.11217, read v1 (18 Nov 2024). https://arxiv.org/abs/2411.11217
- Location: Abstract; Sec 3.2 Hierarchical Roofline Model (Fig 3-5); Sec 6 Table 4 (HELM), Fig 7.
- Claim: A hierarchical roofline model (GPU / CPU / PCIe levels) predicts whether an operator is memory-bound at which tier, and finds batch/micro-batch/offload policies; with CGOPipe pipelining and CPU attention it reaches up to 10.3x higher throughput than FlexGen-class systems for Mixtral 8x7B on one T4 and hits the memory-bound ceiling with 2-3x less CPU memory.
- Conditions: BATCH THROUGHPUT (large effective batches, MTBench/HELM/synthetic), T4 16 GB / L4 24 GB, 192 GB CPU RAM, Mixtral 8x7B/8x22B/DBRX, int4 and fp16; not single-request latency.
- Applicability: target = inspiration only (the roofline method is reusable, numbers are not); single-request decode sits at the bottom-left of their roofline (memory-bound at every tier). Proxy = adapt the method.
- Hypothesis: Single-token decode on our proxy is memory-bound at each tier, so measured tok/s matches min over tiers of (tier BW / bytes touched) within 25%.
- Experiment: build a 3-parameter roofline (VRAM BW, RAM/PCIe BW, SSD BW) from microbenchmarks and compare to measured decode in GPU-resident, CPU-offload, and page-cache-limited runs. Refuted if error >25% in any regime.
  Transfer: the method transfers directly; constants change (unified memory bandwidth, SSD).
- Status: source-supported.

---------------------------------------------------------------------------------------------------

## (a) Claim -> campaign method family map
(Q=quantization depth, F=frequency-based precision, H=hot-expert residency, S=SSD streaming, P=prefetch, C=structural compression; X = primary, o = secondary)

| Claim | Q | F | H | S | P | C |
|---|---|---|---|---|---|---|
| P01 Fiddler CPU vs transfer | | | o | | | |
| P02 MoE-Infinity per-request skew | | o | X | | o | |
| P03 MoE-Infinity trace cache | | | X | | X | |
| P04 FlexGen SSD/throughput | | | | X | | |
| P05 FlexGen 4-bit | X | | | o | | |
| P06 PowerInfer hot/cold | | o | X | | | |
| P07 vLLM paging | | | | | | o (KV) |
| P08 MxMoE granularity/frequency | X | X | | | | |
| P09 MxMoE low-bit | X | o | | | | |
| P10 HOBBIT miss substitution | X | X | X | X | o | |
| P11 HOBBIT lookahead | | | | o | X | |
| P12 ProMoE uniform access | | | X | | o | |
| P13 ProMoE predictor | | | | o | X | |
| P14 Pre-gated MoE | | | | o | X | o (arch change) |
| P15 MoE-Lightning roofline | | | o | X | | |

No paper in this set addresses structural compression (expert merging/pruning/low-rank) directly; that family has NO source support in this register (gap).

## (b) Top 5 claims to test next (highest value)
1. P12 (+P02, P06 folded in): Is expert routing on granite skewed enough, per request vs pooled and vs random baseline, to justify any residency scheme? Offline, minutes, uses existing traces. Directly checks our 70% hit-rate against the paper that says fine-grained decoder-only MoEs are flat. Gates the whole hot-expert-residency family.
2. P11 (+P13, P14): Training-free router lookahead recall@8 at 1-3 layers. Offline hooks, small model. Prefetch is the only way to hide SSD latency at the target; if recall is <60% prefetch needs a trained predictor.
3. P05: Does decode tok/s track bytes under RAM limit (Q8 vs Q4 vs Q3 in the collapse regime)? Cheap, reuses the existing container setup, links quantization depth to SSD streaming - the actual target bottleneck.
4. P08: Equal-bytes allocation across matrix types/layers (ffn_down vs gate/up) vs uniform Q4 and our 4/8 mix, measured by KLD. Cheap with llama-quantize overrides; tests frequency/sensitivity-aware precision without custom kernels.
5. P10: Low-importance-expert substitution on miss. Closest to target (unified memory + SSD + batch 1 + llama.cpp) and potentially the biggest SSD-traffic reduction, but needs custom emulation, so do it after items 1-4 establish miss rates and importance proxies.
(P15 roofline model is a cheap supporting analysis for 3 and 5.)

## (c) Not retrieved / not verified / limits
- I read abstracts via the arXiv listing and body text via pdftotext of the arXiv PDF. Several figures (e.g. PowerInfer Fig 4 percentages, HOBBIT Fig 3/7, MoE-Infinity Fig 2, Pre-gated Fig 10-12, MoE-Lightning Fig 7/Table 4) were seen only through text extraction, not visually; numbers taken from figure text should be double-checked against the figures.
- Section numbers were taken from headings/captions in the extracted text; a few (P01 Appendix A, P04 Appx Table 24, P08 Sec 5.4, P14 Sec numbering, P15 Sec 6) are approximate.
- Papers were not read cover-to-cover; I did not verify ablations, baseline fairness, or implementation details, and did not open any GitHub code (GitHub MCP failed to connect; not needed).
- MoE-Infinity Table 1 column header reads "GPU idle time / TPOT Avg / TPOT Tail"; I used the avg column for the 3.1x-16.7x ratios (155 vs 485 and 2590 -> 3.1x and 16.7x), which matches the paper's stated range.
- PowerInfer: exact percentage-of-activations for the 26%/43%/69% figures was cut in extraction (80% inferred from footnote 3); verify in Fig 4.
- MxMoE abstract's "3.4x speedup" and "29.4%" were not traced to a specific table (kernel throughput, Fig 5 text); the PPL numbers were checked in Table 1.
- ProMoE llama.cpp-codebase decode speedup (1.34x average) comes from the intro text; Sec 7.2 text lists 1.09-1.49x for static/LRU comparisons in llama.cpp; read the exact table before quoting.
- Candidates NOT used: ExpertFlow, DeepSpeed-MoE inference, EdgeMoE, SwapMoE, Mixtral-Offloading (Eliseev & Mazur) - not opened by me. (HOBBIT, ProMoE, Pre-gated MoE, MoE-Lightning were opened.)
- No paper here evaluates an SSD tier for MoE experts at batch 1 with a model of 10x+ DRAM size; HOBBIT (Jetson SSD) and FlexGen (dense, batched) are the closest. The 500 GB -> 128 GB + SSD regime is therefore extrapolation from all of these.
