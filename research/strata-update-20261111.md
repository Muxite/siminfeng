# Strata update check (2026-11-11)

Local ref/Strata stays at 504b1433 (pinned for the earlier notes). Fetched origin/HEAD = 6c642df7 (2026-11-11), 136 commits ahead.
Read-only; nothing built or run.

## What changed in the 136 commits
Mostly product/robustness: network timeouts (lane 15), Windows/macOS test hermeticity, picture reader fallback, images on white,
release tooling. Engine-relevant only:
- Automatic RAM copy of experts + `STRATA_RESIDENT_PIN=0` covers it (feaa928f, c28474dc, 27108a4f); "lend region comes after the ranked experts".
- Expert cache: failed free-memory query reported with its error, not as 0 free (e008709d).
- Shape engines: `SHAPE=<geometry>` builds per model shape for CUDA x86_64 and aarch64/Spark, HIP, SYCL (8b92e9ac); engine refuses
  if it cannot answer `--geometry` (b2f26840). Compile-time specialisation per model geometry.
- SYCL fp32 router for cards without fp64 (3801ec40). Conversation-cache eviction returns RAM (8c71c492).

## Evidence relevant to "is kernel optimisation worth it on a DGX Spark" (origin/HEAD docs/DGX_SPARK_LINUX.md, docs/DEEPSEEK41.md)
- In-memory decode on GB10 (Qwen3.8-Flash-Next, all experts resident): ~40-48 tok/s, "limited by the GPU's work, about 930 kernel
  launches per token. No configuration switch measured here moved it more than a few percent." Fusion switches: HC_UPMIX decode +5.4%,
  SIDE_MIXER +6.7%, SPEC_COST +4.7%; the "SM121 perf table" ON vs OFF: decode 0.996x (IQ3_S) / 1.02x (NVFP4).
- Prompt (prefill) is where kernels paid on Spark: fused IQ prompt kernels +10-27%, NVFP4 w4a4 prompt +16%; table 1.20x / 1.24x.
  These two change logits (mean KL 0.019 / 0.032), quality suite "no loss we could measure".
- Over-memory on 128 GB unified: DeepSeek-V4.1-Flash 2.00 bpw (138 GB experts + 111 GB n-gram table on SSD):
  Spark 11,187/15,744 experts in GPU share, rest on CPU lane from files: 13-23 tok/s. Strix Halo 128 GB: 109 GB experts in pool +
  23 GB from SSD: 19-21 tok/s code, 11 chat. Levers used: 2 bpw format, expert placement/ranking, speculative drafter (DSpark/MTP).
- Spark UMA sizing uses MemAvailable - 6 GiB (`DS4_UMA_HEADROOM_GIB`) because page cache dominates cudaMemGetInfo — same lesson as our D030/D032.
- Strix: async verify mailbox "sized for cards that hold most of the experts" costs 31% when experts stream from SSD → scheduling, not kernels.
