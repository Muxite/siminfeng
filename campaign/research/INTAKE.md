# Repeatable research intake (MoE inference factory)

Cadence: weekly sweep, or after any D0xx result that changes a bottleneck. Output: research/intake-N.json.

## Channels (record which worked; each fails differently)
1. paper-gatherer MCP: paper_gather(topic) per query; paper_get for abstracts. Semantic Scholar 429s often; retry later, arXiv/OpenAlex still return titles. Abstracts of top items must be read before scoring.
2. GitHub via gh: merged PRs since the pin date (`--search "<q> merged:>YYYY-MM-DD"`), open issues for moe-cache bugs, ik_llama.cpp merged PRs. Read PR bodies: they carry benchmarks and hardware.
3. HF API (no auth): `https://huggingface.co/api/models?search=<q>&limit=20&sort=downloads` for variants (MTP, REAP, imatrix).
(github MCP plugin may be down; gh CLI is the fallback.)

## Query list
Papers: MoE expert offloading batch-1; expert caching/prefetch; MoE speculative decoding; expert pruning/merging; mixed-precision experts; MoE edge/unified memory; kernel fusion MoE decode.
PRs: moe, expert, n-cpu-moe, moe-cache, fusion, mtp, speculative, "cuda" +MoE.
HF: mtp+gguf, reap+<model>, <model>+imatrix.

## Triage rubric (score each 0-2, sum)
- Batch-1 decode relevant (not throughput/prefill/batched serving)? 
- Hardware match (16GB consumer GPU + AVX2 DRAM, or 8GB unified)?
- Baseline is strong (static --n-cpu-moe / current upstream, not naive offload)?
- Cheap probe (<=6h, <=25GB)?
- Orthogonal to shipped stack (release-stack-1) and fits byte-bandwidth model?
Priority 1 = >=8, 2 = 5-7, 3 = else. Discount claims vs weak baselines (HOBBIT-style) and server-GPU results.

## Pipeline
1. Intake item (JSON: source, mechanism, claim, their hw/baseline, probe, cost).
2. Probe: smallest offline or one-shot experiment (trace replay, model prediction from PL4, single run). Output: go/no-go note, no decision number.
3. Indicative screen: 2 alternating rounds on quiet host, engagement check, label INDICATIVE (as D041).
4. Frozen experiment: pre-registered protocol (decision-note + frozen-experiment skills), ABBA x3, KLD/PPL gate for numeric changes, verdict, component registry entry.
Rule: upstream PRs enter by rebasing the pin on a branch and re-running the shipped engagement/identity checks; never mix with a frozen run.
