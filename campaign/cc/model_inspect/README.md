# Model inspection

`inspect_model.py` uses Python 3.12, numpy, and the pinned local
`ref/llama.cpp/gguf-py`. It reads that reference tree without writing caches.
It never reads, dequantizes, copies, or hashes tensor payloads. A read-only
file mapping provides metadata parsing; its tensor-construction hook is
replaced with shape/type accounting, and the mapping closes after inspection.
No builds, GPU work, worker pools, or model inference are required.

From the campaign root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python cc/model_inspect/inspect_model.py \
  ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf \
  --trace experiments/routing/prose.json \
  --out artifacts/codexMI/granite-mixE4K.json --summary
```

The Python API is `inspect(gguf_path, trace=None) -> dict`, with
`schema: "model-stats/0"`. Without `--out` or `--summary`, the CLI writes JSON
to stdout. `--summary` prints a human-readable Markdown table; use `--out`
alongside it to retain the full machine report. Missing paths or invalid
headers/traces produce a nonzero exit. Output parent directories must exist.

## Accounting conventions

- Shapes use GGML order (`ne[0]` is the contiguous dimension), the reverse of
  numpy shape order. Bytes are exact tensor payload bytes from integer shape
  products and `GGML_QUANT_SIZES`; file headers and alignment gaps are excluded.
- `ROLE_PATTERNS` is the explicit ordered classification table. It covers
  Granite, Mixtral/llama MoE, Qwen2/3 MoE, DeepSeek2 MLA and leading dense FFNs,
  GPT-OSS fused gate/up and biases, and hybrid hot/cold tensors and ID maps.
  Unknown tensors contribute to `other`, remain in `tensors`, and appear in
  `unknown_tensors` with a warning. Tied embedding/output storage is counted
  once; absent output tensors do not create a fictitious second embedding.
- Shared expert count/dimensions come from metadata when supplied. A shared
  FFN without count metadata is flagged rather than guessing a count from a
  fused matrix. Expert FFN width falls back to bank shapes when needed.
- `per_layer_experts` reports physical, real and null expert bytes and each
  projection's per-expert storage. Hot/cold banks with different encodings
  retain separate byte rates; a single common rate is `null` when unequal.
  Hybrid metadata declares one null slot in every hot bank; these slots count
  toward physical storage, placement floors and the proportional storage
  estimate; a separate active estimate excludes the null overhead.
- Active bytes are non-expert bytes (including shared FFNs once) plus
  `top_k/E * stored_expert_bytes`, as requested. The additional
  `real_experts_only_bytes` field uses real expert bytes without null padding.
  These estimate proportional weight storage, not measured memory traffic.
  They assume uniformly sized experts. Missing top-k
  makes this estimate unavailable rather than inventing a value.
- GPU strategy floors place all non-expert weights, including shared FFNs,
  on GPU. Fractions apply to stored expert bytes and allow fractional expert
  counts. They exclude KV, runtime, compute buffers, allocator padding, host
  duplicates and transfer staging. Hybrid reports additionally include the
  actual hot-bank placement floor, including null slots and router ID maps.
- KV assumes one full-context sequence and the same cache type for K/V.
  Block sizes and bytes come from ggml. Rows with invalid block alignment
  yield `null` plus a reason, never fractional blocks. Current DeepSeek MLA
  `key_length_mla`/`value_length_mla` metadata triggers the pinned engine's
  K-only compressed latent/rope allocation; legacy GGUF dimensions use separate
  K/V storage. Backend support for a listed cache type is not implied. There
  is no sliding-window reduction or allocator/context padding. The requested
  contexts are reported even when they exceed a model's context limit.

## Routing traces

`--trace` accepts the JSONL records documented by
`cc/routing_trace/README.md` or the replay JSON produced by `trace_replay.py`
(including `experiments/routing/*.json`). JSONL validates model-specific
expert ranges/top-k, weights, layer/phase, and duplicate positions. Replay
JSON uses its histograms, not simulated cache hits. Raw and replay reports
agree on the real prose trace.

Metrics count unweighted expert requests: frequency, Shannon entropy (bits),
Gini over all E experts including unused ones, and cumulative coverage by
descending frequency (expert ID breaks ties). `coverage_curve[H-1]` is the
fraction covered by the top H experts; recommendations are minimum H for
50%, 75%, and 90%. Empty histograms have unavailable rates/recommendations.

Phases and optional per-row `split` labels remain separate. The official raw
format has no train/eval split; it is explicitly `unspecified`, with observed
sequence/position ranges recorded. Replay split/train/evaluation/source
annotations are preserved when present. No split is inferred from filenames.
Every report warns that train≠eval and that completeness is not established.
Recommendations fitted to these traces need disjoint evaluation before use.

## Verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q cc/model_inspect/test_inspect_model.py \
  --basetemp=artifacts/codexMI/pytest-tmp \
  -o cache_dir=artifacts/codexMI/pytest-cache
```

Tests inspect real BF16, Q8_0, Q4_K_M and mixE4K Granite files and the real
hybrid split. They verify D013's approximately 90% expert share, mixE4K's
Q4_K experts with the Q8_0 source's non-expert types unchanged, null slots,
byte totals, placement floors, independent KV block-size calculations,
header-only access, synthetic shared-expert/MLA headers, routing metrics,
trace validation, real raw/replay equivalence, and CLI outputs. Real-file
tests skip with explicit reasons on machines without the campaign models.
