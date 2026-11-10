# routing_trace

Capture routed expert IDs and aligned weights with llama.cpp's graph callback.
The generic patch is `patches/routing-trace-generic.patch`, standalone against
**31288c9c66e59377f7788d920678a565bb89dd37**. Apply this patch alone, not on top
of `routing-trace.patch`. It adds `examples/routing-trace/` and its CMake
registration; the executable target remains **llama-routing-trace**.
`ref/llama.cpp` must stay read-only; apply patches only in a separate build tree.

## Build CC integration

From the campaign root, the build owner can use:

```bash
python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp --commit 31288c9c6 \
  --patch cc/routing_trace/patches/routing-trace-generic.patch \
  --extra-target llama-routing-trace \
  --tag campaign/llamacpp:routing-trace-generic \
  --out artifacts/routing-trace-generic/build
```

Supply the digest-pinned development/runtime images required by Build CC.
Its current `--extra-target` interface builds, packages and records the extra
binary; the old README's instruction to edit Dockerfile/TOOLS is obsolete.
Keep `LLAMA_BUILD_COMMON=ON` and `LLAMA_BUILD_EXAMPLES=ON`. The generic patch
has **not been built or run** by J-OM3. Per `research/own-repo-assets.md`, D014
records a real ccache build (rt1) and three real Granite traces using the legacy
patch; the previous blanket “not compile-linked” statement was stale.
Those runs do not validate this generic patch or Qwen capture.

## Capture

```bash
/app/llama-routing-trace -m /models/model.gguf \
  -p 'Explain how a cache works.' -n 64 -ngl 0 -c 2048 \
  --trace-out /out/routing.jsonl > /out/generated.txt 2> /out/capture.log
python3 cc/routing_trace/validate_generic_trace.py /out/routing.jsonl
python3 cc/expert_replay/expert_replay.py --help
```

Use a separate output path per run: `--trace-out` truncates an existing file.
Only one text sequence is supported (`seq=0`). Prompt-file handling (`-f`) and
BOS tokenization come from common; no chat template is added. `-n` defaults to
32 and must be finite/nonnegative. Generation uses raw-logit argmax, ignores
sampling settings, and evaluates exactly N generated tokens, including EOG;
`-n 0` captures only prefill. `tok` is the absolute evaluated input position,
including BOS, and starts at zero. Phases describe submitted tokens, so a
single-token prompt chunk remains prefill. Context overflow is refused.

The first line is metadata, emitted after the first submission has validated:

```json
{"type":"metadata","arch":"example","n_layer":2,"layers":2,"model_path":"/models/model.gguf","model_sha256":null,"experts":16,"n_expert":[16,0],"k":[2,0],"shared_experts_routed":false,"shared_experts_present":[true,false]}
```

`n_expert` and `k` are arrays indexed by **model layer ID**, with zero for dense
layers. `experts` is the maximum per-layer expert count, and `layers` aliases
`n_layer` for `cc/expert_replay`. A full-file SHA is omitted (`null`) to avoid
reading a multi-gigabyte model just for capture identity; record an existing
manifest hash alongside the run when available. Model path and architecture
strings are JSON escaped. Each routing row is:

```json
{"tok":0,"layer":0,"experts":[3,12],"weights":[0.4,0.6],"phase":"prefill","seq":0}
```

The last line is emitted only after successful generation and capture:

```json
{"type":"metadata","trace_complete":true,"tokens":1,"records":1}
```

The footer's `type` lets the existing expert_replay parser skip it as metadata
without counting it as malformed input. `tokens` counts evaluated input
positions, not just generated tokens; `records` counts routing rows. Missing
footer, partial JSON, wrong counts, missing tokens/layers, or data after the
footer are rejected by `validate_generic_trace.py`. Always validate before
replay: expert_replay itself does not enforce this completion protocol. Its
coverage diagnostic assumes every model layer routes; mixed dense/MoE models
therefore report incomplete layer coverage even when capture is complete.
Its global expert bound also cannot enforce individual per-layer bounds; the
validator does. Use `cc/expert_replay`, not the legacy `trace_replay.py`, for
generic traces. The latter retains Granite-only 32-expert/top-8 assumptions and
accepts only legacy `pos` records without metadata/footer.

## Callback contract and limits

The tool derives `n_expert` from `ffn_moe_probs-N` shape `{E,T,1,1}` and `k`
from `ffn_moe_topk-N` shape `{K,T,1,1}`, never from hardcoded model dimensions
or tensor strides. It copies I32 IDs using `nb[]` (top-k may be a strided view).
F32 weights must be `{1,K,T,1}` or `{K,T,1,1}`. It observes exact stages
`ffn_moe_weights-N`, `_softmax-N`, `_norm-N`, `_scaled-N`, retaining the last
stage. Sum/clamp helpers are excluded. Weights must be finite/nonnegative;
IDs must be distinct and within that layer's router dimension.

Shared expert presence is discovered through `ffn_shexp-N`,
`ffn_shexp_out-N`, `ffn_moe_shexp-N`, or `ffn_shared_out-N` callbacks. These
branches are always computed when present (possibly sigmoid gated), are not
top-k selections, and never appear in routed expert arrays. Architectures
without these shared-branch callback names require a hook extension before
interpreting a false presence flag. Custom graph naming, remapped expert IDs
(e.g. grouped GroveMoE), and architectures bypassing `build_moe_ffn` need
separate validation; “generic” means shape independent within this callback
contract, not a guarantee for every architecture.

Qwen35moe requires routing in every main layer even on the first submission.
For other architectures, first-submission discovery permits dense layers; later submissions must have
identical routed-layer coverage, dimensions and shared presence. Missing,
duplicate or reordered tensors fail capture. A model whose first graph omits
an entire routed layer cannot be distinguished from a dense layer by callback
observation alone. Unexpected physical batch splitting is rejected.

Submissions contain at most `min(n_batch,n_ubatch)` tokens and are synchronized
before emission. Owned data is bounded by one submission's layer/top-k values
plus one observed tensor's device readback, rather than total generated tokens.
Output is ordered by token then model layer, and flushed after **each token**.
Host buffers are read after callback synchronization; device buffers use
`ggml_backend_tensor_get`. No routing tensors are mutated.

Tracing has material overhead: selected callbacks force scheduler range
synchronizations and device readbacks. Logits are requested for every prompt
token to prevent final-layer pruning. Warmup and backend sampling are disabled.
This can inhibit fusion and add output/LM-head work. Captures are routing
evidence, not performance-neutral timing runs; numerical equivalence has not
been established. Any failed decode, callback validation, output write or
close returns nonzero; reject the trace even if a footer is visible after an
I/O failure. Flush is not fsync or an atomic completion transaction.

## Jetson / qwen35moe

At the pin, the architecture lives in `src/models/qwen35moe.cpp` (the model
factory and public layer-count accessor are in `src/llama-model.cpp`). Its main
graph loops `il=0..n_layer-1`, selects DeltaNet or full attention, then
unconditionally calls `build_layer_ffn`, which asserts a router and calls
`build_moe_ffn`. Thus **every main decoder layer emits routing**, including
DeltaNet layers: IDs 0..39 for the 40-layer variant. Default full-attention
interval is four (3,7,...,39); metadata may override the layer pattern.
`llama_model_n_layer` returns `hparams.n_layer()`, excluding NextN/MTP layers;
this single-model driver does not run the separate MTP graph. Shared expert
branches are recorded as present and excluded from routed choices.

Build on the Jetson lane with its configured CUDA architecture/image and
ccache via Build CC, using the same extra target. Do not copy the x86 CUDA
architecture 120 setting. Pin `-c` and explicit offload settings (`-ngl`, and
common's CPU-MoE options when applicable) to match the experiment; confirm the
built binary's `--help`. Preserve mmap defaults for over-memory experiments.
Do not infer physical DRAM pressure on UMA from Docker `--memory` or simulated
cache hit rates. Record the model manifest/hash, command, chunk sizes, successful
exit status and validation result. Qwen3.6's expected 256 experts/top-8 are a
sanity check, never capture constants. Capture only; no GPU/build was run for
this assignment.

## Verification (J-OM3)

```bash
git -C ref/llama.cpp apply --check \
  "$PWD/cc/routing_trace/patches/routing-trace-generic.patch"
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/routing_trace/test_generic_trace.py \
  cc/routing_trace/test_trace_replay.py cc/expert_replay/test_expert_replay.py \
  -p no:cacheprovider
```

On the pinned reference: apply check passed; **56 tests passed, 5 subtests
passed**. New fixture tests cover variable top-k, IDs through 255, dense layers,
replay compatibility, schema failures and truncated/missing/mismatched footers.
These are Python protocol tests, not C++ callback execution tests. No builds,
GPU execution or model capture were performed; integration remains with the
build owner. The original patch and replay tests remain for legacy Granite
reproducibility.
