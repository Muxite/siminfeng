# routing_trace

Use this component to collect complete top-8 expert decisions from a single text sequence in llama.cpp, then estimate logical per-layer cache hit rates. The patch adds only `examples/routing-trace/` and its registration in `examples/CMakeLists.txt`. It does not modify the router, experts, backend, or public API. Replay uses the Python standard library; tests require pytest.

## Apply and build

The required source is **31288c9c66e59377f7788d920678a565bb89dd37**. Keep `ref/llama.cpp` read-only. All commands below start at the campaign root and put writable files under this assignment's locations.

The Build CC accepts the patch through its existing interface:

```bash
# Integration example for the build orchestrator; not executed for T-F.
# Supply digest-pinned BASE_DEVEL / BASE_RUNTIME as documented by Build CC.
python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp --commit 31288c9c6 \
  --patch cc/routing_trace/routing-trace.patch \
  --tag campaign/llamacpp:routing-trace \
  --out artifacts/codexF/build-engine
```

**Build CC packaging prerequisite:** the current `builds/llamacpp/Dockerfile` explicitly builds and copies seven binaries. The command above applies the patch but, with that recipe unchanged, does **not** build or package `/app/llama-routing-trace`. Before using it for tracer deployment, the Build CC owner must add `llama-routing-trace` to both the `cmake --build --target` list and the runtime copy loop, retain `LLAMA_BUILD_COMMON=ON` and `LLAMA_BUILD_EXAMPLES=ON`, and add it to `TOOLS` in `cc/build_engine/build_engine.py` so the build record hashes it. These files are outside T-F's write locations and were not edited. Passing a source patch cannot change a Dockerfile supplied separately by Build CC. A successful unextended Build CC build is not evidence that the tracer exists.

For a local build without changing Build CC, make a separate archive copy and apply there (or reuse the already verified copy at `artifacts/codexF/verification-tree`):

```bash
mkdir -p artifacts/codexF/local-source
git -C ref/llama.cpp archive 31288c9c6 > artifacts/codexF/local-source.tar
tar -xf artifacts/codexF/local-source.tar -C artifacts/codexF/local-source
git -C artifacts/codexF/local-source apply --check \
  "$PWD/cc/routing_trace/routing-trace.patch"
git -C artifacts/codexF/local-source apply \
  "$PWD/cc/routing_trace/routing-trace.patch"
cmake -S artifacts/codexF/local-source -B artifacts/codexF/local-build \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build artifacts/codexF/local-build --target llama-routing-trace -j 4
```

For GPU use, configure `GGML_CUDA=ON` and the appropriate CUDA architecture instead (campaign hardware: `CMAKE_CUDA_ARCHITECTURES=120`). No CMake build, Docker invocation, download, or network access was performed for this assignment.

## Capture and replay

```bash
artifacts/codexF/local-build/bin/llama-routing-trace \
  -m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf \
  -p 'Explain how a cache works.' -n 64 -ngl 0 -c 2048 \
  --trace-out artifacts/codexF/granite.jsonl \
  > artifacts/codexF/generated.txt 2> artifacts/codexF/capture.log

python3 cc/routing_trace/trace_replay.py artifacts/codexF/granite.jsonl \
  --json-out artifacts/codexF/replay.json \
  --markdown-out artifacts/codexF/replay.md
```

Use `-ngl 99` for full GPU offload with a CUDA build. `-f PATH` uses common's prompt-file handling, including stripping one trailing newline. Prompts are raw text, with the tokenizer's BOS policy and special-token parsing; no chat template is added. `--trace-out PATH` is required and truncates an existing file. `-n` defaults to 32; only finite nonnegative counts are accepted. Generation takes the argmax of raw logits (first vocabulary ID on ties); sampling, grammar, repetition penalties and other completion features are not applied. The tool deliberately evaluates exactly `n` generated tokens, including EOG if selected, so every generated token has a routing record. It does not stop early at EOG. Generated pieces go to stdout, diagnostics to stderr, trace records to the specified file. `-n 0` collects only prefill.

Example line (no abbreviation of either array):

```json
{"seq":0,"pos":0,"layer":0,"experts":[0,1,2,3,4,5,6,7],"weights":[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],"phase":"prefill"}
```

`pos` is the absolute model input position, including BOS. Prompt positions are `0..P-1`; generated-token positions are `P..P+n-1`. With the expected 24-layer Granite model, success produces `24*(P+n)` lines. Each row contains IDs and aligned weights in router rank order. Lines are ordered by position, then layer. Phases describe the submitted tokens, not the shape of the batch: a one-token prompt chunk is still prefill. Cache replay defaults to machine-readable JSON on stdout and its short Markdown table on stderr; the output flags write separate files instead.

## Callback and model assumptions

This is coupled to the pinned C++ headers, `llama_batch_ext` / `llama_process` API, and graph tensor naming. Build the example and libraries together; mixing libraries from another commit is unsupported. The intended model is Granite MoE, 24 layers, 32 experts, top-8. A compatible decoder-only Granite/Qwen-style graph can work if every layer exposes exactly eight IDs in `[0,32)` and the named weight stages below. Other top-k values, hybrid/dense layers, multimodal inputs, multiple sequences, encoder models, shared-expert decisions and alternate graph conventions are outside scope. A non-Granite model with fewer than 32 experts is not identified automatically; replay still assumes 32 and should not be used to infer its uniform baseline.

The observer asks only for exact tensor names ending in a numeric layer index:

- `ffn_moe_topk-N`, I32 `{8,T,1,1}`. This can be a view with the full 32-expert argsort's row stride. The code uses `nb[]`, never assumes packed IDs.
- `ffn_moe_weights-N`, F32 `{1,8,T,1}`, and optional `ffn_moe_weights_softmax-N`, `ffn_moe_weights_norm-N`, `ffn_moe_weights_scaled-N`, F32 in `{8,T,1,1}` or `{1,8,T,1}` form. The last observed stage in that order replaces earlier weights before emitting. Thus Granite records normalized (and, when present, scaled) mixture weights, not merely the initial selected probabilities. It does not assume their sum is one after scaling. Sum/clamp helper tensors are excluded.

During `ask=true`, the callback selects names only; it reads no data. During `ask=false`, it checks shape/type and copies each value into owned storage. Host buffers are read after the scheduler's synchronization; other buffers use `ggml_backend_tensor_get` before their storage can be reused. Floats are written with `max_digits10` and the classic locale. Missing/duplicate/incompatible tensors, invalid IDs/weights, failed decode or failed output writes cause a nonzero exit; earlier complete chunks may remain in a partial trace. Do not replay a failed run as complete evidence.

Prompt submissions are bounded by `min(llama_n_batch, llama_n_ubatch)` and contain one sequence in position order. This avoids reconstructing physical microbatch order from callback tensor names. Each submission is synchronized and validated before emission; unexpected physical splitting fails instead of silently assigning incorrect positions. Context overflow is refused; there is no context shift or KV reuse across runs.

The callback never writes tensors or changes routing math. **Execution overhead is material:** `ggml/src/ggml-backend.cpp:1899-1935` computes graph ranges ending at requested nodes and synchronizes each range, including the remainder of a split. It does not synchronize every node as the unfiltered stock debug callback does, but it reduces asynchrony and can inhibit fusion. Device readbacks and JSON output add further overhead. Warmup is disabled to exclude artificial routing decisions. The driver requests logits for **every prompt token** because `src/models/granite.cpp:175-178` otherwise prunes earlier prompt tokens before the last FFN. This preserves each evaluated token's math but adds final-layer/LM-head work and output storage compared with last-logit-only prefill. Full all-layer tracing therefore cannot be treated as a performance-neutral observation of the usual pruned graph. Chunk size and scheduling may also affect floating-point kernel choices; bitwise equivalence was not tested.

## Replay conventions and limits

Every report starts with: **SIMULATION on a logical trace: no residency/transfer semantics; Granite fully fits GPU memory.** Neither hit rates nor observed routing locality prove a GPU memory bottleneck, transfer reduction, latency improvement or speedup.

The JSON provides per-phase, per-layer 32-bin unweighted usage histograms, Shannon entropy in bits (maximum 5), and mean consecutive-position top-8 intersection count and fraction (intersection/8). Only adjacent positions within the same layer, sequence and phase form overlap pairs; gaps do not. A missing pair or absent phase produces `null` rates, not zero. There is no weight-weighted hit metric.

LRU and LFU are simulated with **4, 8, 12, 16, 24 slots per layer**, separately for prefill and decode. Every phase/layer/sequence starts empty; decode is **not** warmed from prefill. Rows are sorted by sequence, position, layer so graph traversal/interleaving does not become an access-order artifact. All eight expert lookups are scored against the slots at token entry, then the eight requests update/admit in recorded rank order. Intra-token eviction does not retroactively change hits. This atomic lookup convention makes an independent uniformly random top-8 trace approach capacity/32, including capacities below eight. It is not a model of eight experts simultaneously pinned in four slots. Router rank order determines final cache contents and can affect locality estimates.

LRU evicts the least recently touched resident. LFU counts only accesses during the current residency, starts at one on admission and discards counts on eviction; the least recently touched resident breaks frequency ties. This is neither global-history LFU nor decayed LFU. Each miss admits its expert; there is no bypass policy. Overall hit rate is total expert hits / total expert requests, not an unweighted average of layer rates. Startup misses are included. Replay checks types, eight distinct expert IDs, finite nonnegative weights, phases and duplicate `(seq,pos,layer)` records. It accepts partial layer/position coverage; completeness must be checked against the capture driver's successful exit and expected token/layer counts. It holds the trace in memory.

Strata's `include/strata/core/expert_cache.hpp:154-166` defines `(layer,expert)->slot or -1`, admission without eviction, and caller-managed replacement after a copy completes. Its optional per-layer admission partitions a physical slot pool. Replay borrows only the layer/expert key concept: its LRU/LFU replacement, independent per-layer capacities and instant admission are hypothetical policies, not a reproduction of Strata's implemented cache, byte packing, residency table, replacement publication, synchronization or CPU/GPU dispatch.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/routing_trace/test_trace_replay.py \
  --basetemp=artifacts/codexF/pytest-tmp \
  -o cache_dir=artifacts/codexF/pytest-cache
```

The retained `artifacts/codexF/verify_patch.py` archives the pinned commit into `artifacts/codexF/verification-tree` (must not already exist), runs `git apply --check`, applies the patch there, and invokes:

```bash
g++ -fsyntax-only -std=c++17 -Wall -Wextra -Wpedantic \
  -Iartifacts/codexF/verification-tree/ggml/include \
  -Iartifacts/codexF/verification-tree/include \
  -Iartifacts/codexF/verification-tree/common \
  -Iartifacts/codexF/verification-tree/tools \
  artifacts/codexF/verification-tree/examples/routing-trace/routing-trace.cpp
```

Results: **29 pytest cases passed**, patch check/application passed on an archive copy, and C++17 syntax-only passed with no diagnostics (GCC 13.3.0). Source tracked status was clean before and after. See `artifacts/codexF/verification.json`, `verification.log`, and `pytest.log`. Tests cover repeated experts, deterministic uniform random routing, LRU/LFU differences and tie behavior, cold phases/sequences/layers, weighted aggregation, overlap gaps, malformed input and the CLI output channels. `synthetic-trace.jsonl` and `synthetic-replay.json` / `.md` in the same evidence directory demonstrate output format only; they are not a model capture.

**Not compile-linked:** `llama-routing-trace`, `llama-common`, `llama`, and ggml backends were not built or linked. CMake configuration, CUDA compilation, real host/device callback execution, model capture, numerical equivalence, GPU performance, and Build CC runtime packaging were not verified. An actual patched build and Granite capture remain integration work for the orchestrator.
