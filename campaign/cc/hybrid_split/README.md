# T-R: Granite hot GPU / cold CPU engine candidate

Private experimental fork of llama.cpp `31288c9c66e59377f7788d920678a565bb89dd37`.
Patch SHA256: `1063ac102a14bff798d56b5bf292d21455f3f4c847f0e2043d98d58126bd1efb`.

The engine patch is `cc/hybrid_split/hybrid-split.patch`; the applied tree and all
execution evidence are under `artifacts/codexR/`. `ref/` was not modified.
No network, Docker, CUDA compilation, or GPU execution was used by this worker.
All local builds used `GGML_CUDA=OFF` and at most `-j4`; inference used four threads.

**CPU result:** original mixE4K versus each of the n18/n12/n6 matched split files:
64/64 greedy tokens identical, maximum absolute logit difference **0**, RMS logit
difference **0**. This establishes CPU numerical correctness for this prompt,
not a GPU speedup or mixed-device numerical equivalence.

## Format and integration conditions

Source: `/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf`,
SHA256 `2e520c4a7b15ce292cc88e92fef00fb99da276dc5d6fc8e8f832b8525f8565a6`.
It is `granitemoe`, 24 layers, 32 experts, top-8, embedding 1024, FFN 512.
Actual GGML dimensions (fastest dimension first) and types are:

| Tensor in each `blk.L` | GGML shape | Type |
| --- | --- | --- |
| `ffn_gate_inp.weight` | `[1024,32]` | F32 |
| `ffn_gate_exps.weight` | `[1024,512,32]` | Q4_K |
| `ffn_up_exps.weight` | `[1024,512,32]` | Q4_K |
| `ffn_down_exps.weight` | `[512,1024,32]` | Q4_K |

Each expert in each projection is 294,912 packed bytes. A complete expert across
three projections is 884,736 bytes (0.84375 MiB).

`split_gguf.py` uses only the pinned `ref/llama.cpp/gguf-py` and installed NumPy.
It takes the `make_hot_set.py` text format: comments plus `layer expert` rows.
Within each layer it sorts the hot original IDs, appends the remaining original
IDs, permutes the F32 router rows consistently, and copies packed expert blocks
without dequantization or requantization. It replaces each expert tensor with:

- `<original-name>.hot`: `[ne0,ne1,H+1]`, with H real hot experts followed by one
  all-zero null expert in the same packed type.
- `<original-name>.cold`: `[ne0,ne1,32-H]`, holding the remaining real experts.
- `blk.L.ffn_expert_ids.ids_hot`: I32 `[1,32]`, original ID to hot slot or H.
- `blk.L.ffn_expert_ids.ids_cold`: I32 `[1,32]`, original ID to cold slot or -1.
- `blk.L.ffn_expert_ids.ids_router`: I32 `[32]`, original ID to permuted router row.

The three ID tensors are per layer, not per expert projection. Metadata records
`hybrid_split.version=1`, source/hot-set SHA256 values, hot counts and a flattened
new-index-to-original-index permutation. Other metadata and tensors are copied.
Only little-endian `granitemoe` with 32 experts is accepted. The transform supports
F32/F16/Q4_K/Q8_0 expert banks with F32 routers; the real validation used Q4_K.
Each layer must have 0..31 hot experts; all-hot layers are rejected because GGUF
zero-sized cold banks are unsupported. Unknown expert/router companion tensors,
duplicate/out-of-range hot entries, existing outputs and already split inputs
are rejected. Production CLI GGUF output paths are restricted to
`artifacts/codexR/models/`; pytest temporary GGUFs are also placed there.

The loader recognizes optional split banks only for Granite MoE. Existing
expert pointers own the hot banks; extra layer pointers hold cold banks and ID
maps. All use existing model buffer contexts and normal tensor ownership. There
are no new backend allocations, callbacks, placement APIs or CUDA kernels.
Missing/partial banks, inconsistent shapes and non-I32 maps fail loading. Map
contents are checked for a bijective router permutation and consistent local
IDs when the graph is built, after weights are loaded.

Required runtime flags are:

```text
-ngl 99 -ot '\.cold$=CPU' --no-op-offload --no-repack --no-warmup
```

`-ot` is the existing tensor buffer override. `--no-op-offload` is necessary:
the scheduler can otherwise move CPU weight operations to CUDA during prefill.
The split graph explicitly rejects op offload. `--no-repack` ensures plain CPU
buffers reach the modified generic CPU kernel; optimized repacked CPU buffers
have separate MUL_MAT_ID implementations. Cold non-host buffers are rejected.
Do not combine split models with `--moe-cache-mib`, LoRA, Hadamard transforms,
expert biases/scales, post-expert normalization, or other activation functions;
the split branch rejects these combinations. No KV or attention code is changed.

## Routing, skips, numerical order, and CUDA graphs

Routing uses the existing router, softmax, top-k, weight normalization and scaling.
After router matmul, transpose/contiguous/GET_ROWS/transpose/contiguous restores
logits to the **original** expert order before softmax and top-k. This preserves
both floating-point reduction order and original-ID tie-breaking despite the
on-disk permutation. It uses existing operations on the GPU, not an I32 REPEAT
(which this CUDA backend does not support). Two further existing I32 GET_ROWS
operations map flattened selected IDs into the hot and cold partitions.

CPU MUL_MAT_ID explicitly zeros each negative-ID destination column and omits it
from the per-expert work lists. Positive bounds use `GGML_ASSERT`, so Release
builds still reject positive out-of-range IDs. Skips perform no expert dot
products or expert-weight accesses. ID scanning, output zeroing, activation
conversion and dense activation work still have a fixed top-8 overhead; total
latency is not claimed to be strictly proportional to the number of cold IDs.
The negative-ID convention is **CPU-only**, and requires plain CPU buffers.
CUDA always receives in-range hot IDs.

The null expert has exactly zero gate, up and down weights. Its unweighted
contribution is zero; original routing weights are applied **once** after adding
hot/cold slot outputs. No separate zero weight mask is needed. Normalization
and expert scaling remain in the shared original code, ahead of this branch.
Slot outputs are reduced in the original selected-slot order.

One null expert cannot safely be used with all hot slots in a single CUDA
MUL_MAT_ID: `ggml-cuda/mmid.cu` compacts one record per token/expert and assumes
unique IDs, while several misses repeat the null ID. To avoid changing CUDA
kernels, GPU-resident hot banks dispatch each selected slot separately and
concatenate the outputs. Every CUDA ID list then has one ID per token. This
requires 24 hot matmuls instead of 3 and 7 concatenations for ordinary top-8,
plus 3 cold matmuls. GPU launch/bandwidth overhead may defeat the hypothesis.
When hot banks are on plain host buffers, one batched hot FFN retains the
original CPU arithmetic and avoids that overhead. The focused test exercises
the separate-slot version even on CPU and can run the same version on CUDA.

An earlier discarded design permuted the routing reduction order and dispatched
CPU hot slots separately. It passed local permutation and matmul tests but
failed the full continuation. Its evidence is retained as `*-v0.*`; the reported
successful results use the corrected graph. Different per-slot matmul shapes
can also change rounding on GPU; CPU bitwise equivalence is not evidence that
GPU or mixed-device logits will be bitwise equivalent.

Sync/transfer costs with hot banks/maps on GPU and cold banks on CPU:

- Graph construction checks three 32-element maps per layer using synchronous
  backend tensor reads (384 bytes/layer). This can synchronize a GPU during graph
  construction/rebuild; cached graph execution does not repeat those reads.
- Integer lookup itself adds no host roundtrip. Routing-order restoration adds
  small GPU copy/gather kernels, not new arithmetic or a new backend operation.
- Execution must transfer token activations and cold IDs GPU to CPU, then cold
  output activations back. IDs are 32 bytes/token/layer at top-8; the shared FFN
  input is 4096 bytes/token/layer and the unreduced cold slot output is up to
  32768 bytes/token/layer. Scheduler split boundaries impose synchronization;
  actual copy counts/order and any intermediate transfers require GPU tracing.
- No cold expert weights need an H2D copy with the required flags. This is a
  source/scheduler design claim, not a measured transfer trace.

A whole mixed-device layer cannot be captured as one CUDA graph. GPU-only
scheduler partitions remain eligible for existing CUDA graph capture after
normal backend warmup if their hot MUL_MAT_ID kernels pass
`ggml_cuda_mul_mat_id_needs_sync`; this patch neither forces nor disables graphs.
CPU partitions and transfers remain outside those captured GPU partitions.
Actual capture/reuse, split counts and backend fallback paths are unverified.

## Create variants matching static expert payloads

Run from the campaign root. These commands use the real prose prefill trace:
sequence 0, positions `[0,1560)`, SHA256
`f8a108bed6c2d086779f63837b9a330f1d3ac64ab4e471d08f433035815b9489`.
This is the TRAIN set for frequency ranking; the numerical continuation uses
`artifacts/traces/in/chat.txt`. No synthetic hot set is used for the real model.

Static n18/n12/n6 keep 6/12/18 full expert layers on GPU. Subtract 24 real hot
experts from the corresponding counts to pay for the 24 per-layer null experts:

| Static comparison | Real hot experts | Real hot fraction | Hot plus null payload | D022 measured total GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| n18 | 168 | 21.875% | 162 MiB | 902 |
| n12 | 360 | 46.875% | 324 MiB | 1064 |
| n6 | 552 | 71.875% | 486 MiB | 1226 |

These match **expert payload**, not measured total GPU memory. Small ID maps,
backend alignment and new activation/concatenation buffers change the complete
footprint. The orchestrator must measure both shim peak and nvidia-smi delta at
`-c 4096`, and trim hot experts if necessary before making an equal-MiB claim.
The selector's budget includes its own single group guard slot only to select
exactly N entries; its reported cache-bank allocation is not this engine layout.

```bash
mkdir -p artifacts/codexR/hot-sets artifacts/codexR/models
for ncpu in 18 12 6; do
  hot=$(( (24 - ncpu) * 32 - 24 ))
  PYTHONDONTWRITEBYTECODE=1 python3 cc/moe_static_hot/make_hot_set.py \
    artifacts/traces/out/prose.jsonl \
    --output "artifacts/codexR/hot-sets/n${ncpu}.txt" \
    --report "artifacts/codexR/hot-sets/n${ncpu}.json" \
    --budget-bytes "$(( (hot + 1) * 884736 ))" \
    --expert-bytes 294912 294912 294912 --alignment 128 \
    --train-seq 0 --train-phase prefill \
    --train-pos-start 0 --train-pos-end 1560
  PYTHONDONTWRITEBYTECODE=1 python3 cc/hybrid_split/split_gguf.py \
    "$HOME/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf" \
    "artifacts/codexR/models/granite-split-n${ncpu}.gguf" \
    --hot-set "artifacts/codexR/hot-sets/n${ncpu}.txt" \
    --report "artifacts/codexR/split-n${ncpu}.json"
done
```

The real split files already exist; the transformer refuses to overwrite them.
For a fresh run, choose new output basenames under the same models directory.
Their SHA256 values are:

```text
n18 02951b0a845f954f8a665f1c48d5f4593fee469769cd992f71ab265571970c4d
n12 fbe26fbb7689df8cc5db4c46f1a81ccf55e99e2ca44f9b29dce7d1dc8d36c494
n6  f7d321d144887310bb863d225af4f4274e4460c05e3d9ec3a69ae8a53e7aa75f
```

## Orchestrator: Build CC and CUDA runs

The following are handoff commands, **not commands executed by this CPU-only
worker**. They require the orchestrator's Docker/CUDA environment and cached,
identified base images. Build CC currently uses `--parallel 10` internally;
that Docker recipe was not run or modified here. All worker compilation used
`-j4`. Build CC's runtime base digest resolution must succeed locally.

```bash
python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp \
  --commit 31288c9c66e59377f7788d920678a565bb89dd37 \
  --patch cc/hybrid_split/hybrid-split.patch \
  --cmake-define GGML_CUDA_NO_VMM=ON \
  --cmake-define LLAMA_BUILD_HYBRID_SPLIT_TESTS=ON \
  --extra-target test-backend-ops \
  --extra-target test-hybrid-split \
  --tag campaign/llamacpp:hybrid-split-r \
  --out artifacts/codexR/cuda-build
```

Build CC fixes `LLAMA_BUILD_TESTS=OFF` and rejects overrides of that built-in key.
The patch's default-OFF `LLAMA_BUILD_HYBRID_SPLIT_TESTS` option exposes the tests
subdirectory for those explicit extra targets despite that fixed setting. This
configuration was compiled locally with CUDA OFF, including both test targets.

```bash
docker run --rm --network none --gpus all \
  campaign/llamacpp:hybrid-split-r /app/test-hybrid-split --backend CUDA0

docker run --rm --network none --gpus all \
  campaign/llamacpp:hybrid-split-r \
  /app/test-backend-ops -b CUDA0 -o GET_ROWS -p 'type=i32' -j 1

for ncpu in 18 12 6; do
  docker run --rm --network none --gpus all \
    -v "$PWD/artifacts/codexR/models:/models:ro" \
    -v "$PWD/artifacts/traces/in:/prompts:ro" \
    campaign/llamacpp:hybrid-split-r /app/llama-completion \
    -m "/models/granite-split-n${ncpu}.gguf" \
    -ngl 99 -ot '\.cold$=CPU' --no-op-offload --no-repack \
    --no-warmup -c 4096 -n 160 -t 4 -tb 4 --temp 0 --seed 1 \
    -f /prompts/chat.txt --no-display-prompt \
    > "artifacts/codexR/cuda-n${ncpu}.stdout" \
    2> "artifacts/codexR/cuda-n${ncpu}.stderr"
done
```

The focused CUDA command keeps negative IDs exclusively on its CPU cold backend,
checks the scheduler assignment, and checks hot/null computation on CUDA. It
covers F32 and Q4_K, one and 17 tokens, broadcast and per-slot activations, mixed,
all-hot and all-cold routing, nontrivial permutations, router restoration, and
NaN-poisoned reused skipped outputs. It does not add negative-ID CUDA support.
Use the existing campaign GPU monitor/budget shim to record memory and transfers
when benchmarking; the commands above alone do not establish equal GPU MiB.

## Reproduce CPU checks

The applied scratch tree already exists. For a clean reproduction extract the
pinned source and apply the patch into a new tree under `artifacts/codexR/`.

```bash
mkdir -p artifacts/codexR/repro-tree
git -C ref/llama.cpp archive 31288c9c6 | tar -x -C artifacts/codexR/repro-tree
GIT_CEILING_DIRECTORIES="$PWD/artifacts/codexR" \
  git -C artifacts/codexR/repro-tree apply "$PWD/cc/hybrid_split/hybrid-split.patch"
cmake -S artifacts/codexR/repro-tree -B artifacts/codexR/repro-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_HYBRID_SPLIT_TESTS=ON
cmake --build artifacts/codexR/repro-build \
  --target llama-cli llama-completion test-backend-ops test-hybrid-split -j4
artifacts/codexR/repro-build/bin/test-hybrid-split

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/hybrid_split/test_split_gguf.py \
  --basetemp=artifacts/codexR/models/pytest-tmp \
  -o cache_dir=artifacts/codexR/pytest-cache

c++ -std=c++17 -O2 cc/hybrid_split/compare_logits.cpp \
  -I artifacts/codexR/repro-tree/include -I artifacts/codexR/repro-tree/ggml/include \
  -L artifacts/codexR/repro-build/bin \
  -Wl,-rpath,"$PWD/artifacts/codexR/repro-build/bin" -lllama \
  -o artifacts/codexR/compare-logits-repro
for ncpu in 18 12 6; do
  artifacts/codexR/compare-logits-repro \
    "$HOME/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf" \
    "artifacts/codexR/models/granite-split-n${ncpu}.gguf" \
    artifacts/traces/in/chat.txt "artifacts/codexR/repro-n${ncpu}.json"
done
```

The comparator uses llama's CPU backend, raw prompt text with the model's BOS
policy, greedy argmax, four generation/batch threads, context 512 and ubatch 128.
It compares every vocabulary logit at each of 64 continuation steps, continuing
through EOS if encountered, and checks finite logits. Each model follows its own
greedy tokens; successful runs have identical prefixes throughout.

Evidence: `artifacts/codexR/{build-final.log,build-cc-mode.log,focused-test.log,
pytest.log,numerical-n18.json,numerical-n12.json,numerical-n6.json}`. There are
8 passing pytest cases and 8 focused configurations with 3 routing modes each.
The source GGUF is read in place; only transformed models are stored locally.

Unverified: CUDA compilation and execution, mixed CPU/GPU numerical accuracy,
CUDA graph capture, GPU allocation/transfer traces, exact total memory matching,
throughput versus n-cpu-moe/LRU, other prompts/models, and optimized CPU buffers.
The extra GPU launches and unreduced cold output transfer are substantial open
performance risks. This is a CPU-verified engine candidate, not evidence that
D014/D019 beats either existing placement strategy.
