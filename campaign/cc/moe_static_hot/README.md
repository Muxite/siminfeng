# Static hot experts in the pinned engine

Candidate against **31288c9c66e59377f7788d920678a565bb89dd37**. `static-hot.patch` changes only `src/llama-moe-cache.cpp`. The source map is in `artifacts/codexO/moe-cache-map.md`.

This is an engine-native **static weight cache**, not the hybrid harness's CPU-miss dispatcher. The original engine has no such dispatcher: its cache misses are uploaded and computed on the GPU. This minimal patch prefills fixed banks and routes every batch through the existing uncached selected-expert path. Hits are copied D2D from the banks into the scheduler's compute copy; misses are uploaded there from host weights. **No admission or eviction happens after initialization, but miss H2D and D2D hit traffic remain.** Removing admission traffic does not mean removing all miss traffic. An end-to-end speedup is an untested hypothesis.

## Apply and enable

Use an archive copy, never edit `ref/llama.cpp`. In an archive nested below another Git checkout, stop Git discovery before the archive's parent; otherwise `git apply` can silently skip paths outside the enclosing repository subdirectory.

```bash
mkdir -p artifacts/codexO/tree
# Only for a fresh, empty scratch tree:
git -C ref/llama.cpp archive 31288c9c6 | tar -x -C artifacts/codexO/tree
GIT_CEILING_DIRECTORIES="$PWD/artifacts/codexO" \
  git -C artifacts/codexO/tree apply --check "$PWD/cc/moe_static_hot/static-hot.patch"
GIT_CEILING_DIRECTORIES="$PWD/artifacts/codexO" \
  git -C artifacts/codexO/tree apply "$PWD/cc/moe_static_hot/static-hot.patch"
```

Set `LLAMA_MOE_CACHE_STATIC=/absolute/path/hot.txt` **and** a positive `--moe-cache-mib N`. The environment variable is deliberately local to cache construction: no public ABI, common parser, server or benchmark option changes are needed. It is read once per context. Unset preserves original LRU execution and log formatting. An empty/missing file path fails construction when the cache is enabled. With cache size zero the constructor is not called and the variable is ignored. Avoid inherited cache configuration between runs.

File format: UTF-8 ASCII-compatible text, zero-based `layer expert` integer pairs, one per line; blank lines and `#` comments allowed. Each pair names **all** participating projections for that expert. Example:

```text
# layer expert; comments are optional
0 3
0 17
1 9
```

Negative/out-of-range IDs, duplicates, extra fields, unavailable layers, or sets exceeding a device/layout group's capacity fail initialization. An empty set is legal with an otherwise valid cache budget; all requests miss. Unlisted experts remain at slot -1, which is never passed to a matmul. Rows are sorted before packing. Multiple groups/devices are validated separately; the included generator supports only one GPU and one homogeneous group.

## Integration conditions and ownership

- Eligibility/layout and budget splitting are inherited unchanged. All participating expert weights must be host-resident WEIGHTS tensors, with a GPU/IGPU assigned to their layer. Tensor-parallel split mode is unsupported. Cache construction requires at least top-k bank slots even for an empty set. Do not request a hot set for fully offloaded or CPU-assigned layers.
- Banks are 3D projection tensors with expert index in dimension 2 and stride `nb[2]`. Group membership requires equal types, shapes, projection list and stride. The extra MMQ guard slot is zero-filled. CPU_REPACK has no host-buffer predicate and is excluded by the original cache eligibility test; this patch adds no repacking or kernels. Use `--no-repack` and leave `--no-host` unset for the controlled cache experiment, and verify the actual loaded layout matches the declared byte accounting.
- Banks, maps, original-weight pointer bindings and scratch views belong to `llama_moe_cache::impl`, owned by the context. Bank writes use synchronous `ggml_backend_tensor_set`, followed by backend synchronization before construction returns. Maps are published only after all projections of that expert have been written. Memory-only `no_alloc` construction validates the file and capacity but does not read weights or prefill zero-sized buffers.
- After init the bank contents and maps stay fixed. `get_slot_map()` returns null in static mode; the LRU planning callback is unreachable and guarded by an assertion. Existing scheduler callbacks reuse resident ranges and upload nonresident ranges into temporary original-ID-indexed tensors. A host-weight split not recognized by the selected-expert callback falls back to normal scheduler copying. No guarantee of cache hits being used is made for an arbitrary model/graph/backend.
- The fallback reads IDs and synchronizes at runtime. Scheduler events protect temporary-buffer reuse; D2D copies execute in backend order before consumers. Pipeline parallelism stays disabled with a cache. One context must not be evaluated concurrently. No change to context-state serialization is required: construct/reconstruct the cache with the same file when recreating a context; it is not a mutable KV-state snapshot.
- Compute graph capture remains the backend's decision. No CUDA implementation is changed. Stable static banks do not remove host callbacks, graph splits, selected-ID readbacks, or `MUL_MAT_ID` capture restrictions. Run with default graphs and with `GGML_CUDA_DISABLE_GRAPHS=1`; capture/replay and overlap need GPU validation.

Persistent bank bytes are **not total GPU bytes**. Host slot maps, alignment, guard storage, KV/context buffers and especially scheduler copies of original-shaped expert weights must be accounted for. The static path may reserve more staging memory than the direct LRU slot path. For a homogeneous group, the bank estimator is `sum(align(nb[2]*(slots+1), backend_alignment))`. Allocation uses the original budget algorithm even if the file selects fewer experts than fit; spare slots stay unused.

## Generate a hot set

`make_hot_set.py` reads raw JSONL in `cc/routing_trace` format and reuses its strict Granite top-8 reader. Aggregated `experiments/routing/*.json` reports are not input traces. It ranks `(layer,expert)` by **unweighted TRAIN frequency**, with deterministic layer/expert tie-breaking, and assigns the global equal-byte pool across layers. This maximizes observed TRAIN hits for equal-size experts; it does not force the same K in every layer. Zero-frequency experts are not selected to fill spare slots.

TRAIN must be declared by sequence IDs, phase and half-open absolute position interval. Other rows are validated but never counted. The output comments and optional JSON report retain the split, input SHA256, capacity, layout assumptions and bank/payload bytes. This is selection on a logical trace, not a measured cache/performance result.

A real example already generated under `artifacts/codexO/hot-sets/` uses **prose seq=0, prefill positions [0,1560)** for TRAIN. All decode records are excluded. Decode of that same prompt is a within-sequence holdout; code/chat decode is a cross-prompt holdout. The BF16-captured routing may differ from mixE4K model routing, so profile transfer must be reported. The local mixE4K GGUF has gate/up/down shapes `[1024,512,32]`, `[1024,512,32]`, `[512,1024,32]` and 294,912 bytes per expert per projection. Pinned CUDA buffer alignment is 128; confirm the loaded engine layout before reusing these values.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 cc/moe_static_hot/make_hot_set.py \
  artifacts/traces/out/prose.jsonl \
  --output artifacts/codexO/hot-sets/prose-prefill-325mib.txt \
  --report artifacts/codexO/hot-sets/prose-prefill-325mib.json \
  --budget-bytes 340787200 --expert-bytes 294912 294912 294912 \
  --alignment 128 --layers 24 --train-seq 0 --train-phase prefill \
  --train-pos-start 0 --train-pos-end 1560
```

Budgets 163/244/325/487 MiB allocate 192/288/384/576 persistent slots respectively, including the extra guard slot in the budget calculation. They correspond to average K=8/12/16/24, not equal per-layer K. The generator refuses a budget below the engine's minimum eight slots. It does not infer strides from GGUF, partition multi-device budgets, or adjust for heterogeneous layout groups.

## Correctness and exact orchestrator experiment

First build the static patch through Build CC using the pinned commit, the patch hash in `artifacts/codexO/patch-verification.json`, CUDA architecture 120 and the campaign's recorded compiler/graph configuration. Use the same binary for cache-off, LRU and static modes. If measuring with the allocation shim, use `GGML_CUDA_NO_VMM=ON` consistently. This assignment did not invoke Docker/Build CC. Record the build identity and all flags.

For correctness, restart a fresh single-slot server for each mode below, using the same model, raw prompt, seed and batch sizes. Native `/completion` supports `return_tokens`, so compare the returned **token ID arrays**, not just displayed text. Example server command for static 325 MiB (run by the orchestrator with its built `llama-server`; model/hot-set paths must exist inside its runtime):

```bash
LLAMA_MOE_CACHE_STATIC="$PWD/artifacts/codexO/hot-sets/prose-prefill-325mib.txt" \
  /path/to/built/llama-server \
  -m /home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf \
  -ngl 99 --n-cpu-moe 24 --moe-cache-mib 325 \
  -t 8 -tb 8 -c 8192 -b 512 -ub 512 -np 1 \
  --no-repack --fit off --no-warmup --no-context-shift --host 127.0.0.1 --port 8080
```

After `/health` is ready, submit this request from the campaign root. Run separately for `prose`, `code`, `chat`, setting `name` and `destination` per mode/repetition. No network beyond the orchestrator's local server is needed.

```python
import json
from pathlib import Path
import urllib.request
name = 'prose'
destination = Path('artifacts/codexO/orchestrator/static-325-prose-r0.json')
destination.parent.mkdir(parents=True, exist_ok=True)
prompt = Path(f'artifacts/traces/in/{name}.txt').read_text()
# Match common's file prompt handling used by the tracer.
if prompt.endswith('\n'):
    prompt = prompt[:-1]
body = dict(prompt=prompt, n_predict=256, temperature=0, seed=1,
            repeat_penalty=1.0, ignore_eos=True, cache_prompt=False,
            return_tokens=True, stream=False)
request = urllib.request.Request('http://127.0.0.1:8080/completion',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)
assert len(result['tokens']) == 256
assert result['timings']['predicted_n'] == 256
destination.write_text(json.dumps(result, indent=2) + '\n')
```

Use `--no-repack --fit off` throughout this controlled matrix, with `--no-host` unset. Also measure the layer-placement baseline with its usual production repacking settings as a separately labeled comparison. Check the cache initialization log names all 24 eligible layers. Use this mode matrix, always restarting the server so no earlier KV/cache residency leaks into a cold run:

| Mode | Environment | Placement | Cache MiB |
|---|---|---|---|
| Uncached host-expert control | variable unset | `--n-cpu-moe 24` | 0 |
| LRU | variable unset | `--n-cpu-moe 24` | 163, 244, 325, 487 |
| Static | corresponding hot-set path | `--n-cpu-moe 24` | 163, 244, 325, 487 |
| Layer placement | variable unset | `--n-cpu-moe 24,18,12,6,0` (separate runs) | 0 |

1. Compare token arrays to the uncached control at temperature zero for all three prompts. Repeat the first static configuration with a comments-only hot set (all misses), and with a complete 768-pair set at 649 MiB (all hits; 648.84375 MiB bank requirement). Exercise decode and prefill `-ub 1,32,512` in separate correctness runs. Missing file, duplicate pair, out-of-range ID, ineligible layer and 193-pair set with a 163 MiB budget must fail before evaluation. Cache-off with the variable unset must retain baseline behavior.
2. Exact source weight bytes, expert identity and original graph math are preserved by construction. Token identity is **expected**, especially static versus uncached host-expert GPU computation, but is not a portable numerical guarantee. CPU/GPU kernels, quantized activation formats, placement, fusion, batch size and accumulation can change rounding and greedy decisions. On divergence, compare logits on identical teacher-forced token inputs (not diverged free-running continuations), inspect top-two margins, and check all-hit/all-miss endpoints before any performance claim. Token/logit identity and runtime file rejection have not been tested here.
3. For timing, use the regular server with no routing/debug observer. After each fresh start, run one unscored request then five measured identical requests with `cache_prompt=False`; retain both cold and steady-state results, including LRU warm residency. Alternate static/LRU order over three server restarts. Record `timings.prompt_ms`, `predicted_ms`, token counts, wall time, medians and p95; do not combine prefill with decode. Retain stderr and static-prefill/D2D logs. Static copy counters count projection copies, not logical router hit rate or miss H2D bytes. Use separate transfer profiling to distinguish init H2D, miss/padding H2D and hit D2D.
4. Repeat the matrix with CUDA graphs disabled by `GGML_CUDA_DISABLE_GRAPHS=1`; for the default-graph run, collect graph eligibility/capture evidence in a separate diagnostic run (`GGML_CUDA_GRAPH_DEBUG=1` plus debug logging). Keep diagnostic logging outside timed samples. Recompute routing only for diagnostics; observer synchronization changes performance.
5. Use the same GPU-byte caps for all modes, initially the campaign's measured layer-placement points near 734/894/1056 MiB, under the orchestrator's allocation recorder/shim. Record **peak total allocated bytes** and bank bytes separately, and whether each run fits the cap. Calibrate static/LRU cache MiB downward as needed to fit each cap, regenerate static sets for the final budgets, then freeze them before timing. For a direct equal-used-bytes comparison, tune cache MiB against the measured layer-placement peak to within 1 MiB (or report the remaining mismatch explicitly). Layer placement is discrete; if no match fits, report the memory/latency Pareto points and cap-feasible comparison instead of claiming equal used bytes. Equal nominal cache MiB alone is insufficient because staging differs. Never count an OOM or changed context length as a valid matched run.

Success means a reproducible token/logit correctness result and lower decode latency at matched measured GPU bytes than both controls. Failure is useful evidence: D2D staging plus miss H2D may outweigh static-residency gains. A true CPU-miss/GPU-hit engine would require additional graph/kernel dispatch work beyond this minimal existing-path patch; no harness-derived throughput estimate is claimed here.

## Verification retained for this assignment

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/moe_static_hot/test_make_hot_set.py \
  --basetemp=artifacts/codexO/pytest-tmp -o cache_dir=artifacts/codexO/pytest-cache
```

22 tests passed: frequency/tie selection, TRAIN exclusion and boundaries, global pooling, guard/alignment bytes, insufficient/invalid budgets/layout, malformed trace/duplicate rejection, CLI metadata/hash and deterministic output. The source was archived under `artifacts/codexO/tree`; isolated `git apply --check`, application and reverse check passed. The final patched source compiled and linked as `libllama.so` with GCC 13.3.0, CMake/Ninja, `GGML_CUDA=OFF`, `GGML_NATIVE=OFF`, `-j4`. See `artifacts/codexO/verification.json`, `patch-verification.json`, `configure.log`, `compile-verified-patch.log` and `pytest.log`.

CPU-only compilation checks all added C++ against the real backend API; cache execution itself requires a GPU. CUDA compilation, real backend init/copy scheduling, malformed-file runtime checks, token/logit equivalence, memory peaks, transfer traffic and performance are open orchestrator checks. The reference checkout remained clean. All assignment writes are under `cc/moe_static_hot/` and `artifacts/codexO/`.
