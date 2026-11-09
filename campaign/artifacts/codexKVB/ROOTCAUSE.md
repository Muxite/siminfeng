# T-KVB2: quantized KV abort during scheduler graph splitting

Assignment base: `31288c9c66e59377f7788d920678a565bb89dd37`.
Patch SHA-256: `dd0e4ffd3ffa1e414353f70addf67312816d167ebb1e9518cc75b543b86e8e6c`.

## Mechanism

`llama_context::graph_max_nodes()` budgets `max(1024, 8 * model.n_tensors())` for this architecture. This model receives graph capacity 1936. `sched_reserve()` passes that capacity both to `llm_graph_result` and to `sched_new()`. `llm_graph_result::reset()` allocates a graph with separate node and leaf arrays, each of that capacity, and `ggml_new_graph_custom()` gives its visited-tensor hash table `ggml_hash_size(2 * capacity)` slots.

However, `ggml_backend_sched_new()` allocates its independent hash table with only `ggml_hash_set_new(graph_size)`. The scheduler tracks both computational nodes and leaf tensors in this table. `graph_size` is a capacity for each graph array, not a bound on their sum. For capacity 1936 the old scheduler hash table rounds up to 2053 slots; the corrected table rounds 3872 up to 4099 slots.

The CPU measurements at the first split are:

| KV types | Nodes | Leaves | Total tensors | Old scheduler slots | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| f16 / f16 | 1475 | 300 | 1775 | 2053 | completes |
| q8_0 / q8_0 | 1763 | 302 | 2065 | 2053 | hash full, SIGABRT |
| q4_0 / q4_0 | 1763 | 302 | 2065 | 2053 | hash full, SIGABRT |
| q8_0 / q8_0, rotation disabled | 1475 | 300 | 1775 | 2053 | completes |

Thus the graph fits both of its arrays, but its tensor population exceeds the scheduler hash capacity by 12. The reported abort is in the scheduler's `ggml_hash_find_or_insert`, not the graph builder's visited hash table. The original GPU log also puts `ggml_backend_sched_split_graph` directly above the abort.

During flash-attention capability probing, `resolve_fused_ops()` calls `graph_reserve(..., split_only=true)`, which directly invokes `ggml_backend_sched_split_graph`. This bypasses the node-plus-leaf capacity assertions in the reserve/allocate entry points, so the first symptom is the generic full-table abort at `ggml-impl.h:330`. CPU baseline reproduces that exact file/line.

## Why quantized KV, and why Granite?

For quantized K/V with head dimensions divisible by 64, `llama_kv_cache` enables Hadamard rotations. `llm_graph_context::build_attn()` rotates Q and K before attention, V before caching, and the attention result back afterward. Each `llama_mul_mat_hadamard()` adds three operations: a reshape or contiguous copy, a matrix multiply, and a reshape back. The four calls add 12 nodes per layer: 24 layers add 288 nodes, with two additional rotation input leaves. These exactly explain the measured difference (1775 + 288 + 2 = 2065).

The ordinary K/V `set_rows` storage paths exist with f16 too; a hypothesis that quantized cache writes alone add the excess conversion nodes is not the observed cause here. Quantized-cache context shifting does contain an explicit dequantize/RoPE/requantize path, but this abort occurs during initial reservation before any shift.

This is a generic scheduler capacity defect, not a Granite-specific numerical/kernel bug. Granite's compact MoE weight representation yields relatively many graph operations per model tensor, so its heuristic budget is close to the tensor population. Other models can encounter the same defect when nodes plus leaves cross the rounded scheduler capacity. Context length changes tensor dimensions but does not remove the extra per-layer rotation operations, explaining failure at both 4096 and 32768.

## Patch and f16 behavior

The patch replaces `ggml_hash_set_new(graph_size)` with `ggml_hash_set_new(graph_size * 2)` and replaces the existing FIXME with the invariant. The two hash-indexed metadata allocations already use `sched->hash_set.size`, so they automatically receive the matching capacity. Graph arrays, operations, scheduling decisions, backend selection, and quantization math are unchanged. This is the same sizing rule already used for the graph's hash table.

F16 generated stdout is byte-identical before and after with seed 42. F16 does receive a larger scheduler hash table and associated metadata; there is an allocation-footprint change, but no intended inference behavior change. For this model the hash capacity increases by 2046 slots, adding roughly `(8 + 4 + 8 * n_backends * n_copies)` bytes per slot plus used bits. This does not double KV storage or compute buffers.

## Evidence and checks

- `build-before2.log`, `build-diagnostic.log`, `build-after.log`: CPU-only RelWithDebInfo builds, reused scratch tree/build, `nice -n 10`, `-j4`.
- `before-summary.jsonl` and `before-*.{stdout,stderr,json}`: clean baseline; q8/q4 exit -6, f16 and q8 without rotations exit 0.
- `diagnostic-summary.jsonl` and `diagnostic-*.stderr`: temporary split-entry counters print capacity, node/leaf counts, and scheduler slots. Temporary diagnostics were removed before the final patch build. The diagnostic header change shifts the abort line by one; clean baseline is line 330.
- `after-summary.jsonl` and `after-*.{stdout,stderr,json}`: all six cases exit 0: f16, q8, q4, q8 without rotations, q8 at 32768, and q8 with auto-fit disabled. Runs generate four tokens from `hi`, use CPU threads 4, and seed 42.
- `test-scheduler-capacity.cpp`: model-independent regression with capacity 600, 550 nodes and 551 leaves. Old hash capacity 1031 aborts; fixed hash capacity 2053 permits split and reserve. `before-capacity.log` records -6; `after-capacity.log` records `reserve=ok`. It changes no operator implementation.
- `patch-apply-check.log`: `git apply --check` succeeds against the read-only reference checkout at the pinned commit.
- `verification.json`: automated checks of case exit codes, f16 stdout equality, patch digest, and scratch-source scope.

GDB was attempted but ptrace is prohibited in this sandbox (`before-q8_0-gdb.log`). Runtime counters were used instead. No network, Docker, or CUDA execution was performed. This artifact is AI-assisted private campaign work, not an upstream submission.

## GPU handoff and limits

The exact Build CC command and CUDA verification loop are in `cc/kv_quant_fix/README.md`. `build-cc-j4.py` uses the existing Build CC with a copied recipe whose build concurrency is four. No shared build recipe is edited.

GPU compilation/execution, other models, multi-backend/pipeline configurations, and long-context quality/perplexity are untested. The CPU evidence proves the capacity failure and its repair for this model, not general numerical quality. The patch does not dynamically grow tables for callers that exceed their declared graph capacity or address unrelated graph-node budget exhaustion.
