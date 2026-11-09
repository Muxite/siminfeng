# Quantized KV scheduler capacity fix

Base: llama.cpp `31288c9c66e59377f7788d920678a565bb89dd37`.
Patch: `cc/kv_quant_fix/kv-quant-fix.patch`.
Mechanism and CPU results: `artifacts/codexKVB/ROOTCAUSE.md`.

The patch sizes the scheduler's tensor hash table for both nodes and leaves, matching `ggml_new_graph_custom`. It changes allocation capacity only; the compute graph and KV quantization are unchanged.

## GPU handoff (orchestrator only)

Run from the campaign root with the campaign's locally available CUDA base images. Build CC resolves and records their digests; `BASE_DEVEL` and `BASE_RUNTIME` can select existing digest-pinned images. No GPU build or Docker command was executed during this assignment.

```bash
nice -n 10 python3 cc/kv_quant_fix/build-cc-j4.py \
  --source ref/llama.cpp \
  --commit 31288c9c66e59377f7788d920678a565bb89dd37 \
  --patch cc/kv_quant_fix/kv-quant-fix.patch \
  --tag campaign/llamacpp:c2-kv-quant-fix \
  --out artifacts/codexKVB/build-gpu-fixed
```

The wrapper invokes the existing `cc/build_engine/build_engine.py` with a private recipe copy in `artifacts/codexKVB/gpu-recipes-j4`. Its only recipe change is `--parallel 10` to `--parallel 4`. Build CC records the copied recipe's actual hash. It does not alter the shared Build CC files. Wrapper argument parsing was checked with `--help`.

Verify all six cache-type/context combinations with the resulting CUDA image:

```bash
for kv_type in f16 q8_0 q4_0; do
  for context in 4096 32768; do
    nice -n 10 docker run --rm --network none --gpus all \
      -v "$HOME/models/campaign/gguf:/models:ro" \
      --entrypoint /app/llama-completion campaign/llamacpp:c2-kv-quant-fix \
      -m /models/granite-3.1-1b-a400m-instruct-mixE4K.gguf \
      -p hi -n 4 -c "$context" -fa on -ctk "$kv_type" -ctv "$kv_type" \
      -ngl 99 -t 4 --seed 42 \
      > "artifacts/codexKVB/gpu-${kv_type}-${context}.stdout" \
      2> "artifacts/codexKVB/gpu-${kv_type}-${context}.stderr" || exit 1
  done
done
```

Require exit status zero, generated tokens, actual CUDA layer offload, and flash-attention enabled in logs. Compare f16 output against an unpatched build using the same seed and toolchain. CUDA kernels, output quality, and multi-GPU pipeline behavior remain GPU verification work.

## CPU replay

```bash
nice -n 10 cmake --build artifacts/codexKVB/build --target llama-completion -j4
nice -n 10 python3 artifacts/codexKVB/run_cpu.py after
nice -n 10 artifacts/codexKVB/test-scheduler-capacity
```

`run_cpu.py` records exact arguments, environment overrides, return codes, stdout, and stderr per case. Baseline logs were captured before applying the patch; the current scratch source and binary contain the clean patch with diagnostics removed.
