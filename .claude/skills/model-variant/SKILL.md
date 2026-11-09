---
name: model-variant
description: Use when producing or evaluating a quantized/mixed-precision GGUF variant of the MoE model (imatrix calibration, per-layer expert precision such as PAP, mixE4K-style recipes) and deciding if it is acceptable. Trigger on "quantize", "llama-quantize", "imatrix", "pap12_q3", "make_pap", "tensor-type", "KLD gate", "inspect the GGUF", "is this variant better".
---

# Model variant: quantize/imatrix -> inspect -> KLD gate -> lab

Representation changes are NOT computation-preserving: they are judged on a quality gate (KLD/top-1 vs bf16) and a speed/memory metric, both frozen beforehand
(`frozen-experiment`). Reference implementation: `cc/pap/` (`make_pap.sh`, `README.md`, `verify_pap.py`), decisions D013, D035. Run from `campaign/`.
Models: `~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-{bf16,Q8_0,Q4_K_M,mixE2K,mixE3K,mixE4K}.gguf`; source for every variant is the **bf16** file (never requantize low-bit from low-bit).

## 1. Tools (CPU only; Codex-safe)
Pinned scratch build from `git archive 31288c9c...` of `ref/llama.cpp` (do not build inside the checkout):
`make_pap.sh build` = cmake `-DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMIT=<sha> -DLLAMA_BUILD_NUMBER=0`, targets `llama-quantize llama-imatrix llama-perplexity`, `nice -n 10`, `-j4`.
(Set the commit/number explicitly or the banner shows the campaign repo hash.) Compiling contaminates GPU timing: not during a timing window.

## 2. Imatrix (default for every low-bit variant - D035)
```bash
nice -n 10 llama-imatrix -m <bf16.gguf> -f ~/models/campaign/data/wikitext/wikitext-2-raw/wiki.train.raw -o <out>.imatrix \
   --output-format gguf -t 6 -tb 6 -ngl 0 -c 512 --chunks 100 --no-ppl --output-frequency 10
```
Calibrate on TRAIN (sha 6707892f...), evaluate on TEST (sha 173c87a5...); never calibrate on TEST. The file is GGUF format despite the `.imatrix` suffix.
Existing: `artifacts/codexPAP/models/granite-wtrain.imatrix` (sha 106f834b..., 100x512 chunks, all 32 experts seen). Results: imatrix -33% KLD at 4-bit, -13% at 3-bit, same bytes.

## 3. Quantize with a placement-aware recipe
```bash
# FIRST matching --tensor-type wins: put the specific first-K-layer regex BEFORE the general one. Base type Q8_0, 4 threads.
llama-quantize --imatrix <im> \
  --tensor-type '^blk\.(0|1|...|K-1)\.ffn_.*_exps\.weight$=q3_K' \
  --tensor-type 'ffn_.*_exps=q4_K' \
  <bf16.gguf> <out.gguf> Q8_0 4
```
Regexes use `regex_search`, types are case-insensitive. Norms and router matrices stay F32 by upstream policy. Layers 0..K-1 are the ones `--n-cpu-moe K`
puts on CPU, so K at inference must equal K in the recipe or the placement-matched comparison is invalid. Rules so far (D035): imatrix always;
lower than 4-bit only for CPU-resident layers and only 3-bit (2-bit KLD 0.19-0.36 rejected). Whole pipeline reproducible: `cc/pap/make_pap.sh {build|calibrate|quantize|verify|sanity}`.

## 4. Inspect (verify what was actually produced)
```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 cc/model_inspect/inspect_model.py M.gguf --summary --out artifacts/<x>/M.json
.claude/skills/model-variant/scripts/verify_types.py M.gguf --expect 'expert_ffn@0-11=Q3_K' --expect 'expert_ffn@12-23=Q4_K' --expect attn_qkv=Q8_0 --expect norm=F32
```
Header-only (no payload loads). Confirms per-tensor types, `unknown_tensors == 0`, bytes by role/type, per-layer expert bytes, KV bytes/token. Verified on pap12_q3
(Q3_K 247.5 MiB, Q4_K 324.0 MiB, Q8_0 127.5 MiB). **MB vs MiB**: weight bytes / 2^20 = MiB; D035 quotes decimal MB for file size (738 MB = 703.9 MiB); the file also holds metadata,
so file size > weight bytes. Say which unit, always. File size is not resident RAM or VRAM (weights are mmap page cache; PAP only changes CPU-layer bytes, GPU MiB is unchanged).

## 5. Quality gate and speed (lab)
Tests (copy from `lab/tests/pap1.json`): `pap_kl_<name>`: `llama-perplexity -c 512 --chunks 20 --kl-divergence-base /out/bf16.kl --kl-divergence`, `needs_data`, mount `artifacts/kl:/out:ro`
(reference logits `artifacts/kl/bf16.kl`, deterministic, one run per model). Decode: `pap_decode_<name>_n<K>` with `--n-cpu-moe K`, controls at the SAME K (mixE4K, mix4_imat).
Submit with `--model <any model in same dir> --data wiki.test.raw` (per-test `model_file` selects the variant); procedure in `frozen-experiment`.
Reference numbers (wikitext 20x512): mixE4K KLD 0.040 / top-1 90.5; mix4_imat 0.027 / 91.9; pap12_q3 0.073 / 87.3; Q4_K_M 0.078 / 85.6. Frozen D032 rule: decode >= +5% vs mixE4K at the
same K AND KLD <= 0.08. KLD is deterministic (solid); speed only counts from a quiet host (D035: pap12_q3 +9.8% but PROVISIONAL, a download ran).
Claim category: representation trade-off (never "computation-preserving"). Wikitext KLD does not establish task quality.

## Done when
- [ ] variant built from bf16 with imatrix (provenance: imatrix sha, TRAIN sha, tool commit) and recipe regexes in the README/manifest
- [ ] `verify_types.py` expectations pass; sizes stated in a named unit
- [ ] KLD/top-1 vs bf16.kl measured for variant AND controls; gate threshold from the frozen note applied
- [ ] speed from a clean-host alternating job, or flagged PROVISIONAL; decision note written (`decision-note`)
