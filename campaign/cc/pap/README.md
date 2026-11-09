# Placement-aware expert precision (T-PAP)

This recipe quantizes only the experts assigned to CPU at lower precision, while keeping the other experts at Q4_K and ordinary quantizable weights at Q8_0. It is a model-production technique; it needs no inference-engine changes. Norm vectors and expert router matrices remain F32 under upstream quantization policy (`src/llama-quant.cpp:295-314`).

The scratch source is a `git archive` of `ref/llama.cpp` commit `31288c9c66e59377f7788d920678a565bb89dd37`, under `artifacts/codexPAP/tree`. The build is CPU-only, uses `nice -n 10`, and limits compilation and quantization to four threads. No Docker or network is used by the recipe. The evaluation JSON is an orchestrator handoff and does not execute Docker here.

The initial calibration build's automatically generated banner used the parent campaign Git hash `bc22a4f`; its source was still the pinned archive. The recipe explicitly sets `LLAMA_BUILD_COMMIT` and archive build number 0 to prevent this cosmetic provenance error. Initial configure/build logs and the calibration executable hash are retained; the tools are rebuilt with the explicit stamp before model production.

## Integration conditions

Use each `papK_*` with `-ngl 99 --n-cpu-moe K`. In the archived source, `common/arg.cpp:2771-2777` documents the first N layers and calls `llm_add_n_cpu_ffn_overrides`; `common/common.h:1255-1261` actually loops over indices `0 <= i < N`, creating CPU buffer overrides. These citations refer to the pinned scratch source, not an arbitrary later upstream version. GPU-capable evaluation builds must preserve those semantics. Changing K at inference changes which device gets the low-bit weights and invalidates a placement-matched comparison. A CPU-only run cannot measure the intended CPU/GPU division.

The recipe places the specific first-K-layer regex before `ffn_.*_exps=q4_K`. Repeated `--tensor-type REGEX=TYPE` arguments are parsed at `tools/quantize/quantize.cpp:317-347`; `src/llama-quant.cpp:774-785` uses `std::regex_search` and breaks on the first match. Types are case-insensitive. Every output is quantized directly from the read-only BF16 source, with Q8_0 as the base type. No low-bit output is requantized from another low-bit model.

| Output basename | Layers 0 through K-1 | Layers K through 23 | Imatrix |
| --- | --- | --- | --- |
| pap12_q3.gguf | 0-11 Q3_K | 12-23 Q4_K | TRAIN |
| pap12_q2.gguf | 0-11 Q2_K | 12-23 Q4_K | TRAIN |
| pap18_q3.gguf | 0-17 Q3_K | 18-23 Q4_K | TRAIN |
| pap18_q2.gguf | 0-17 Q2_K | 18-23 Q4_K | TRAIN |
| pap24_q3.gguf | 0-23 Q3_K | none | TRAIN |
| pap24_q2.gguf | 0-23 Q2_K | none | TRAIN |
| pap24_q3_noimat.gguf | 0-23 Q3_K | none | none |
| mix4_imat.gguf | all Q4_K | all Q4_K | TRAIN |

`pap24_q3_noimat` is the mixE3K-equivalent recipe control. `mix4_imat` isolates the effect of calibration at four bits and is evaluated at all three K values. The existing `granite-3.1-1b-a400m-instruct-mixE4K.gguf` is an external prerequisite, evaluated at K=12,18,24. It is not copied into the output directory; the orchestrator must include that basename in the common model directory.

## Reproduction and provenance

Run `cc/pap/make_pap.sh` from any directory. It builds, calibrates, quantizes eight models, verifies actual GGUF tensors using the pinned gguf-py reader, hashes the files, and emits the manifest and lab tests. Individual stages are `build`, `calibrate`, `quantize`, and `verify`. The optional `sanity` stage runs only the CPU pap12_q3 KLD check; run `verify` afterward to include its result in the manifest. Re-running a stage overwrites that stage's outputs. Python requires the already installed NumPy; CMake and the local C/C++ toolchain are used without dependency downloads. Python bytecode writes outside the assignment roots are disabled.

Calibration command (paths expanded in the script):

```sh
nice -n 10 artifacts/codexPAP/build/bin/llama-imatrix \
  -m ~/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf \
  -f ~/models/campaign/data/wikitext/wikitext-2-raw/wiki.train.raw \
  -o artifacts/codexPAP/models/granite-wtrain.imatrix \
  --output-format gguf -t 6 -tb 6 -ngl 0 -c 512 --chunks 100 --no-ppl \
  --output-frequency 10
```

The imatrix uses only the first 100 processed TRAIN chunks; TEST is not used for calibration. It is GGUF format despite the requested `.imatrix` suffix. `MANIFEST.json` records the imatrix metadata, completed chunks, TRAIN and TEST SHA256 hashes, source and output hashes, sizes, and each tensor's actual type and dimensions. `logs/imatrix.log` records the run; calibration start/end files record UTC timestamps. The train/test distinction is dataset provenance, not a claim of zero overlapping natural-language strings.

## Evaluation handoff

`artifacts/codexPAP/lab-tests-pap.json` contains 22 tests using exactly the existing lab entry shapes. All `model_file` references are basenames.

The orchestrator must register this JSON in the lab's test-definition directory, place the eight outputs and existing mixE4K baseline in one model directory, and supply `--data ~/models/campaign/data/wikitext/wikitext-2-raw/wiki.test.raw` when submitting KLD tests. Registration and lab execution are outside this assignment's write locations. Existing absolute read-only prompt/KL mounts match the supplied lab templates.

- Six placement-matched decode runs for calibrated PAP outputs; one for `pap24_q3_noimat` at K=24; three for `mix4_imat`; three for mixE4K. Decode copies `lab/tests/match1.json`: `llama-completion -f /tin/chat.txt -n 160 -c 4096 --temp 0 -s 1 -ngl 99 -no-cnv --no-warmup --n-cpu-moe K`, with the read-only `artifacts/traces/in:/tin:ro` mount.
- Nine KLD runs, covering every produced model and mixE4K. Each copies `lab/tests/mem2.json`'s `kl_q8`: `llama-perplexity -c 512 --chunks 20 -ngl 99 --kl-divergence-base /out/bf16.kl --kl-divergence`, with `needs_data: true`, parser `kl`, and read-only `artifacts/kl:/out:ro` mount. The lab supplies WikiText-2 TEST through its standard data handling. Saved KL data also carries its evaluation tokens.

The full GPU lab suite is defined here for orchestrator execution. Any optional CPU sanity outcome is recorded separately in `artifacts/codexPAP/cpu-sanity.json` and in the manifest; it is not a substitute for the suite. The BF16 KL header is checked against the pinned reader's format (`tools/perplexity/perplexity.cpp:1688-1714`).

## Claims allowed

Verified type placement, byte reductions, calibration provenance, and content hashes are factory properties. The hypothesis is that smaller CPU expert weights reduce DRAM traffic during decode; speedup need not scale with file size because kernels, routing, caching, and GPU work also contribute. File size does not establish resident host RAM or VRAM. Calibration is intended to reduce low-bit quality loss; that benefit must be measured against the controls. Do not claim throughput improvement, memory residency savings, acceptable quality for all variants, or an imatrix quality improvement without completed corresponding lab results. A WikiText CPU sanity KLD, if present, supports only that model and dataset, not task-level quality.

## Delivered measurements

All eight GGUFs passed the gguf-py tensor verification. Sizes below include GGUF metadata; complete SHA256 hashes and all 242 tensor records per model are in `artifacts/codexPAP/MANIFEST.json`.

| File | Bytes | MiB |
| --- | ---: | ---: |
| pap12_q3.gguf | 738,044,192 | 703.854 |
| pap12_q2.gguf | 676,702,496 | 645.354 |
| pap18_q3.gguf | 697,936,160 | 665.604 |
| pap18_q2.gguf | 605,923,616 | 577.854 |
| pap24_q3.gguf | 657,828,128 | 627.354 |
| pap24_q2.gguf | 535,144,736 | 510.354 |
| pap24_q3_noimat.gguf | 657,827,808 | 627.353 |
| mix4_imat.gguf | 818,260,256 | 780.354 |
| granite-wtrain.imatrix | 16,787,104 | 16.009 |

Imatrix SHA256: `106f834b2452b33a0a4eed7d69270a60c5b2330879f2c292a5331bc8450f77a9`. Calibration ran from 2026-11-09T23:08:34Z to 2026-11-09T23:14:28Z, with 100 chunks of 512 tokens and all 32 experts observed for every expert tensor. TRAIN SHA256: `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`. TEST SHA256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.

The optional CPU pap12_q3 check completed successfully against the existing BF16 KL reference (20 chunks, context 512, six inference threads). Mean KLD was **0.072459 +/- 0.002828**, PPL(Q) **10.747627**, PPL(base) **10.006522**, and PPL ratio **1.074062**. Same-top-token agreement in the raw log was **86.961 +/- 0.472%**. This is a finite-statistics sanity result, with no quality acceptance threshold. It does not establish a calibration improvement because the corresponding controls have not been evaluated here. The exact command and timestamps are in `cpu-sanity.json`; the raw output is in `logs/cpu-sanity.log`.

Remaining work for the orchestrator: register and execute the 22 GPU lab tests, provide the existing mixE4K baseline beside the outputs, and compare runtime throughput, memory, and quality before making performance or quality claims. No GPU decode or other model KLD result is claimed. Production and verification finished before the 50-minute checkpoint.
