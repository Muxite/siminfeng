---
name: engine-candidate
description: Use when a llama.cpp engine change (patch) must be built into a Build CC image, shown to engage, checked for correctness, and only then timed on the GPU. Trigger on "build the patch", "Build CC", "build_engine.py", "run_ref", "engagement check", "test-backend-ops", "text identity", "new engine candidate", "kvfix/q8reuse/hsplit style build".
---

# Engine candidate pipeline: patch -> Build CC -> engagement -> correctness -> timed

Pinned base: llama.cpp `31288c9c66e59377f7788d920678a565bb89dd37` in `ref/llama.cpp` (clean checkout, read-only). Run from `campaign/`.
Order is strict; a candidate that fails a step is recorded and stops (D024: "compiles + default path unchanged" does NOT show the new path engages).

## 1. Patch hygiene
```bash
git -C ref/llama.cpp status --short            # must be clean; HEAD must resolve to 31288c9c6
git -C ref/llama.cpp apply --check $PWD/cc/<name>/<name>.patch   # ABSOLUTE path (relative fails: -C changes cwd)
sha256sum cc/<name>/<name>.patch               # record in the decision note and build record
```
Patches live in `cc/<name>/` with a README (mechanism, switch/env, GPU handoff). Prefer a default-OFF env switch so one binary gives OFF/ON arms (D026).

## 2. Build CC (needs Docker; Codex cannot do this)
```bash
export CAMPAIGN_ID=c1
export BASE_RUNTIME=$(python3 -c "import json;print(json.load(open('builds/results/b0-baseline-5/build-record.json'))['base_images']['runtime']['reference'])")
# BASE_DEVEL defaults to the recorded devel digest. Floating tags are never used.
nohup bin/run_op --agent claude-code --task X4 --desc "build <name>" -- python3 cc/build_engine/build_engine.py \
  --source ref/llama.cpp --commit 31288c9c6 --patch cc/<name>/<name>.patch \
  --extra-target test-backend-ops [--extra-target test-<custom>] [--cmake-define GGML_CUDA_NO_VMM=ON] [--linkage] \
  --tag campaign/llamacpp:<name> --out builds/results/<name> > artifacts/x2/<name>-build.out 2>&1 &
```
- Use a NEW `--out` each attempt (existing evidence is never overwritten); also build an unpatched control from the same commit/bases
  (`q8base-N` vs `q8reuse-N`) if the baseline image predates the toolchain. Any `test-*` extra target auto-enables `LLAMA_BUILD_TESTS=ON`.
- Compiles are long and loud: NO timing, no other builds, no Codex compiles meanwhile. `kv_quant_fix/build-cc-j4.py` = same CLI with `--parallel 4` for a shared host.
- Accept the build only if `build-record.json` has `status: success` (`failure_stage` null) - a record with a later failing stage (e.g. `linkage` on shot-2,
  `inspect_image` on novmm-1) is NOT a release even if the image exists (D023). `patch_sha256` must equal your sha256.
- Register the image by **`image.run_ref`**, NOT `config_digest` (containerd store: `docker run` rejects the bare config digest, exit 125; D016, MANAGER-DESIGN c2 §5):
```bash
RR=$(python3 -c "import json;print(json.load(open('builds/results/<name>/build-record.json'))['image']['run_ref'])")
bin/lab image add llamacpp-<name> $RR          # immutable name->id; keep config_digest in the decision note as the identity key
```
Old record `b0-baseline-5` has no `run_ref` (pre-run_ref schema); `llamacpp-b0` is already registered - do not rebuild it.

## 3. Engagement check (untimed, first GPU test)
Show the new code path RUNS, using a counter/log or the greedy-text fingerprint. Submit a single `gen` test (copy `c1_q8_gen_*` from `lab/tests/c1q8.json`) with a
fresh key, `bin/lab run`, then `scripts/text_fp.py JOB --filter gen`. Fingerprints known (Granite, chat prompt, -n 160, temp 0, seed 1): all-GPU mixE4K `950b46b1`,
Q8_0 `65894776`, CPU-computed experts `76dc9422`. Expert placement changes numerics, so the fingerprint tells which backend computed what.
If the candidate should change behaviour and the fingerprint/speed equals the baseline -> NOT engaged -> reject (D024), do not time it.

## 4. Correctness gates (before any timing)
- `test-backend-ops` on CUDA for the touched op (e.g. `MUL_MAT`: 2445/2445 OK in D029), via `docker run --rm --gpus all --entrypoint /app/test-backend-ops <run_ref> test -b CUDA0 -o <OP>` (candidate custom tests the same way, e.g. `test-hybrid-split --backend CUDA0`).
- Computation-preserving claims: greedy text sha identical OFF/ON/base (`text_fp.py`) and PPL identical to printed precision.
- Representation/numerics-changing claims: KLD/top-1 vs `artifacts/kl/bf16.kl` (`llama-perplexity -c 512 --chunks 20 --kl-divergence-base /out/bf16.kl --kl-divergence`, wikitext TEST, `--data`), PPL ratio; thresholds frozen in the protocol.
- Patch must be inert when its switch is OFF (compare unpatched build vs patched-OFF text).
- Failing correctness stops the pipeline; open it as a defect (delegate via `codex-assignment`, reword in correctness terms).

## 5. Timed comparison
Only after 2-4 pass. Same binary for OFF/ON when possible; one GPU job at a time; follow `frozen-experiment` (protocol, alternating order, host monitor, agg).
Record `run_ref`, config digest, patch sha256, build-record path, job id in the decision note (`decision-note`).

## Done when
- [ ] `apply --check` clean, patch sha256 recorded; build-record `status=success`, patch sha matches
- [ ] image registered from `run_ref`; control build/baseline identified
- [ ] engagement shown by fingerprint/counter (not just "default path unchanged")
- [ ] correctness gates passed (or candidate recorded as failed at the first failing gate)
- [ ] timed run done under `frozen-experiment`; claim category assigned
