# CC kernel screen

Tool CC screens per-op, per-weight-type CUDA performance before spending a full
model build/A/B cycle on a dispatch rule. Python 3.10+, stdlib only; pytest for
tests. Accepts a host `test-backend-ops` executable or an image containing
`/app/test-backend-ops`. It does not build or fetch anything.

## Prepare the binary (operator action)

Inspected source: `ref/llama.cpp` at `31288c9c66e59377f7788d920678a565bb89dd37`.
`tests/CMakeLists.txt:316` registers `llama_build(test-backend-ops.cpp LINK ggml)`:
the executable target is **test-backend-ops**.

The unmodified revision has **neither requested model's perf shapes** and registers
`MUL_MAT_VEC_FUSION` only in `make_test_cases_eval`, not `make_test_cases_perf`.
`-p` selects registered cases; it cannot create arbitrary dimensions. The supplied
`preset-perf-cases.patch` adds both presets to the perf registry, including gate/up
SWIGLU graphs, isolated dense/expert matmuls, and F16/F32 SWIGLU. The patch stays
under this tool directory; the source checkout is unchanged. Build CC applies it
to an isolated snapshot. Operators may combine it with their dispatch patch before
passing `--patch` (Build CC accepts one patch file).

```bash
# Operator only: build/package on the intended host. No build was run for J-T6.
DOCKER_CMD_PREFIX='sg docker -c' python3 cc/build_engine/build_engine.py \
  --target jetson-orin --source ref/llama.cpp --commit 31288c9c6 \
  --with-tests --extra-target test-backend-ops \
  --patch cc/kernel_screen/preset-perf-cases.patch \
  --tag campaign/llamacpp:kernel-screen --out cc/build_engine/results/kernel-screen
```

`--with-tests` sets `-DLLAMA_BUILD_TESTS=ON` for either Build CC target;
`--extra-target` selects and ships the executable. Both are recorded, alongside
CMake flags and executable hashes. x86 modifies only its staged recipe, recording
the staged recipe hashes. The original shared recipe is unchanged.

## Inputs and invocation

```bash
DOCKER_CMD_PREFIX='sg docker -c' python3 cc/kernel_screen/kernel_screen.py \
  --image campaign/llamacpp:kernel-screen \
  --preset cc/kernel_screen/presets/granite.json \
  --types Q4_K,Q5_K,Q6_K --ops MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION \
  --arm 'A={"GGML_CUDA_J_UNFUSE_IDS_TYPES":null}' \
  --arm 'A2={"GGML_CUDA_J_UNFUSE_IDS_TYPES":null}' \
  --arm 'B={"GGML_CUDA_J_UNFUSE_IDS_TYPES":"q4_k,q5_k"}' \
  --repeats 3 --out cc/kernel_screen/results/granite
```

The environment variable's accepted value syntax belongs to the binary's dispatch
patch; use its documented spelling. An unrecognized variable may have no effect.
Default arms are A={}, A2={}, B={GGML_CUDA_DISABLE_FUSION:"1"}. A2 is an explicit
interleaved A/A control. All names mentioned by any arm, plus the two CUDA knobs
above, are removed before each run; null means unset, not empty string. Other host
environment variables are inherited for host mode. Docker starts with the image's
environment, and `/usr/bin/env -u` removes managed variables even if baked into it.
Use digest-pinned images and retain Build CC records for reproducibility.

Docker uses `docker run --rm --runtime nvidia --network=none`, NVIDIA visibility
and compute/utility capabilities, and `/app/test-backend-ops perf -o OP -b CUDA0
-p REGEX`. Image must have `/usr/bin/env`. `DOCKER_CMD_PREFIX` follows Build CC's
launcher convention: `sg docker -c` receives one shell-quoted command. Host mode
uses `--binary /absolute/path/test-backend-ops`; set library paths as needed.
`--backend` defaults to CUDA0 and can select another device for host screening.

Use `presets/qwen36-35b-a3b.json` for Qwen: embedding 2048, expert FFN **512**,
256 experts, 8 selected, verified from `experiments/jetson/OM/qwen36-layout.json`
metadata and tensor shapes. Granite uses 1024/512/32/8. Both include tokens 1 and
8 and up/down directions. Alternatively supply all five comma-separated options
`--n-embd`, `--n-ff`, `--n-expert`, `--n-used`, `--n-tokens`; their Cartesian
product is screened. `--direction up,down` is the default.

Matmul mapping: up has k=embedding, width=FFN; down reverses them. MUL_MAT and
MUL_MAT_ID use m=width,n=tokens; MUL_MAT_VEC_FUSION uses m=tokens,n=width,
use_id=1,with_gate=1. Expert dimensions map to n_mats/n_used. Dense ops have no
expert dimension; all matching layout/broadcast variants are reported separately.
`--filter` ANDs an additional params regex with the shape/type filter. Filters use
ECMAScript-compatible lookaheads and run before graph allocation.

Supported selectors include `MUL_MAT_ID_FUSION` and SWIGLU/GEGLU/REGLU/GLU.
For standalone SWIGLU, use a separate run with `--ops SWIGLU --types F16,F32`:
these types describe activations, not quantized weights. SWIGLU selects contiguous
packed operands `[2*width,tokens,1,1]`. The optional patch registers SWIGLU only.
`MUL_MAT_ID_FUSION` constructs expert matmuls followed optionally by ADD/MUL;
o=1,mul=0 is just one matmul despite its name. Inspect params before attributing
a change to fusion. `MUL_MAT_VEC_FUSION` constructs actual gate/up matmuls,
SWIGLU (or other registered GLU), optional biases/scaling, and final scaling.
Down-direction gate graphs are synthetic probes, not the model's down projection.

## Outputs and interpretation

Output must be new/empty. `results.json` retains requested shapes/types/ops,
arms, commands, repeats, raw per-case samples (us/run, normalized GFLOPS or GB/s),
unsupported rows, errors and medians. `00000.log`, etc. retain merged subprocess
stdout/stderr. `results.md` shows per-case median times, within-arm spread and
A/arm time ratios: >1 means that arm is faster. A/A spread is reported both as
A and A2 max/min spread and their median ratio. No significance threshold is
implied. Each process has a configurable `--timeout` (default 120 seconds).

Repeats interleave A,A2,B then B,A2,A for each shape/op/type, using a fresh process
each time because CUDA environment knobs may be cached at startup. Minimum two
repeats. Each backend case itself runs for at least one second after warmup.
Missing cases, unsupported-only results, partial arm coverage, duplicate cases,
timeouts and nonzero subprocess exits make the screen fail; missing coverage is
never a speedup. Ratios are suppressed for incomplete repeat coverage. Results
are saved after every invocation, so failures leave evidence. No automatic reruns.

The console printer emits either FLOPS (k/M/G/T normalized to GFLOPS here) or
bandwidth, not both. Bandwidth labeled GB/s is calculated using 1024^3 in source.
Fixture text is crafted from `print_perf_console` printf statements and
`VARS_TO_STR` serializers; it is not a GPU measurement.

## What this screen covers

`test_case::eval_perf` builds a forward graph and calls
`ggml_backend_graph_compute` for warmup and timing: **backend graph-level fusion
can run** when a registered case constructs the corresponding pattern. Perf does
not universally disable graph fusion. However, it appends repeated references
to the output node, rather than independent copies of the whole graph. For
multi-node fusion cases, upstream-node work and fusion eligibility can differ
from full-graph inference; reported us/run is normalized by those output repeats,
not necessarily the cost of independent complete MoE blocks. Treat gate/up perf
ratios as leads requiring full-graph validation.

Isolated MUL_MAT_ID cannot establish the effect of the D027 gate/up fusion rule.
This screen does not cover model graph construction/scheduling, full-model fusion
decisions, shared-expert interactions, routing distributions, weight residency,
KV cache, transfer/offload, end-to-end latency/throughput or accuracy. Random IDs
and synthetic tensors are used. Env arms do not prove that the requested dispatch
branch was taken. Platform dispatch rules still require model-level A/B tests.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  cc/kernel_screen/test_kernel_screen.py cc/build_engine/test_build_engine.py \
  -p no:cacheprovider
git -C ref/llama.cpp apply --check ../../cc/kernel_screen/preset-perf-cases.patch
```

All subprocess execution in runner tests is mocked. No builds, Docker or GPU
commands are executed by this validation.
