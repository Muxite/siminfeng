#!/usr/bin/env bash
# OPERATOR ONLY. Queue class: build. Expected: 15-40 min first CUDA build, <2 min incremental.
# Outputs: build-operator/bin/expert-crossover, build-operator/probe_uma, build-operator.log, build-manifest.json.
set -euo pipefail
source "$(dirname "$0")/operator_common.sh"
require_job
check_source
command -v cmake >/dev/null
NVCC=${F19_NVCC:-/usr/local/cuda/bin/nvcc}
[[ -x "$NVCC" ]] || { echo "NVCC_MISSING: $NVCC" >&2; exit 2; }
BUILD=$F19_ROOT/build-operator
mkdir -p "$BUILD"
exec > >(tee "$F19_ROOT/build-operator.log") 2>&1
"$NVCC" --version
cmake -S "$F19_ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_SOURCE="$F19_LLAMA" -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_CUDA_GRAPHS=ON -DBUILD_SHARED_LIBS=OFF
cmake --build "$BUILD" --target expert-crossover -j2
"$NVCC" -O3 -std=c++17 -arch=sm_87 --cudart=static "$F19_ROOT/probe_uma.cu" -o "$BUILD/probe_uma"
python3 - "$F19_ROOT" "$F19_PIN" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
files = ['build-operator/bin/expert-crossover', 'build-operator/probe_uma', 'crossover.cpp', 'probe_uma.cu']
record = {'pin': sys.argv[2], 'execution': 'not run', 'sha256': {f: hashlib.sha256((p/f).read_bytes()).hexdigest() for f in files}}
(p/'build-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
PY
echo BUILD_OK
