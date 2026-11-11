#!/usr/bin/env bash
# Build hybrid_moe_bench inside campaign/cuda-tools:c1. Sources are mounted READ-ONLY; build output goes to a separate dir.
# Usage: build.sh [target]   (default target: hybrid_moe_bench)
set -euo pipefail
CAMPAIGN=/home/muk/projects/siminfeng
SRC=$CAMPAIGN/cc/hybrid_expert
LLAMA=$CAMPAIGN/ref/llama.cpp
BUILD=${BUILD_DIR:-$CAMPAIGN/artifacts/opus-hybrid/build}
TARGET=${1:-hybrid_moe_bench}
mkdir -p "$BUILD"
docker run --rm --label campaign=c1 --cpus 8 --memory 16g --pids-limit 512 --network none \
  --user "$(id -u):$(id -g)" \
  -v "$SRC":/src:ro -v "$LLAMA":/llama:ro -v "$BUILD":/build \
  campaign/cuda-tools:c1 bash -c "
    set -e
    [ -f /build/build.ninja ] || cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/llama
    cmake --build /build --target $TARGET --parallel 8"
