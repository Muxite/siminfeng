#!/usr/bin/env bash
set -euo pipefail
cd /home/muk/projects/siminfeng/campaign
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/codexPL4/tree
git -C ref/llama.cpp archive 31288c9c6 | tar -x -C artifacts/codexPL4/tree
nice -n 10 cmake -S artifacts/codexPL4/tree -B artifacts/codexPL4/tree/build \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_COMMIT=31288c9c6 \
  > artifacts/codexPL4/configure.log 2>&1
nice -n 10 cmake --build artifacts/codexPL4/tree/build --target llama-bench -j4 \
  > artifacts/codexPL4/build.log 2>&1
nice -n 10 python3 cc/moe_planner/cpu_type_screen.py > artifacts/codexPL4/cpu-screen.log 2>&1
nice -n 10 python3 cc/moe_planner/fit_decode.py > artifacts/codexPL4/fit-summary.json
nice -n 10 python3 cc/moe_planner/validate_decode.py > artifacts/codexPL4/scenario-summary.json
nice -n 10 python3 -m pytest -q cc/moe_planner \
  --basetemp=artifacts/codexPL4/pytest-tmp -o cache_dir=artifacts/codexPL4/pytest-cache \
  > artifacts/codexPL4/pytest.txt 2>&1
