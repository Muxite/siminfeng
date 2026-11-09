#!/usr/bin/env bash
# Run INSIDE campaign/cuda-tools:c1. Source can be mounted read-only.
set -euo pipefail
src_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out_dir="${1:-/out}"
cuda_dir="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p -- "$out_dir"
export TMPDIR="$out_dir/.tmp"
mkdir -p -- "$TMPDIR"
"${CC:-gcc}" -std=c99 -O2 -g -fPIC -shared -Wall -Wextra -Werror \
  -I"$cuda_dir/include" "$src_dir/gpubudget.c" \
  -Wl,-z,defs -Wl,-soname,libgpubudget.so -ldl -pthread \
  -o "$out_dir/libgpubudget.so"
"$cuda_dir/bin/nvcc" -std=c++14 -O2 --cudart shared \
  "$src_dir/test_budget.cu" -L"$cuda_dir/lib64/stubs" -lcuda \
  -Xlinker -rpath -Xlinker "$cuda_dir/lib64" -o "$out_dir/test_budget"
printf 'Built %s/libgpubudget.so and %s/test_budget\n' "$out_dir" "$out_dir"
