#!/usr/bin/env bash
set -euo pipefail
out_dir="${1:-/out}"
out_dir="$(cd -- "$out_dir" && pwd)"
export GPU_BUDGET_MIB=250 GPU_BUDGET_RESERVE_MIB=0 GPU_BUDGET_COUNT_MANAGED=1
export GPU_BUDGET_LOG="${GPU_BUDGET_LOG:-$out_dir/test-budget.jsonl}"
export LD_PRELOAD="$out_dir/libgpubudget.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$out_dir/test_budget"
