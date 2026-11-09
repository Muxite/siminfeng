#!/usr/bin/env bash
# Host-only evidence; does not validate CUDA ABI or GPU execution.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$here/../../.." && pwd)"
out="$here/out"
mkdir -p "$out"
export TMPDIR="$out/.tmp"
mkdir -p "$TMPDIR"
gcc -std=c99 -O2 -g -fPIC -shared -Wall -Wextra -Werror \
  -I"$here" "$root/cc/gpu_budget/gpubudget.c" -ldl -pthread -Wl,-z,defs -o "$out/libgpubudget.so"
gcc -std=c99 -fPIC -shared -Wall -Wextra -Werror -I"$here" "$here/fake_cuda.c" -o "$out/libfakecuda.so"
g++ -x c++ -std=c++14 -Wall -Wextra -Werror -I"$here" \
  "$root/cc/gpu_budget/test_budget.cu" -L"$out" -lfakecuda -Wl,-rpath,"$out" -o "$out/test_budget"
gcc -std=c99 -Wall -Wextra -Werror -I"$here" "$here/host_checks.c" \
  -L"$out" -lfakecuda -pthread -Wl,-rpath,"$out" -o "$out/host_checks"
GPU_BUDGET_LOG="$out/table.jsonl" bash "$root/cc/gpu_budget/run_test.sh" "$out"
export LD_PRELOAD="$out/libgpubudget.so" GPU_BUDGET_MIB=4 GPU_BUDGET_RESERVE_MIB=0 GPU_BUDGET_COUNT_MANAGED=1
export GPU_BUDGET_LOG="$out/host.jsonl"
for check in threads failures syncfail; do "$out/host_checks" "$check"; done
GPU_BUDGET_RESERVE_MIB=1 "$out/host_checks" reserve
GPU_BUDGET_COUNT_MANAGED=0 "$out/host_checks" managed0
GPU_BUDGET_MIB=0 "$out/host_checks" zero
GPU_BUDGET_RESERVE_MIB=5 "$out/host_checks" zero
env -u GPU_BUDGET_MIB "$out/host_checks" disabled
for bad in -1 +1 1.5 18446744073709551615 ''; do
  set +e
  GPU_BUDGET_MIB="$bad" "$out/host_checks" zero
  status=$?
  set -e
  test "$status" -eq 125
done
unset LD_PRELOAD GPU_BUDGET_LOG
python3 - "$out/table.jsonl" "$out/host.jsonl" <<'PY'
import json
import sys
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    assert rows and any(r['event'] == 'deny' for r in rows)
    assert any(r['event'] == 'exit' for r in rows)
    for r in rows:
        assert r['used'] <= r['peak']
        if r['budget']:
            assert r['used'] <= r['capacity']
    print(f'PASS JSONL parse and accounting invariants: {path} ({len(rows)} events)')
PY
