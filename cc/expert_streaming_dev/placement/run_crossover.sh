#!/usr/bin/env bash
# OPERATOR ONLY. Queue class: gpu. Expected: 2-8 min; timeout: 900 s.
# Outputs: 04-crossover/curve.csv (18 rows), stderr.txt, container-command.txt, build-manifest.json.
set -euo pipefail
source "$(dirname "$0")/operator_common.sh"
require_job
require_image
[[ -x $F19_ROOT/build-operator/bin/expert-crossover ]] || { echo 'Run run_build.sh first.' >&2; exit 2; }
new_output 04-crossover
cp "$F19_ROOT/build-manifest.json" "$OUT/"
engine_container -c 'set -euo pipefail; export LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}; /placement/build-operator/bin/expert-crossover --threads 6 > /out/curve.csv 2> /out/stderr.txt'
python3 - "$OUT/curve.csv" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 18, len(rows)
for shape in ('up_gate', 'down'):
    assert sorted(int(r['n_tokens']) for r in rows if r['shape'] == shape) == [1,2,4,8,16,32,64,128,512]
for r in rows:
    assert all(math.isfinite(float(r[k])) and float(r[k]) > 0 for k in ('cpu_ms','cpu_boundary_ms','gpu_resident_ms','gpu_staged_ms'))
    assert 0 <= float(r['nmse']) < 5e-4
print('CROSSOVER_OK rows=18 numerical_check=PASS')
PY
cp /proc/meminfo "$OUT/meminfo.after.txt"
