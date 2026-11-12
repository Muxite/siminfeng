#!/usr/bin/env bash
# OPERATOR ONLY. Queue class: gpu. Expected: 10-45 s; timeout: 180 s.
# Outputs: 03-uma/probe.tsv (attributes + four cases, optional fifth), stderr.txt, container-command.txt.
set -euo pipefail
source "$(dirname "$0")/operator_common.sh"
require_job
require_image
[[ -x $F19_ROOT/build-operator/probe_uma ]] || { echo 'Run run_build.sh first.' >&2; exit 2; }
new_output 03-uma
engine_container -c 'set -euo pipefail; /placement/build-operator/probe_uma "$1" 67108864 0 > /out/probe.tsv 2> /out/stderr.txt' \
    f19 "/om/$F19_MODEL"
cp /proc/meminfo "$OUT/meminfo.after.txt"
echo "UMA_OK $OUT/probe.tsv"
