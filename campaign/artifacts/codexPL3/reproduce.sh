#!/usr/bin/env bash
set -euo pipefail
cd /home/muk/projects/siminfeng/campaign
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 python3 -B cc/moe_planner/build_quality_ledger.py
nice -n 10 python3 -B -m pytest cc/moe_planner -q -p no:cacheprovider --basetemp=artifacts/codexPL3/pytest-tmp > artifacts/codexPL3/pytest.txt
for scenario in 900:4096 1200:32768 1317:32768; do
    gpu_mib=${scenario%:*}
    ctx=${scenario#*:}
    extra_flags=()
    if [[ "$ctx" != 4096 ]]; then extra_flags+=(--allow-extrapolation); fi
    nice -n 10 python3 -B cc/moe_planner/plan.py \
        --stats artifacts/codexPL3/factory-stats.json \
        --host artifacts/codexPL3/factory-host.json \
        --resource-sample artifacts/codexPL3/factory-sample.json \
        --components cc/factory/components.json \
        --budget-gpu-mib "$gpu_mib" --ctx "$ctx" --max-kld 0.08 --top 3 \
        "${extra_flags[@]}" \
        --emit-lab-tests "artifacts/codexPL3/lab-tests-$gpu_mib.json" \
        > "artifacts/codexPL3/plan-$gpu_mib.json"
done
