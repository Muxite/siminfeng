#!/usr/bin/env bash
# Pinned vs upstream engine, same tool/config (llama-bench -lm mmap -ngl 99 -ncmoe 40), counterbalanced P,U,U,P. Waits for other GPU sweeps.
cd "$(dirname "$0")/../.."; export DOCKER_CMD_PREFIX='sg docker -c'
while pgrep -f "python3 cc/build_engine/build_engine.py" >/dev/null || pgrep -f "cc/om_run/sweep_trace.sh|cc/om_run/sweep_upstream_v.sh|cc/om_run/sweep_baselines.sh" >/dev/null; do sleep 30; done
O=experiments/jetson/OM/cmp-$(date +%Y%m%d-%H%M)
B="llama-bench -m /om/model.gguf -p 128 -n 64 -r 2 -o json -lm mmap -ngl 99 -ncmoe 40"
run(){ name=$1; img=$2; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM-C --desc "OM engine cmp $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $img "" $B >/dev/null 2>&1
  python3 - $O/$name <<'PY'
import json,sys
d=sys.argv[1]; r=json.load(open(d+'/result.json'))
try: b=[(x['n_prompt'],x['n_gen'],round(x['avg_ts'],3),[round(s,3) for s in x['samples_ts']]) for x in json.load(open(d+'/stdout.txt'))]
except Exception as e: b=str(e)[:60]
print(d.split('/')[-1], r['status'], r['wall_s'], b, flush=True)
PY
}
run pin-a campaign/llamacpp:jetson-c1-p03
run up-a  campaign/llamacpp:jetson-up1108
run up-b  campaign/llamacpp:jetson-up1108
run pin-b campaign/llamacpp:jetson-c1-p03
echo CMP_DONE $O
