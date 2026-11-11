#!/usr/bin/env bash
# Scored over-memory baselines (quiet CPU: waits for build_engine to finish). Pinned engine image with patch 03 (env unset = upstream behaviour).
cd "$(dirname "$0")/../.."; export DOCKER_CMD_PREFIX='sg docker -c'
while pgrep -f "python3 cc/build_engine/build_engine.py" >/dev/null || pgrep -f "cc/om_run/sweep_upstream.sh" >/dev/null; do sleep 30; done
IMG=${IMG:-campaign/llamacpp:jetson-c1-p03}; O=experiments/jetson/OM/base-$(date +%Y%m%d-%H%M)
B="llama-bench -m /om/model.gguf -p 128 -n 64 -r 2 -o json"
run(){ name=$1; shift; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM-S --desc "OM baseline $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $IMG "" $B "$@" >/dev/null 2>&1
  python3 - $O/$name <<'PY'
import json,sys
d=sys.argv[1]; r=json.load(open(d+'/result.json'))
try: b=[(x['n_prompt'],x['n_gen'],round(x['avg_ts'],2)) for x in json.load(open(d+'/stdout.txt'))]
except Exception as e: b=str(e)[:80]
print(d.split('/')[-1], r['status'], r['wall_s'], b, flush=True)
PY
}
# ABBA-ish ordering of the two main configs, plus single runs of the others
run mmap-ncmoe40-a -lm mmap -ngl 99 -ncmoe 40
run mmap-ncmoe38   -lm mmap -ngl 99 -ncmoe 38
run mmap-ncmoe40-b -lm mmap -ngl 99 -ncmoe 40
run none-ncmoe40   -lm none -ngl 99 -ncmoe 40
run mmap-ngl0      -lm mmap -ngl 0
run auto-default   -ngl 99 -ncmoe 40
echo SWEEP_DONE $O
