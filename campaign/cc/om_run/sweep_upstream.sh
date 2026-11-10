#!/usr/bin/env bash
# Upstream moe-cache sweep (llama-completion; llama-bench lacks cache flags). Waits for builds + pinned baseline sweep.
cd "$(dirname "$0")/../.."; export DOCKER_CMD_PREFIX='sg docker -c'
while pgrep -f "python3 cc/build_engine/build_engine.py" >/dev/null || pgrep -f "cc/om_run/sweep_baselines.sh" >/dev/null; do sleep 30; done
IMG=${IMG:-campaign/llamacpp:jetson-up1108}; O=experiments/jetson/OM/up-$(date +%Y%m%d-%H%M)
C="llama-completion -m /om/model.gguf -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap"
run(){ name=$1; shift; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM-U --desc "OM upstream $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $IMG "" $C "$@" >/dev/null 2>&1
  r=$(python3 -c "import json;print(json.load(open('$O/$name/result.json'))['status'])")
  pe=$(grep -E "prompt eval time" $O/$name/stderr.txt | tail -1 | grep -oE "[0-9.]+ tokens per second"); ev=$(grep -E "^.*\beval time" $O/$name/stderr.txt | grep -v prompt | tail -1 | grep -oE "[0-9.]+ tokens per second")
  mc=$(grep -iE "moe.cache|moe_cache" $O/$name/stderr.txt | tail -3 | tr '\n' ' ' | cut -c1-300)
  echo "$name $r prompt=[$pe] eval=[$ev] cache=[$mc]"; }
run base-cmoe-a -cmoe
run cache1024 -cmoe --moe-cache-mib 1024
run cache2048 -cmoe --moe-cache-mib 2048
run cache2048-cpu50 -cmoe --moe-cache-mib 2048 --moe-cache-cpu-share 0.5
run base-cmoe-b -cmoe
echo UPSWEEP_DONE $O
