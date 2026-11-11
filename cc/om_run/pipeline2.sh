#!/usr/bin/env bash
# After all GPU sweeps + fio: build upstream + cache components, then A/B on the over-memory model.
cd "$(dirname "$0")/../.."; export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH DOCKER_CMD_PREFIX='sg docker -c' BUILD_JOBS=3
while pgrep -f "cc/om_run/sweep_|cc/om_run/fio_ceiling.sh" >/dev/null || pgrep -f "python3 cc/build_engine/build_engine.py" >/dev/null; do sleep 30; done
bin/run_op --agent opus-jetson --task J-BQ --desc "build upstream cf34d311 + moe-cache components 01+02+03" --timeout 7200 -- python3 cc/build_engine/build_engine.py --target jetson-orin --source ref/llama.cpp-up1108 --commit cf34d311332e362198f0a410c466d1e940287d02 --patch cc/moe_cache_profile/patches/combined-01-02-03.patch --tag campaign/llamacpp:jetson-up1108-mc --out cc/build_engine/results/jetson-up1108-mc >/dev/null 2>&1
python3 -c "import json;r=json.load(open('cc/build_engine/results/jetson-up1108-mc/build-record.json'));print('BUILD',r['status'],r.get('failure_stage'),str(r.get('error'))[:300],round(r.get('wall_seconds') or 0))"
grep -iE "error" cc/build_engine/results/jetson-up1108-mc/build.stderr | head -5
st=$(python3 -c "import json;print(json.load(open('cc/build_engine/results/jetson-up1108-mc/build-record.json'))['status'])"); [ "$st" = success ] || { echo PIPE2_STOP_BUILD_FAILED; exit 1; }
IMG=campaign/llamacpp:jetson-up1108-mc; O=experiments/jetson/OM/mc-$(date +%Y%m%d-%H%M)
C="llama-completion -m /om/model.gguf -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap -cmoe -lv 4"
run(){ name=$1; shift; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM-MC --desc "OM components $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $IMG "" $C "$@" >/dev/null 2>&1
  r=$(python3 -c "import json;print(json.load(open('$O/$name/result.json'))['status'])")
  ev=$(grep -E "\beval time" $O/$name/stderr.txt | grep -v prompt | tail -1 | grep -oE "[0-9.]+ tokens per second")
  mc=$(grep -hE "llama_moe_cache:|direct|dedupe|pinned" $O/$name/stderr.txt | tail -4 | tr '\n' ' ' | cut -c1-260)
  echo "$name $r eval=[$ev] [$mc]"; sha256sum < $O/$name/stdout.txt | cut -c1-12; }
run base-a
run cache2048 --moe-cache-mib 2048
run cache2048-dedupe --moe-cache-mib 2048 --moe-cache-uma-dedupe
run cache2048-direct --moe-cache-mib 2048 --moe-cache-direct-fill
run cache2048-direct-cpu50 --moe-cache-mib 2048 --moe-cache-direct-fill --moe-cache-cpu-share 0.5
run cache1024-direct --moe-cache-mib 1024 --moe-cache-direct-fill
run base-b
echo PIPE2_DONE $O
