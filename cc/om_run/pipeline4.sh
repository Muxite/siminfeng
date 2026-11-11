#!/usr/bin/env bash
# Kernel-screen predictor test: does test-backend-ops perf (seconds) predict the end-to-end D027 results?
cd "$(dirname "$0")/../.."; export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH DOCKER_CMD_PREFIX='sg docker -c' BUILD_JOBS=3
while pgrep -f "python3 cc/build_engine/build_engine.py|cc/om_run/pipeline3.sh|cc/om_run/pipeline2.sh" >/dev/null; do sleep 30; done
bin/run_op --agent opus-jetson --task J-BQ --desc "build pin + patch03 + perf cases --with-tests (kernel screen image)" --timeout 7200 -- python3 cc/build_engine/build_engine.py --target jetson-orin --source ref/llama.cpp --commit 31288c9c66e59377f7788d920678a565bb89dd37 --patch cc/kernel_screen/combined-p03-perf.patch --with-tests --extra-target test-backend-ops --tag campaign/llamacpp:jetson-c1-ks --out cc/build_engine/results/jetson-c1-ks >/dev/null 2>&1
python3 -c "import json;r=json.load(open('cc/build_engine/results/jetson-c1-ks/build-record.json'));print('BUILD',r['status'],r.get('failure_stage'),str(r.get('error'))[:300],round(r.get('wall_seconds') or 0))"
[ "$(python3 -c "import json;print(json.load(open('cc/build_engine/results/jetson-c1-ks/build-record.json'))['status'])")" = success ] || exit 1
O=experiments/jetson/KS-$(date +%Y%m%d-%H%M); mkdir -p $O
for t in bf16 q4_K q8_0; do
  T=$(echo $t | tr a-z A-Z | sed 's/Q4_K/Q4_K/;s/BF16/BF16/')
  bin/run_op --agent opus-jetson --task J-KS --desc "kernel screen granite $t base vs unfuse" --timeout 1800 -- python3 cc/kernel_screen/kernel_screen.py --image campaign/llamacpp:jetson-c1-ks --preset cc/kernel_screen/presets/granite.json --types $T --ops MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION --arm 'A={"GGML_CUDA_J_UNFUSE_IDS_TYPES":null}' --arm 'A2={"GGML_CUDA_J_UNFUSE_IDS_TYPES":null}' --arm "B={\"GGML_CUDA_J_UNFUSE_IDS_TYPES\":\"$t\"}" --repeats 3 --out $O/$t > $O/$t.log 2>&1
  echo "== $t rc=$?"; tail -15 $O/$t.log | cut -c1-200
done
echo KS_DONE $O
