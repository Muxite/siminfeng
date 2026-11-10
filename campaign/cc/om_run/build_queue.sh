#!/usr/bin/env bash
# Sequential Build CC queue (one heavy build at a time on jetson-arm). Waits for any running build_engine first.
cd "$(dirname "$0")/../.."; export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH DOCKER_CMD_PREFIX='sg docker -c' BUILD_JOBS=3
while pgrep -f "build_engine.py" >/dev/null; do sleep 20; done
b(){ bin/run_op --agent opus-jetson --task J-BQ --desc "build queue: $2" --timeout 7200 -- python3 cc/build_engine/build_engine.py --target jetson-orin "$@" > /dev/null 2>&1
     python3 -c "import json,sys;r=json.load(open(sys.argv[1]+'/build-record.json'));print(sys.argv[1],r['status'],r.get('failure_stage'),str(r.get('error'))[:160],round(r.get('wall_seconds') or 0))" "${@: -1}"; }
b --source ref/llama.cpp --commit 31288c9c66e59377f7788d920678a565bb89dd37 --patch cc/routing_trace/patches/combined-trace-p03.patch --extra-target llama-routing-trace --tag campaign/llamacpp:jetson-c1-tr03 --out cc/build_engine/results/jetson-c1-tr03
b --source ref/llama.cpp-pr31469 --commit fc4a5a77a61b6ff965a5aa8b0f9e6d6a9c796583 --tag campaign/llamacpp:jetson-pr31469 --out cc/build_engine/results/jetson-pr31469
echo QUEUE_DONE
