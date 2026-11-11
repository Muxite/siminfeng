#!/usr/bin/env bash
# Routing-trace capture on the over-memory model (image jetson-c1-tr03 = pin + generic trace + patch03). Waits for other GPU sweeps.
cd "$(dirname "$0")/../.."; export DOCKER_CMD_PREFIX='sg docker -c'
while pgrep -f "python3 cc/build_engine/build_engine.py" >/dev/null || pgrep -f "cc/om_run/sweep_upstream.sh|cc/om_run/sweep_baselines.sh" >/dev/null; do sleep 30; done
O=$PWD/experiments/jetson/OM/trace-$(date +%Y%m%d-%H%M); mkdir -p $O
for t in wiki code chat math; do
  mkdir -p $O/$t; chmod 777 $O/$t
  OM_MAX_S=3000 bin/run_op --agent opus-jetson --task J-OM-T --desc "routing trace $t (Qwen3.6 over-memory, -lm mmap -cmoe)" --timeout 3600 -- cc/om_run/om_run.sh $O/$t/run campaign/llamacpp:jetson-c1-tr03 "-v $O/$t:/out" llama-routing-trace -m /om/model.gguf -f /om/prompts/t-$t.txt -n 128 -c 4096 -ngl 99 -lm mmap -cmoe --temp 0 --trace-out /out/routing.jsonl >/dev/null 2>&1
  python3 cc/routing_trace/validate_generic_trace.py $O/$t/routing.jsonl 2>&1 | tail -2 | tr '\n' ' '; echo " [$t] $(cat $O/$t/run/result.json | tr -d ' \n')"
done
echo TRACE_DONE $O
