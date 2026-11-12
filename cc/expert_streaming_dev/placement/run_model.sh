#!/usr/bin/env bash
# OPERATOR ONLY. Queue class: gpu. Expected: 2-6 min per invocation; timeout: 900 s.
# Usage: run_model.sh baseline|no-offload|adaptive|baseline-replay
# Outputs: NN-MODE/{summary.json,stdout.txt,stderr.txt,result.json,telemetry.jsonl,watchdog.jsonl,page_cache.*.json,cmd.txt}.
set -euo pipefail
source "$(dirname "$0")/operator_common.sh"
require_job
require_image
MODE=${1:?baseline|no-offload|adaptive|baseline-replay}
FLAGS=(--op-offload)
THRESHOLD=32
case "$MODE" in
    baseline) LABEL=01-baseline ;;
    no-offload) LABEL=02-no-offload; FLAGS=(--no-op-offload) ;;
    adaptive) LABEL=05-adaptive ;;
    baseline-replay) LABEL=06-baseline-replay ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
[[ ${OM_NSYS:-0} == 0 ]] || { echo 'Use untraced timings for this campaign.' >&2; exit 2; }
new_output "$LABEL"
sha256sum "$F19_MODEL_DIR/prompts/p1.txt" > "$OUT/prompt.sha256"
if [[ $MODE == adaptive ]]; then
    AUTO_ARGS=()
    [[ ${F19_ENABLE_AUTO:-0} == 1 ]] && AUTO_ARGS=(--enable)
    python3 "$F19_ROOT/select_policy.py" --baseline "$F19_RESULTS/01-baseline/summary.json" \
        --cpu "$F19_RESULTS/02-no-offload/summary.json" --curve "$F19_RESULTS/04-crossover/curve.csv" \
        --prompt-sha256 "$(cut -d ' ' -f 1 "$OUT/prompt.sha256")" \
        "${AUTO_ARGS[@]}" > "$OUT/policy.json"
    THRESHOLD=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$OUT/policy.json")
    FLAGS=("$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["flag"])' "$OUT/policy.json")")
fi
# The existing strict LLAMA_EXPERT_* validator accepts exactly these settings at the pin.
ENV_FLAGS="-e GGML_OP_OFFLOAD_MIN_BATCH=$THRESHOLD -e GGML_CUDA_REGISTER_HOST=0 -e LLAMA_EXPERT_FILL=1 -e LLAMA_EXPERT_FILL_RANDOM=decode -e LLAMA_EXPERT_FILL_THREADS=16 -e LLAMA_EXPERT_FILL_CHUNK_KIB=2048 -e LLAMA_EXPERT_FILL_MIN_TOKENS=0 -e LLAMA_EXPERT_FILL_MAX_TOKENS=1 -e LLAMA_EXPERT_FILL_DIAG=0 -e LLAMA_EXPERT_FILL_POPULATE=0 -e LLAMA_EXPERT_PREFETCH=0 -e LLAMA_EXPERT_PREFETCH_NEXT=0 -e LLAMA_EXPERT_PREGATE=1 -e LLAMA_EXPERT_PREGATE_K=8 -e LLAMA_EXPERT_PREGATE_ORDER=after -e LLAMA_EXPERT_PREGATE_MAX_INFLIGHT=4 -e LLAMA_EXPERT_PREGATE_LOOKAHEAD=1 -e LLAMA_MMAP_ADVICE=default"
export OM_MODELDIR=$F19_MODEL_DIR
export OM_CACHE_PROTOCOL=cold
export OM_CACHE_MODEL=$F19_MODEL_DIR/$F19_MODEL
export OM_MAX_S=900 OM_MIN_AVAIL_MIB=700 OM_MAX_SWAP_MIB=1536
export DOCKER_CMD_PREFIX='sg docker -c'
# om_run writes only into OUT; the model is mounted read-only. Cold uses per-file fadvise, never global drop_caches.
bash "$F19_LAB/cc/om_run/om_run.sh" "$OUT" "$F19_IMAGE" "$ENV_FLAGS" llama-completion \
    -m "/om/$F19_MODEL" -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv \
    -ngl 99 -cmoe -lm mmap -b 512 -ub 512 -t 6 -tb 6 -lv 4 "${FLAGS[@]}"
python3 "$F19_ROOT/summarize_model.py" "$OUT"
cp /proc/meminfo "$OUT/meminfo.after.txt"
echo "MODEL_OK $OUT/summary.json"
