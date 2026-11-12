#!/usr/bin/env bash
# J-XS4 activity capture: ONE observability run (not scored) on factory/llamacpp:xs-trace
# (cf34d3113 + patches 01-08,10,11,16 + p12 selection trace). Same model/prompt/argv as contract qwen36-bare-greedy-v1.
set -u
D=$(cd "$(dirname "$0")" && pwd)
M=$HOME/models/campaign/qwen36-35b-a3b
python3 /home/muk/projects/siminfeng/cc/page_cache/page_cache.py resident $M/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --json > $D/page_cache.start.json
sg docker -c "docker run --rm --runtime nvidia --entrypoint /app/llama-completion \
  -v $M:/om:ro -v $D:/prof \
  -e LLAMA_EXPERT_FILL=1 -e LLAMA_EXPERT_FILL_RANDOM=decode -e LLAMA_EXPERT_FILL_THREADS=16 -e LLAMA_EXPERT_FILL_CHUNK_KIB=2048 \
  -e LLAMA_EXPERT_FILL_MAX_TOKENS=1 -e LLAMA_EXPERT_PREGATE=1 -e LLAMA_EXPERT_PREGATE_K=8 \
  -e LLAMA_EXPERT_READINESS=1 -e LLAMA_EXPERT_PHASE_STATS=1 -e LLAMA_EXPERT_TRACE=/prof/expert_trace.csv \
  factory/llamacpp:xs-trace -m /om/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap -cmoe" > $D/stdout.txt 2> $D/stderr.txt
echo rc=$? > $D/status
sha256sum $D/stdout.txt > $D/stdout.sha256
sg docker -c "docker image inspect --format '{{.Id}}' factory/llamacpp:xs-trace" > $D/image.id
