#!/bin/bash
# Indicative (unfrozen) screen: static --n-cpu-moe vs upstream --moe-cache-mib on Qwen3-30B-A3B, matched VRAM.
R=sha256:5b8a6a22981a798362a8a2d32b04270c8ab8de00c99a457307e18dd20b6b277e
D=/home/muk/projects/siminfeng/campaign/artifacts/mc-qwen
run(){ name=$1; shift
  ( peak=0; while sleep 0.5; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -gt "$peak" ] && peak=$m && echo $peak > $D/$name.peak; done ) & mon=$!
  docker run --rm --label campaign=c1 --gpus all --network=none -v /home/muk/models/campaign/qwen3-30b-a3b:/m:ro -v /home/muk/projects/siminfeng/campaign/artifacts/traces/in:/tin:ro --entrypoint /app/llama-completion $R \
    -m /m/Qwen3-30B-A3B-Q4_K_M.gguf -f /tin/chat.txt -n 128 -c 4096 --temp 0 -s 1 -ngl 99 -fa on -no-cnv --no-warmup "$@" > $D/$name.out 2> $D/$name.err
  kill $mon; echo "$name rc=$? peakMiB=$(cat $D/$name.peak) $(grep -E 'eval time' $D/$name.err | tail -1 | grep -oE '[0-9.]+ tokens per second')  $(grep -E 'prompt eval time' $D/$name.err | grep -oE '[0-9.]+ ms /' )"
  sha256sum < $D/$name.out | cut -c1-8
}
for round in 1 2; do
run st_k30_r$round --n-cpu-moe 30
run mc_6000_r$round --n-cpu-moe 48 --moe-cache-mib 6000
run st_k42_r$round --n-cpu-moe 42
run mc_2200_r$round --n-cpu-moe 48 --moe-cache-mib 2200
done
