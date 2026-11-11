#!/usr/bin/env bash
# Run hybrid_moe_bench inside campaign/cuda-tools:c1 (GPU). Waits up to 10 min for the GPU to have no compute apps.
# Usage: run.sh <out.json> [extra args, e.g. --threads 4 --layer 12]
set -euo pipefail
CAMPAIGN=/home/muk/projects/siminfeng
BUILD=${BUILD_DIR:-$CAMPAIGN/artifacts/opus-hybrid/build}
OUT=${1:?out json}; shift
OUTDIR=$(dirname "$(readlink -f "$OUT")"); mkdir -p "$OUTDIR"
for i in $(seq 1 60); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  [ "$i" = 60 ] && { echo "GPU busy for 10 min, giving up" >&2; exit 75; }
  echo "GPU busy, waiting ($i)" >&2; sleep 10
done
docker run --rm --label campaign=c1 --cpus 8 --memory 16g --pids-limit 512 --network none --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$BUILD":/build:ro -v /home/muk/models/campaign/gguf:/models:ro -v "$CAMPAIGN/artifacts/traces/out":/traces:ro -v "$OUTDIR":/out \
  campaign/cuda-tools:c1 timeout 120 /build/hybrid_moe_bench --out "/out/$(basename "$OUT")" "$@"
