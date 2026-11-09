#!/usr/bin/env bash
# Reproduce T-PAP without Docker or network access. All writes stay in the two assignment roots.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
OUT="$ROOT/artifacts/codexPAP"
TREE="$OUT/tree"
BUILD="$OUT/build"
MODELS="$OUT/models"
SOURCE="$HOME/models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf"
TRAIN="$HOME/models/campaign/data/wikitext/wikitext-2-raw/wiki.train.raw"
IMAT="$MODELS/granite-wtrain.imatrix"
REF=31288c9c66e59377f7788d920678a565bb89dd37
mkdir -p "$OUT/logs" "$MODELS"
export PYTHONDONTWRITEBYTECODE=1
build() {
    git -C ref/llama.cpp archive "$REF" | sha256sum > "$OUT/logs/source-archive.sha256"
    if [[ ! -f "$TREE/CMakeLists.txt" ]]; then
        mkdir -p "$TREE"
        git -C ref/llama.cpp archive "$REF" | tar -x -C "$TREE"
    fi
    nice -n 10 cmake -S "$TREE" -B "$BUILD" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_COMMIT="$REF" -DLLAMA_BUILD_NUMBER=0 > "$OUT/logs/configure.log" 2>&1
    nice -n 10 cmake --build "$BUILD" --target llama-quantize llama-imatrix llama-perplexity -j4 > "$OUT/logs/build.log" 2>&1
}
calibrate() {
    date -u +%FT%TZ > "$OUT/logs/calibrate.start"
    sha256sum "$BUILD/bin/llama-imatrix" > "$OUT/logs/calibration-binary.sha256"
    nice -n 10 "$BUILD/bin/llama-imatrix" -m "$SOURCE" -f "$TRAIN" -o "$IMAT" \
        --output-format gguf -t 6 -tb 6 -ngl 0 -c 512 --chunks 100 --no-ppl \
        --output-frequency 10 > "$OUT/logs/imatrix.log" 2>&1
    date -u +%FT%TZ > "$OUT/logs/calibrate.end"
}
quantize_one() {
    local name=$1 k=$2 bits=$3 calibrated=$4
    local opts=()
    if [[ "$calibrated" == yes ]]; then opts+=(--imatrix "$IMAT"); fi
    if (( k > 0 )); then
        local layers
        layers=$(seq -s '|' 0 "$((k - 1))")
        opts+=(--tensor-type "^blk\\.($layers)\\.ffn_.*_exps\\.weight$=q${bits}_K")
    fi
    opts+=(--tensor-type 'ffn_.*_exps=q4_K')
    date -u +%FT%TZ > "$OUT/logs/$name.start"
    nice -n 10 "$BUILD/bin/llama-quantize" "${opts[@]}" "$SOURCE" "$MODELS/$name.gguf" Q8_0 4 \
        > "$OUT/logs/$name.log" 2>&1
    date -u +%FT%TZ > "$OUT/logs/$name.end"
}
quantize() {
    rm -f "$OUT/cpu-sanity.json"
    for k in 12 18 24; do
        for bits in 3 2; do quantize_one "pap${k}_q${bits}" "$k" "$bits" yes; done
    done
    quantize_one pap24_q3_noimat 24 3 no
    quantize_one mix4_imat 0 4 yes
}
case "${1:-all}" in
    build) build ;;
    calibrate) calibrate ;;
    quantize) quantize ;;
    sanity) python3 cc/pap/cpu_sanity.py ;;
    verify) python3 cc/pap/verify_pap.py ;;
    all) build; calibrate; quantize; python3 cc/pap/verify_pap.py ;;
    *) echo 'usage: make_pap.sh [all|build|calibrate|quantize|sanity|verify]' >&2; exit 2 ;;
esac
