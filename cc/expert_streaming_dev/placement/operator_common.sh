#!/usr/bin/env bash
# Shared by operator scripts; never executes a GPU job when sourced.
set -euo pipefail
F19_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
F19_LLAMA=${F19_LLAMA:-$(dirname "$F19_ROOT")/llama-place}
F19_RESULTS=${F19_RESULTS:-$F19_ROOT/results}
F19_MODEL_DIR=${F19_MODEL_DIR:-$HOME/models/campaign/qwen36-35b-a3b}
F19_MODEL=${F19_MODEL:-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
F19_LAB=${F19_LAB:-/home/muk/projects/siminfeng}
F19_PIN=6e9d96cf4e007c74aab91bb59767247737a80f0f

require_job() {
    [[ -n ${JOBQ_JOB_ID:-} ]] || { echo 'Submit through bin/jobq; JOBQ_JOB_ID is missing.' >&2; exit 2; }
}
require_image() {
    : "${F19_IMAGE:?Set F19_IMAGE to the CUDA 12.6 engine image verified against the pin.}"
}
check_source() {
    [[ $(git -C "$F19_LLAMA" rev-parse HEAD) == "$F19_PIN" ]] || { echo 'Wrong llama-place commit' >&2; exit 2; }
    git -C "$F19_LLAMA" diff --exit-code --quiet -- ggml src common
}
new_output() {
    OUT=$F19_RESULTS/$1
    mkdir -p "$F19_RESULTS"
    mkdir "$OUT" || { echo "Output already exists: $OUT (use a fresh F19_RESULTS)." >&2; exit 2; }
    printf '%s\n' "$JOBQ_JOB_ID" > "$OUT/job-id.txt"
    cp /proc/meminfo "$OUT/meminfo.before.txt"
}
engine_container() {
    local command
    # %q preserves each argv element through sg's single command-string argument.
    printf -v command '%q ' docker run --rm --runtime nvidia --network none \
        --entrypoint /bin/bash \
        -v "$F19_ROOT:/placement:ro" -v "$OUT:/out:rw" \
        -v "$F19_MODEL_DIR:/om:ro" "$F19_IMAGE" "$@"
    printf '%s\n' "$command" > "$OUT/container-command.txt"
    sg docker -c "$command"
}
