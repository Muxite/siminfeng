#!/usr/bin/env bash
set -euo pipefail
# Run from any directory; cd/pwd/printf/read/wait are shell control builtins.
ROOT=$(cd -- "${BASH_SOURCE[0]%/*}/../.." && pwd)
cd "$ROOT"
export CAMPAIGN_ID=c1 PYTHONDONTWRITEBYTECODE=1
export CAMPAIGN_ROOT="$ROOT/artifacts/codexDEMO1"
unset LAB_FAULT  # This rehearsal kills a sandbox, never its controller.
DRY=${DEMO_DRY:-0}
HELPER=cc/demo/inspect.py
BUILD=builds/results/novmm-4/build-record.json
HOSTS=cc/engine_handoff/hosts
MODEL=${DEMO_MODEL:-/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf}

op() {
    printf '+ CAMPAIGN_ID=c1 CAMPAIGN_ROOT=%q LAB=%q bin/run_op --agent demo --task DEMO --' "$CAMPAIGN_ROOT" "${LAB:-}" >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    if [[ "$DRY" != 1 ]]; then bin/run_op --agent demo --task DEMO -- "$@"; fi
}
step() {
    if [[ "$1" != 1 && ${DEMO_PAUSE:-0} == 1 && "$DRY" != 1 ]]; then
        read -r -p 'Press Enter for the next step… ' _
    fi
    printf '\n=== %s/6: %s ===\n' "$1" "$2"
}

step 1 'Build identity: content, local execution reference, transport, linkage'
op python3 "$HELPER" build "$BUILD"

step 2 'Submit a fresh immutable job with the existing mixE4K tests'
if [[ "$DRY" == 1 ]]; then
    op date -u +%Y%m%dT%H%M%S.%NZ
    stamp=DRY-TIMESTAMP
else
    stamp=$(op date -u +%Y%m%dT%H%M%S.%NZ)
fi
key="demo-${stamp}-$$"
OUT="$CAMPAIGN_ROOT/$key"
export LAB="$OUT/lab"
op python3 "$HELPER" setup "$OUT" lab
if [[ "$DRY" == 1 ]]; then
    op bin/lab submit --image llamacpp-novmm4 --model "$MODEL" --tests m1_static_n12,m1_static_n0 --key "$key"
    job=job-DRY-SUBMISSION
else
    job=$(op bin/lab submit --image llamacpp-novmm4 --model "$MODEL" --tests m1_static_n12,m1_static_n0 --key "$key")
    [[ "$job" =~ ^job-[0-9a-f]{10}$ ]]
    printf 'New job: %s\n' "$job"
fi
JDIR="$LAB/jobs/$job"

step 3 'One proposal: accept x86; reject the Jetson template with reasons'
op python3 "$HELPER" policy "$OUT/policy.json" "$LAB/export/$job/att-1" "$BUILD"
op python3 cc/engine_handoff/envelopes.py proposal --build-record "$BUILD" --job-spec "$JDIR/job.json" --host-profile "$HOSTS/x86-main.json" --policy "$OUT/policy.json" --out "$OUT/proposal.json"
op python3 "$HELPER" proposal "$OUT/proposal.json"
op python3 cc/engine_handoff/envelopes.py admit --proposal "$OUT/proposal.json" --host-profile "$HOSTS/x86-main.json" --out "$OUT/admit-x86.json"
op python3 "$HELPER" admission "$OUT/admit-x86.json" accepted
# Rejection is a documented CLI exit status, not a reason to disable errexit.
if op python3 cc/engine_handoff/envelopes.py admit --proposal "$OUT/proposal.json" --host-profile "$HOSTS/jetson-arm.TEMPLATE.json" --out "$OUT/admit-jetson.json"; then
    [[ "$DRY" == 1 ]] || { printf 'ERROR: expected rejection (exit 2)\n' >&2; exit 1; }
else
    rc=$?
    [[ "$rc" == 2 ]] || exit "$rc"
fi
op python3 "$HELPER" admission "$OUT/admit-jetson.json" rejected

step 4 'Interrupt only the owned test sandbox; verify durable exported evidence'
if [[ "$DRY" == 1 ]]; then
    op bin/lab run "$job"
    printf '# Above run starts in background; wait for its att-1 test container.\n'
    op python3 "$HELPER" wait-running "$JDIR/att-1/state.jsonl"
    container=lab-c1-job-DRY-SUBMISSION-att-1-m1_static_n12-0
    op docker inspect --format '{{.State.Running}}' "$container"
else
    op bin/lab run "$job" &
    runner=$!
    container=$(op python3 "$HELPER" wait-running "$JDIR/att-1/state.jsonl")
    # The journal entry follows Popen, not Docker's actual container start.
    ready=0
    for ((poll=0; poll<200; poll++)); do
        if running=$(op docker inspect --format '{{.State.Running}}' "$container") && [[ "$running" == true ]]; then
            ready=1
            break
        fi
        op sleep 0.05
    done
    [[ "$ready" == 1 ]] || { printf 'Container never became running; no chaos sent.\n' >&2; wait "$runner"; exit 1; }
fi
# kill-sandbox is the action implemented by chaos; it is NOT a positional arg.
op bin/lab chaos "$job" att-1
if [[ "$DRY" != 1 ]]; then wait "$runner"; else printf '# wait for background run_op/runner to finish\n'; fi
op tail -n 12 "$JDIR/att-1/state.jsonl"
op python3 "$HELPER" verify "$LAB/export/$job/att-1"
op python3 cc/engine_handoff/envelopes.py result --job-dir "$JDIR" --attempt att-1 --claim-category interrupted-observation --out "$OUT/result-att-1.json"
op python3 "$HELPER" result "$OUT/result-att-1.json" interrupted_sandbox_terminated
op python3 "$HELPER" snapshot "$JDIR/att-1" "$LAB/export/$job/att-1" "$OUT/att-1-before.json"

step 5 'Retry as att-2; complete both tests and preserve att-1 byte for byte'
op bin/lab run "$job"
op tail -n 12 "$JDIR/att-2/state.jsonl"
op python3 "$HELPER" verify "$LAB/export/$job/att-2"
op python3 cc/engine_handoff/envelopes.py result --job-dir "$JDIR" --attempt att-2 --claim-category infrastructure-validation --out "$OUT/result-att-2.json"
op python3 "$HELPER" result "$OUT/result-att-2.json" completed
op python3 "$HELPER" compare "$JDIR/att-1" "$LAB/export/$job/att-1" "$OUT/att-1-before.json"

step 6 'Five-line summary'
op python3 "$HELPER" summary "$job" "$OUT"
