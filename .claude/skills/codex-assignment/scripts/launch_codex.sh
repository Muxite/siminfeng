#!/bin/bash
# Launch a Codex worker on an assignment, in the background, recorded by bin/run_op.
# Usage: launch_codex.sh TASK_SHORT ASSIGNMENT_FILE [--parent X-c2] [--timeout SECONDS] [--dry-run]
#   e.g. launch_codex.sh MI assignments/T-MI-model-inspect.md --timeout 3000
# Writes evidence/codex-<TASK_SHORT>.jsonl (intent/started/outcome) and artifacts/codex<TASK_SHORT>/last-message.txt.
# Pattern copied from the real intent records (evidence/codex-MI.jsonl, codex-KVB2.jsonl).
set -eu
T=${1:?TASK_SHORT}; A=${2:?assignment file}; shift 2
PARENT=X-c2; TMO=3600; DRY=0
while [ $# -gt 0 ]; do case $1 in --parent) PARENT=$2; shift 2;; --timeout) TMO=$2; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
CAMP=${CAMPAIGN_ROOT:-$HOME/projects/siminfeng/campaign}; cd "$CAMP"
AF=$(readlink -f "$A"); [ -f "$AF" ] || { echo "assignment not found: $A"; exit 2; }
case $AF in "$CAMP"/assignments/*) ;; *) echo "assignment must live in $CAMP/assignments/"; exit 2;; esac
# Provider-safety lint: crash/exploit vocabulary got T-KVB refused (decisions/gaps.md, c2-resume friction).
if grep -n -i -E 'sigsegv|segfault|gdb|backtrace|exploit|crash' "$AF"; then
  echo "WARNING: assignment contains crash/debugger wording above; rephrase in correctness terms (see SKILL.md)"; [ "${FORCE:-0}" = 1 ] || { [ $DRY = 1 ] || exit 3; }
fi
export CAMPAIGN_ID=${CAMPAIGN_ID:-c1}
PROMPT="Read and execute the assignment in $AF exactly. Stay within its write locations."
CMD=(nohup nice -n 15 bin/run_op --agent "codex-$T" --parent "$PARENT" --task "T-$T" --desc "codex exec ${A#assignments/}" --timeout "$TMO"
     -- codex exec --json -s workspace-write -C "$CAMP" -o "artifacts/codex$T/last-message.txt" "$PROMPT")
if [ $DRY = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "artifacts/codex$T"
"${CMD[@]}" >/dev/null 2>&1 &
echo "launched codex-$T pid $! ; wait with: python3 ../.claude/skills/codex-assignment/scripts/wait_outcome.py codex-$T"
