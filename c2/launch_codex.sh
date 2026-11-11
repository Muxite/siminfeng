#!/usr/bin/env bash
# launch_codex.sh TASK ASSIGNMENT_FILE [EXTRA_PROMPT] — runs codex on an assignment, logs start/end to c2/worklog.jsonl
set -u; cd "$(dirname "$0")/.."
T=$1; A=$2; X=${3:-}
mkdir -p artifacts/codex
S=$(date -u +%FT%TZ); S0=$(date +%s)
echo "{\"event\":\"start\",\"task\":\"$T\",\"worker\":\"codex\",\"assignment\":\"$A\",\"ts\":\"$S\"}" >> c2/worklog.jsonl
codex exec -s workspace-write -C "$PWD" --add-dir /tmp -o artifacts/codex/$T.final.txt \
  "$(cat $A)
$X
You are a Codex worker in campaign c2. Follow the assignment exactly; acceptance commands will be run verbatim by a checker." \
  < /dev/null > artifacts/codex/$T.log 2>&1
RC=$?
TOK=$(grep -A1 -i 'tokens used' artifacts/codex/$T.log | tail -1 | tr -dc 0-9)
echo "{\"event\":\"end\",\"task\":\"$T\",\"worker\":\"codex\",\"rc\":$RC,\"wall_s\":$(( $(date +%s)-S0 )),\"tokens\":\"$TOK\",\"ts\":\"$(date -u +%FT%TZ)\"}" >> c2/worklog.jsonl
