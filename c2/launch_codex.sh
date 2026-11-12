#!/usr/bin/env bash
# launch_codex.sh TASK ASSIGNMENT_FILE [EXTRA_PROMPT] — runs codex on an assignment, logs start/end to c2/worklog.jsonl
set -u; cd "$(dirname "$0")/.."
T=$1; A=$2; X=${3:-}
mkdir -p artifacts/codex
S=$(date -u +%FT%TZ); S0=$(date +%s)
echo "{\"event\":\"start\",\"task\":\"$T\",\"worker\":\"codex\",\"assignment\":\"$A\",\"ts\":\"$S\"}" >> c2/worklog.jsonl
codex exec ${CODEX_ARGS:-} -s workspace-write -C "${CODEX_CWD:-$PWD}" --add-dir /tmp --add-dir /home/muk/projects/siminfeng/lab/jobq ${CODEX_ADD_DIR:+--add-dir "$CODEX_ADD_DIR"} -o artifacts/codex/$T.final.txt \
  "$(cat $A)
$X
You are a Codex worker (siminfeng c2 / Satay factory). Follow the assignment exactly; acceptance commands will be run verbatim by a checker.
Any compile/build (cmake, make, nvcc, cargo) MUST run via /home/muk/projects/siminfeng/bin/heavy so it is serialized with lab jobs." \
  < /dev/null > artifacts/codex/$T.log 2>&1
RC=$?
TOK=$(grep -A1 -i 'tokens used' artifacts/codex/$T.log | tail -1 | tr -dc 0-9)
echo "{\"event\":\"end\",\"task\":\"$T\",\"worker\":\"codex\",\"rc\":$RC,\"wall_s\":$(( $(date +%s)-S0 )),\"tokens\":\"$TOK\",\"ts\":\"$(date -u +%FT%TZ)\"}" >> c2/worklog.jsonl
