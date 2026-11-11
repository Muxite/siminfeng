#!/usr/bin/env bash
# codex_job.sh TASK ASSIGNMENT [EXTRA] — submit a Codex worker as a jobq cpu job (never overlaps gpu timing jobs), start a worker.
set -u; cd "$(dirname "$0")/.."
J=$(bin/jobq submit --class cpu --name codex-$1 --timeout 7200 -- c2/launch_codex.sh "$@"); echo $J
nohup bin/jobq worker --once >> c2/worker-codex.log 2>&1 &
