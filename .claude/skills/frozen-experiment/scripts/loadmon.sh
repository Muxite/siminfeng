#!/bin/bash
# Host-load monitor for a timing window (generalised from artifacts/x2/chain3.sh).
# Usage:  loadmon.sh LOGFILE &   MON=$!   ... timed jobs ...   kill $MON ; loadmon.sh --summary LOGFILE
# One line / 5 s: UTC time, loadavg 1/5/15, cc=<compiler procs>, dl=<hf download procs>.
# The bracket trick ('[c]c1plus') stops pgrep matching this script's own command line (self-match bug fixed in
# artifacts/x2: the first monitor reported a phantom compiler). Validate with a positive control (see SKILL.md).
if [ "$1" = "--summary" ]; then
  echo "samples: $(wc -l < "$2")  load1 max: $(awk '{print $2}' "$2" | sort -n | tail -1)"
  awk '{print $(NF-1),$NF}' "$2" | sort | uniq -c; exit 0
fi
LOG=${1:?logfile}; INTERVAL=${INTERVAL:-5}
while true; do
  echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-3 /proc/loadavg) cc=$(pgrep -c -f '[c]c1plus|[n]vcc|[c]make --build|[c]ompile_cuda') dl=$(pgrep -c -f '[h]f download|[h]uggingface-cli')"
  sleep "$INTERVAL"
done >> "$LOG" 2>&1
