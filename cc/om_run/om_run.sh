#!/usr/bin/env bash
# om_run.sh — guarded over-memory engine run on jetson-arm.
# usage: om_run.sh OUTDIR IMAGE "ENV_FLAGS" TOOL ARGS...
#   e.g. om_run.sh experiments/jetson/OM/b1-r1 campaign/llamacpp:jetson-c1-p03 "" llama-bench -m /om/model.gguf -lm mmap -ngl 99 -cmoe -p 64 -n 32 -r 2
# Model dir ~/models/campaign/qwen36-35b-a3b is mounted read-only at /om. Watchdog kills ONLY this container
# when MemAvailable < OM_MIN_AVAIL_MIB (default 700) or zram swap used > OM_MAX_SWAP_MIB (default 1536).
set -u
OUT=$1; IMG=$2; ENVF=$3; TOOL=$4; shift 4
R=$(cd "$(dirname "$0")/../.." && pwd); mkdir -p "$OUT"
NAME="om-$(date +%s)-$$"
MODELDIR=${OM_MODELDIR:-$HOME/models/campaign/qwen36-35b-a3b}
DATADIR=$HOME/models/campaign/data/wikitext/wikitext-2-raw
DP=${DOCKER_CMD_PREFIX:-}
dk(){ if [ -n "$DP" ]; then $DP "docker $*"; else docker "$@"; fi; }
printf '%s\n' "$IMG" "$ENVF" "$TOOL $*" > "$OUT/cmd.txt"
dk run -d --name "$NAME" --runtime nvidia --network none --label campaign=c1 --label om=1 $ENVF \
   -v "$MODELDIR":/om:ro -v "$DATADIR":/d:ro "$IMG" /app/$TOOL "$@" > "$OUT/container.id" || { echo "START_FAIL" > "$OUT/status"; exit 2; }
if [ "${OM_THREADS:-0}" = 1 ]; then
    python3 "$R/cc/thread_sampler/thread_sampler.py" --container "$NAME" \
        --out "$OUT/threads.jsonl" --summary "$OUT/threads.summary.json" \
        >"$OUT/threads.log" 2>&1 & TS=$!
fi
if [ -n "$DP" ]; then KILLCMD="$DP 'docker kill $NAME'"; else KILLCMD="docker kill $NAME"; fi
python3 "$R/cc/resource_observe/resource_observe.py" watchdog --min-available-mib ${OM_MIN_AVAIL_MIB:-600} \
   --max-swap-used-mib ${OM_MAX_SWAP_MIB:-2560} --max-swap-rate-mibs ${OM_MAX_SWAP_RATE:-1000} --interval-ms 250 --kill-cmd "$KILLCMD" --out "$OUT/watchdog.jsonl" & WD=$!
python3 "$R/cc/resource_observe/resource_observe.py" sample --backends io,proc,tegrastats --path "$MODELDIR" --interval-ms 500 \
   --duration-s ${OM_MAX_S:-1800} --out "$OUT/telemetry.jsonl" >/dev/null 2>&1 & SM=$!
T0=$(date +%s.%N)
RC=$(dk wait "$NAME"); T1=$(date +%s.%N)
kill $WD $SM 2>/dev/null; wait $WD 2>/dev/null; WDRC=$?
dk logs "$NAME" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
if [ "${OM_THREADS:-0}" = 1 ]; then
    wait "$TS" || echo "thread sampler failed; see threads.log" >&2
    if [ -s "$OUT/threads.jsonl" ]; then
        python3 "$R/cc/thread_sampler/thread_sampler.py" --summarize "$OUT/threads.jsonl" \
            --stderr "$OUT/stderr.txt" --summary "$OUT/threads.summary.json" \
            >>"$OUT/threads.log" 2>&1 || echo "thread summary failed; see threads.log" >&2
    fi
fi
OOM=$(dk inspect "$NAME" --format '{{.State.OOMKilled}}')
dk rm "$NAME" > /dev/null
TRIP=$(grep -c watchdog_trip "$OUT/watchdog.jsonl" 2>/dev/null); TRIP=${TRIP:-0}
python3 - "$OUT" "$RC" "$T0" "$T1" "$OOM" "$TRIP" "$IMG" "$ENVF" "$TOOL" "$@" <<'PY'
import json,sys,os,socket
o,rc,t0,t1,oom,trip,image,env,*argv=sys.argv[1:]
code=int(rc.strip() or -1)
tripped=int(trip)>0
if tripped and code in (137, -9):
    status="watchdog_tripped_killed"
elif tripped and code == 0:
    status="watchdog_tripped_survived"
elif oom.strip()=="true":
    status="oom_killed"
else:
    status="ok" if code == 0 else "failed"
try:
    with open(os.path.expanduser("~/.config/siminfeng/host_id")) as f:
        host_id=f.read().strip() or socket.gethostname()
except OSError:
    host_id=socket.gethostname()
json.dump(dict(exit_code=code,wall_s=round(float(t1)-float(t0),2),
  oom_killed=oom.strip()=="true",watchdog_trips=int(trip),status=status,
  host_id=host_id,image=image,env=env,argv=argv),open(o+"/result.json","w"),indent=1)
print(open(o+"/result.json").read())
PY
