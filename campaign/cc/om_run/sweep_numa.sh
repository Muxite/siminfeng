#!/usr/bin/env bash
# Hypothesis H-RA: llama.cpp mmap advice (FADV_SEQUENTIAL + WILLNEED whole file) causes readahead pollution on random expert access.
# Zero-code test: --numa distribute => prefetch=0 + MADV_RANDOM on the whole mapping (llama-mmap.cpp:486-525 @cf34d311).
cd "$(dirname "$0")/../.."; export DOCKER_CMD_PREFIX='sg docker -c'
while pgrep -f "python3 cc/build_engine/build_engine.py|cc/om_run/pipeline2.sh|cc/om_run/fio_ceiling.sh|cc/om_run/sweep_cmp.sh" >/dev/null; do sleep 30; done
IMG=campaign/llamacpp:jetson-up1108; O=experiments/jetson/OM/numa-$(date +%Y%m%d-%H%M)
C="llama-completion -m /om/model.gguf -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap -cmoe"
run(){ name=$1; shift; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM-RA --desc "readahead test $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $IMG "" $C "$@" >/dev/null 2>&1
  python3 - $O/$name <<'PY'
import json,sys,re
d=sys.argv[1]; r=json.load(open(d+'/result.json')); se=open(d+'/stderr.txt',errors='replace').read()
pe=re.findall(r'prompt eval time.*?([0-9.]+) tokens per second',se); ev=[m for m in re.findall(r'\n.*?\beval time.*?([0-9.]+) tokens per second',se)]
v={}
for l in open(d+'/telemetry.jsonl'):
    try: x=json.loads(l)
    except: continue
    for m in x['metrics']:
        if isinstance(m['value'],(int,float)): v.setdefault(m['name'],[]).append(m['value'])
rd=(max(v['device_read'])-min(v['device_read']))/1024
print(d.split('/')[-1], r['status'], 'wall',r['wall_s'],'prompt',pe[-1] if pe else '-','eval',ev[-1] if ev else '-','NVMe GiB',round(rd,1),'minAvail',round(min(v['mem_available'])), flush=True)
PY
  sha256sum < $O/$name/stdout.txt | cut -c1-12; }
run base-a
run numa-a --numa distribute
run numa-b --numa distribute
run base-b
echo NUMA_DONE $O
