#!/usr/bin/env bash
# H-RA real test: upstream + env-gated mmap advice patch; ABBA base vs LLAMA_MMAP_ADVICE=random on the over-memory model.
cd "$(dirname "$0")/../.."; export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH DOCKER_CMD_PREFIX='sg docker -c' BUILD_JOBS=3
while pgrep -f "python3 cc/build_engine/build_engine.py|cc/om_run/pipeline2.sh|cc/om_run/sweep_numa.sh" >/dev/null; do sleep 30; done
bin/run_op --agent opus-jetson --task J-BQ --desc "build upstream cf34d311 + mmap advice patch (J-RA)" --timeout 7200 -- python3 cc/build_engine/build_engine.py --target jetson-orin --source ref/llama.cpp-up1108 --commit cf34d311332e362198f0a410c466d1e940287d02 --patch cc/mmap_advice/patches/01-mmap-advice-random.patch --tag campaign/llamacpp:jetson-up1108-ra --out cc/build_engine/results/jetson-up1108-ra >/dev/null 2>&1
python3 -c "import json;r=json.load(open('cc/build_engine/results/jetson-up1108-ra/build-record.json'));print('BUILD',r['status'],r.get('failure_stage'),str(r.get('error'))[:200],round(r.get('wall_seconds') or 0))"
[ "$(python3 -c "import json;print(json.load(open('cc/build_engine/results/jetson-up1108-ra/build-record.json'))['status'])")" = success ] || exit 1
IMG=campaign/llamacpp:jetson-up1108-ra; O=experiments/jetson/OM/ra-$(date +%Y%m%d-%H%M)
C="llama-completion -m /om/model.gguf -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap -cmoe -lv 4"
run(){ name=$1; env=$2; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM-RA --desc "H-RA $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $IMG "$env" $C >/dev/null 2>&1
  python3 - $O/$name <<'PY'
import json,sys,re
d=sys.argv[1]; r=json.load(open(d+'/result.json')); se=open(d+'/stderr.txt',errors='replace').read()
pe=re.findall(r'prompt eval time.*?([0-9.]+) tokens per second',se); ev=re.findall(r'\n.*?\beval time.*?([0-9.]+) tokens per second',se)
v={}
for l in open(d+'/telemetry.jsonl'):
    try: x=json.loads(l)
    except: continue
    for m in x['metrics']:
        if isinstance(m['value'],(int,float)): v.setdefault(m['name'],[]).append(m['value'])
rd=(max(v['device_read'])-min(v['device_read']))/1024; io=(max(v['device_read_ios'])-min(v['device_read_ios']))
print(d.split('/')[-1], r['status'], 'wall',r['wall_s'],'prompt',pe[-1] if pe else '-','eval',ev[-1] if ev else '-','NVMe GiB',round(rd,1),'avg KiB/io',round(rd*1048576/max(io,1)),'minAvail',round(min(v['mem_available'])),'advice_log',int('LLAMA_MMAP_ADVICE=random' in se), flush=True)
PY
  sha256sum < $O/$name/stdout.txt | cut -c1-12; }
run base-a ""
run rand-a "-e LLAMA_MMAP_ADVICE=random"
run rand-b "-e LLAMA_MMAP_ADVICE=random"
run base-b ""
echo RA_DONE $O
