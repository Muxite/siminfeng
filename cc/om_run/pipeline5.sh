#!/usr/bin/env bash
# J-OM7: router-informed expert WILLNEED prefetch; ABCCBA base/pf/pfn, then kernel-screen (was pipeline4; its pgrep wait loop self-matched the launching shell).
cd "$(dirname "$0")/../.."; export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH DOCKER_CMD_PREFIX='sg docker -c' BUILD_JOBS=3
bin/run_op --agent opus-jetson --task J-BQ --desc "build upstream cf34d311 + mmap advice 01+02 router prefetch (J-OM7)" --timeout 7200 -- python3 cc/build_engine/build_engine.py --target jetson-orin --source ref/llama.cpp-up1108 --commit cf34d311332e362198f0a410c466d1e940287d02 --patch cc/mmap_advice/patches/combined-01-02.patch --tag campaign/llamacpp:jetson-up1108-om7 --out cc/build_engine/results/jetson-up1108-om7 >/dev/null 2>&1
python3 -c "import json;r=json.load(open('cc/build_engine/results/jetson-up1108-om7/build-record.json'));print('BUILD',r['status'],r.get('failure_stage'),str(r.get('error'))[:200],round(r.get('wall_seconds') or 0))"
[ "$(python3 -c "import json;print(json.load(open('cc/build_engine/results/jetson-up1108-om7/build-record.json'))['status'])")" = success ] || exit 1
IMG=campaign/llamacpp:jetson-up1108-om7; O=experiments/jetson/OM/om7-$(date +%Y%m%d-%H%M)
C="llama-completion -m /om/model.gguf -f /om/prompts/p1.txt -n 64 -c 4096 --temp 0 -s 1 -no-cnv -ngl 99 -lm mmap -cmoe -lv 4"
run(){ name=$1; env=$2; OM_MAX_S=2400 bin/run_op --agent opus-jetson --task J-OM7 --desc "J-OM7 $name" --timeout 3000 -- cc/om_run/om_run.sh $O/$name $IMG "$env" $C >/dev/null 2>&1
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
print(d.split('/')[-1], r['status'], 'wall',r['wall_s'],'prompt',pe[-1] if pe else '-','eval',ev[-1] if ev else '-','NVMe GiB',round(rd,1),'avg KiB/io',round(rd*1048576/max(io,1)),'minAvail',round(min(v['mem_available'])),'pf',[l.strip()[-160:] for l in se.splitlines() if 'prefetch' in l.lower()][-3:], flush=True)
PY
  sha256sum < $O/$name/stdout.txt | cut -c1-12; }
E="-e LLAMA_EXPERT_PREFETCH=1"; N="-e LLAMA_EXPERT_PREFETCH=1 -e LLAMA_EXPERT_PREFETCH_NEXT=1"
run base-a ""; run pf-a "$E"; run pfn-a "$N"; run pfn-b "$N"; run pf-b "$E"; run base-b ""
echo OM7_DONE $O
bin/run_op --agent opus-jetson --task J-BQ --desc "build pin + patch03 + perf cases --with-tests (kernel screen image)" --timeout 7200 -- python3 cc/build_engine/build_engine.py --target jetson-orin --source ref/llama.cpp --commit 31288c9c66e59377f7788d920678a565bb89dd37 --patch cc/kernel_screen/combined-p03-perf.patch --with-tests --extra-target test-backend-ops --tag campaign/llamacpp:jetson-c1-ks --out cc/build_engine/results/jetson-c1-ks >/dev/null 2>&1
python3 -c "import json;r=json.load(open('cc/build_engine/results/jetson-c1-ks/build-record.json'));print('BUILD',r['status'],r.get('failure_stage'),str(r.get('error'))[:300],round(r.get('wall_seconds') or 0))"
[ "$(python3 -c "import json;print(json.load(open('cc/build_engine/results/jetson-c1-ks/build-record.json'))['status'])")" = success ] || exit 1
O=experiments/jetson/KS-$(date +%Y%m%d-%H%M); mkdir -p $O
for t in bf16 q4_K q8_0; do
  T=$(echo $t | tr a-z A-Z | sed 's/Q4_K/Q4_K/;s/BF16/BF16/')
  bin/run_op --agent opus-jetson --task J-KS --desc "kernel screen granite $t base vs unfuse" --timeout 1800 -- python3 cc/kernel_screen/kernel_screen.py --image campaign/llamacpp:jetson-c1-ks --preset cc/kernel_screen/presets/granite.json --types $T --ops MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION --arm 'A={"GGML_CUDA_J_UNFUSE_IDS_TYPES":null}' --arm 'A2={"GGML_CUDA_J_UNFUSE_IDS_TYPES":null}' --arm "B={\"GGML_CUDA_J_UNFUSE_IDS_TYPES\":\"$t\"}" --repeats 3 --out $O/$t > $O/$t.log 2>&1
  echo "== $t rc=$?"; tail -15 $O/$t.log | cut -c1-200
done
echo KS_DONE $O
