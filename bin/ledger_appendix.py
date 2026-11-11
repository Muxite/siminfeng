import json,glob,os,csv,html,re,collections,statistics as st
R=os.path.expanduser('~/projects/siminfeng'); os.chdir(R)
E=lambda s: html.escape(str(s))
out=[]
def table(title,cols,rows,note=None):
    out.append(f'<h3>{E(title)}</h3>')
    if note: out.append(f'<p class="small">{E(note)}</p>')
    out.append('<div class="tbl"><table><thead><tr>'+''.join(f'<th>{E(c)}</th>' for c in cols)+'</tr></thead><tbody>')
    for r in rows: out.append('<tr>'+''.join(f'<td class="num">{E(x)}</td>' if isinstance(x,(int,float)) or re.match(r'^[-+0-9.,/ %x]+$',str(x) or 'a') else f'<td>{E(x)}</td>' for x in r)+'</tr>')
    out.append('</tbody></table></div>')
# 1 build attempts
rows=[]
for f in sorted(glob.glob('cc/build_engine/results/*/build-record.json')):
    r=json.load(open(f)); d=f.split('/')[-2]; c=r.get('ccache') or {}
    a=(c.get('after') or {}); b=(c.get('before') or {})
    rows.append([d,r.get('target_profile') or 'x86-5070ti',r['status'],r.get('failure_stage') or '',(r.get('source') or {}).get('requested_commit','')[:10] if isinstance(r.get('source'),dict) else '',(r.get('patch_sha256') or '')[:10],round(r.get('wall_seconds') or 0),(a.get('hits',0)-b.get('hits',0)) if a else '',(a.get('misses',0)-b.get('misses',0)) if a else '',((r.get('archive') or {}).get('bytes') or '')]) 
table('Build CC attempts (all, incl. failures)',['out dir','target','status','fail stage','commit','patch sha','wall s','ccache hits','ccache misses','archive bytes'],rows)
# 2 granite bench runs
def bench(f):
    try: return json.load(open(f))
    except Exception: return []
rows=[]
for d,label in [('experiments/jetson/J-E004-fusion-screen','fusion screen (jetson-c1)'),('experiments/jetson/J-E005-patch-ab','patch A/B (jetson-c1-j3)'),('experiments/jetson/J-E006-quant-track','quant track (p01q)'),('experiments/jetson/J-E007-unfuse-by-type','unfuse by type (p03)')]:
    for f in sorted(glob.glob(d+'/*.json')):
        for x in bench(f):
            if not isinstance(x,dict) or 'samples_ts' not in x: continue
            s=x['samples_ts']; rows.append([label,os.path.basename(f)[:-5],x.get('model_type','')[:28],x['n_prompt'],x['n_gen'],x.get('n_depth',0),round(x['avg_ts'],2),round(st.mean(s[1:]),2) if len(s)>1 else '',' '.join(f'{v:.1f}' for v in s)])
table('Granite-1B llama-bench runs (every run file)',['experiment','run','model','pp','tg','depth','avg t/s','steady t/s','samples'],rows,'steady = mean of samples 2..n (first sample is GPU clock ramp)')
# 3 quality
rows=[]
for f in sorted(glob.glob('experiments/jetson/*/kl-*.txt'))+sorted(glob.glob('experiments/jetson/*/ppl*.txt')):
    t=open(f,errors='replace').read()
    g=lambda p: (re.search(p,t) or [None,''])[1]
    rows.append([f.split('/')[-2],os.path.basename(f),g(r'Final estimate: PPL = ([0-9.]+)') or g(r'Mean PPL\(Q\)\s*:\s*([0-9.]+)'),g(r'Mean\s+KLD:\s+(-?[0-9.]+)'),g(r'99\.0%\s+KLD:\s+([0-9.]+)'),g(r'Maximum KLD:\s+([0-9.]+)'),g(r'Same top p:\s+([0-9.]+)')])
table('Quality runs (ppl / KL vs bf16 base logits)',['experiment','file','ppl','mean KLD','p99 KLD','max KLD','top-1 %'],rows)
# 4 OM runs
rows=[]
for d in sorted(glob.glob('experiments/jetson/OM/*/*/result.json')):
    run=os.path.dirname(d); r=json.load(open(d)); cmd=open(run+'/cmd.txt').read().split('\n')
    so=open(run+'/stdout.txt',errors='replace').read() if os.path.exists(run+'/stdout.txt') else ''
    se=open(run+'/stderr.txt',errors='replace').read() if os.path.exists(run+'/stderr.txt') else ''
    pp=tg=''
    try:
        for x in json.loads(so):
            if x['n_prompt']: pp=round(x['avg_ts'],3)
            else: tg=round(x['avg_ts'],3)
    except Exception:
        m=re.findall(r'prompt eval time.*?([0-9.]+) tokens per second',se); n=re.findall(r'\n.*?\beval time.*?([0-9.]+) tokens per second',se)
        pp=m[-1] if m else ''; tg=[v for v in n][-1] if n else ''
    hc=re.search(r'hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB',se)
    v={}
    if os.path.exists(run+'/telemetry.jsonl'):
        for l in open(run+'/telemetry.jsonl'):
            try: rr=json.loads(l)
            except Exception: continue
            for m in rr['metrics']:
                if isinstance(m['value'],(int,float)): v.setdefault(m['name'],[]).append(m['value'])
    rd=(max(v['device_read'])-min(v['device_read'])) if v.get('device_read') else ''
    rows.append([run.split('/OM/')[1],cmd[0].split(':')[-1],' '.join(cmd[2].split()[1:])[:90],r['status'],r['wall_s'],pp,tg,(hc[1]+'%') if hc else '',(round(float(hc[2])/1024,1)) if hc else '',round(rd/1024,1) if rd!='' else '',round(max(v['device_read_rate']),0) if v.get('device_read_rate') else '',round(min(v['mem_available']),0) if v.get('mem_available') else '',round(max(v['pgmajfault_rate']),0) if v.get('pgmajfault_rate') else '',round(max(v['swap_used']),0) if v.get('swap_used') else ''])
table('Over-memory runs: Qwen3.6-35B-A3B UD-Q4_K_XL 20.8 GiB on 7.4 GiB (every attempt)',['run','image','args','status','wall s','pp t/s','tg t/s','cache hit','uploaded GiB','NVMe read GiB','peak MiB/s','min MemAvail MiB','peak majflt/s','peak zram MiB'],rows,'upv-* timings contaminated by a concurrent simulation; cmp-* partially overlapped a nice-19 simulation')
# 5 replay
rows=[]
for r in csv.DictReader(open('experiments/jetson/OM/replay/sweep.csv')): rows.append([r['workload'],r['cap'],r['policy'],r['overall'],r['decode'],r['decode_mb_per_tok']])
table('Expert-cache replay sweep (decode hit rate; static-profile = leave-one-out profile from the other 3 workloads)',['workload','capacity (experts of 10240)','policy','overall hit','decode hit','decode MB moved/token'],rows,'expert = 1.92 MB; capacity 1300 ≈ 2.5 GB ≈ 12.7%')
# 6 routing stats
rows=[]
for w in ['wiki','code','chat','math']:
    f=glob.glob(f'experiments/jetson/OM/trace-*/{w}/routing.jsonl')[0]; cnt=collections.Counter(); toks=set(); per_layer=collections.defaultdict(collections.Counter); ph=collections.Counter()
    for l in open(f):
        if not l.startswith('{"seq'): continue
        x=json.loads(l); toks.add((x['phase'],x['tok'])); ph[x['phase']]+=1
        for e in x['experts']: cnt[(x['layer'],e)]+=1; per_layer[x['layer']][e]+=1
    tot=sum(cnt.values()); top=sorted(cnt.values(),reverse=True)
    cov=lambda k: round(sum(top[:k])/tot,3)
    used=len(cnt)
    ent=[]
    import math
    for L,c in per_layer.items():
        s=sum(c.values()); ent.append(-sum(v/s*math.log2(v/s) for v in c.values()))
    rows.append([w,len(toks),ph['prefill']//40,ph['decode']//40,tot,used,round(used/10240,3),cov(512),cov(1024),cov(2048),round(st.mean(ent),2)])
table('Routing trace statistics (40 layers × 256 experts, top-8)',['workload','tokens','prefill tok','decode tok','expert requests','distinct (layer,expert)','fraction touched','share served by top-512','top-1024','top-2048','mean per-layer entropy (bits, max 8)'],rows,'uniform routing would give top-1024 share ≈ 0.10 if all experts were touched; entropy 8.0 = uniform over 256')
# 7 fusion/kernels
rows=[]
for f in sorted(glob.glob('experiments/jetson/J-P00*/*_cuda_gpu_kern_sum.csv'))+sorted(glob.glob('experiments/jetson/J-P001-nsys-decode/stats_cuda_gpu_kern_sum.csv')):
    try: rr=list(csv.DictReader(open(f)))
    except Exception: continue
    for r in rr[:4]: rows.append([f.split('/')[-2]+'/'+os.path.basename(f).split('_cuda')[0],r['Time (%)'],r['Instances'],round(float(r['Avg (ns)'])/1000,1),r['Name'][:90]])
for f in sorted(glob.glob('experiments/jetson/J-P003-nsys-quant/*.top.txt')):
    for l in open(f):
        p=l.split()
        if len(p)>8 and p[0][0].isdigit(): rows.append(['J-P003/'+os.path.basename(f).split('.')[0],p[0],p[2],round(float(p[3].replace(',',''))/1000,1),' '.join(p[8:])[:90]])
table('Top decode kernels (nsys, tg64 under profiler)',['profile','time %','instances','avg µs','kernel'],rows)
# 8 fio
rows=[]
for f in sorted(glob.glob('experiments/jetson/OM/fio/*.json')):
    try: j=json.load(open(f))['jobs'][0]['read']; rows.append([os.path.basename(f)[:-5],round(j['bw']/1024,1),round(j['iops']),round(j['lat_ns']['mean']/1000)])
    except Exception: pass
if rows: table('NVMe ceiling (fio, O_DIRECT reads of the model file)',['pattern','MiB/s','IOPS','mean lat µs'],rows)
else: out.append('<h3>NVMe ceiling (fio)</h3><p class="small">Queued behind the engine comparison; not yet run.</p>')
# 9 lab jobs
rows=[]
for jd in sorted(glob.glob('lab/jobs/*')):
    a=os.path.join(jd,'admission.json')
    if not os.path.exists(a): continue
    ad=json.load(open(a)); spec=json.load(open(jd+'/job.json'))['spec']
    atts=sorted(glob.glob(jd+'/att-*'))
    res=[]
    for at in atts:
        if os.path.exists(at+'/result.json'):
            rj=json.load(open(at+'/result.json')); res.append(os.path.basename(at)+':'+rj['execution']['status'])
        else:
            s=[json.loads(l) for l in open(at+'/state.jsonl')] if os.path.exists(at+'/state.jsonl') else []
            res.append(os.path.basename(at)+':'+(s[-1].get('execution','') if s else '?'))
    rows.append([os.path.basename(jd),spec['image'],','.join(spec['tests']),ad['decision'],'; '.join(dict.fromkeys(ad['reasons']))[:80],' '.join(res)])
table('Lab jobs on jetson-arm with admission decisions',['job','image','tests','admission','reasons','attempts'],rows)
# 10 host inventory
h=json.load(open('hosts/jetson-arm.json')); rows=[]
for g,d in h.get('facts',{}).items():
    for k,v in d.items():
        val=v.get('value') if isinstance(v,dict) else v
        rows.append([g,k,str(val)[:110]])
table('Host inventory (hosts/jetson-arm.json)',['group','fact','value'],rows[:60])
# 11 decisions
rows=[]
for f in sorted(glob.glob('decisions/D0*.md')):
    t=open(f).read().split('\n')[0].lstrip('# ')
    rows.append([os.path.basename(f)[:4],t[:150]])
table('Decision log',['id','title'],rows)
open(os.path.expanduser('/tmp/claude-1000/-home-muk-projects-siminfeng/dea92124-f703-47de-bb9e-9c93188bff0f/scratchpad/appendix.html'),'w').write('\n'.join(out))
print(len(out),'html chunks')
