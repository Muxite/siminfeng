import json,sys,statistics as st,collections,hashlib,os
for j in sys.argv[1:]:
  d=f'lab/jobs/{j}/att-1'; r=json.load(open(d+'/result.json')); print('==',j,r['execution'],r['evidence']['coverage']['tests_completed_cleanly'],'/',r['evidence']['coverage']['tests_expected'])
  g=collections.defaultdict(lambda: collections.defaultdict(list))
  for k,v in r['quality']['tests'].items():
    n=k.split('#')[0]; m=g[n]; m['mib'].append(v['mem']['gpu_delta_mib']); m['host'].append(v['mem']['host_cgroup_peak_mib'])
    for x,y in (('decode_tps','dec'),('prefill_tps','pre'),('ppl','ppl'),('kld','kld'),('top1_pct','top1')):
      if v.get(x) is not None: m[y].append(v[x])
    if not v['ok']: m['bad'].append(k)
  for n,m in g.items():
    print(f"{n:34s} gpu {max(m['mib']):5d} host {max(m['host']):7.0f}", *(f"{x} {st.median(m[x]):.3f}" if x in('kld',) else f"{x} {st.median(m[x]):.1f}" for x in ('dec','pre','ppl','kld','top1') if m[x]), ('BAD '+str(len(m['bad']))) if m['bad'] else '')
