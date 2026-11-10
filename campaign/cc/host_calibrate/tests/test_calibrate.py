import copy
import json
from pathlib import Path
import sys
import pytest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import calibrate as c
from apply import apply
F=Path(__file__).parent/'fixtures'

@pytest.mark.parametrize('fixture,op,weight_bytes',[
    ('cpu-perf-real.txt','MUL_MAT',4096*14336//256*144),
    ('cpu-expert-real.txt','MUL_MAT_ID',768*2048//256*144*8)])
def test_real_perf(fixture,op,weight_bytes):
    row=c.parse_perf((F/fixture).read_text())[0]
    assert row['op']==op and row['kind']=='Q4_K'
    assert row['weight_gbs']==pytest.approx(weight_bytes/row['time_us']/1000)
    assert row['printed_gbs'] is None

@pytest.mark.parametrize('text',['','MUL_MAT(): not supported','ERROR: allocation failed'])
def test_no_measurement(text):
    with pytest.raises(ValueError): c.parse_perf(text)

@pytest.mark.parametrize('change',['n=2','bs=[2,1]'])
def test_wrong_shape(change):
    text=(F/'cpu-perf-real.txt').read_text()
    text=text.replace('n=1',change) if change=='n=2' else text.replace('bs=[1,1]',change)
    with pytest.raises(ValueError): c.parse_perf(text)


def test_real_bench():
    vals=c.parse_bench((F/'bench-real.json').read_text(),n_gen=32,coord=0)
    assert len(vals)==3 and vals[0]==pytest.approx(1e6/64.1803)
    with pytest.raises(ValueError): c.parse_bench((F/'bench-real.json').read_text())


def test_real_cpu_identity():
    host=c.identity('',(F/'lscpu-real.txt').read_text())
    assert '3700X' in host['cpu_model'] and len(host['id'])==64
    assert host['nvidia_smi_text']==''


def test_plan_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'model_profile',lambda _: {'n_layer':24})
    model=tmp_path/'small.gguf';model.write_bytes(b'GGUF')
    jobs=c.command_plan('sha256:release',model,0,4,c.OUT/'test')
    assert len(jobs)==10
    for j in jobs:
        a=j['argv']
        assert a[:3]==['docker','run','--rm']
        assert '--memory=2g' in a and a[a.index('timeout')+3]=='25s'



def test_write_escape():
    with pytest.raises(ValueError): c.output_path(c.ROOT/'cc/moe_planner/calibration.json')


def test_h2d_synthetic_explicit():
    rows=[dict(direction='H2D',pinned=True,bytes=2**i*1048576,repeat=r,ms=1) for i in range(9) for r in range(3)]
    text='\n'.join(map(json.dumps,rows))
    assert len(c.parse_h2d(text))==27
    with pytest.raises(ValueError): c.parse_h2d('\n'.join(map(json.dumps,rows[:-1])))


def test_apply_consumed_by_real_decode_model():
    sys.path.insert(0,str(c.ROOT/'cc/moe_planner'))
    import decode_model as dm
    base=c.read(c.ROOT/'cc/moe_planner/calibration.json')
    before=copy.deepcopy(base)
    host=dict(schema='host-calibration/1',host={'id':'synthetic'},constants=dict(
        bw_gpu_gbs=100.,bw_cpu_gbs={'Q4_K':10.},handoff_us=2.,bw_h2d_gbs=5.,cache_miss_us=0.))
    result=apply(host,base)
    s=dict(n_layer=2,n_expert=4,n_expert_used=1,expert_bytes=4000,nonexpert_bytes=1000,representation='mixE4K')
    pred=dm.predict(s,'static',1,result['physical_decode']['constants'])
    assert pred['decode_us']==pytest.approx(.015+.05+2)
    assert base==before and result['physical_decode']['evaluation']['summary']=={}
    host['constants']['bw_gpu_gbs']=float('nan')
    with pytest.raises(ValueError): apply(host,base)


def test_handoff_pipeline_synthetic_explicit(tmp_path):
    # Synthetic mechanism test; benchmark formats separately have real captures above.
    names=['nvidia','lscpu','cpu-stream','cuda-stream','h2d','bench-0','bench-1']
    texts={'nvidia':'GPU 0: NVIDIA GeForce RTX 5070 Ti (synthetic)', 'lscpu':(F/'lscpu-real.txt').read_text()}
    texts.update({'cpu-stream':stream_text('cpu'),'cuda-stream':stream_text('cuda')})
    texts['h2d']='\n'.join(json.dumps(dict(direction='H2D',pinned=True,bytes=2**i*1048576,repeat=r,ms=1)) for i in range(9) for r in range(3))
    bench=json.loads((F/'bench-real.json').read_text())
    bench[0].update(n_gen=64,n_cpu_moe=0,samples_ts=[1000,1000])
    texts['bench-0']=json.dumps(bench)
    bench[0].update(n_cpu_moe=1,samples_ts=[500,500])
    texts['bench-1']=json.dumps(bench)
    for name,text in texts.items():
        (tmp_path/f'{name}.stdout').write_text(text)
        (tmp_path/f'{name}.status.json').write_text('{"returncode":0}')
    stats=dict(n_layer=2,n_expert=4,n_expert_used=1,expert_bytes=4000,nonexpert_bytes=1000,
               tensors=[dict(role='expert_ffn',layer=i,ggml_type='Q4_K',bytes=2000) for i in range(2)])
    manifest=dict(image='synthetic',cpu_layers=[0,1],jobs=[{'name':n} for n in names])
    result=c.parse_capture(manifest,tmp_path,stats)
    terms=result['handoff_terms']
    assert result['constants']['handoff_us']==pytest.approx(1000-terms['delta_cpu_us']-terms['delta_gpu_us'])
    assert terms['delta_gpu_us']<0
    (tmp_path/'h2d.status.json').write_text('{"returncode":124}')
    with pytest.raises(ValueError,match='h2d failed'): c.parse_capture(manifest,tmp_path,stats)


def test_real_model_header_bounds():
    model=Path('/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf')
    if not model.is_file():
        pytest.skip('campaign real model absent')
    assert c.model_profile(model)['n_layer']==24


# Explicitly synthetic stream schema/algebra cases, distinct from real CAL captures.
def stream_text(device):
    pool=512*1048576
    return '\n'.join(json.dumps(dict(probe=device+'-stream-v1',threads=t,repeat=r,
        working_set_bytes=pool,cache_bytes=32*1048576,buffers=8,bytes_read=pool,
        ms=pool/(20*1e6),checksum=100.,gbs=20.))
        for t in range(1,9 if device=='cpu' else 2) for r in range(3))


@pytest.mark.parametrize('name,device',[
    ('CUDA0-MUL_MAT','gpu'), ('CPU-MUL_MAT_ID','cpu')])
def test_real_cal_cache_residency_rejected(name,device):
    f=F/'cal2'
    limits=c.bandwidth_ceilings((f/'nvidia.stdout').read_text(),(f/'lscpu.stdout').read_text())
    rows=c.parse_perf((f/(name+'.stdout')).read_text())
    q4=next(r for r in rows if r['kind']=='Q4_K')
    assert q4['weight_gbs']>limits[device+'_gbs']
    with pytest.raises(ValueError,match='implausible bandwidth.*ceiling'):
        c.guard_bandwidth({name:rows},limits)


def test_real_cal_capture_fails_before_model_algebra():
    f=F/'cal2'
    with pytest.raises(ValueError,match='implausible bandwidth'):
        c.parse_capture(c.read(f/'manifest.json'),f,{})


def test_identity_ceilings_fail_closed():
    assert c.bandwidth_ceilings('Jetson Orin Nano','')['gpu_gbs']==68
    with pytest.raises(ValueError,match='unknown/ambiguous GPU'):
        c.bandwidth_ceilings('unlisted GPU','')
    limits=c.bandwidth_ceilings('RTX 5070 Ti','',dict(channels=2,rate_mts=3200))
    assert limits['cpu_gbs']==51.2
    with pytest.raises(ValueError):
        c.bandwidth_ceilings('RTX 5070 Ti','',dict(channels=0,rate_mts=3200))
    with pytest.raises(ValueError):
        c.guard_bandwidth({'CPU-stream':[dict(weight_gbs=float('nan'))]},limits)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stream_accounting(device):
    text=stream_text(device)
    rows=c.parse_stream(text,device)
    assert all(r['weight_gbs']==pytest.approx(20.) for r in rows)
    for key,value in [('working_set_bytes',32*1048576),('cache_bytes',128*1048576),
                      ('bytes_read',1),('gbs',4000),('checksum',0),('ms',30000)]:
        bad=json.loads(text.splitlines()[0]); bad[key]=value
        lines=text.splitlines(); lines[0]=json.dumps(bad)
        with pytest.raises(ValueError): c.parse_stream('\n'.join(lines),device)
    with pytest.raises(ValueError): c.parse_stream('\n'.join(text.splitlines()[:-1]),device)


def test_new_plan_stream_commands(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'model_profile',lambda _: {'n_layer':24})
    model=tmp_path/'m.gguf';model.write_bytes(b'GGUF')
    jobs={j['name']:j['argv'] for j in c.command_plan('release',model,0,12,c.OUT/'gpu')}
    assert not any('MUL_MAT' in n for n in jobs)
    assert 'campaign/cuda-tools:c1' in jobs['build-bw']
    assert all(flag in jobs['build-bw_cpu'] for flag in ('gcc','-O3','-mavx2','-fopenmp'))
    assert jobs['cpu-stream'][-2:]==['/probe/bw_cpu','8']
    with pytest.raises(ValueError): c.output_path(c.ROOT/'artifacts/codexCAL/forbidden.json')


def test_native_cpu_stream_real_capture():
    rows=c.parse_stream((F/'cpu-stream-native-real.jsonl').read_text(),'cpu')
    assert len(rows)==24
    assert rows[0]['working_set_bytes']>=8*rows[0]['cache_bytes']
    limits=c.bandwidth_ceilings((F/'cal2/nvidia.stdout').read_text(),'')
    c.guard_bandwidth({'CPU-stream':rows},limits)


def test_real_cal_h2d_and_benches_preserved():
    rows=c.parse_h2d((F/'cal2/h2d.stdout').read_text())
    bw=c.summary([r['gbs'] for r in rows if r['bytes']>=64*1048576])['median']
    assert bw==pytest.approx(28.7,rel=.02)
    for k in (0,12):
        assert len(c.parse_bench((F/f'cal2/bench-{k}.stdout').read_text(),coord=k))==2


def test_ceiling_boundary_and_stream_upper_allocation():
    limits=c.bandwidth_ceilings('RTX 5070 Ti','')
    c.guard_bandwidth({'CPU-stream':[dict(weight_gbs=60)],'CUDA-stream':[dict(weight_gbs=896)]},limits)
    for name,value in [('CPU-stream',60.01),('CUDA-stream',896.01)]:
        with pytest.raises(ValueError,match='implausible bandwidth'):
            c.guard_bandwidth({name:[dict(weight_gbs=value)]},limits)
    rows=[json.loads(line) for line in stream_text('cuda').splitlines()]
    for row in rows:
        row['working_set_bytes']=2*1024**3
    with pytest.raises(ValueError): c.parse_stream('\n'.join(map(json.dumps,rows)),'cuda')
