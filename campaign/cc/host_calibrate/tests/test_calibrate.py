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


def test_plan_bounded_and_filters(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'model_profile',lambda _: {'n_layer':24})
    model=tmp_path/'small.gguf';model.write_bytes(b'GGUF')
    jobs=c.command_plan('sha256:release',model,0,4,c.OUT/'test')
    assert len(jobs)==10
    for j in jobs:
        a=j['argv']
        assert a[:3]==['docker','run','--rm']
        assert '--memory=2g' in a and a[a.index('timeout')+3]=='25s'
    for j in jobs:
        if '-p' in j['argv'] and 'perf' in j['argv']:
            import re
            fixture='cpu-expert-real.txt' if j['name'].endswith('_ID') else 'cpu-perf-real.txt'
            params=c.parse_perf((F/fixture).read_text())[0]['params']
            assert re.search(j['argv'][-1],params)


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
    names=['nvidia','lscpu','CPU-MUL_MAT','CPU-MUL_MAT_ID','CUDA0-MUL_MAT','CUDA0-MUL_MAT_ID','h2d','bench-0','bench-1']
    dense=(F/'cpu-perf-real.txt').read_text()
    expert=(F/'cpu-expert-real.txt').read_text()
    texts={'nvidia':'synthetic GPU', 'lscpu':(F/'lscpu-real.txt').read_text()}
    for backend in ('CPU','CUDA0'):
        texts[backend+'-MUL_MAT']='\n'.join(dense.replace('q4_K',q) for q in c.TYPES)
        texts[backend+'-MUL_MAT_ID']='\n'.join(expert.replace('q4_K',q) for q in ('q4_K','q6_K','q8_0'))
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
