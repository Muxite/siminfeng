"""Physical accounting and genuine cross-model holdout checks, all CPU-only."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import decode_model as dm
from fit_decode import evaluate, fit
ROOT = HERE.parents[1]
OUT = ROOT/'artifacts/codexPL4'


def read(path): return json.loads(Path(path).read_text())


@pytest.fixture
def stats():
    return dict(n_layer=2,n_expert=4,n_expert_used=2, expert_bytes=12000,
                nonexpert_bytes=1000,representation='mixE4K',tensors=[
                    dict(role='expert_ffn',layer=0,bytes=4000,ggml_type='Q4_K'),
                    dict(role='expert_ffn',layer=1,bytes=8000,ggml_type='Q3_K')])


def test_first_cpu_layers_use_actual_quantized_tensor_bytes(stats):
    p = dm.traffic(stats,'static',1)
    assert p['cpu_active_bytes']=={'Q4_K':2000,'Q3_K':0}
    assert p['gpu_active_bytes']==5000
    assert p['resident_expert_bytes']==8000
    assert p['cpu_touching_layers']==1
    all_gpu=dm.traffic(stats,'static',0)
    assert all_gpu['gpu_active_bytes']==7000
    all_cpu=dm.traffic(stats,'static',2)
    assert all_cpu['gpu_active_bytes']==1000
    assert sum(all_cpu['cpu_active_bytes'].values())==6000


def test_hybrid_coverage_reduces_bytes_but_touches_all_cold_layers(stats):
    p=dm.traffic(stats,'hybrid',4,coverage=.75)
    assert sum(p['cpu_active_bytes'].values())==1500
    assert p['gpu_active_bytes']==5500
    assert p['cpu_touching_layers']==2
    assert dm.traffic(stats,'hybrid',8,coverage=1)['cpu_touching_layers']==0


def test_lru_misses_are_gpu_expert_transfers_not_cpu_gemv(stats):
    p=dm.traffic(stats,'lru',0,coverage=.5)
    assert p['cpu_active_bytes']=={}
    assert p['gpu_active_bytes']==7000
    assert p['h2d_bytes']==3000
    assert p['cpu_touching_layers']==0


def test_component_times_have_consistent_units(stats):
    constants=dict(bw_gpu_gbs=10,bw_cpu_gbs={'Q4_K':2,'Q3_K':1},handoff_us=7)
    p=dm.predict(stats,'static',1,constants)
    assert p['gpu_us']==.5
    assert p['cpu_us']==1
    assert p['handoff_us']==7
    assert p['decode_us']==8.5
    assert p['decode_tps']==pytest.approx(1e6/8.5)


def test_sibling_quantization_does_not_reuse_stale_tensor_types(stats):
    stats.update(expert_bytes=6000,representation='mixE2K')
    p=dm.traffic(stats,'static',2)
    assert p['cpu_active_bytes']=={'Q2_K':3000}


@pytest.mark.parametrize('coord',[-1,3,.5])
def test_invalid_cpu_layer_count_fails(stats,coord):
    with pytest.raises(ValueError): dm.traffic(stats,'static',coord)


def test_recorded_evidence_and_strict_model_holdouts():
    rows=read(OUT/'measured-points.json')
    assert len(rows)==165
    assert {r['decision'] for r in rows}=={'D022','D029','D034','D035','D039','D040'}
    for held_out in ('Granite','Qwen'):
        training=[r for r in rows if r['model']!=held_out]
        model=fit(training)
        assert held_out not in model['training_models']
        held_jobs={r['job_id'] for r in rows if r['model']==held_out}
        assert not held_jobs.intersection(model['training_jobs'])
        assert model['training_point_count']==len(training)
    # A one-model Granite CPU slope cannot identify handoff separately from bandwidth.
    assert fit([r for r in rows if r['model']=='Granite'])['handoff_prior']


def test_qwen_fixes_order_of_magnitude_extrapolation_failure():
    rows=read(OUT/'measured-points.json'); models=read(OUT/'model-stats.json')
    constants=read(HERE/'calibration.json')['physical_decode']['constants']
    # Frozen D039 decision values, independent of the regression implementation.
    for k,measured in [(18,54.62),(30,36.58),(42,27.73)]:
        tps=dm.predict(models['qwen'],'static',k,constants)['decode_tps']*1.02
        assert tps==pytest.approx(measured,rel=.1)
    # Granite-only training must improve on the previous -85% transfer error.
    fitted=fit([r for r in rows if r['model']=='Granite'])
    result=evaluate([r for r in rows if r['model']=='Qwen'],models,fitted)
    assert result['summary']['Qwen']['mape_pct']<20


def test_cpu_screen_covers_required_types_without_changing_clean_fits():
    cal=read(HERE/'calibration.json')['physical_decode']
    screen=read(OUT/'cpu-screen/report.json')
    assert set(screen['effective_gbs'])=={'Q4_K','Q3_K','Q2_K','Q8_0'}
    assert len(screen['results'])==7
    for r in screen['results']:
        assert r['returncode']==0 and len(r['samples_tps'])==3
        assert r['load_samples'] and r['load_before']['loadavg']
        assert r['bench']['n_gpu_layers']==0
        assert r['bench']['n_gen']==32
    assert cal['constants']['screen_bw_cpu_gbs']==screen['effective_gbs']
    assert cal['constants']['bw_cpu_gbs']['Q4_K']!=screen['effective_gbs']['Q4_K']
