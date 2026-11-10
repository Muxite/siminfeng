"""Meaningful CPU-only checks against independent byte and trace accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

from model import decode_runs, implied_length, pool_curves, predict, recommend, union_curves

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pl4_decode_model', ROOT/'cc/moe_planner/decode_model.py')
pl4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl4)


@pytest.fixture(scope='module')
def real():
    stats = json.loads((ROOT/'artifacts/codexPL4/model-stats.json').read_text())['qwen']
    constants = json.loads((ROOT/'cc/moe_planner/calibration.json').read_text())['physical_decode']['constants']
    return stats, constants


@pytest.fixture
def tiny():
    stats = dict(n_layer=2, n_expert=4, n_expert_used=2, expert_bytes=12000,
                 nonexpert_bytes=1000, tensors=[
                     dict(role='expert_ffn', layer=0, bytes=4000, ggml_type='Q4_K'),
                     dict(role='expert_ffn', layer=1, bytes=8000, ggml_type='Q3_K')])
    constants = dict(bw_gpu_gbs=10, bw_cpu_gbs={'Q4_K':2,'Q3_K':1}, handoff_us=7)
    return stats, constants


@pytest.mark.parametrize('k', [0,18,30,42,48])
def test_d1_equals_unmodified_pl4_at_every_placement(real, k):
    stats, constants = real
    got = predict(stats, constants, k, 1, 1., [8]*48)
    expected = pl4.predict(stats, 'static', k, constants)
    for ours, theirs in [('tps','decode_tps'),('step_us','decode_us'),
                         ('cpu_us','cpu_us'),('gpu_us','gpu_us'),('handoff_us','handoff_us')]:
        assert got[ours] == pytest.approx(expected[theirs], rel=1e-12)


def test_exact_bytes_fixed_cost_and_acceptance(tiny):
    stats, constants = tiny
    p = predict(stats, constants, 1, 2, .5, [3,4])
    assert p['cpu_bytes'] == {'Q4_K':3000}
    assert p['gpu_expert_bytes'] == 8000
    assert p['gpu_bytes'] == 9000  # nonexpert weights once, not expert-scaled
    assert p['cpu_us'] == 1.5
    assert p['gpu_us'] == .9
    assert p['handoff_us'] == 7  # one handoff per CPU layer per batch
    assert p['step_us'] == 9.4
    assert p['accepted_tokens'] == 1
    assert p['tps'] == pytest.approx(1e6/9.4)
    rejected = predict(stats, constants, 1, 2, 0., [3,4])
    assert rejected['step_us'] == p['step_us']
    assert rejected['tps'] == 0  # reward counts accepted drafts only
    all_cpu = predict(stats, constants, 2, 2, 1., [3,4])
    assert all_cpu['cpu_bytes'] == {'Q4_K':3000, 'Q3_K':8000}
    assert all_cpu['cpu_us'] == 9.5
    assert all_cpu['gpu_bytes'] == 1000
    assert all_cpu['handoff_us'] == 14


def test_cost_and_whole_gpu_sensitivities(tiny):
    stats, constants = tiny
    p = predict(stats, constants, 1, 2, 1., [3,4], draft_us=3, extra_us_per_token=5)
    assert p['step_us'] == 22.4
    assert p['additional_us'] == 13
    whole = predict(stats, constants, 1, 2, 1., [3,4], scale_nonexpert=True)
    assert whole['nonexpert_bytes'] == 1750
    assert whole['step_us'] == pytest.approx(9.475)


def write_rows(tmp_path, rows):
    path = tmp_path/'trace.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def row(seq, pos, layer, ids=(0,1), phase='decode'):
    return dict(seq=seq, pos=pos, layer=layer, experts=list(ids), phase=phase)


def test_prefill_gaps_sequences_and_record_order(tmp_path):
    rows = [row(0,0,0,phase='prefill')]
    for seq, positions in [(0,[10,11,13,14]),(1,[0,1])]:
        for pos in positions:
            rows.extend([row(seq,pos,0),row(seq,pos,1,(1,2))])
    runs = decode_runs(write_rows(tmp_path,list(reversed(rows))),2,4,2)
    assert [len(r) for r in runs] == [2,2,2]
    curves = union_curves(runs,2)
    assert curves[1]['windows'] == 6
    assert curves[2]['windows'] == 3
    assert curves[2]['per_layer'] == [2,2]
    with pytest.raises(ValueError,match='no consecutive windows'):
        union_curves(runs,3)


@pytest.mark.parametrize('rows,match', [
    ([row(0,0,0)], 'incomplete'),
    ([row(0,0,0),row(0,0,0)], 'duplicate'),
    ([row(0,0,0,(1,1))], 'invalid'),
    ([row(0,0,0,(0,4))], 'invalid'),
    ([row(0,0,2)], 'invalid'),
    ([row(0,-1,0)], 'invalid'),
    ([row(0,0,0,phase='prefill')], 'no decode'),
])
def test_bad_or_empty_decode_is_rejected(tmp_path, rows, match):
    with pytest.raises(ValueError,match=match):
        decode_runs(write_rows(tmp_path,rows),2,4,2)


def test_real_decode_unions_against_independent_brute_force():
    curves = []
    for domain, expected16 in [('prose',40.02891156462585),
                               ('chat',43.667942176870746),('code',38.45195578231293)]:
        path = ROOT/f'artifacts/codexQRT/{domain}.jsonl'
        runs = decode_runs(path)
        assert len(runs) == 1 and len(runs[0]) == 64
        curve = union_curves(runs)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        decode = {(r['pos'],r['layer']):r['experts'] for r in records if r['phase']=='decode'}
        positions = sorted({p for p,l in decode})
        for d in range(1,17):
            assert curve[d]['windows'] == 65-d
            for layer in range(48):
                counts = []
                for i in range(65-d):
                    counts.append(len({expert for pos in positions[i:i+d]
                                       for expert in decode[pos,layer]}))
                assert curve[d]['per_layer'][layer] == pytest.approx(sum(counts)/len(counts))
            assert 8 <= curve[d]['mean'] <= min(128,d*8)
        assert curve[1]['mean'] == 8
        assert curve[16]['mean'] == pytest.approx(expected16,abs=.001)
        curves.append(curve)
    pooled = pool_curves(curves)
    assert pooled[16]['windows'] == 147
    assert pooled[16]['mean'] == pytest.approx(sum(c[16]['mean'] for c in curves)/3)


def test_window_pooling_weights_samples_without_bridging():
    a = union_curves([[[{0,1}], [{0,1}]]],2)
    b = union_curves([[[{0,1}], [{2,3}], [{0,1}]]],2)
    p = pool_curves([a,b])
    assert p[2]['windows'] == 3
    assert p[2]['mean'] == pytest.approx(10/3)


def test_inversion_uses_mixture_of_times_and_tokens():
    got = implied_length(2.4,{4:200.,5:250.},120.)
    # Both endpoints have the same speedup: no unique implied length.
    assert got is None
    got = implied_length(2.4,{4:200.,5:230.},100.)
    assert got is None  # beyond both endpoint rates
    got = implied_length(2.1,{4:200.,5:230.},100.)
    f = got['upper_fraction']
    assert 0 < f < 1
    assert (4+f)*100/(200+30*f) == pytest.approx(2.1)
    assert implied_length(5.,{1:100.,2:150.},100.) is None


def test_recommendation_fit_engagement_acceptance_and_cost_gates(tiny):
    stats, constants = tiny
    curve = {d:dict(per_layer=[2,2]) for d in range(1,17)}
    good = recommend(stats,constants,1,[curve],1.)
    assert good['enable'] and good['draft_max'] == 16
    for kwargs in [dict(fits=False),dict(engaged_fraction=0),dict(draft_us=1e9)]:
        assert not recommend(stats,constants,1,[curve],1.,**kwargs)['enable']
    assert not recommend(stats,constants,1,[curve],0.)['enable']
    assert not recommend(stats,constants,1,[curve],1.,engaged_fraction=.01)['enable']
    other = {d:dict(per_layer=[min(4,2*d)]*2) for d in range(1,17)}
    robust = recommend(stats,constants,1,[curve,other],1.)
    assert robust['predicted_speedup'] < good['predicted_speedup']
    short = recommend(stats,constants,1,[curve],1.,feasible_lengths={2,3,4})
    assert short['enable'] and short['draft_max'] == 4
    acceptance = {d:(1. if d<=4 else .01) for d in curve}
    assert recommend(stats,constants,1,[curve],acceptance)['draft_max'] == 4
    costs = {d:(0. if d<=3 else 1e9) for d in curve}
    assert recommend(stats,constants,1,[curve],1.,draft_us=costs)['draft_max'] == 3
    assert not recommend(stats,constants,1,[curve],1.,feasible_lengths={1})['enable']


@pytest.mark.parametrize('kwargs', [dict(k=-1),dict(k=3),dict(d=0),dict(a=-.1),
    dict(a=1.1),dict(a=float('nan')),dict(unions=[1,2]),dict(unions=[2]),
    dict(unions=[5,2]),dict(draft_us=-1),dict(extra_us_per_token=float('inf'))])
def test_invalid_forecast_inputs(tiny,kwargs):
    stats, constants = tiny
    args = dict(k=1,d=2,a=.8,unions=[3,4])
    args.update(kwargs)
    with pytest.raises(ValueError):
        predict(stats,constants,**args)
