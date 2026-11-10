"""Evidence rules and fail-closed regression checks; CPU only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_plan import planner, stats, host, run

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('builder', HERE/'build_quality_ledger.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def ledger():
    return planner.load_quality_ledger()


def resolve(ledger, **kw):
    defaults = dict(rep='mixE4K', family='static', coord=18, kv='f16', flash=False, components=[])
    defaults.update(kw)
    return planner.resolve_quality(ledger, **defaults)


def test_ledger_is_reproducible_and_metrics_match_hashed_results(ledger):
    assert builder.build() == ledger
    assert len(ledger['entries']) == 64
    for row in ledger['entries']:
        source = row['source']
        path = planner.ROOT/source['result_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['result_sha256']
        raw = json.loads(path.read_text())['quality']['tests'][source['test']]
        field = 'top1_pct' if row['metric'] == 'top1' else row['metric']
        assert row['value'] == raw[field]
        assert row['n_chunks'] > 0
    assert {r['representation'] for r in ledger['entries'] if r['source']['decision'] == 'D035'} >= {'pap12_q3', 'mix4_imat'}
    assert ledger['binary_evidence']['decision'] == 'D036'
    assert ledger['binary_evidence']['job_id'] is None
    assert not ledger['binary_evidence']['paired_ppl']


def test_measured_quality_and_exact_precision(ledger):
    q = resolve(ledger, coord=19)
    assert q['status'] == 'measured' and q['kld'] == .040017
    assert any('D034' in e for e in q['evidence'])
    assert resolve(ledger, rep='Q4_K_M', coord=0)['kld'] == .077526


def test_rule_a_requires_text_and_ppl_identity_for_the_representation(ledger):
    q = resolve(ledger, components=['q8_reuse'])
    assert q['kld'] == .040017 and 'a' in q['rules']
    assert any('D029' in e for e in q['evidence'])
    for flag in ('text_identity', 'ppl_identity'):
        bad = copy.deepcopy(ledger)
        for proof in bad['computation_preserving']:
            proof[flag] = False
        assert resolve(bad, components=['q8_reuse'])['kld'] is None
    assert resolve(ledger, rep='Q4_K_M', coord=0, components=['q8_reuse'])['kld'] is None
    assert resolve(ledger, components=['unknown_patch'])['kld'] is None


def test_rule_b_requires_multiple_equal_placements_of_same_representation(ledger):
    q = resolve(ledger)
    assert q['status'] == 'inherited' and q['kld'] == .040017
    assert 'b' in q['rules'] and any('D034' in e for e in q['evidence'])
    assert resolve(ledger, coord=24)['kld'] == .040017
    single = copy.deepcopy(ledger)
    single['entries'] = [r for r in ledger['entries'] if r['source']['decision'] != 'D034']
    assert resolve(single)['kld'] is None
    conflicting = copy.deepcopy(ledger)
    next(r for r in conflicting['entries'] if r['source']['decision'] == 'D034' and r['metric'] == 'kld')['value'] = .1
    assert resolve(conflicting)['kld'] is None
    assert resolve(ledger, rep='Q8_0')['kld'] is None
    assert resolve(ledger, family='lru', coord=640)['kld'] is None
    assert resolve(ledger, family='hybrid')['kld'] is None


def test_rule_c_q8_inherits_by_ppl_policy_and_q4_fails(ledger):
    q = resolve(ledger, kv='q8_0', flash=True, components=['kv_quant_fix'])
    assert q['kld'] == .040017 and 'c' in q['rules']
    assert q['ppl_status'] == 'within-budget'
    assert q['ppl_relative_change'] == pytest.approx(7.5392/7.5385-1)
    assert any('D033' in e for e in q['evidence'])
    q4 = resolve(ledger, kv='q4_0', flash=True, components=['kv_quant_fix'])
    assert q4['kld'] is None and q4['status'] == 'quality-failing'
    assert q4['ppl_relative_change'] > .079
    assert resolve(ledger, kv='q8_0', flash=True)['kld'] is None
    assert resolve(ledger, kv='q4_0', flash=True, components=['kv_quant_fix'], max_ppl_change=.1)['kld'] is None
    assert resolve(ledger, rep='Q8_0', coord=0, kv='q8_0', flash=True, components=['kv_quant_fix'])['kld'] is None
    strict = resolve(ledger, kv='q8_0', components=['kv_quant_fix'], max_ppl_change=0)
    assert strict['status'] == 'quality-failing' and strict['kld'] is None


def test_rule_d_fa_only_ppl_is_a_caveat_not_a_block(ledger):
    q = resolve(ledger, flash=True, ctx=32768)
    assert q['kld'] == .040017 and 'd' in q['rules']
    assert any('PPL measured with FA only' in c for c in q['caveats'])
    assert any('context transfer is unvalidated' in c for c in q['caveats'])
    assert resolve(ledger, flash=False)['kld'] == q['kld']


@pytest.mark.parametrize('changes', [dict(rep='unknown'), dict(matched=False), dict(coord=-1), dict(coord=25), dict(kv='unknown'), dict(components=['unknown'])])
def test_unknown_combinations_fail_closed(ledger, changes):
    assert resolve(ledger, **changes)['kld'] is None


def test_empty_ledger_cannot_use_calibration_priors(stats, host, ledger):
    ledger['entries'] = []
    assert not run(stats, host, quality_ledger=ledger, allow_extrapolation=True,
                   budget=dict(gpu_mib=20000, ctx=4096, max_quality_loss=.08))['plans']


def test_factory_900_gate_and_no_q4_under_strict_budget(stats, host):
    r = run(stats, host, components={'q8_reuse': True, 'kv_quant_fix': True},
            budget=dict(gpu_mib=900, ctx=4096, max_quality_loss=.08))
    assert r['plans'] and r['plans'][0]['id'].startswith('mixE4K-static19-f16-')
    assert 'a' in r['plans'][0]['quality']['rules']
    long = run(stats, host, components={'q8_reuse': True, 'kv_quant_fix': True}, allow_extrapolation=True,
               budget=dict(gpu_mib=1400, ctx=32768, max_quality_loss=.08), top=1000)
    assert long['plans'] and all(p['kv_type'] == 'q8_0' for p in long['plans'])
    assert all(p['kld'] == .040017 for p in long['plans'])
    assert any(p.get('quality', {}).get('status') == 'quality-failing' for p in long['rejected'])


def test_1200_32k_memory_limit_is_independent_of_quality(stats, host):
    kw = dict(components={'q8_reuse': True, 'kv_quant_fix': True}, allow_extrapolation=True)
    r = run(stats, host, budget=dict(gpu_mib=1200, ctx=32768, max_quality_loss=.08), **kw)
    assert not r['plans']
    r = run(stats, host, budget=dict(gpu_mib=1400, ctx=32768, max_quality_loss=.08), **kw)
    smallest = min(r['pareto_set'], key=lambda p: p['gpu_mib']+p['gpu_guard_mib'])
    assert smallest['gpu_mib']+smallest['gpu_guard_mib'] == pytest.approx(1316.605, abs=.02)
