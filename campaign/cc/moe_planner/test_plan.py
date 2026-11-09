"""CPU-only checks. All test scratch files belong under artifacts/codexPL."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('planner', HERE/'plan.py')
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


@pytest.fixture
def cal():
    return planner.read_json(HERE/'calibration.json')


@pytest.fixture
def stats():
    return planner.read_json(ROOT/'artifacts/codexPL/granite-fixture.json')


@pytest.fixture
def host():
    return planner.read_json(ROOT/'cc/engine_handoff/hosts/x86-main.json')


def run(stats, host, **kw):
    budget = kw.pop('budget', dict(gpu_mib=900, ctx=4096))
    return planner.plan(stats, host, budget, **kw)


def test_d022_three_independent_samples_and_raw_provenance(cal):
    raw = planner.read_json(ROOT/'lab/jobs/job-ad883810b3/att-1/result.json')['quality']['tests']
    assert len(cal['points']) == 27
    for family, xs in [('static', [0,6,12,18,24]), ('lru', [128,256,384,640])]:
        for x in xs:
            rows = [p for p in cal['points'] if p['family'] == family and p['x'] == x]
            assert len(rows) == 3
            for p in rows:
                r = raw[p['test']]
                assert p['job_id'] == 'job-ad883810b3' and 'D022' in p['source']
                assert p['gpu_mib'] == r['mem']['gpu_delta_mib']
                assert p['decode_tps'] == r['decode_tps']
    # Independent, frozen rounded decision numbers, not snapshots of the fitter.
    assert [round(p['decode_tps']) for p in planner.grouped(cal, 'static')] == [684,323,197,152,128]
    assert [round(p['gpu_mib']) for p in planner.grouped(cal, 'static')] == [1388,1226,1064,902,742]


@pytest.mark.parametrize('family,memory_bound,speed_bound', [('static',2.001,16.25), ('lru',.001,39.27)])
def test_leave_one_coordinate_out_reproduces_d022(cal, family, memory_bound, speed_bound):
    rows = planner.grouped(cal, family)
    for i, held in enumerate(rows):
        trained = planner.fit(rows[:i]+rows[i+1:], family)
        predicted = planner.predict_fit(trained, held['x'])
        assert abs(predicted['gpu_mib']-held['gpu_mib']) <= memory_bound
        assert abs(100*(predicted['decode_tps']/held['decode_tps']-1)) <= speed_bound
    report = planner.residual_report(cal)[family]
    assert report['max_loo_gpu_mib'] <= memory_bound
    assert report['max_loo_decode_pct'] <= speed_bound


def test_900_budget_selects_n19_and_accounts_for_overhead(stats, host):
    result = run(stats, host)
    p = result['plans'][0]
    assert p['id'].startswith('mixE4K-static19-')
    assert p['gpu_mib'] == pytest.approx(875.872, abs=.01)
    assert p['decode_tps'] == pytest.approx(149.633, abs=.001)
    assert p['host_mib'] == pytest.approx(1178.7, abs=.01)
    assert all(p['gpu_mib']+p['gpu_guard_mib'] <= 900 for p in result['plans'])
    assert all(p['confidence'] == 'calibrated' for p in result['plans'])
    assert all(p['metric_confidence']['host_mib'] == 'extrapolated' for p in result['plans'])
    assert '--fit' in p['flags'] and p['flags'][p['flags'].index('--fit')+1] == 'off'


def test_budget_guard_and_inclusive_host_boundary(stats, host):
    first = run(stats, host, top=100)['plans'][0]
    # Two MiB residual guard prevents calling this marginal point feasible.
    r = run(stats, host, budget=dict(gpu_mib=876, ctx=4096), top=100)
    assert all('-static19-' not in p['id'] for p in r['plans'])
    assert not run(stats, host, budget=dict(gpu_mib=900,host_mib=1000,ctx=4096))['plans']
    r = run(stats, host, budget=dict(gpu_mib=900,host_mib=1178.701,ctx=4096))
    assert r['plans']
    assert not run(stats, host, budget=dict(gpu_mib=600,ctx=4096))['plans']


def test_strict_kld_unknown_never_passes(stats, host):
    assert not run(stats, host, budget=dict(gpu_mib=900,ctx=4096,max_quality_loss=.05))['plans']
    r = run(stats, host, budget=dict(gpu_mib=1500,ctx=4096,max_quality_loss=.05), top=100)
    assert r['plans'] and all(p['kld'] is not None and p['kld'] <= .05 for p in r['plans'])
    assert all('-static0-f16-' in p['id'] for p in r['plans'])
    assert not run(stats, host, budget=dict(gpu_mib=1500,ctx=4096,max_quality_loss=0))['plans']


@pytest.mark.parametrize('change', ['ctx','host','model','kv'])
def test_no_silent_extrapolation(stats, host, change):
    budget = dict(gpu_mib=20000,ctx=4096)
    if change == 'ctx':
        budget['ctx'] = 8192
    elif change == 'host':
        host['memory_model'] = 'unified'
    elif change == 'model':
        stats['expert_bytes'] *= 2
    else:
        stats['kv_bytes_per_token'] = {'f16':49153,'q8_0':26112}
    assert not run(stats, host, budget=budget)['plans']
    r = run(stats, host, budget=budget, allow_extrapolation=True)
    assert r['plans'] and all(p['confidence'] == 'extrapolated' for p in r['plans'])


def test_quant_kv_flash_attention_and_batch_candidates(stats, host):
    r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), allow_extrapolation=True, top=10000)
    quant = [p for p in r['plans'] if 'kv:q8_0' in p['techniques'] or 'kv:q4_0' in p['techniques']]
    assert quant
    assert all(p['flags'][p['flags'].index('-fa')+1] == 'on' for p in quant)
    assert all(p['confidence'] == 'extrapolated' and p['kld'] is None for p in quant)
    assert any('ubatch:128' in p['techniques'] for p in r['plans'])
    assert {p['representation'] for p in r['plans']} == {'bf16','Q8_0','Q4_K_M','mixE4K','mixE3K','mixE2K'}


def test_lru_policy_uses_resident_coverage_not_guessed_hit_rate(stats, host):
    r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), top=10000)
    assert any(p['placement'] == 'lru' and '--moe-cache-mib' in p['flags'] for p in r['plans'])
    assert not any(p['placement'] == 'lru' and any(f'lru{x}-' in p['id'] for x in (128,256,384)) for p in r['plans'])
    assert any('coverage <60%' in ' '.join(p['reasons']) for p in r['rejected'])


def test_hybrid_has_real_per_layer_hot_counts_and_null_bank_cost(stats, host, cal):
    r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), allow_extrapolation=True, top=10000)
    hybrid = [p for p in r['plans'] if p['placement'] == 'hybrid' and '-f16-' in p['id'] and '-b2048-u512' in p['id']]
    assert len(hybrid) == 3
    for p in hybrid:
        assert len(p['hot_counts_per_layer']) == 24
        assert '--no-op-offload' in p['flags'] and '--no-repack' in p['flags']
        assert p['confidence'] == 'extrapolated' and p['kld'] is None
        assert any('CUDA correctness failure' in s for s in p['prerequisites'])
    smallest = min(hybrid, key=lambda p:p['gpu_mib'])
    # 168 real + 24 null expert slots = 162 MiB; not 141.75 MiB.
    floor = planner.predict_fit(planner.fit(planner.grouped(cal,'static'),'static'),24)['gpu_mib']
    assert smallest['gpu_mib']-floor == pytest.approx(162, abs=.001)


def test_fixed_hot_ids_applied_to_decode_not_reranked(stats, cal):
    spec = cal['hybrid_specs'][0]
    stats['routing_trace'] = {'groups':[dict(split='eval',phase='decode',layers={
        str(l):dict(frequency=[1 if e in spec['hot_ids'][l] else 0 for e in range(32)]) for l in range(24)})]}
    hit, note = planner.hybrid_hit_rate(stats,spec)
    assert hit == 1 and 'split=eval' in note


def test_resource_sample_missing_is_not_zero(stats, host):
    sample = dict(schema='resource-sample/0',fields={})
    r = run(stats, host, sample=sample)
    assert r['plans'] and len(r['warnings']) >= 7
    sample['fields'] = {'gpu.memory.total':dict(status='ok',value=1000,unit='MiB'),
                        'gpu.memory.used':dict(status='ok',value=500,unit='MiB'),
                        'host.mem.available':dict(status='ok',value=3000*planner.MIB,unit='B')}
    assert not run(stats, host, sample=sample)['plans']
    sample['fields']['gpu.memory.used']['unit']='B'
    with pytest.raises(ValueError,match='unit'):
        run(stats,host,sample=sample)


@pytest.mark.parametrize('objective', ['decode','ttft','balanced'])
def test_objectives_are_deterministic(stats, host, objective):
    r = run(stats,host,budget=dict(gpu_mib=1500,ctx=4096,objective=objective),top=100)
    plans = r['plans']
    scores = [p['decode_tps'] if objective == 'decode' else -p['ttft_ms'] if objective == 'ttft' else p['decode_tps']/p['ttft_ms'] for p in plans]
    assert scores == sorted(scores,reverse=True)


@pytest.mark.parametrize('key,value', [('ctx',0),('ctx',131073),('gpu_mib',float('nan')),('host_mib',-1),('max_quality_loss',-.1)])
def test_invalid_budget_fails_closed(stats,host,key,value):
    budget=dict(gpu_mib=900,ctx=4096);budget[key]=value
    with pytest.raises(ValueError): run(stats,host,budget=budget)


def test_real_inspector_contract(stats, host):
    # The parallel T-MI producer has landed; read its artifact without changing it.
    raw = planner.read_json(ROOT/'artifacts/codexPL/granite-inspected.json')
    s = planner.adapt_stats(raw)
    assert s['expert_bytes'] == 648*planner.MIB
    assert s['kv_bytes_per_token']['q8_0'] == 26112
    assert run(raw,host)['plans'][0]['id'] == run(stats,host)['plans'][0]['id']


def test_totals_adapter_and_unknown_role_accounted(stats):
    raw = copy.deepcopy(stats)
    del raw['expert_bytes'],raw['nonexpert_bytes']
    raw['totals']={'by_role':{'expert_ffn':648*planner.MIB,'other':100,'shared_ffn':200}}
    s=planner.adapt_stats(raw)
    assert s['nonexpert_bytes'] == 300
    del raw['kv_bytes_per_token']
    with pytest.raises(ValueError,match='kv_bytes'): planner.adapt_stats(raw)


def test_emission_matches_real_lab_loader_and_schema(stats,host,tmp_path):
    result=run(stats,host)
    definitions=planner.emit_lab_tests(result)
    reference=planner.read_json(ROOT/'lab/tests/match1.json')
    # bin/lab has no JSON Schema: validate actual consumer-required fields and types.
    allowed=set().union(*(set(d) for f in (ROOT/'lab/tests').glob('*.json') for d in planner.read_json(f).values()))
    for td in definitions.values():
        assert set(td) <= allowed
        assert td['tool'] in ('llama-completion','llama-perplexity')
        assert td['parser'] in ('completion','kl')
        assert isinstance(td['args'],list) and all(isinstance(v,str) for v in td['args'])
        assert Path(td['model_file']).name == td['model_file']
        assert '-m' not in td['args']  # lab injects model_file exactly once
        assert td['args'][td['args'].index('--fit')+1] == 'off'
        if td['parser']=='kl':
            assert td['needs_data'] is True
            assert td['args'][td['args'].index('-c')+1] == '512'
        else:
            assert set(reference['m1_static_n24']) == set(td)
    # Run the real load_tests function in isolation; do not import the CLI or run Docker.
    (tmp_path/'tests').mkdir()
    (tmp_path/'tests/planner.json').write_text(json.dumps(definitions))
    tree=ast.parse((ROOT/'bin/lab').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_tests')
    namespace=dict(json=json,LAB=str(tmp_path),os=__import__('os'))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'bin/lab','exec'),namespace)
    assert namespace['load_tests']() == definitions


def test_cli_output_and_write_boundary(stats,host,capsys):
    dest=ROOT/'artifacts/codexPL/cli-test-definitions.json'
    argv=['--stats',str(ROOT/'artifacts/codexPL/granite-fixture.json'),'--host',str(ROOT/'cc/engine_handoff/hosts/x86-main.json'),
          '--budget-gpu-mib','900','--ctx','4096','--emit-lab-tests',str(dest)]
    assert planner.main(argv)==0
    assert json.loads(capsys.readouterr().out)['plans']
    assert planner.read_json(dest)
    argv[-1]=str(ROOT/'lab/tests/forbidden-planner.json')
    with pytest.raises(SystemExit) as e: planner.main(argv)
    assert e.value.code==2 and not (ROOT/'lab/tests/forbidden-planner.json').exists()
