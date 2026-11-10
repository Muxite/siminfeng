"""CPU-only checks. All test scratch files belong under artifacts/codexPL4."""
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
    kw.setdefault('components', {})  # deterministic even as the factory registry lands
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
    # Independent D034 n19 median; the physical model should stay within 5%.
    assert p['decode_tps'] == pytest.approx(139.64, rel=.05)
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
    measured = run(stats, host, budget=dict(gpu_mib=900,ctx=4096,max_quality_loss=.05))['plans']
    assert measured and measured[0]['kld'] == .040017
    unknown = copy.deepcopy(stats)
    unknown['expert_bytes'] *= 2
    assert not run(unknown, host, budget=dict(gpu_mib=20000,ctx=4096,max_quality_loss=.05), allow_extrapolation=True)['plans']
    r = run(stats, host, budget=dict(gpu_mib=1500,ctx=4096,max_quality_loss=.05), top=100)
    assert r['plans'] and all(p['kld'] is not None and p['kld'] <= .05 for p in r['plans'])
    assert all(p['quality']['status'] in ('measured', 'inherited') for p in r['plans'])
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
    r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), allow_extrapolation=True, top=10000,
            components={'kv_quant_fix': {'enabled':True, 'status':'pending'}})
    quant = [p for p in r['plans'] if 'kv:q8_0' in p['techniques'] or 'kv:q4_0' in p['techniques']]
    assert quant
    assert all(p['flags'][p['flags'].index('-fa')+1] == 'on' for p in quant)
    assert all(p['confidence'] == 'extrapolated' for p in quant)
    assert all(p['kld'] is None for p in quant if p['kv_type'] == 'q4_0')
    assert any(p['kld'] == .040017 and p['quality']['status'] == 'inherited' for p in quant)
    assert any('ubatch:128' in p['techniques'] for p in r['plans'])
    assert {p['representation'] for p in r['plans']} == {'bf16','Q8_0','Q4_K_M','mixE4K','mixE3K','mixE2K'}


def test_lru_policy_uses_resident_coverage_not_guessed_hit_rate(stats, host):
    r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), top=10000)
    assert any(p['placement'] == 'lru' and '--moe-cache-mib' in p['flags'] for p in r['plans'])
    assert not any(p['placement'] == 'lru' and any(f'lru{x}-' in p['id'] for x in (128,256,384)) for p in r['plans'])
    assert any('coverage <60%' in ' '.join(p['reasons']) for p in r['rejected'])


def test_hybrid_has_real_per_layer_hot_counts_and_null_bank_cost(stats, host, cal):
    r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), allow_extrapolation=True, top=10000)
    assert not any(p['placement'] == 'hybrid' for p in r['plans'])
    assert any('D030/D031' in ' '.join(p['reasons']) for p in r['rejected'])
    # Isolate accounting under a hypothetical qualified crossover calibration.
    # Keep each real transform separately because plans now contains family winners.
    for spec in cal['hybrid_specs']:
        hypothetical = copy.deepcopy(cal)
        hypothetical['planner_v2']['hybrid_crossover']['required_margin'] = 0
        hypothetical['hybrid_specs'] = [spec]
        r = run(stats, host, budget=dict(gpu_mib=16000,ctx=4096), allow_extrapolation=True,
                top=10000, cal=hypothetical)
        hybrid = [p for p in r['plans'] if p['placement'] == 'hybrid']
        assert len(hybrid) == 1
        p = hybrid[0]
        assert len(p['hot_counts_per_layer']) == 24
        assert '--no-op-offload' in p['flags'] and '--no-repack' in p['flags']
        assert p['confidence'] == 'extrapolated' and p['kld'] is None
        assert any('CUDA correctness failure' in s for s in p['prerequisites'])
        floor = planner.predict_fit(planner.fit(planner.grouped(cal,'static'),'static'),24)['gpu_mib']
        # At extrapolated ties the winner may use a smaller ubatch.
        ub = int(p['flags'][p['flags'].index('-ub')+1])
        payload = (sum(spec['hot_counts'])+24)*.84375
        assert p['gpu_mib']-floor-(108*ub/512-108) == pytest.approx(payload, abs=.001)


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
    dest=ROOT/'artifacts/codexPL4/cli-test-definitions.json'
    argv=['--stats',str(ROOT/'artifacts/codexPL/granite-fixture.json'),'--host',str(ROOT/'cc/engine_handoff/hosts/x86-main.json'),
          '--budget-gpu-mib','900','--ctx','4096','--emit-lab-tests',str(dest)]
    assert planner.main(argv)==0
    assert json.loads(capsys.readouterr().out)['plans']
    assert planner.read_json(dest)
    argv[-1]=str(ROOT/'lab/tests/forbidden-planner.json')
    with pytest.raises(SystemExit) as e: planner.main(argv)
    assert e.value.code==2 and not (ROOT/'lab/tests/forbidden-planner.json').exists()


def test_diversity_best_per_family_and_top_is_maximum(stats, host):
    r = run(stats, host, allow_extrapolation=True, top=5)
    assert len(r['plans']) == 5
    assert len({p['technique_family'] for p in r['plans']}) == 5
    assert len({p['representation'] for p in r['plans']}) >= 4
    assert r['plans'] == r['family_best'][:5]
    # A calibrated-only singleton family must not be padded with n20..n23.
    calibrated = run(stats, host, all_representations=False)
    assert len(calibrated['plans']) == 1
    assert calibrated['plans'][0]['id'].startswith('mixE4K-static19-')
    assert len(calibrated['pareto_set']) == 6
    assert any('top is a maximum' in w for w in calibrated['warnings'])


def test_pareto_is_global_and_unknown_quality_is_not_zero():
    def p(name, gpu, speed, risk):
        return dict(id=name, gpu_mib=gpu, decode_tps=speed, quality_risk=risk)
    points = [p('dominated', 100, 10, None), p('unknown-fast', 100, 20, None),
              p('measured', 100, 20, .04), p('small', 90, 15, .08),
              p('quality', 120, 10, 0)]
    assert {r['id'] for r in planner.pareto_front(points)} == {'measured', 'small', 'quality'}
    assert {r['id'] for r in planner.pareto_front(list(reversed(points)))} == {'measured', 'small', 'quality'}


@pytest.mark.parametrize('ctx,flash', [(4096,False), (8192,True), (32768,True)])
def test_flash_attention_threshold_and_kv_bytes(stats, host, ctx, flash):
    r = run(stats, host, budget=dict(gpu_mib=16000, ctx=ctx), allow_extrapolation=True, top=10000)
    assert r['plans']
    for p in r['plans']:
        assert p['flash_attention'] is flash
        assert p['kv_mib'] == pytest.approx(ctx*stats['kv_bytes_per_token'][p['kv_type']]/planner.MIB)
        if flash:
            assert p['flags'][p['flags'].index('-fa')+1] == 'on'
            assert any('D028' in note for note in p['notes'])
            assert p['confidence'] == 'extrapolated'
            if p['kld'] is not None:
                assert 'd' in p['quality']['rules']
                assert any('no non-FA PPL' in note for note in p['quality']['caveats'])
    if ctx == 32768:
        allgpu = next(p for p in r['family_best'] if p['technique_family'].startswith('mixE4K|static|'))
        # The all-GPU D028 anchor includes its attention-buffer observation.
        assert '--n-cpu-moe' not in allgpu['flags']
        ub = int(allgpu['flags'][allgpu['flags'].index('-ub')+1])
        assert allgpu['gpu_mib'] == pytest.approx(2762+108*ub/512-108, abs=.01)
        assert allgpu['prefill_tps'] == pytest.approx(7001, abs=.001)


def test_long_context_quant_kv_requires_enabled_fix(stats, host):
    budget = dict(gpu_mib=1200, ctx=32768)
    assert not run(stats, host, budget=budget, allow_extrapolation=True)['plans']
    pending = {'components': {'kv_quant_fix': {'status':'pending'}}}
    assert not run(stats, host, budget=budget, allow_extrapolation=True, components=pending)['plans']
    pending['components']['kv_quant_fix']['enabled'] = True
    r = run(stats, host, budget=budget, allow_extrapolation=True, components=pending)
    assert r['plans'] and len({p['technique_family'] for p in r['plans']}) == len(r['plans'])
    for p in r['plans']:
        assert p['kv_type'] in ('q8_0','q4_0')
        assert p['kv_mib'] == pytest.approx(32768*stats['kv_bytes_per_token'][p['kv_type']]/planner.MIB)
        assert p['gpu_mib']+p['gpu_guard_mib'] <= 1200
        assert 'kv_quant_fix' in p['required_engine']
        assert p['kld'] is None and p['quality_risk'] is None
        assert 'kv_quant_fix' in p['components']
    strict = dict(budget, max_quality_loss=.08)
    assert not run(stats, host, budget=strict, allow_extrapolation=True, components=pending)['plans']


def test_components_absence_registry_formats_and_fail_closed(tmp_path, monkeypatch):
    # Missing default is tolerated, but a misspelled explicit path is an error.
    monkeypatch.setattr(planner, 'ROOT', tmp_path)
    assert planner.load_components() == {}
    with pytest.raises(FileNotFoundError):
        planner.load_components(tmp_path/'missing.json')
    registry = tmp_path/'components.json'
    registry.write_text(json.dumps({'schema':'factory-components/0', 'components':[
        dict(id='q8_reuse', status='accepted'), dict(id='kv_quant_fix', status='pending'),
        dict(id='hybrid_split',status='rejected-small-expert-regime'),
        dict(id='pap',status='pending',enabled=False)]}))
    assert set(planner.load_components(registry)) == {'q8_reuse'}
    assert set(planner.load_components(['q8_reuse', 'kv_quant_fix'])) == {'q8_reuse','kv_quant_fix'}
    with pytest.raises(ValueError, match='boolean'):
        planner.load_components({'q8_reuse': {'enabled':'false'}})


def test_q8_reuse_multiplies_only_decode_and_emits_env(stats, host):
    kw = dict(budget=dict(gpu_mib=900,ctx=4096), allow_extrapolation=True, all_representations=False)
    base = run(stats, host, **kw)['plans'][0]
    patched_result = run(stats, host, components={'q8_reuse': {'status':'accepted'}}, **kw)
    patched = patched_result['plans'][0]
    assert patched['decode_tps'] == pytest.approx(base['decode_tps']*1.02, abs=.001)
    assert patched['prefill_tps'] == base['prefill_tps']
    assert patched['prefill_ms'] == base['prefill_ms']
    assert patched['gpu_mib'] == base['gpu_mib'] and patched['host_mib'] == base['host_mib']
    assert patched['ttft_ms'] == pytest.approx(patched['prefill_ms']+1000/patched['decode_tps'], abs=.001)
    assert patched['runtime_env'] == {'GGML_CUDA_Q8_REUSE':'1'}
    assert 'q8_reuse' in patched['technique_family'] and patched['id'] != base['id']
    for definition in planner.emit_lab_tests(patched_result).values():
        assert 'GGML_CUDA_Q8_REUSE=1' in definition['docker_args']


def test_prefill_is_independent_of_decode_and_scales_prompt(stats, host, cal):
    a = planner.prefill_prediction(cal, 'static', 12, 62, 4096, False)
    modified = copy.deepcopy(cal)
    for row in modified['points']:
        row['decode_tps'] *= 100
    assert planner.prefill_prediction(modified, 'static',12,62,4096,False) == a
    doubled = planner.prefill_prediction(cal,'static',12,124,4096,False)
    assert doubled['prefill_tps'] == a['prefill_tps']
    assert doubled['prefill_ms'] == a['prefill_ms']*2
    off = planner.prefill_prediction(cal,'static',0,1560,32768,False)
    on = planner.prefill_prediction(cal,'static',0,1560,32768,True)
    assert off['prefill_tps'] == pytest.approx(5865)
    assert on['prefill_tps'] == pytest.approx(7001)
    assert on['prefill_ms'] == pytest.approx(1560/7001*1000)


def test_crossover_reports_measured_fixed_cost_and_large_expert_condition(stats, cal):
    c = planner.crossover_model(planner.adapt_stats(stats), cal)
    assert c['per_layer_fixed_us'] == pytest.approx(2030/12)
    assert c['crossover_expert_mib'] == pytest.approx(4.664522)
    assert c['expert_bytes']/planner.MIB == .84375
    assert not c['predicted_beneficial']
    large = copy.deepcopy(stats)
    large['expert_bytes'] *= 20
    assert planner.crossover_model(planner.adapt_stats(large), cal)['predicted_beneficial']
    pap = cal['planner_v2']['techniques']['pap']
    assert pap['calibration_points'] == [] and pap['status'] == 'pending-calibration'
    assert {'job_id','test','decode_tps','prefill_tps','kld','cpu_layers'} <= set(pap['point_fields'])


def test_cli_components_read_and_old_write_location_rejected(tmp_path, capsys):
    registry = tmp_path/'components.json'
    registry.write_text(json.dumps({'components':{'kv_quant_fix':{'enabled':True,'status':'pending'}}}))
    args = ['--stats',str(ROOT/'artifacts/codexPL/granite-inspected.json'),
            '--host',str(ROOT/'cc/engine_handoff/hosts/x86-main.json'),
            '--budget-gpu-mib','1200','--ctx','32768','--allow-extrapolation','--components',str(registry)]
    assert planner.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['plans'] and result['enabled_components'] == ['kv_quant_fix']
    with pytest.raises(SystemExit):
        planner.main(args+['--emit-lab-tests',str(ROOT/'artifacts/codexPL/forbidden-pl2.json')])
    assert not (ROOT/'artifacts/codexPL/forbidden-pl2.json').exists()
