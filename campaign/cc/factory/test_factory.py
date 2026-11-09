"""CPU-only integration checks. Use the README basetemp to keep all writes scoped."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('factory', HERE/'factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)
MODEL = Path('/home/muk/models/campaign/gguf/granite-3.1-1b-a400m-instruct-mixE4K.gguf')
REAL_RESULT = ROOT/'lab/jobs/job-ad883810b3/att-1/result.json'


def save(path, value):
    factory.write(path, value)
    return path


def cli(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    return subprocess.run([sys.executable, '-B', str(HERE/'factory.py'), *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.skipif(not MODEL.is_file(), reason='real Granite mixE4K GGUF unavailable')
def test_real_full_dry_run_resume_and_output_integrity(tmp_path):
    run = tmp_path/'full'
    args = ['run', '--model', MODEL, '--host', 'x86-main', '--budget-gpu-mib', '900',
            '--ctx', '4096', '--dry-run', '--run-dir', run, '--backends', 'linux']
    first = cli(*args)
    assert first.returncode == 0, first.stderr
    assert [line.split(':')[0] for line in first.stdout.splitlines() if ': wrote ' in line] == list(factory.STAGES)
    before = {s: (run/(s+'.json')).stat().st_mtime_ns for s in factory.STAGES}
    for stage in factory.STAGES:
        record = factory.read(run/(stage+'.json'))
        assert record['inputs_sha256'] == factory.digest(record['inputs'])
        assert record['outputs']['data_sha256'] == factory.digest(record['data'])
        assert all(factory.sha(p) == h for p, h in record['outputs']['files'].items())
    recipe = factory.read(run/'recipe.json')['data']
    assert recipe['recipes'], '900 MiB should admit a static placement hypothesis'
    assert all(c['status'] == 'accepted' for c in recipe['components'])
    assert [p['component'] for p in recipe['recipes'][0]['engine']['ordered_patches']] == ['q8_reuse']
    assert recipe['recipes'][0]['model']['sha256'] == factory.sha(MODEL)
    assert recipe['recipes'][0]['runtime_argv'][1:3] == ['-m', '/m/' + MODEL.name]
    assert recipe['recipes'][0]['runtime_mounts'][0]['access'] == 'ro'
    assert 'DRY-RUN env LAB=' in first.stdout
    assert not list((run/'lab/jobs').glob('*'))
    assert not (run/'builds').exists()
    for r in factory.read(run/'report.json')['data']['rows']:
        assert r['claim_category'] == 'prediction' and r['quality_accepted'] is None
    resumed = cli(*args)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.count('skipped (inputs and outputs unchanged)') == 7
    assert before == {s: (run/(s+'.json')).stat().st_mtime_ns for s in factory.STAGES}
    # Resume verifies emitted outputs as well as the JSON payload.
    (run/'plan-tests.json').write_text('{}\n')
    repaired = cli(*args, '--stages', 'plan')
    assert repaired.returncode == 0, repaired.stderr
    assert 'plan: wrote' in repaired.stdout
    assert factory.read(run/'plan-tests.json')
    # A changed quality constraint invalidates plan and all affected downstream stages.
    constrained = cli(*args, '--max-kld', '0.05')
    assert constrained.returncode == 0, constrained.stderr
    assert 'inspect: skipped' in constrained.stdout and 'system: skipped' in constrained.stdout
    assert factory.read(run/'plan.json')['data']['plans'] == []
    assert factory.read(run/'build.json')['data']['status'] == 'no-feasible-plans'


def test_component_gates_and_environment():
    registry = factory.read(HERE/'components.json')
    default = factory.allowed_components(registry)
    pending = factory.allowed_components(registry, True)
    assert {c['id'] for c in default} == {'q8_reuse', 'gpu_budget', 'flash-attn'}
    assert {c['id'] for c in pending} == {'q8_reuse', 'gpu_budget', 'flash-attn', 'kv_quant_fix', 'pap'}
    assert all(c['status'] != 'rejected' for c in pending)
    td = factory.lab_definition(dict(docker_args=['-e', 'GGML_CUDA_Q8_REUSE=1',
                 '-v', '/host:/tin:ro'], args=['-f', '/tin/chat.txt']), {'GGML_CUDA_Q8_REUSE': '1'})
    assert td['env'] == {'GGML_CUDA_Q8_REUSE': '1'}
    assert td['docker_args'] == ['-v', '/host:/tin:ro']
    with pytest.raises(ValueError, match='conflict'):
        factory.lab_definition(dict(docker_args=['-e', 'X=0']), {'X': '1'})


def test_real_result_join_explicit_occurrences_and_error_flags(tmp_path):
    assert REAL_RESULT.is_file()
    predictions = save(tmp_path/'predictions.json', {'plans': [dict(
        id='fabricated-n18-prediction', placement='static', gpu_mib=800,
        gpu_guard_mib=2, host_mib=1200, decode_tps=10, prefill_tps=10,
        ttft_ms=100, kld=None, flags=['--n-cpu-moe', '18'],
        test_names=['m1_static_n18'])]})
    out = tmp_path/'joined.json'
    result = cli('report', '--predictions', predictions, '--result', REAL_RESULT, '--out', out)
    assert result.returncode == 0, result.stderr
    data = factory.read(out)['data']
    assert data['result_envelopes'][0]['format'] == 'campaign-job-result/0'
    assert data['result_envelopes'][0]['evidence_completeness']['clean'] == 27
    rows = data['rows']
    assert len(rows) == 3  # retain independent rounds, never join/average them
    assert len(data['unmatched']) == 24
    assert all(r['clean'] and r['job'] == 'job-ad883810b3' and r['attempt'] == 'att-1' for r in rows)
    for r in rows:
        assert r['measured']['gpu_delta_mib'] == 902
        assert r['measured']['gpu_peak_mib'] == 933
        assert r['measured']['ttft_ms'] is None
        assert r['measured']['warm_ttft_proxy_ms'] > 200
        assert r['measured']['kld'] is None and r['quality_accepted'] is None
        assert r['prediction_errors']['gpu_mib']['flagged']
        assert r['prediction_errors']['decode_tps']['flagged']
        assert r['claim_category'] == 'runtime-placement/offload trade-off'
        assert r['transfer_class'] == 'discrete-only'
        assert r['inputs_bound'] is False  # this historical job lacks full input identities
        assert 'host_mib' not in r['prediction_errors']
    zero_error = factory.report_join({'plans': [dict(id='exact', placement='static',
        gpu_mib=902, gpu_guard_mib=2, decode_tps=141.05,
        test_names=['m1_static_n18'])]}, [REAL_RESULT])
    first = next(r for r in zero_error['rows'] if r['test'] == 'm1_static_n18#1')
    assert not first['prediction_error']


def test_incomplete_results_never_supply_metrics(tmp_path):
    prediction = {'plans': [dict(id='partial', placement='static', gpu_mib=12,
                                 kld=0, test_names=['t'])]}
    for execution, partial, ok in [('interrupted', False, True), ('completed', True, True),
                                    ('completed', False, False)]:
        result = save(tmp_path/'partial.json', dict(job='j', attempt='att-1',
            execution={'status': execution}, quality={'tests': {'t#0': dict(
                exit_code=0, partial=partial, ok=ok, kld=0, mem={'gpu_delta_mib': 12})}}))
        row = factory.report_join(prediction, [result])['rows'][0]
        assert not row['clean'] and row['measured'] == {}
        assert row['claim_category'] == 'failed candidate'
        assert row['quality_accepted'] is None
    ambiguous = {'plans': [dict(id='a', test_names=['t']), dict(id='b', test_names=['t'])]}
    with pytest.raises(ValueError, match='ambiguous'):
        factory.report_join(ambiguous, [])


def test_writes_cannot_escape_assignment_roots(tmp_path):
    with pytest.raises(ValueError, match='outside assignment'):
        factory.writable(ROOT/'lab/tests/factory.json')
    with pytest.raises(ValueError, match='outside assignment'):
        factory.writable(ROOT/'artifacts/factory/run/inspect.json')
    link = tmp_path/'escape'
    link.symlink_to(ROOT/'lab', target_is_directory=True)
    with pytest.raises(ValueError, match='outside assignment'):
        factory.write(link/'tests/new.json', {})


def test_real_lab_submission_with_embedded_definitions_no_docker(tmp_path):
    # Submission hashes tiny real files and runs the existing lab CLI. No execution is requested.
    model = tmp_path/'model.gguf'
    model.write_bytes(b'test payload; no inference')
    data = tmp_path/'wiki.test.raw'
    data.write_text('test data')
    run = tmp_path/'submit'
    record_path = save(run/'build-record.json', {'status': 'success',
        'image': {'config_digest': 'sha256:' + '1'*64}})
    defs = {f'pl_t_{phase}': dict(tool='llama-completion' if phase != 'quality512' else 'llama-perplexity',
            parser='completion' if phase != 'quality512' else 'kl',
            model_file=model.name, docker_args=[], args=['-c', '512'],
            env={'GGML_CUDA_Q8_REUSE': '1'}, **({'needs_data': True} if phase == 'quality512' else {}))
            for phase in ('engagement', 'timing', 'quality512')}
    recipe = dict(plan_id='tiny', model={'path': str(model), 'sha256': factory.sha(model)}, tests=defs)
    save(run/'recipe.json', {'data': {'recipes': [recipe]}})
    save(run/'build.json', {'data': {'builds': [{'record': str(record_path)}]}})
    args = argparse_args(run, model, data)
    f = factory.Factory(args)
    payload, files = f.do_measure()
    assert len(payload['jobs']) == 3
    assert all(job['status'] == 'submitted-not-run' for job in payload['jobs'])
    for item in payload['jobs']:
        job = factory.read(run/'lab/jobs'/item['job']/'job.json')
        assert job['spec']['inputs']['model_files'][model.name] == factory.sha(model)
        assert job['spec']['test_definitions']
        assert len(job['spec']['tests']) == (3 if item['phase'] == 'timing' else 1)
        assert not list((run/'lab/jobs'/item['job']).glob('att-*'))
    assert f.do_measure()[0] == payload  # immutable, idempotent lab submission


def argparse_args(run, model, data=None):
    from argparse import Namespace
    return Namespace(run_dir=str(run), model=str(model), host='x86-main',
        components=str(HERE/'components.json'), data=str(data) if data else None,
        dry_run=False, include_pending=False, ctx=4096)


def test_pending_patch_bundle_and_pap_proposal(tmp_path):
    model = tmp_path/'granite-3.1-1b-a400m-instruct-mixE4K.gguf'
    model.write_bytes(b'model')
    run = tmp_path/'pending'
    p = dict(id='mix-static18', representation='mixE4K', model_path=str(model),
             flags=['-m', str(model), '--n-cpu-moe', '18'], prerequisites=[])
    tests = save(run/'tests.json', {factory.test_prefix(p)+'_timing': dict(
        model_file=model.name, args=p['flags'][2:], docker_args=[])})
    save(run/'plan.json', {'data': {'plans': [p], 'emitted_tests': str(tests)}})
    args = argparse_args(run, model)
    args.include_pending = True
    payload, outputs = factory.Factory(args).do_recipe()
    patches = payload['recipes'][0]['engine']['ordered_patches']
    assert [p['component'] for p in patches] == ['q8_reuse', 'kv_quant_fix']
    assert outputs[0].read_bytes() == b''.join(Path(p['path']).read_bytes().rstrip(b'\n')+b'\n' for p in patches)
    assert len(payload['transform_alternatives']) == 2
    proposal = payload['transform_alternatives'][0]
    assert proposal['input'].endswith('bf16.gguf')
    assert proposal['selected'] is False and proposal['output_sha256'] is None
    argv = proposal['argv']
    assert argv[argv.index('--tensor-type')+1].endswith('=q2_K')
    assert argv.index('ffn_.*_exps=q4_K') > argv.index('--tensor-type')


def test_build_calls_existing_build_cc_and_forwards_variant_arguments(tmp_path, monkeypatch):
    model = tmp_path/'m.gguf'
    model.write_bytes(b'model')
    run = tmp_path/'build'
    engine = dict(base_commit=factory.read(HERE/'components.json')['base_commit'],
                  ordered_patches=[], patch_bundle=None, cmake_defines=['GGML_CUDA_NO_VMM=ON'],
                  extra_targets=['test-backend-ops'], linkage=True, env={})
    save(run/'recipe.json', {'data': {'recipes': [{'plan_id': 'p', 'engine': engine}]}})
    f = factory.Factory(argparse_args(run, model))
    original = f.command
    calls = []
    def invoke(argv, env=None):
        if str(ROOT/'cc/build_engine/build_engine.py') not in argv:
            return original(argv, env)
        calls.append(argv)
        out = Path(argv[argv.index('--out')+1])
        save(out/'build-record.json', {'status': 'success', 'image': {'config_digest': 'sha256:'+'1'*64}})
        return ''
    monkeypatch.setattr(f, 'command', invoke)
    built, files = f.do_build()
    assert built['builds'][0]['status'] == 'built'
    assert calls[0][:3] == ['nice', '-n', '10']
    assert calls[0][calls[0].index('--cmake-define')+1] == 'GGML_CUDA_NO_VMM=ON'
    assert calls[0][calls[0].index('--extra-target')+1] == 'test-backend-ops'
    assert '--linkage' in calls[0]
    assert any(p.name == 'build-record.json' for p in files)
