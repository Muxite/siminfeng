"""Offline regressions over retained real Build CC / Lab Manager records."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import envelopes as e

ROOT = Path(__file__).resolve().parent
FIX = ROOT / 'fixtures/envelopes'
JOB = FIX / 'lab/jobs/job-ad883810b3'
RETRY = FIX / 'lab/jobs/job-8df3c7bb27'


@pytest.fixture
def host():
    return e.read_json(ROOT / 'hosts/x86-main.json')


@pytest.fixture
def build():
    return e.read_json(FIX / 'build-record.json')


@pytest.fixture
def policy():
    return {'campaign_id': 'c2-resume', 'experiment_id': 'M1', 'attempt_id': 'att-1',
            'release_role': 'engine', 'build_record_ref': 'builds/results/b0-baseline-5/build-record.json',
            'wall_time_s': 900, 'capture_bytes': 8 * 1024**2,
            'memory': {'domain': 'host-ram', 'budget_bytes': 16 * 1024**3,
                       'guarantee': 'cgroup-host', 'accounting_method': 'cgroup-v2-memory'},
            'evidence': {'controller_outside_test_sandbox': True,
                         'durable_destination_ref': 'lab/export/job-ad883810b3/att-1',
                         'outputs': ['stdout', 'stderr', 'state.jsonl', 'result.json']}}


@pytest.fixture
def submitted():
    # Explicitly supplemented declaration, NOT a claim that legacy job had inputs.
    job = e.read_json(JOB / 'job.json')
    s = job['spec']
    # The referenced real job used another image. A fresh proposed submission
    # must deliberately name the build being proposed; do not rewrite fixtures.
    s['image_id'] = e.read_json(FIX / 'build-record.json')['image']['config_digest']
    defs = e.read_json(FIX / 'test-definitions.json')
    s['test_definitions'] = defs
    s['inputs'] = {'model_files': {Path(s['model']).name: s['model_sha256']},
                   'tests': {k: e.canonical_sha(v) for k, v in defs.items()},
                   'mounts': {'/home/muk/projects/siminfeng/campaign/artifacts/traces/in':
                              {'chat.txt': 'a' * 64}}, 'data_files': {}}
    return job


@pytest.fixture
def proposal(build, submitted, host, policy):
    return e.make_proposal(build, submitted, host, policy)


def test_real_legacy_gaps(build, host):
    p = e.make_proposal(build, e.read_json(JOB / 'job.json'), host, {})
    assert not p['dispatchable']
    assert any('inputs.tests' in v for v in p['unresolved'])
    assert p['inputs'] == {}
    assert p['invocation']['tests'][0]['argv'] is None
    assert p['policy']['wall_time_s'] is None  # per-test timeout != whole-job budget


def test_project_real_build(proposal, build):
    assert proposal['format'] == 'campaign-job-proposal/0'
    assert proposal['release']['image_config_id'] == build['image']['config_digest']
    assert proposal['release']['image_config_id'] != build['image']['Id']
    assert proposal['release']['index_digest'] is None
    assert proposal['release']['patch_sha256'] is None  # valid unpatched source
    assert proposal['release']['cmake_defines']['CMAKE_CUDA_ARCHITECTURES'] == '120'
    assert proposal['target']['platform_compat']['cuda_arch'] == ['sm_120']
    assert proposal['release']['transport']['bytes'] == 2210050048
    assert proposal['release']['executables_sha256'] == build['app_files']


def test_role_projection_and_repeated_argv(proposal):
    assert proposal['dispatchable'], proposal['unresolved']
    assert len(proposal['invocation']['tests']) == 27
    assert proposal['invocation']['tests'][0]['argv'][:3] == [
        '/app/llama-completion', '-m', '/m/granite-3.1-1b-a400m-instruct-mixE4K.gguf']
    assert proposal['invocation']['tests'][5]['environment'] == {'LLAMA_ARG_MOE_CACHE_MIB': '128'}
    assert {'role': 'prompt-0', 'access': 'ro', 'target': '/tin'} in proposal['invocation']['mounts']
    assert '/home/' not in json.dumps(proposal['invocation'])
    assert '/home/' not in json.dumps(proposal['inputs'])
    assert proposal['inputs']['mounts']['prompt-0']['chat.txt'] == 'a' * 64


def test_accepted_is_receipt(proposal, host):
    assert e.check_admission(proposal, host) == {'decision': 'accepted', 'reasons': []}
    assert 'success' not in proposal


def test_amd64_rejected_on_jetson(proposal):
    jetson = e.read_json(ROOT / 'hosts/jetson-arm.TEMPLATE.json')
    result = e.check_admission(proposal, jetson)
    assert result['decision'] == 'rejected'
    assert any('platform mismatch' in s for s in result['reasons'])
    assert 'cuda_arch mismatch' in result['reasons']


def test_missing_transport_sha_needs_information(proposal, host):
    proposal['release']['transport']['sha256'] = None
    proposal['dispatchable'] = True  # admission must recompute
    assert e.check_admission(proposal, host)['decision'] == 'needs-information'


def test_missing_transport_at_generation(build, submitted, host, policy):
    del build['archive']['sha256']
    p = e.make_proposal(build, submitted, host, policy)
    assert not p['dispatchable']
    assert e.check_admission(p, host)['decision'] == 'needs-information'


@pytest.mark.parametrize('guarantee,domain,method', [
    ('hard', 'gpu-device-allocation', 'gpu_budget-shim'),
    ('cgroup-host', 'gpu-device-allocation', 'cgroup-v2-memory'),
    ('cooperative-intercept', 'gpu-device-allocation', 'tegrastats'),
])
def test_unsupported_memory(proposal, host, guarantee, domain, method):
    proposal['policy']['memory'].update(guarantee=guarantee, domain=domain, accounting_method=method)
    assert e.check_admission(proposal, host)['decision'] == 'rejected'


def test_missing_inputs_rejected(proposal, host):
    proposal['inputs']['tests'] = {}
    assert e.check_admission(proposal, host)['decision'] == 'rejected'


def test_missing_prompt_identity_rejected(proposal, host):
    proposal['inputs']['mounts']['prompt-0'] = {}
    assert e.check_admission(proposal, host)['decision'] == 'rejected'


def test_unrelated_build_job_pair(build, submitted, host, policy):
    submitted['spec']['image_id'] = e.read_json(JOB / 'job.json')['spec']['image_id']
    p = e.make_proposal(build, submitted, host, policy)
    assert not p['dispatchable']
    assert any('job_spec.image_id' in v for v in p['unresolved'])


def test_unmapped_argv_path_rejected(proposal, host):
    proposal['invocation']['tests'][0]['argv'].extend(['-f', '/home/muk/private.txt'])
    assert e.check_admission(proposal, host)['decision'] == 'rejected'


def test_driver_floor(proposal, host):
    proposal['target']['platform_compat']['driver_min'] = '596.0.0'
    assert e.check_admission(proposal, host)['decision'] == 'rejected'


def test_host_paths_rejected(proposal, host):
    proposal['invocation']['mounts'][0]['host_path'] = '/home/muk/models'
    assert e.check_admission(proposal, host)['decision'] == 'rejected'


def test_definition_changed(build, submitted, host, policy):
    submitted['spec']['test_definitions']['m1_static_n24']['args'].append('--changed')
    p = e.make_proposal(build, submitted, host, policy)
    assert not p['dispatchable']
    assert any('definition differs' in s for s in p['unresolved'])


def test_schema_two(build, submitted, host, policy):
    build['schema_version'] = 2
    build['cmake_defines'] = {'GGML_CUDA_NO_VMM': 'ON'}
    assert e.make_proposal(build, submitted, host, policy)['release']['cmake_defines']['GGML_CUDA_NO_VMM'] == 'ON'


def test_completed_real_result():
    r = e.make_result(JOB, 'att-1', 'performance-observation')
    assert r['execution_outcome']['status'] == 'completed'
    assert r['test_assessment']['all_ok'] is True
    assert r['claim_category'] == 'performance-observation'
    c = r['evidence_completeness']
    assert (c['expected'], c['started'], c['clean'], c['missing']) == (27, 27, 27, [])
    assert r['stop_confirmed'] is True
    rows = (JOB / 'att-1/state.jsonl').read_text().splitlines()
    assert r['last_durable_sequence'] == {'line_count': len(rows), 'last_ts': json.loads(rows[-1])['ts']}
    manifest = e.read_json(FIX / 'lab/export/job-ad883810b3/att-1/MANIFEST.json')
    assert {Path(a['path']).name: a['sha256'] for a in r['artifact_refs']} == manifest
    assert 'success' not in r


def test_interrupted_and_retry_stay_separate():
    first = e.make_result(RETRY, 'att-1')
    retry = e.make_result(RETRY, 'att-2')
    assert first['execution_outcome']['status'] == 'interrupted'
    assert first['test_assessment']['assessed'] is False
    assert first['evidence_completeness']['expected'] == 10
    assert first['evidence_completeness']['started'] == 9
    assert first['evidence_completeness']['clean'] == 0  # no durable assessment, not inferred from files
    assert len(first['evidence_completeness']['missing']) == 10
    assert first['claim_category'] == 'unclassified'
    assert retry['execution_outcome']['status'] == 'completed'
    assert retry['evidence_completeness']['clean'] == 10
    assert retry['evidence_completeness']['missing'] == []
    assert first['resource_generation'] != retry['resource_generation']
    assert first['resource_generation']['owner_token_sha256'] == e.sha_text(
        e.read_json(RETRY / 'att-1/owner.json')['token'])
    assert e.read_json(RETRY / 'att-1/owner.json')['token'] not in json.dumps(first)


def copy_job(tmp_path):
    path = tmp_path / 'lab/jobs' / JOB.name
    shutil.copytree(JOB, path)
    return path


def test_cancel_request_is_not_confirmation(tmp_path):
    path = copy_job(tmp_path)
    rfile = path / 'att-1/result.json'
    r = e.read_json(rfile)
    r['cleanup'] = {'confirmed': False, 'owned_containers_left': ['alive']}
    rfile.write_text(json.dumps(r))
    (path / 'att-1/cancel_requested').touch()
    assert e.make_result(path, 'att-1')['stop_confirmed'] is False


def test_result_token_conflict(tmp_path):
    path = copy_job(tmp_path)
    owner = path / 'att-1/owner.json'
    value = e.read_json(owner)
    value['token'] = 'foreign'
    owner.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='token mismatch'):
        e.make_result(path, 'att-1')


def test_missing_result_and_torn_journal(tmp_path):
    path = copy_job(tmp_path)
    (path / 'att-1/result.json').unlink()
    with (path / 'att-1/state.jsonl').open('a') as stream:
        stream.write('{"torn":')
    r = e.make_result(path, 'att-1')
    assert r['execution_outcome']['status'] == 'unknown'
    assert r['test_assessment']['assessed'] is False
    assert r['evidence_completeness']['journal_errors']
    assert r['evidence_completeness']['manifest_present'] is False
    assert r['stop_confirmed'] is False


def test_cli_all_commands(tmp_path, build, submitted, host, policy):
    args = []
    for flag, value in [('build-record', build), ('job-spec', submitted), ('host-profile', host), ('policy', policy)]:
        path = tmp_path / (flag + '.json')
        path.write_text(json.dumps(value))
        args += ['--' + flag, str(path)]
    out = tmp_path / 'proposal.json'
    run = lambda a: subprocess.run([sys.executable, '-B', str(ROOT / 'envelopes.py'), *a], capture_output=True, text=True)
    assert run(['proposal', *args, '--out', str(out)]).returncode == 0
    verdict = run(['admit', '--proposal', str(out), '--host-profile', str(tmp_path / 'host-profile.json')])
    assert verdict.returncode == 0
    assert json.loads(verdict.stdout)['decision'] == 'accepted'
    p = e.read_json(out)
    p['release']['transport']['sha256'] = None
    out.write_text(json.dumps(p))
    assert run(['admit', '--proposal', str(out), '--host-profile', str(tmp_path / 'host-profile.json')]).returncode == 2
    result = run(['result', '--job-dir', str(RETRY), '--attempt', 'att-1', '--claim-category', 'interrupted-observation'])
    assert result.returncode == 0
    assert json.loads(result.stdout)['claim_category'] == 'interrupted-observation'
