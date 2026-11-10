"""Host contracts; all subprocess access is mocked, never execute a container/GPU."""
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lab(monkeypatch, tmp_path):
    monkeypatch.setenv('LAB', str(tmp_path / 'lab'))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('DOCKER_CMD_PREFIX', raising=False)
    loader = importlib.machinery.SourceFileLoader('lab_hosts_under_test', str(ROOT / 'bin/lab'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    def forbidden(*args, **kwargs):
        pytest.fail('unexpected subprocess invocation')
    monkeypatch.setattr(mod.subprocess, 'run', forbidden)
    monkeypatch.setattr(mod.subprocess, 'Popen', forbidden)
    return mod


def profile(lab, name='jetson-arm'):
    return lab.load_host(name)[0]


def test_profiles_and_default(lab, monkeypatch, tmp_path):
    host, digest = lab.load_host()
    assert host['host_id'] == 'x86-main'
    assert digest == hashlib.sha256((ROOT / 'lab/hosts/x86-main.json').read_bytes()).hexdigest()
    home = tmp_path / 'home/.config/siminfeng'
    home.mkdir(parents=True)
    (home / 'host_id').write_text('jetson-arm\n')
    host, digest = lab.load_host()
    assert host['platform'] == 'linux/arm64'
    assert host['resources_max'] == {'cpus': 6, 'memory': '5g'}
    assert host['l4t_major'] == 36
    (home / 'host_id').write_text('')
    assert lab.load_host()[0]['host_id'] == 'x86-main'
    with pytest.raises(ValueError): lab.load_host('../escape')
    with pytest.raises(FileNotFoundError): lab.load_host('unknown')


def test_x86_golden_argv(lab):
    # Literal golden list transcribed from bin/lab before J-T4.
    lab.CID = 'c1'
    spec = dict(model='/models/a.gguf', data='/data/p.txt', resources=dict(cpus='8', memory='16g', pids='512'))
    td = dict(tool='llama-perplexity', env={'A': 'two words'}, needs_data=True,
              memory='2g', model_file='other.gguf', docker_args=['--network', 'none'], args=['-c', '512'])
    golden = ['docker', 'run', '--rm', '--name', 'sandbox', '--label', 'campaign=c1', '--label',
              'lab_job=job-x', '--label', 'lab_attempt=att-1', '--label', 'lab_token=token',
              '-e', 'A=two words', '--gpus', 'all', '--cpus', '8', '--memory', '2g',
              '--memory-swap', '2g', '--pids-limit', '512', '-v', '/models:/m:ro', '--network', 'none',
              '-v', '/data:/d:ro', '--entrypoint', '/app/llama-perplexity', 'sha256:abc', '-m',
              '/m/other.gguf', '-f', '/d/p.txt', '-c', '512']
    args = ('job-x', 'att-1', 'token', 'sandbox', spec, {'image_id': 'sha256:abc'}, td)
    assert lab.container_argv(profile(lab, 'x86-main'), *args) == golden
    jetson = lab.container_argv(profile(lab), *args)
    at = golden.index('--gpus')
    assert jetson == golden[:at] + ['--runtime', 'nvidia', '-e', 'NVIDIA_VISIBLE_DEVICES=all',
                                   '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility'] + golden[at+2:]
    assert lab.docker_argv(golden) == golden


def test_prefix_is_one_shell_command(lab, monkeypatch):
    argv = ['docker', 'inspect', 'name with spaces;$(touch /tmp/no)', '-f', '{{.Id}}', "a'b"]
    monkeypatch.setenv('DOCKER_CMD_PREFIX', 'sg docker -c')
    wrapped = lab.docker_argv(argv)
    assert wrapped[:3] == ['sg', 'docker', '-c']
    assert len(wrapped) == 4
    assert shlex.split(wrapped[-1]) == argv
    seen = []
    monkeypatch.setattr(lab.subprocess, 'run', lambda cmd, **kw: seen.append(cmd))
    lab.sh(argv)
    assert seen == [wrapped]
    assert lab.docker_argv(['nvidia-smi']) == ['nvidia-smi']


@pytest.mark.parametrize('backend,text,value', [
    ('nvml', '123\n', 123), ('nvml', '0', 0), ('nvml', '[N/A]', None),
    ('nvml', '', None), ('nvml', '1\n2', None), ('nvml', 'nonsense', None),
    ('tegrastats', 'RAM 234/7620MB (lfb 1x4MB) CPU [0%@100]', 234),
    ('tegrastats', 'RAM 0/7620MB', 0), ('tegrastats', 'RAM [N/A]', None),
    ('tegrastats', 'truncated', None), ('tegrastats', '', None),
])
def test_parse_memory(lab, backend, text, value):
    got, reason = lab.parse_memory(text, backend)
    assert got == value
    assert bool(reason) == (value is None)


def test_sampler_nvml_failures(lab, monkeypatch):
    sampler = lab.MemorySampler(profile(lab, 'x86-main'))
    monkeypatch.setattr(lab.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='[N/A]', stderr='', returncode=0))
    assert sampler.sample()[0] is None
    assert lab.gpu_state(profile(lab, 'x86-main'))['unavailable']
    monkeypatch.setattr(lab.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='', stderr='denied', returncode=1))
    assert sampler.sample() == (None, 'nvidia-smi failed: denied')
    def absent(*a, **k): raise FileNotFoundError('missing tool')
    monkeypatch.setattr(lab.subprocess, 'run', absent)
    assert sampler.sample() == (None, 'missing tool')
    assert lab.gpu_state(profile(lab, 'x86-main'))['gpu'] is None
    sampler.close()


def test_tegrastats_sampler_and_state(lab, monkeypatch):
    calls = []
    class Child:
        stdout = iter([])
        def terminate(self): calls.append('terminate')
        def wait(self, **kwargs): calls.append('wait')
    def popen(cmd, **kwargs):
        calls.append(cmd)
        return Child()
    monkeypatch.setattr(lab.subprocess, 'Popen', popen)
    sampler = lab.MemorySampler(profile(lab))
    # Read directly from the queue to avoid depending on thread scheduling.
    sampler.reader.q.put('RAM 100/7620MB')
    value, reason = sampler.sample()
    assert value == 100 and reason is None
    assert sampler.sample()[0] is None
    sampler.close()
    assert calls == [['tegrastats', '--interval', '250'], 'terminate', 'wait']
    state = lab.gpu_state(profile(lab))
    assert state['gpu'] is None and state['compute_apps'] is None and state['unavailable']
    def absent(*a, **k): raise FileNotFoundError('tegrastats missing')
    monkeypatch.setattr(lab.subprocess, 'Popen', absent)
    sampler = lab.MemorySampler(profile(lab))
    assert sampler.sample() == (None, 'tegrastats missing')
    sampler.close()


@pytest.fixture
def admission_case(lab, tmp_path):
    model = tmp_path / 'a.gguf'; model.write_bytes(b'model')
    data = tmp_path / 'data.txt'; data.write_text('data')
    img = dict(image_id='sha256:abc', platform='linux/arm64', target_requirements={
        'platform': 'linux/arm64', 'cuda_arch': 'sm_87', 'l4t_major': 36,
        'runtime': 'nvidia (NVIDIA_VISIBLE_DEVICES required)'})
    spec = dict(model=str(model), data=str(data), tests=['smoke'], resources={'cpus': '6', 'memory': '5g'})
    return img, spec


@pytest.mark.parametrize('change,expected', [
    ('none', 'accepted'), ('platform', 'rejected'), ('req_platform', 'rejected'),
    ('cuda_arch', 'rejected'), ('l4t_major', 'rejected'), ('cpus', 'rejected'),
    ('memory', 'rejected'), ('bad_cpu', 'rejected'), ('bad_memory', 'rejected'),
    ('no_platform', 'needs_information'), ('no_requirements', 'needs_information'),
    ('no_runtime', 'needs_information'), ('model', 'needs_information'),
    ('data', 'needs_information'), ('needs_data', 'needs_information'),
    ('test_memory', 'rejected'), ('test_model', 'needs_information'),
    ('reject_over_missing', 'rejected'),
])
def test_admission_matrix(lab, admission_case, change, expected):
    img, spec = admission_case
    if change == 'platform': img['platform'] = 'linux/amd64'
    if change == 'req_platform': img['target_requirements']['platform'] = 'linux/amd64'
    if change in ('cuda_arch', 'l4t_major'): img['target_requirements'][change] = 'wrong'
    if change == 'cpus': spec['resources']['cpus'] = '7'
    if change == 'memory': spec['resources']['memory'] = '5121m'
    if change == 'bad_cpu': spec['resources']['cpus'] = 'nan'
    if change == 'bad_memory': spec['resources']['memory'] = 'oops'
    if change == 'no_platform': del img['platform']
    if change == 'no_requirements': del img['target_requirements']
    if change == 'no_runtime': del img['target_requirements']['runtime']
    if change in ('model', 'data'): spec[change] += '.missing'
    if change == 'needs_data':
        spec['data'] = None; lab.TESTS['smoke'] = {'needs_data': True}
    if change == 'test_memory': lab.TESTS['smoke'] = {'memory': '6g'}
    if change == 'test_model': lab.TESTS['smoke'] = {'model_file': 'missing.gguf'}
    if change == 'reject_over_missing':
        img['platform'] = 'linux/amd64'; spec['model'] += '.missing'
    decision = lab.admission(img, spec, profile(lab), 'digest')
    assert decision['decision'] == expected
    assert decision['host_profile_sha256'] == 'digest'
    assert decision['host_id'] == 'jetson-arm'
    assert bool(decision['reasons']) == (expected != 'accepted')


def test_x86_legacy(lab, admission_case):
    img, spec = admission_case
    spec['resources'] = {'cpus': '8', 'memory': '16g'}
    result = lab.admission({'image_id': 'legacy'}, spec, profile(lab, 'x86-main'), 'digest')
    assert result['decision'] == 'accepted'
    assert result['reasons'] == ['legacy image record: platform unverified']
    img['platform'] = 'linux/amd64'
    img['target_requirements'] = {'platform': 'linux/amd64', 'cuda_arch': 'sm_120', 'runtime': 'nvidia'}
    assert lab.admission(img, spec, profile(lab, 'x86-main'), 'digest')['decision'] == 'accepted'


@pytest.mark.parametrize('decision', ['accepted', 'rejected', 'needs_information'])
def test_submit_persists_readonly_idempotent(lab, monkeypatch, admission_case, decision, capsys):
    img, spec = admission_case
    host = profile(lab)
    monkeypatch.setattr(lab, 'load_host', lambda: (host, 'digest'))
    images = Path(lab.LAB) / 'images'; images.mkdir(parents=True)
    if decision == 'rejected': img['platform'] = 'linux/amd64'
    if decision == 'needs_information': spec['model'] += '.missing'
    (images / 'test.json').write_text(json.dumps(img))
    args = SimpleNamespace(image='test', model=spec['model'], model_sha256=None,
                           data=spec['data'], tests='smoke', key='key', res=None)
    lab.cmd_submit(args)
    jid = capsys.readouterr().out.strip()
    jobdir = Path(lab.LAB) / 'jobs' / jid
    saved = {}
    for name in ('job.json', 'admission.json'):
        path = jobdir / name
        assert path.stat().st_mode & 0o222 == 0
        saved[name] = path.read_bytes()
    assert json.loads(saved['admission.json'])['decision'] == decision
    assert json.loads(saved['job.json'])['spec']['resources']['memory'] == '5g'
    lab.cmd_submit(args)
    assert 'duplicate submission' in capsys.readouterr().out
    assert all((jobdir / name).read_bytes() == content for name, content in saved.items())
    args.res = 'cpus=1'
    with pytest.raises(SystemExit, match='DIFFERENT spec'): lab.cmd_submit(args)
    lab.cmd_status(SimpleNamespace(job=jid))
    assert 'admission=' + decision in capsys.readouterr().out


def test_image_requirements(lab, monkeypatch, tmp_path):
    build = json.loads((ROOT / 'cc/build_engine/results/jetson-c1/build-record.json').read_text())
    record = tmp_path / 'build.json'; record.write_text(json.dumps(build))
    monkeypatch.setenv('DOCKER_CMD_PREFIX', 'sg docker -c')
    seen = []
    def inspect(cmd, **kw):
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout='sha256:abc|[]|today', stderr='')
    monkeypatch.setattr(lab.subprocess, 'run', inspect)
    args = SimpleNamespace(name='jetson', ref=build['tag'], requirements=str(record), platform=None)
    lab.cmd_image(args)
    result = json.loads((Path(lab.LAB) / 'images/jetson.json').read_text())
    assert result['platform'] == 'linux/arm64'
    assert result['target_requirements'] == build['target_requirements']
    assert shlex.split(seen[0][-1]) == ['docker', 'image', 'inspect', build['tag'], '-f', '{{.Id}}|{{json .RepoDigests}}|{{.Created}}']


@pytest.mark.parametrize('stored,current', [
    ('rejected', 'accepted'), ('needs_information', 'accepted'),
    ('accepted', 'rejected'), ('accepted', 'needs_information'),
    (None, 'rejected'),
])
def test_run_admission_guard_without_running_job(lab, stored, current, capsys):
    # Exercise the pre-attempt guard alone; never invoke lab run.
    make = lambda decision: {'decision': decision, 'reasons': ['guard reason']} if decision else None
    with pytest.raises(SystemExit) as ex:
        lab.require_admission(make(stored), make(current))
    assert ex.value.code == 10
    assert 'guard reason' in capsys.readouterr().out


def test_run_admission_guard_accepts_legacy_and_current(lab):
    lab.require_admission(None, {'decision': 'accepted', 'reasons': []})
    lab.require_admission({'decision': 'accepted'}, {'decision': 'accepted'})
