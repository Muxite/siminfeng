import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('observe', Path(__file__).parents[1] / 'observe.py')
o = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(o)
F = Path(__file__).parent / 'fixtures'


def fixture(name):
    return (F / name).read_text()


def test_gpu_units_devices_and_zero():
    fields, uuid = o.parse_gpu(fixture('nvidia-gpu-synthetic.csv'))
    assert uuid == 'GPU-test-0'
    assert fields['gpu.memory.used']['value'] == 123
    assert fields['gpu.power']['value'] == 61.25
    assert fields['gpu.clocks.memory']['value'] == 7001
    assert all(v['status'] == 'ok' for v in fields.values())
    other, uuid = o.parse_gpu(fixture('nvidia-gpu-synthetic.csv'), 'GPU-test-1')
    assert other['gpu.memory.used']['value'] == 0
    assert other['gpu.memory.used']['status'] == 'ok'
    assert uuid == 'GPU-test-1'


def test_unsupported_and_malformed():
    fields, _ = o.parse_gpu(fixture('nvidia-gpu-unsupported.csv'))
    for name in ['gpu.memory.used', 'gpu.util', 'gpu.power', 'gpu.clocks.sm']:
        assert fields[name]['status'] == 'unavailable'
        assert fields[name]['value'] is None
        assert fields[name]['reason']
    assert fields['gpu.temp']['status'] == 'error'
    assert fields['gpu.clocks.memory']['value'] == 0
    fields, _ = o.parse_gpu(fixture('nvidia-gpu-malformed.csv'))
    assert all(v['status'] == 'error' for v in fields.values())


@pytest.mark.parametrize('gpu', ['0', '99', 'GPU-absent'])
def test_empty_gpu(gpu):
    fields, uuid = o.parse_gpu(fixture('empty.txt'), gpu)
    assert uuid is None
    assert all(v['value'] is None and v['status'] == 'unavailable' for v in fields.values())


def test_apps_attribution_and_unknown_memory():
    metric = o.parse_apps(fixture('nvidia-apps-synthetic.csv'), 'GPU-test-0')
    assert [p['pid'] for p in metric['value']] == [123, 456]
    assert metric['value'][0]['used_memory']['value'] == 100
    assert metric['value'][1]['used_memory']['value'] is None
    assert metric['value'][1]['used_memory']['status'] == 'unavailable'
    assert o.parse_apps(fixture('nvidia-apps-synthetic.csv'), 'GPU-other')['value'] == []
    assert o.parse_apps(fixture('empty.txt'), 'GPU-test-0')['status'] == 'unavailable'
    assert o.parse_apps(fixture('nvidia-apps-synthetic.csv'), None)['status'] == 'unavailable'
    assert o.parse_apps('gpu_uuid,pid,used_memory\nGPU-test-0,x,10 MiB\n', 'GPU-test-0')['status'] == 'error'


def test_real_meminfo():
    fields = o.parse_meminfo(fixture('meminfo-real.txt'))
    total_kib = int(next(l for l in fixture('meminfo-real.txt').splitlines() if l.startswith('MemTotal:')).split()[1])
    assert fields['host.mem.total']['value'] == total_kib * 1024
    assert all(v['status'] == 'ok' and v['unit'] == 'B' for v in fields.values())


def test_missing_malformed_meminfo():
    fields = o.parse_meminfo(fixture('meminfo-malformed.txt'))
    assert fields['host.mem.total']['status'] == 'error'
    assert fields['host.mem.available']['value'] == 0
    assert fields['host.mem.swap.total']['status'] == 'unavailable'
    assert fields['host.mem.swap.free']['value'] is None
    assert o.number('9007199254740993', 'B', 'test', 'host')['value'] == 9007199254740993


def test_linux_cgroup_process(tmp_path):
    proc = tmp_path / 'proc'
    proc.mkdir()
    (proc / 'meminfo').write_text(fixture('meminfo-real.txt'))
    (proc / '123').mkdir()
    (proc / '123' / 'status').write_text('Name: test\nVmRSS: 512 kB\n')
    cg = tmp_path / 'cg'
    cg.mkdir()
    for filename in o.CGROUP_SPECS.values():
        (cg / filename).write_text(fixture('io-valid.txt') if filename == 'io.stat' else '0\n')
    (cg / 'memory.peak').unlink()
    fields = o.linux_backend(proc, cg, 123)
    assert fields['process.rss']['value'] == 512 * 1024
    assert fields['cgroup.io.rbytes']['value'] == 124
    assert fields['cgroup.memory.current']['value'] == 0
    assert fields['cgroup.memory.peak']['status'] == 'unavailable'
    assert fields['cgroup.memory.peak']['value'] is None
    (cg / 'io.stat').write_text(fixture('io-malformed.txt'))
    assert o.linux_backend(proc, cg, 999)['cgroup.io.rbytes']['status'] == 'error'
    assert o.linux_backend(proc, cg, 999)['process.rss']['status'] == 'unavailable'
    assert o.parse_io('', 'io.stat')['value'] == 0
    fields = o.linux_backend(proc / 'absent', None, None)
    assert all(v['status'] == 'unavailable' for v in fields.values())


def test_nvidia_absent(monkeypatch):
    absent = json.loads(fixture('nvidia-absent.json'))
    def run(*args, **kwargs):
        raise FileNotFoundError(absent['reason'])
    monkeypatch.setattr(o.subprocess, 'run', run)
    fields, uuid = o.nvidia_backend('0', 1)
    assert uuid is None
    assert all(v['status'] == 'unavailable' and v['value'] is None and v['reason'] == absent['reason'] for v in fields.values())


def test_captured_real_queries(monkeypatch):
    meta = json.loads(fixture('capture.json'))
    def run(argv, **kwargs):
        name = 'nvidia-gpu-real.csv' if argv[1].startswith('--query-gpu=') else 'nvidia-apps-real.csv'
        assert argv == meta['commands'][name]['argv']
        return subprocess.CompletedProcess(argv, meta['commands'][name]['returncode'], fixture(name), fixture(name + '.stderr'))
    monkeypatch.setattr(o.subprocess, 'run', run)
    fields, uuid = o.nvidia_backend('0', 2)
    if meta['commands']['nvidia-gpu-real.csv']['returncode']:
        assert uuid is None
        assert all(v['status'] == 'error' and v['value'] is None and 'exit 9' in v['reason'] for v in fields.values())
    else:
        assert uuid
        assert fields['gpu.memory.total']['status'] == 'ok'


def test_timeout_and_permission(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 2)
    monkeypatch.setattr(o.subprocess, 'run', run)
    assert o.run_query('query-gpu=uuid', 2)[1] == 'error'
    def read(*args):
        raise PermissionError('denied')
    monkeypatch.setattr(o.Path, 'read_text', read)
    assert o.read_text('/proc/meminfo')[1:] == ('error', 'denied')


def test_schema_disabled_and_extensions():
    result = o.sample(backends=())
    assert result['schema'] == 'resource-sample/0'
    assert result['ts']['utc'].endswith('+00:00')
    assert result['host_id']
    assert all(set(v) == {'value', 'unit', 'source', 'scope', 'status', 'reason'} for v in result['fields'].values())
    assert all(v['status'] == 'unavailable' for v in result['fields'].values())
    metric = o.field(5, 'B', 'tegrastats:RAM', 'shared-dram')
    result = o.sample(backends=('jetson',), extra_backends={'jetson': lambda: {'platform.jetson.ram.used': metric}})
    assert result['fields']['platform.jetson.ram.used'] == metric
    with pytest.raises(ValueError):
        o.sample(backends=('jetson',), extra_backends={'jetson': lambda: {'host.mem.total': metric}})
    with pytest.raises(ValueError):
        o.sample(backends=('unknown',))
    with pytest.raises(ValueError):
        o.sample(pid=0)


class FakeTime:
    def __init__(self):
        self.t = 100.0
    def clock(self):
        return self.t
    def sleep(self, seconds):
        self.t += seconds


def test_series_cadence():
    fake = FakeTime()
    def sampler():
        fake.t += .1
        return {'schema': 'resource-sample/0'}
    records = list(o.sample_series(1, 3, sampler=sampler, clock=fake.clock, sleep=fake.sleep))
    summary = records[-1]
    assert len(records) == 4
    assert summary['expected_samples'] == summary['obtained_samples'] == 3
    assert summary['lost_samples'] == 0
    assert summary['max_gap_seconds'] == 1


def test_series_overrun_and_failure():
    fake = FakeTime()
    calls = 0
    def sampler():
        nonlocal calls
        calls += 1
        fake.t += 2.2
        if calls == 2:
            raise OSError('sampler failed')
        return {'schema': 'resource-sample/0'}
    records = list(o.sample_series(1, until=106, sampler=sampler, clock=fake.clock, sleep=fake.sleep))
    summary = records[-1]
    assert summary['expected_samples'] == 6
    assert summary['obtained_samples'] == 2
    assert summary['lost_samples'] == 4
    assert summary['max_gap_seconds'] == pytest.approx(4.4)
    assert summary['collection_errors'][0]['slot'] == 2


@pytest.mark.parametrize('interval,duration,until', [(0, 1, None), (float('nan'), 1, None), (1, -1, None), (1, None, None), (1, 1, 2)])
def test_invalid_series(interval, duration, until):
    with pytest.raises(ValueError):
        list(o.sample_series(interval, duration, until=until))


def test_series_zero_and_fractional_endpoint():
    fake = FakeTime()
    summary = list(o.sample_series(.1, .3, sampler=lambda: {}, clock=fake.clock, sleep=fake.sleep))[-1]
    assert summary['expected_samples'] == summary['obtained_samples'] == 3
    summary = list(o.sample_series(1, 0, clock=fake.clock, sleep=fake.sleep))[-1]
    assert summary['expected_samples'] == summary['obtained_samples'] == 0
    assert summary['max_gap_seconds'] is None


def test_cli_jsonl():
    script = str(Path(__file__).parents[1] / 'observe.py')
    for args in [('once',), ('watch', '--interval', '.01', '--duration', '.025')]:
        proc = subprocess.run(['python3', '-B', script, *args, '--backend', 'linux', '--out', 'jsonl'], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        rows = [json.loads(line) for line in proc.stdout.splitlines()]
        assert rows[0]['schema'] == 'resource-sample/0'
        if args[0] == 'watch':
            assert rows[-1]['schema'] == 'resource-series/0'
            assert rows[-1]['expected_samples'] == rows[-1]['obtained_samples'] + rows[-1]['lost_samples']


@pytest.mark.parametrize('duration,interval,expected', [(.3, .05, 6), (.07, .01, 7), (1e-14, 1, 1)])
def test_series_exact_count_at_large_monotonic(duration, interval, expected):
    fake = FakeTime()
    fake.t = 700000.0
    summary = list(o.sample_series(interval, duration, sampler=lambda: {}, clock=fake.clock, sleep=fake.sleep))[-1]
    assert summary['expected_samples'] == expected


def test_success_backend_queries_and_uuid_filter(monkeypatch):
    def run(argv, **kwargs):
        filename = 'nvidia-gpu-synthetic.csv' if argv[1].startswith('--query-gpu=') else 'nvidia-apps-synthetic.csv'
        return subprocess.CompletedProcess(argv, 0, fixture(filename), '')
    monkeypatch.setattr(o.subprocess, 'run', run)
    result = o.sample(backends=('nvidia-smi',), gpu='1')
    assert result['gpu_id'] == 'GPU-test-1'
    assert result['fields']['gpu.memory.used']['value'] == 0
    apps = result['fields']['gpu.compute_apps']['value']
    assert [p['pid'] for p in apps] == [789]
