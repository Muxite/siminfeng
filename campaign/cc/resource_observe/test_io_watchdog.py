import json
import os
from pathlib import Path
from types import SimpleNamespace
import signal
import subprocess
import sys

import pytest
import io_watchdog as io
import resource_observe as ro

FX = Path(__file__).parent / 'fixtures/io'


class Clock:
    def __init__(self):
        self.t = 0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def tree(tmp_path):
    proc, sysroot, cg = (tmp_path / n for n in ('proc', 'sys', 'cg'))
    proc.mkdir(); cg.mkdir()
    (proc / '42').mkdir()
    for dest, src in [('vmstat', 'vmstat'), ('42/stat', 'proc-stat'),
                      ('42/status', 'status'), ('42/io', 'proc-io')]:
        (proc / dest).write_text((FX / src).read_text())
    for f in ('memory.stat', 'memory.events', 'io.stat'):
        (cg / f).write_text((FX / f).read_text())
    (cg / 'memory.current').write_text('3145728\n')
    (cg / 'cgroup.procs').write_text('42\n')
    disk = sysroot / 'devices/pci/block/nvme0n1'
    partition = disk / 'nvme0n1p1'
    partition.mkdir(parents=True)
    (partition / 'partition').write_text('1')
    (disk / 'stat').write_text((FX / 'diskstat').read_text())
    (sysroot / 'dev/block').mkdir(parents=True)
    (sysroot / 'dev/block/259:1').symlink_to(partition)
    (sysroot / 'block').mkdir()
    (sysroot / 'block/nvme0n1').symlink_to(disk)
    return proc, sysroot, cg, disk


def names(metrics):
    return {m['name']: m for m in metrics}


def test_device_resolution(tree):
    _, sysroot, _, _ = tree
    assert io.resolve_device('/models', str(sysroot), lambda p: SimpleNamespace(st_dev=os.makedev(259, 1))) == 'nvme0n1'
    with pytest.raises(OSError, match='no sysfs block device'):
        io.resolve_device('/models', str(sysroot), lambda p: SimpleNamespace(st_dev=os.makedev(0, 12)))


def test_scopes_units_and_rates(tree):
    proc, sysroot, cg, disk = tree
    clock = Clock()
    s = io.IOSampler(device='nvme0n1', pid_from_cgroup=str(cg), cgroup=str(cg),
                     procroot=str(proc), sysroot=str(sysroot), clock=clock)
    m, u = s.sample()
    d = names(m)
    assert d['device_read']['value'] == 2
    assert d['pid_minflt']['value'] == 100 and d['pid_majflt']['value'] == 10
    assert d['pid_read_bytes']['value'] == 2
    assert d['pid_VmRSS']['value'] == 4
    assert d['cgroup_file']['value'] == 2
    assert d['cgroup_rbytes_259:0']['value'] == 4
    assert {x['scope'] for x in m} == {'system', 'device', 'process', 'cgroup'}
    assert not any(x['name'].endswith('_rate') for x in m)
    assert any('first observation' in x['reason'] for x in u)
    clock.t = 2
    (proc / 'vmstat').write_text((FX / 'vmstat').read_text().replace('pgmajfault 10', 'pgmajfault 16'))
    (disk / 'stat').write_text('14 0 8192 10 0 0 0 0 0 700 700')
    d = names(s.sample()[0])
    assert d['pgmajfault_delta']['value'] == 6 and d['pgmajfault_rate']['value'] == 3
    assert d['device_read_rate']['value'] == 1 and d['device_read_rate']['unit'] == 'MiB/s'
    assert d['device_read_ios_rate']['value'] == 2
    assert d['device_util']['value'] == 25
    clock.t = 3
    (disk / 'stat').write_text((FX / 'diskstat').read_text())
    m, u = s.sample()
    assert 'device_read_rate' not in names(m)
    assert any(x['name'] == 'device_read_rate' and x['reason'] == 'counter reset' for x in u)
    clock.t = 4
    assert names(s.sample()[0])['device_read_rate']['value'] == 0


def test_permission_and_no_partial_totals(tree, monkeypatch):
    proc, sysroot, cg, _ = tree
    real = io.read
    def denied(path):
        if str(path).endswith('/42/io'):
            raise PermissionError('permission denied: proc io')
        return real(path)
    monkeypatch.setattr(io, 'read', denied)
    s = io.IOSampler(pid=42, device='nvme0n1', procroot=str(proc), sysroot=str(sysroot))
    m, u = s.sample()
    assert 'pid_read_bytes' not in names(m)
    assert any('permission denied' in x['reason'] for x in u)
    assert 'pid_VmRSS' in names(m)
    s.pid = None; s.pid_from_cgroup = str(cg)
    (cg / 'cgroup.procs').write_text('42\n43\n')
    assert 'pid_VmRSS' not in names(s.sample()[0])


def test_membership_and_pid_reuse_reset_baseline(tree):
    proc, sysroot, cg, _ = tree
    clock = Clock()
    s = io.IOSampler(pid_from_cgroup=str(cg), device='nvme0n1', procroot=str(proc), sysroot=str(sysroot), clock=clock)
    s.sample(); clock.t = 1
    assert 'pid_minflt_rate' in names(s.sample()[0])
    (proc / '42/stat').write_text((FX / 'proc-stat').read_text().replace('1000', '2000'))
    clock.t = 2
    assert 'pid_minflt_rate' not in names(s.sample()[0])
    (cg / 'cgroup.procs').write_text('')
    clock.t = 3
    assert 'pid_minflt' not in names(s.sample()[0])


def args(**kwargs):
    d = dict(min_available_mib=900, max_swap_used_mib=512, max_swap_rate_mibs=20,
             interval_ms=250, kill_pid=None, kill_cgroup=None, kill_cmd=None)
    d.update(kwargs)
    return SimpleNamespace(**d)


def run_watch(monkeypatch, samples, config=None):
    clock, events, actions = Clock(), [], []
    def read(path):
        index = min(round(clock.t * 4), len(samples) - 1)
        avail, used, pages = samples[index]
        if path.endswith('meminfo'):
            return 'MemAvailable: %s kB\nSwapTotal: 1048576 kB\nSwapFree: %s kB\n' % (avail * 1024, (1024 - used) * 1024)
        return 'pswpin %s\npswpout 0\n' % pages
    monkeypatch.setattr(io, 'read', read)
    code = io.watchdog(config or args(), lambda event, data: events.append(dict(event=event, **data)),
                       clock=clock, sleep=clock.sleep, action=lambda a: actions.append(a) or {'ok': True})
    return code, events, actions, clock


@pytest.mark.parametrize('limit,samples', [
    ('min_available_mib', [(800, 0, 0), (1000, 0, 0), (800, 0, 0), (800, 0, 0)]),
    ('max_swap_used_mib', [(1000, 600, 0), (1000, 600, 0)]),
    ('max_swap_rate_mibs', [(1000, 0, 0), (1000, 0, 100000), (1000, 0, 200000)]),
])
def test_watchdog_debounce_and_once(monkeypatch, limit, samples):
    code, events, actions, clock = run_watch(monkeypatch, samples, args(kill_cmd='docker kill test'))
    assert code == 3 and len(actions) == 1
    trips = [e for e in events if e['event'] == 'watchdog_trip']
    assert len(trips) == 1 and trips[0]['limits'] == [limit]
    assert len([e for e in events if e['event'] == 'watchdog_sample']) == len(samples)


def test_no_action_without_target(monkeypatch):
    code, events, actions, _ = run_watch(monkeypatch, [(1, 0, 0)] * 2)
    assert code == 3 and not actions
    assert events[-1]['action'] == 'none'


@pytest.mark.parametrize('target', ['pid', 'cgroup'])
def test_normal_exit(monkeypatch, target):
    clock = Clock(); events = []
    monkeypatch.setattr(io, 'pid_identity', lambda *a: None)
    monkeypatch.setattr(io, 'cgroup_pids', lambda *a: [])
    a = args(kill_pid=42) if target == 'pid' else args(kill_cgroup='/cg')
    assert io.watchdog(a, lambda *e: events.append(e), clock, clock.sleep) == 0
    assert events[0][0] == 'watchdog_end' and not clock.sleeps


def test_watchdog_read_error(monkeypatch):
    def denied(path):
        raise PermissionError('denied')
    monkeypatch.setattr(io, 'read', denied)
    events = []
    assert io.watchdog(args(), lambda *e: events.append(e)) == 2
    assert events[0][0] == 'watchdog_error'


def test_signal_grace_and_reuse(monkeypatch):
    calls = []; clock = Clock()
    monkeypatch.setattr(io, 'pid_identity', lambda p: 'start')
    monkeypatch.setattr(io.os, 'kill', lambda p, sig: calls.append((p, sig)))
    result = io.kill_target(args(kill_pid=42), clock.sleep)
    assert calls == [(42, signal.SIGTERM), (42, signal.SIGKILL)]
    assert clock.sleeps == [3] and not result['errors']
    calls.clear()
    def sleep(_):
        monkeypatch.setattr(io, 'pid_identity', lambda p: 'reused')
    io.kill_target(args(kill_pid=42), sleep)
    assert calls == [(42, signal.SIGTERM)]


def test_cgroup_only_targets_and_command(monkeypatch):
    calls = []
    monkeypatch.setattr(io, 'cgroup_pids', lambda path: [42, 43])
    monkeypatch.setattr(io, 'pid_identity', lambda p: 'start')
    monkeypatch.setattr(io.os, 'kill', lambda p, sig: calls.append((p, sig)))
    io.kill_target(args(kill_cgroup='/explicit'), lambda _: None)
    assert calls == [(42, signal.SIGTERM), (43, signal.SIGTERM), (42, signal.SIGKILL), (43, signal.SIGKILL)]
    runs = []
    def run(command, **kwargs):
        runs.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr='failed')
    monkeypatch.setattr(io.subprocess, 'run', run)
    result = io.kill_target(args(kill_cmd='docker kill "test name"'))
    assert runs == [['docker', 'kill', 'test name']] and result['returncode'] == 1


def test_cli_invalid_pid_and_io_contract(tmp_path):
    script = str(Path(ro.__file__))
    out = tmp_path / 'out.jsonl'
    r = subprocess.run([sys.executable, script, 'watchdog', '--kill-pid', '0', '--out', str(out)], capture_output=True)
    assert r.returncode == 2 and not out.exists()
    r = subprocess.run([sys.executable, script, 'sample', '--backends', 'io', '--duration-s', '.01',
                        '--interval-ms', '10', '--out', str(out)], capture_output=True)
    assert r.returncode == 0, r.stderr
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) >= 2
    assert all(r['schema'] == 'resource-observe/0' and r['backend'] == 'io' for r in records)


def test_cgroup_recursive_membership(tmp_path):
    child = tmp_path / 'child'
    child.mkdir()
    (tmp_path / 'cgroup.procs').write_text('42\n')
    (child / 'cgroup.procs').write_text('43\n42\n')
    assert io.cgroup_pids(str(tmp_path)) == [42, 43]


def test_watchdog_equal_thresholds_and_different_limits(monkeypatch):
    samples = [(900, 512, 0), (800, 0, 0), (1000, 600, 0), (1000, 600, 0)]
    code, events, _, _ = run_watch(monkeypatch, samples)
    assert code == 3
    assert events[0]['crossed'] == []
    assert len([e for e in events if e['event'] == 'watchdog_sample']) == 4
    assert events[-2]['limits'] == ['max_swap_used_mib']


def test_action_failure_still_trip_exit(monkeypatch):
    def failed(command, **kwargs):
        raise PermissionError('execution denied')
    monkeypatch.setattr(io.subprocess, 'run', failed)
    result = io.kill_target(args(kill_cmd='docker kill named'))
    assert result['action'] == 'command' and 'execution denied' in result['error']
    code, _, actions, _ = run_watch(monkeypatch, [(800, 0, 0)] * 2, args(kill_cmd='docker kill named'))
    assert code == 3 and len(actions) == 1


def test_pid_monitor_permission_error_is_recorded(monkeypatch):
    def denied(*a):
        raise PermissionError('stat denied')
    monkeypatch.setattr(io, 'pid_identity', denied)
    events = []
    assert io.watchdog(args(kill_pid=42), lambda *e: events.append(e)) == 2
    assert events == [('watchdog_error', {'reason': 'stat denied'})]
