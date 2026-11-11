"""CPU-only tests, including labeled real proc captures from this host."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from cc.thread_sampler import thread_sampler as ts

FIXTURE = Path(__file__).parent / 'fixtures/host-proc-stat.json'


def test_real_host_stat():
    fixture = json.loads(FIXTURE.read_text())
    assert fixture['kernel'].endswith('tegra')
    for record in fixture['records']:
        s = ts.parse_stat(record['stat'])
        assert s['comm'] == record['comm']
        assert s['state'] == 'R'
        assert s['processor'] >= 0
        assert int(record['stat_mode'], 8) & 4
        assert int(record['comm_mode'], 8) & 4
    first = ts.parse_stat(fixture['records'][0]['stat'])
    assert (first['utime'], first['stime'], first['minflt'], first['majflt'], first['processor']) == (3, 1, 1336, 3, 4)
    assert ts.parse_stat(fixture['records'][1]['stat'])['comm'] == 'own (test) task'


@pytest.mark.parametrize('comm,tid,group', [
    ('llama-completion', 10, 'main'), ('llama-completion', 11, 'cpu_workers'),
    ('ggml-worker', 11, 'cpu_workers'), ('OpenMP', 11, 'cpu_workers'),
    ('omp-worker', 11, 'cpu_workers'), ('xfill-15', 11, 'expert_fill'),
    ('cuda-EvtHandlr', 11, 'cuda_driver'), ('nv_queue', 11, 'cuda_driver'),
    ('python3', 11, 'others'), ('python3', 10, 'main')])
def test_classification(comm, tid, group):
    assert ts.classify(comm, tid, 10) == group


def write_capture(path):
    rows = [{'type': 'header', 'pid': 10, 'clock_ticks': 100}]
    for t, state in [(1, 'S'), (2, 'R'), (3, 'D'), (4, 'S')]:
        rows.append({'t': t, 'tasks': [
            [10, state, 10 if t > 1 else 0, 0, 0, 2 if t > 1 else 0, 0, 'llama-completio'],
            [11, 'R', 20 if t > 1 else 0, 0, 0, 0, 1, 'llama-completio'],
            [12, 'S', 0, 0, 0, 4 if t > 1 else 0, 2, 'xfill-0']]})
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_decode_window(tmp_path):
    path = tmp_path / 'threads.jsonl'
    write_capture(path)
    s = ts.aggregate(path, eval_ms_total=1500, tokens=2)
    assert s['phase_window']['decode_start'] == 2.5
    assert s['phase_window']['decode_end'] == 4
    assert s['groups']['main']['fractions'] == {'R': 0, 'D': .5, 'S': .5}
    assert s['groups']['main']['cpu_seconds'] == pytest.approx(.15)
    assert s['cpu_workers_r_fraction'] == 1
    assert s['main_workers_d_fraction'] == .25
    assert s['majflt_per_token'] == 4.5  # 1.5 seconds * 6 faults / 2 tokens
    assert s['rd_histogram'] == {'1,0': 1, '1,1': 1}
    assert s['groups']['expert_fill']['majflt_per_token'] == 3
    assert ts.aggregate(path)['groups']['main']['cpu_seconds'] == pytest.approx(.3)


def test_last_eval_line(tmp_path):
    path = tmp_path / 'threads.jsonl'
    write_capture(path)
    stderr = tmp_path / 'stderr.txt'
    stderr.write_text('eval time = 3000 ms / 3 runs\nprompt eval time = 9000 ms / 2 tokens\neval time = 1500 ms / 2 runs\n')
    s = ts.summarize(path, tmp_path / 'summary.json', stderr)
    assert s['phase_window']['decode_start'] == 2.5
    assert s['majflt_per_token'] == 4.5


def test_overhead_own_process_one_second(tmp_path):
    path = tmp_path / 'own.jsonl'
    stats = ts.sample(os.getpid(), path, seconds=1)
    assert 450 <= stats['samples'] <= 510
    assert stats['self_cpu_s'] > 0
    assert stats['self_cpu_percent'] == pytest.approx(100 * stats['self_cpu_s'] / stats['elapsed_s'])
    assert ts.aggregate(path)['threads'][str(os.getpid())]['samples'] > 450
    print('own-process 500Hz self-CPU:', stats)


def test_busy_process_cli(tmp_path):
    busy = subprocess.Popen([sys.executable, '-c', 'while True: pass'])
    try:
        cmd = [sys.executable, ts.__file__, '--pid', str(busy.pid), '--seconds', '2',
               '--out', str(tmp_path / 'busy.jsonl'), '--summary', str(tmp_path / 'summary.json')]
        s = json.loads(subprocess.check_output(cmd, text=True))
        assert s['threads'][str(busy.pid)]['fractions']['R'] > .8
        assert s['overhead']['self_cpu_percent'] < 3
    finally:
        busy.terminate()
        busy.wait()


def test_container_prefix(monkeypatch):
    seen = []
    monkeypatch.setenv('DOCKER_CMD_PREFIX', 'sg docker -c')
    monkeypatch.setattr(ts.subprocess, 'check_output', lambda cmd, **kw: seen.append(cmd) or '123\n')
    assert ts.container_pid('test container') == 123
    assert seen == [['sg', 'docker', '-c', "docker inspect -f '{{.State.Pid}}' 'test container'"]]


def test_until_exit_and_dynamic_tasks(tmp_path):
    program = '''import ctypes,threading,time
libc=ctypes.CDLL(None)
def worker(i):
    libc.prctl(15, ('xfill-%d'%i).encode(), 0,0,0)
    time.sleep(.2)
time.sleep(.1)
threads=[threading.Thread(target=worker,args=(i,)) for i in range(2)]
for t in threads:t.start()
for t in threads:t.join()
'''
    child = subprocess.Popen([sys.executable, '-c', program])
    try:
        path = tmp_path / 'tasks.jsonl'
        stats = ts.sample(child.pid, path)  # no seconds: terminate on exit, even if unreaped
        result = ts.aggregate(path)
        assert stats['elapsed_s'] < 3
        assert result['groups']['expert_fill']['samples'] > 50
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert any(len(r.get('tasks', [])) == 3 for r in rows)
        # Hot-loop parsing and the fixture-tested parser must agree on counter order.
        for row in rows:
            for tid, state, ut, st, minor, major, cpu, comm in row.get('tasks', []):
                assert min(ut, st, minor, major, cpu) >= 0
    finally:
        child.wait(timeout=3)
