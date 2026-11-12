"""Host-contention evidence invalidates scoring; all fixtures are temporary."""
import json
import os

import pytest

from cc.abrun import gate, run_summary
from cc.om_run import contention


@pytest.fixture
def contended_run(tmp_path):
    (tmp_path / 'result.json').write_text(json.dumps(dict(status='ok', wall_s=1)))
    (tmp_path / 'stdout.txt').write_text('same output\n')
    (tmp_path / 'stderr.txt').write_text(
        'eval time = 1000 ms / 120 runs (120 tokens per second)\n')
    (tmp_path / 'contention.json').write_text(json.dumps(dict(
        max_foreign_heavy=2, names=['cc1plus', 'ninja'], samples=[dict(
            time=1, foreign_heavy=[dict(pid=42, ppid=1, comm='ninja')])])))
    return tmp_path


def test_summary_contention(contended_run):
    summary = run_summary.summarize(contended_run, write=False)
    assert summary['host_contention'] is True
    assert summary['max_foreign_heavy'] == 2


@pytest.mark.parametrize('record', [None, {'max_foreign_heavy': 0}])
def test_summary_without_contention(tmp_path, record):
    if record is not None:
        (tmp_path / 'contention.json').write_text(json.dumps(record))
    summary = run_summary.summarize(tmp_path, write=False)
    assert summary['host_contention'] is False
    assert summary['max_foreign_heavy'] == 0


@pytest.mark.parametrize('arm', ['base', 'other'])
def test_contention_invalidates_gate(contended_run, arm):
    arms = {'base': [dict(status='ok', eval_tps=100)],
            'other': [dict(status='ok', eval_tps=120)]}
    assert gate.evaluate(arms)['comparisons']['other']['verdict'] == 'improved'
    arms[arm][0].update(run_summary.summarize(contended_run, write=False))
    verdict = gate.evaluate(arms)
    assert f'{arm}: host_contention' in verdict['reasons']
    assert verdict['comparisons']['other']['verdict'] == 'null'


def test_sampler_excludes_descendants_and_zombies(tmp_path):
    # Synthetic proc tree: the run's build descendants are allowed; siblings
    # and processes owned by other users are still foreign.
    for pid, ppid, comm, state in [(10, 1, 'bash', 'S'), (11, 10, 'make', 'S'),
                                    (12, 11, 'cc1plus', 'R'), (20, 1, 'ninja', 'S'),
                                    (21, 20, 'rustc', 'R'), (22, 1, 'ld', 'Z'),
                                    (23, 1, 'python3', 'R')]:
        path = tmp_path / str(pid)
        path.mkdir()
        (path / 'stat').write_text(f'{pid} ({comm}) {state} {ppid} 0 0\n')
    foreign = contention.foreign_heavy(10, tmp_path)
    assert [(p['pid'], p['comm']) for p in foreign] == [(20, 'ninja'), (21, 'rustc')]
    assert all(p['uid'] == os.getuid() for p in foreign)
