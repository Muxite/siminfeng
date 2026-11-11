"""Small offline fixtures: no Docker, GPU, or production input writes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('shape_report', Path(__file__).with_name('shape_report.py'))
shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape)


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in value) if lines else json.dumps(value))


def test_interval_math():
    assert shape.union([(5, 9), (0, 3), (2, 5), (12, 12), (15, 14)]) == [(0, 9)]
    assert shape.duration([(0, 10), (2, 5), (8, 12)]) == 12
    assert shape.intersection([(0, 10), (20, 30)], [(5, 25)]) == [(5, 10), (20, 25)]
    assert shape.duration(shape.intersection([(0, 10)], [(10, 12)])) == 0
    assert shape.union([]) == []


@pytest.mark.parametrize('text, expected', [
    ('error: invalid option', 'flag_error'), ('unknown argument --foo', 'flag_error'),
    ('invalid argument during build_engine.py', 'flag_error'),
    ('CUDA out of memory', 'oom/watchdog'), ('watchdog tripped', 'oom/watchdog'),
    ('OOM-killed', 'oom/watchdog'), ('docker: Error response from daemon', 'docker_start'),
    ('Cannot connect to the Docker daemon', 'docker_start'),
    ('ninja: build stopped', 'build_fail'), ('build_engine.py --target x', 'build_fail'),
    ('timed out', 'timeout'), ('timeout', 'timeout'), ('exit 7', 'other'),
])
def test_failure_classification(text, expected):
    assert shape.classify_failure(text) == expected


def test_worklog_aggregation():
    rows = [
        dict(event='start', task='H1', worker='codex', ts='2026-11-09T12:00Z'),
        dict(event='end', task='H1', wall_s=10, tokens='20', ts='2026-11-10T12:00Z'),
        dict(event='checkin', task='H1', verdict='REWORK', issues=['a','b'], rework_rounds=1, ts='2026-11-10T12:01Z'),
        dict(event='end', task='H1', worker='codex', wall_s=5, tokens=8, ts='2026-11-10T12:02Z'),
        dict(event='checkin', task='H1', verdict='PASS', issues=[], rework_rounds=1, opus_fix=True, ts='2026-11-10T12:03Z'),
        dict(event='start', task='H2', worker='other', ts='2026-11-09T12:00Z'),
    ]
    result = shape.aggregate_worklog(rows, shape.timestamp('2026-11-10'))
    assert len(result['tasks']) == 1
    task = result['tasks'][0]
    assert (task['worker'], task['wall_s'], task['tokens'], task['verdict']) == ('codex', 15, 28, 'PASS')
    assert (task['issues_count'], task['rework_rounds'], task['opus_fix']) == (2, 1, True)
    assert result['workers']['codex'] == dict(tasks=1, wall_s=15, tokens=28, issues_count=2, rework_rounds=1, opus_fixes=1)


@pytest.fixture
def fixture_root(tmp_path):
    day = shape.timestamp('2026-11-10')
    records = []

    def operation(op, task, argv, start, end, failed=False, stderr=None):
        records.extend([
            dict(kind='outcome', op=op, ts=shape.iso(day+end), status='error' if failed else 'ok',
                 exit_code=1 if failed else 0, elapsed_s=end-start, stderr={'path': stderr} if stderr else {}),
            dict(kind='intent', op=op, task=task, agent='codex', argv=argv, ts=shape.iso(day+start)),
            dict(kind='started', op=op, ts=shape.iso(day+start)),
        ])
    operation('gpu1', 'J-OM1', ['om_run.sh'], 0, 3600)
    operation('gpu2', 'J-KS1', ['kernel_screen.py'], 1800, 5400)
    operation('gpu3', 'J-SPEC1', ['om_spec.py'], 36000, 39600)
    operation('build', 'build-x', ['build_engine.py'], 2700, 7200)
    error = tmp_path/'artifacts/err'
    error.parent.mkdir()
    error.write_text('unknown argument OLD\n'+'x'*5000+'\nwatchdog tripped\n')
    operation('bad', 'H1', ['python3','tool.py'], 10, 11, True, 'artifacts/err')
    records.append(dict(kind='intent', op='unfinished', task='J-OM2', agent='codex', argv=['om_run.sh'], ts=shape.iso(day+8000)))
    write(tmp_path/'evidence/jetson-arm/worker.jsonl', records, True)
    write(tmp_path/'evidence/jetson-arm/claude-code.jsonl', [
        dict(event='PreToolUse', ts=shape.iso(day), payload={'tool_name':'Bash'}, argv=['om_run.sh'], op='hook'),
        dict(event='PostToolUse', ts=shape.iso(day), payload={'tool_name':'Bash'})], True)
    write(tmp_path/'c2/worklog.jsonl', [dict(event='end', task='H1', worker='codex', wall_s=2, tokens='9', ts=shape.iso(day))], True)
    write(tmp_path/'experiments/jetson/OM/one/a/result.json', {'status':'ok'})
    write(tmp_path/'experiments/jetson/arbitrary-run/verdict.json', {'status':'completed', 'comparisons':{'B':{'verdict':'improved'}, 'C':{'verdict':'null'}}})
    write(tmp_path/'experiments/handoff/verdict.json', {'state':'rejected'})
    return tmp_path, day


def test_fixture_report_and_queue(fixture_root):
    root, day = fixture_root
    report = shape.generate(root, now=day+43200)
    r = report['timeline']['2026-11-10']
    assert r['gpu_busy_h'] == 2.5
    assert r['build_h'] == 1.25
    assert r['overlap_h'] == .75
    assert r['longest_idle_gap_h'] == 8.5
    assert r['longest_queued_idle_gap_h'] is None
    assert report['totals']['ops'] == 6
    assert report['totals']['failed_ops'] == 1
    assert report['failures']['oom/watchdog']['examples'] == ['bad']
    assert report['tool_calls_per_day'] == {'2026-11-10':{'Bash':1}}
    assert report['experiments']['abrun_verdict'] == {'improved':1, 'null':1}
    assert report['totals']['abrun_count'] == 1
    assert report['experiments']['result_status'] == {'ok':1}
    assert len(shape.markdown(report).splitlines()) <= 200
    write(root/'lab/jobq/queued.json', dict(id='queued', **{'class':'gpu'}, state='running', submitted=int((day+7200)*1e9), start=int((day+32400)*1e9), end=int((day+34200)*1e9), concurrent_jobs=['build']))
    with_queue = shape.generate(root, now=day+43200)
    q = with_queue['timeline']['2026-11-10']
    assert q['gpu_busy_h'] == 3
    assert q['longest_idle_gap_h'] == 7.5
    assert q['longest_queued_idle_gap_h'] == 7
    assert with_queue['jobs'][0]['concurrent_jobs'] == ['build']
    clipped = shape.generate(root, since=day+3600, now=day+43200)
    assert clipped['timeline']['2026-11-10']['gpu_busy_h'] == 2
    assert shape.timestamp('2026-11-10T01:00:00+01:00') == day


def test_cli(fixture_root, tmp_path):
    root, _ = fixture_root
    out, data = tmp_path/'out.md', tmp_path/'out.json'
    process = subprocess.run([sys.executable, str(Path(__file__).with_name('shape_report.py')),
                              '--root', str(root), '--out', str(out), '--json', str(data)],
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert len([line for line in out.read_text().splitlines() if line.startswith('## ')]) == 5
    assert json.loads(data.read_text())['totals']['ops'] == 6


# Verbatim fixture copied from experiments/jetson/OM/base-20261110-0435/mmap-ncmoe40-a/stderr.txt
REAL_OM_FLAG_STDERR = 'ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7607 MiB):\n  Device 0: Orin, compute capability 8.7, VMM: yes, VRAM: 7607 MiB\nerror: invalid parameter for argument: -c\n'


@pytest.mark.parametrize('wrapper_failed', [False, True])
def test_om_run_flag_stderr_real_fixture(tmp_path, wrapper_failed):
    directory = tmp_path/'runs/bad-bench'
    directory.mkdir(parents=True)
    (directory/'stderr.txt').write_text(REAL_OM_FLAG_STDERR)
    # Captured run result also says failed despite run_op's historical rc=0.
    write(directory/'result.json', dict(exit_code=1, wall_s=.5, status='failed',
                                      oom_killed=False, watchdog_trips=0))
    (tmp_path/'wrapper.stderr').write_text('wrapper finished\n')
    write(tmp_path/'evidence/jetson-arm/ops.jsonl', [
        dict(kind='intent', op='flag', argv=['bash', 'cc/om_run/om_run.sh', 'runs/bad-bench',
             'image', '', 'llama-bench', '-c', '4096'], task='J-OM-S', ts='2026-11-10T00:00Z'),
        dict(kind='outcome', op='flag', status='error' if wrapper_failed else 'ok',
             exit_code=1 if wrapper_failed else 0, stderr={'path': 'wrapper.stderr'},
             ts='2026-11-10T00:01Z')], True)
    report = shape.generate(tmp_path)
    assert report['failures']['flag_error']['count'] == 1
    assert report['operations'][0]['failed']
    assert report['operations'][0]['runner_results'][0]['exit_code'] == 1
    assert shape.classify_failure('usage: llama-bench [options]') == 'flag_error'


def test_lead_events_are_separate(fixture_root):
    root, _ = fixture_root
    rows = [dict(task='J-OM8', event=event, detail=event, ts='2026-11-10T12:00Z')
            for event in ('contamination', 'result', 'failure')]
    # A lead event on the same task must not change its delegated worker.
    rows += [dict(task='J-OM8', event='start', worker='codex', ts='2026-11-10T11:00Z')]
    write(root/'c2/worklog.jsonl', rows, True)
    report = shape.generate(root)
    assert len(report['delegation']['tasks']) == 1
    assert report['delegation']['tasks'][0]['worker'] == 'codex'
    assert len(report['delegation']['lead_events']) == 3
    assert all(r['worker'] == 'opus' for r in report['delegation']['lead_events'])
    assert set(report['delegation']['workers']) == {'codex'}
    assert '### Lead events' in shape.markdown(report)
    clipped = shape.aggregate_worklog(rows, shape.timestamp('2026-11-11'))
    assert clipped['lead_events'] == []


def test_known_task_names_mapped():
    names = ['J-FIO', 'J-ISO1', 'J-Q', 'J-UP', 'J1', 'J1-container', 'J2-build',
             'J2-image-qual', 'J2-model', 'J2-repro', 'J5', 'J5-regress',
             'codex-H2b-H4b', 'codex-J-OM8b', 'sleep', 'true']
    assert all(shape.task_class(name) != 'other' for name in names)
    assert shape.task_class('sleep') == shape.task_class('true') == 'test-noise'
    assert shape.task_class('new-unmapped-task') == 'other'
