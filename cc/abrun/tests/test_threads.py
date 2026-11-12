"""Thread capture spec, preflight, summary and shell integration without Docker."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
from cc.abrun import abrun, gate, run_summary
from cc.thread_sampler.test_thread_sampler import write_capture

ROOT = Path(__file__).resolve().parents[3]


def spec():
    return json.loads((ROOT / 'cc/abrun/specs/J-OM12.json').read_text())


def test_om12_plan(monkeypatch):
    monkeypatch.delenv('ABRUN_RUNNER', raising=False)
    s = spec()
    abrun.validate(s)
    _, runs = abrun.plan(s)
    assert [r['arm'] for r in runs] == ['base', 'pg8', 'pg8t4', 'pg8t6', 'pg8t6', 'pg8t4', 'pg8', 'base']
    for run in runs:
        assert 'OM_THREADS=1' in run['command']
        if run['arm'] in ('pg8t4', 'pg8t6'):
            assert run['command'][-2:] == ['-t', run['arm'][-1]]
        else:
            assert '-t' not in run['command']
    s.pop('threads')
    assert all('OM_THREADS=1' not in r['command'] for r in abrun.plan(s)[1])
    assert s['arms']['base']['env'] == {}


@pytest.mark.parametrize('field,value', [('threads', 1), ('threads', 'yes'), ('extra_argv', '-t 4'), ('extra_argv', [4])])
def test_validation(field, value):
    s = spec()
    (s if field == 'threads' else s['arms']['pg8'])[field] = value
    with pytest.raises(ValueError, match=field):
        abrun.validate(s)


def test_extra_argv_preflight(tmp_path, monkeypatch):
    s = spec()
    calls = []
    def check(image, tool, argv):
        calls.append(argv)
        if '-t' in argv:
            raise ValueError('unsupported flags: -t')
    monkeypatch.setattr(abrun.preflight, 'check', check)
    assert abrun.execute(s, tmp_path / 'run', []) == 1
    assert any(a[-2:] == ['-t', '4'] for a in calls)
    v = json.loads((tmp_path / 'run/verdict.json').read_text())
    assert 'unsupported flags: -t' in v['invalid_reasons']


def test_summary_secondary(tmp_path):
    write_capture(tmp_path / 'threads.jsonl')
    (tmp_path / 'stderr.txt').write_text('eval time = 1500 ms / 2 runs (750 ms per token, 1.3 tokens per second)\n')
    s = run_summary.summarize(tmp_path)
    assert s['threads.cpu_workers_r_fraction'] == 1
    assert s['threads.main_workers_d_fraction'] == .25
    assert s['threads.majflt_per_token'] == 4.5
    v = gate.evaluate({'base': [s], 'pg8': [s]}, 'eval_ms_per_token')
    assert v['secondary_metrics']['base']['threads.majflt_per_token'] == 4.5
    assert 'Threads majflt/token' in gate.secondary_table(v)
    assert json.loads((tmp_path / 'threads.summary.json').read_text())['phase_window']['tokens'] == 2


def test_om_run_commands_unchanged(tmp_path):
    # Stub executables only; never invoke Docker, resource observers, or a GPU.
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    docker = bindir / 'docker'
    docker.write_text(f'''#!{sys.executable}
import json,os,sys,time
args=sys.argv[1:]
with open(os.environ['DOCKER_LOG'],'a') as f:f.write(json.dumps(args)+'\\n')
if args[0]=='run': print('fake-container')
elif args[0]=='wait': time.sleep(.2); print('0')
elif args[0]=='inspect': print(os.getppid() if '-f' in args else 'false')
elif args[0]=='logs': print('eval time = 100 ms / 2 runs',file=sys.stderr)
''')
    docker.chmod(0o755)
    python = bindir / 'python3'
    python.write_text(f'''#!{sys.executable}
import os,sys,time
if len(sys.argv)>1 and sys.argv[1].endswith('resource_observe.py'):
    if '--out' in sys.argv:open(sys.argv[sys.argv.index('--out')+1],'w').close()
    time.sleep(60)
elif len(sys.argv)>1 and sys.argv[1].endswith('thread_sampler.py'):
    # The real sampler itself is covered by local busy-process tests. No proc
    # namespace/container dependency belongs in this shell command comparison.
    if '--out' in sys.argv:open(sys.argv[sys.argv.index('--out')+1],'w').close()
elif len(sys.argv)>1 and sys.argv[1].endswith('run_protocol.py'):
    # The copied baseline has no helper tree; cache behavior is tested separately.
    pass
else:os.execv({sys.executable!r},[{sys.executable!r}]+sys.argv[1:])
''')
    python.chmod(0o755)
    original = subprocess.check_output(['git', 'show', 'HEAD:./cc/om_run/om_run.sh'], cwd=ROOT, text=True)
    baseline = tmp_path / 'original.sh'
    baseline.write_text(original)
    commands = []
    current = ROOT / 'cc/om_run/om_run.sh'
    for script, threads, guard in [(baseline, None, None), (current, None, None),
                                   (current, '1', None), (current, None, '0')]:
        idx = len(commands)
        out = tmp_path / f'out{idx}'
        log = tmp_path / f'docker{idx}.jsonl'
        env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'], DOCKER_LOG=str(log))
        env.pop('DOCKER_CMD_PREFIX', None)
        env.pop('OM_THREADS', None)
        env.pop('OM_CONTENTION_GUARD', None)
        if guard is not None:
            env['OM_CONTENTION_GUARD'] = guard
        if threads:
            env['OM_THREADS'] = threads
        subprocess.run(['bash', str(script), str(out), 'test-image', '-e TEST=1', 'llama-completion', '-t', '4'], env=env, check=True, capture_output=True, timeout=10)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        run = next(row for row in rows if row[0] == 'run')
        commands.append([re.sub(r'om-\d+-\d+', 'CONTAINER', arg) for arg in run])
        assert (out / 'threads.jsonl').exists() == bool(threads)
        if script == current:
            assert (out / 'contention.json').exists() == (guard != '0')
            if guard != '0':
                assert json.loads((out / 'contention.json').read_text())['samples']
    assert all(command == commands[0] for command in commands[1:])
