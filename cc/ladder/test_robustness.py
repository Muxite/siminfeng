"""Offline regressions: captured Qwen fit evidence plus mocked process boundaries."""
import argparse
import copy
from pathlib import Path

import pytest

from cc.ladder import ladder

CAPTURE = ladder.HERE / 'fixtures/qwen35-9b-fit'
SOURCE = ladder.ROOT / 'experiments/jetson/LADDER/qwen35-9b/fit-20261111T194923.506211Z'


@pytest.fixture
def offline_fit(tmp_path, monkeypatch):
    spec = ladder.read(ladder.HERE / 'models.json')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'SHAPE_ONLY model identity; never loaded')
    spec['models']['qwen35-9b']['path'] = str(model)
    spec_path = tmp_path / 'spec.json'
    ladder.write(spec_path, spec)
    out = tmp_path / 'results'
    captured = ladder.read(CAPTURE / 'summary.json')['trials'][0]['metrics']
    stderr = (CAPTURE / 'gpu-stderr.txt').read_text()
    calls = []

    def bounded(command, deadline, env=None):
        calls.append(command)
        if Path(command[1]).name == 'preflight.py':
            return 0
        assert Path(command[1]).name == 'om_run.sh'
        assert env['OM_MIN_AVAIL_MIB'] == str(ladder.watchdog_floor())
        trial = Path(command[2])
        trial.mkdir(parents=True, exist_ok=True)
        (trial / 'stderr.txt').write_text(stderr)
        ladder.write(trial / 'result.json', dict(watchdog_trips=0, oom_killed=False))
        return 1

    monkeypatch.setenv('OM_MIN_AVAIL_MIB', '600')
    monkeypatch.setattr(ladder, 'bounded', bounded)
    monkeypatch.setattr(ladder, 'refuse_busy', lambda: None)
    monkeypatch.setattr(ladder, 'clean_owned', lambda out: None)
    monkeypatch.setattr(ladder.run_summary, 'summarize', lambda *a: copy.deepcopy(captured))

    def run(*extra):
        return ladder.main(['fit', 'qwen35-9b', '--spec', str(spec_path), '--out', str(out), *extra])

    return argparse.Namespace(run=run, calls=calls, spec=spec, spec_path=spec_path,
                              out=out, parent=out / 'qwen35-9b')


@pytest.mark.parametrize('change', ['argv', 'env', 'switch_regex', 'order', 'protocol', 'floor', 'image', 'model'])
def test_key_change_creates_new_dir(offline_fit, monkeypatch, change):
    case = offline_fit
    assert case.run() == 1
    first = next(case.parent.glob('fit-*'))
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    assert ladder.read(first / 'summary.json')['chosen'] is None
    count = len(case.calls)
    assert case.run() == 1
    assert len(case.calls) == count  # Same-key terminal failure is reused.
    model = case.spec['models']['qwen35-9b']
    # Change the last candidate: hashing only the first/selected config is insufficient.
    if change == 'argv':
        model['configs'][-1]['argv'] += ['-t', '2']
    elif change == 'env':
        model['configs'][-1]['env']['SHAPE_ONLY'] = '1'
    elif change == 'switch_regex':
        model['configs'][-1]['switch_regex'] += ['SHAPE_ONLY']
    elif change == 'order':
        model['configs'].reverse()
    elif change == 'protocol':
        monkeypatch.setattr(ladder, 'PROTOCOL', ladder.PROTOCOL + '-changed')
    elif change == 'floor':
        monkeypatch.setenv('OM_MIN_AVAIL_MIB', '128')
    elif change == 'image':
        case.spec['image'] += '-changed'
    elif change == 'model':
        Path(model['path']).write_bytes(b'changed SHAPE_ONLY identity')
    ladder.write(case.spec_path, case.spec)
    assert case.run() == 1
    dirs = sorted(case.parent.glob('fit-*'))
    assert len(dirs) == 2
    assert ladder.read(dirs[0] / 'summary.json')['identity_key'] != ladder.read(dirs[1] / 'summary.json')['identity_key']
    assert before == {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}


def test_refit_creates_new_dir(offline_fit):
    case = offline_fit
    assert case.run() == 1
    first = next(case.parent.glob('fit-*'))
    before = (first / 'summary.json').read_bytes()
    assert case.run('--refit') == 1
    dirs = sorted(case.parent.glob('fit-*'))
    assert len(dirs) == 2
    assert ladder.read(dirs[0] / 'summary.json')['identity_key'] == ladder.read(dirs[1] / 'summary.json')['identity_key']
    assert (first / 'summary.json').read_bytes() == before


@pytest.mark.parametrize('after_success', [False, True])
@pytest.mark.parametrize('rejection', ['nonzero', 'unsupported_flags'])
def test_preflight_abort_exit_3(offline_fit, monkeypatch, after_success, rejection):
    case = offline_fit
    commands = []
    preflights = []

    def bounded(command, deadline, env=None):
        commands.append(command)
        if Path(command[1]).name == 'preflight.py':
            preflights.append(command)
            if after_success and len(preflights) == 1:
                return 0
            if rejection == 'unsupported_flags':
                raise RuntimeError('preflight: unsupported flags: --chat-template-kwargs')
            return 1
        # Only reached for the first, successful trial in the --all case.
        directory = Path(command[2])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'stderr.txt').write_text('')
        return 0

    good = ladder.read(CAPTURE / 'summary.json')['trials'][-1]['metrics']
    assert ladder.successful(good)
    monkeypatch.setattr(ladder.run_summary, 'summarize', lambda *a: copy.deepcopy(good))
    monkeypatch.setattr(ladder, 'bounded', bounded)
    assert case.run('--all') == 3
    result = ladder.read(next(case.parent.glob('fit-*/summary.json')))
    assert result['status'] == 'harness_error' and result['chosen'] is None
    assert len(preflights) == (2 if after_success else 1)
    assert len(commands) == (3 if after_success else 1)  # Never run rejected candidate.
    trial = result['trials'][-1]
    assert trial['fits'] is None  # Harness rejection is not a does-not-fit observation.
    assert trial['metrics']['status'] == 'harness_error'
    assert trial['metrics']['watchdog_floor_mib'] == 600
    assert trial['metrics']['mem_available_min_mib'] is None
    assert case.run('--all') == 3  # Retain the exit status on same-key reuse.


def test_oom_classification_from_captured_stderr(offline_fit):
    # Byte-exact captures; no changes to experiments/ and no manufactured timings.
    assert (CAPTURE / 'summary.json').read_bytes() == (SOURCE / 'summary.json').read_bytes()
    assert (CAPTURE / 'gpu-stderr.txt').read_bytes() == (SOURCE / 'gpu/stderr.txt').read_bytes()
    stderr = (CAPTURE / 'gpu-stderr.txt').read_text()
    assert 'cudaMalloc failed: out of memory' in stderr
    assert offline_fit.run() == 1
    result = ladder.read(next(offline_fit.parent.glob('fit-*/summary.json')))
    captured = ladder.read(CAPTURE / 'summary.json')['trials'][0]['metrics']
    for trial in result['trials']:
        assert trial['fits'] is False
        assert trial['metrics']['failure_kind'] == 'oom_error'
        assert trial['metrics']['watchdog_floor_mib'] == 600
        assert trial['metrics']['mem_available_min_mib'] == captured['mem_available_min_mib']


@pytest.mark.parametrize('step', ['fit', 'speed', 'acc'])
def test_all_step_keys_include_full_ladder_and_floor(offline_fit, monkeypatch, step):
    case = offline_fit
    monkeypatch.setattr(ladder, 'speed_step', lambda *a: dict(complete=True))
    monkeypatch.setattr(ladder, 'acc_step', lambda *a: dict(complete=True))
    data = case.out.parent / 'SHAPE_ONLY.jsonl'
    data.write_text('{}\n')
    # Explicit selection avoids requiring a successful fit for this identity test.
    command = [step, 'qwen35-9b', '--config', 'gpu', '--spec', str(case.spec_path), '--out', str(case.out)]
    if step == 'acc':
        command += ['--data', str(data)]
    ladder.main(command)
    case.spec['models']['qwen35-9b']['configs'][-1]['env']['SHAPE_ONLY'] = 'changed'
    ladder.write(case.spec_path, case.spec)
    ladder.main(command)
    monkeypatch.setenv('OM_MIN_AVAIL_MIB', '128')
    ladder.main(command)
    dirs = list(case.parent.glob(f'{step}-*'))
    assert len(dirs) == 3
    assert len({ladder.read(p / 'step.json')['key'] for p in dirs}) == 3


@pytest.mark.parametrize('change', ['argv', 'env', 'switch_regex', 'protocol', 'floor', 'legacy'])
def test_chosen_config_requires_current_key(offline_fit, monkeypatch, change):
    case = offline_fit
    assert case.run() == 1
    path = next(case.parent.glob('fit-*/summary.json'))
    record = ladder.read(path)
    model = case.spec['models']['qwen35-9b']
    # Exercise selection with a completed successful envelope, retaining the real config.
    record['chosen'] = model['configs'][0]
    ladder.write(path, record)
    args = argparse.Namespace(out=case.out, model_id='qwen35-9b', config=None, dry_run=False)
    assert ladder.chosen_config(args, model, case.spec['image']) == record['chosen']
    if change == 'argv':
        model['configs'][-1]['argv'] += ['-t', '2']
    elif change == 'env':
        model['configs'][-1]['env']['SHAPE_ONLY'] = 'changed'
    elif change == 'switch_regex':
        model['configs'][-1]['switch_regex'] += ['SHAPE_ONLY']
    elif change == 'protocol':
        monkeypatch.setattr(ladder, 'PROTOCOL', 'changed')
    elif change == 'floor':
        monkeypatch.setenv('OM_MIN_AVAIL_MIB', '128')
    elif change == 'legacy':
        record.pop('identity_inputs')
        ladder.write(path, record)
    with pytest.raises(ValueError, match='current identity key'):
        ladder.chosen_config(args, model, case.spec['image'])


@pytest.mark.parametrize('summary,stderr,expected', [
    ({'status': 'failed', 'watchdog_trips': 1}, '', 'watchdog_trip'),
    ({'status': 'failed', 'oom_killed': True}, '', 'oom_error'),
    ({'status': 'failed'}, 'engine: unable to allocate buffer', 'oom_error'),
    ({'status': 'failed'}, 'unrelated error', 'other'),
])
def test_failure_kinds(summary, stderr, expected):
    assert ladder.failure_kind(summary, stderr) == expected
