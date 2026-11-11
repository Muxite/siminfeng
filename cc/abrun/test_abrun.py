"""CPU-only acceptance and failure-path tests; fixture writes are confined to tmp."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import pytest
from cc.abrun import abrun, gate, preflight, run_summary

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'experiments/jetson/OM/om7-20261110-1555'
HELP = (Path(__file__).parent / 'fixtures/llama-bench-help.txt').read_text()


def test_real_base_summary(tmp_path):
    target = tmp_path / 'base-a'
    shutil.copytree(FIXTURE / 'base-a', target)
    summary = run_summary.summarize(target)
    assert summary['status'] == 'ok'
    assert summary['eval_tps'] == 1.18
    assert summary['prompt_tps'] == 5.69
    assert summary['nvme_gib'] == pytest.approx(57.2, abs=0.5)
    assert summary['avg_kib_per_io'] == pytest.approx(95, abs=3)
    assert summary['load_s'] == pytest.approx(50.26553)
    assert summary['eval_tokens'] == 63
    assert summary['nvme_gib_per_token'] == summary['nvme_gib'] / 63
    assert summary['output_sha'] == hashlib.sha256((target / 'stdout.txt').read_bytes()).hexdigest()[:12]
    assert json.loads((target / 'summary.json').read_text()) == summary


@pytest.mark.parametrize('arm', ['pf-a', 'pf-b'])
def test_prefetch_fixture(tmp_path, arm):
    target = tmp_path / arm
    shutil.copytree(FIXTURE / arm, target)
    summary = run_summary.summarize(target, [r'expert_prefetch:.*enabled'])
    assert summary['avg_kib_per_io'] == pytest.approx(5, abs=1)
    assert len(summary['switch_lines']) == 1


def test_llama_bench(tmp_path):
    source = ROOT / 'experiments/jetson/J-E007-unfuse-by-type/bf16-base-a.json'
    shutil.copyfile(source, tmp_path / 'stdout.txt')
    summary = run_summary.summarize(tmp_path)
    row = json.loads(source.read_text())[0]
    assert summary['eval_tps'] == row['avg_ts']
    assert summary['tg_samples_ts'] == row['samples_ts']
    assert summary['tg_steady'] == statistics.mean(row['samples_ts'][1:])
    assert 'eval_tps' not in summary['reasons']


def test_bench_pp_and_tg(tmp_path):
    rows = [{'n_prompt': 64, 'n_gen': 0, 'avg_ts': 20, 'samples_ts': [10, 25, 25]},
            {'n_prompt': 0, 'n_gen': 32, 'avg_ts': 5, 'samples_ts': [3, 6, 6]}]
    (tmp_path / 'stdout.txt').write_text(json.dumps(rows))
    s = run_summary.summarize(tmp_path, write=False)
    assert (s['prompt_tps'], s['eval_tps'], s['pp_steady'], s['tg_steady']) == (20, 5, 25, 6)


def test_last_perf_lines(tmp_path):
    (tmp_path / 'stderr.txt').write_text('\n'.join([
        'common_perf_print: prompt eval time = 12 ms / 2 tokens (1 tokens per second)',
        'common_perf_print: eval time = 30 ms / 3 runs (2 tokens per second)',
        'common_perf_print: prompt eval time = 12 ms / 2 tokens (7 tokens per second)',
        'common_perf_print: eval time = 30 ms / 3 runs (8 tokens per second)']))
    s = run_summary.summarize(tmp_path, write=False)
    assert (s['prompt_tps'], s['eval_tps']) == (7, 8)


def test_missing_and_malformed(tmp_path):
    (tmp_path / 'result.json').write_text('oops')
    (tmp_path / 'telemetry.jsonl').write_text('broken\nnull\n{"metrics":[null]}\n')
    s = run_summary.summarize(tmp_path, write=False)
    for key in ('status', 'wall_s', 'eval_tps', 'nvme_gib', 'avg_kib_per_io',
                'mem_available_min_mib', 'swap_max_mib', 'gpu_freq_median', 'output_sha'):
        assert s[key] is None
        assert s['reasons'][key]
    assert not (tmp_path / 'summary.json').exists()


@pytest.mark.parametrize('unit,factor', [('KiB', 1), ('MiB', 1024)])
def test_telemetry_units(tmp_path, unit, factor):
    rows = []
    for i in (0, 1, 2):
        rows.append({'metrics': [
            {'name': 'device_read', 'value': 100 + i * 1024 / factor, 'unit': unit},
            {'name': 'device_read_ios', 'value': 10 + i * 16},
            {'name': 'mem_available', 'value': 3000 - i, 'unit': 'MiB'},
            {'name': 'swap_total', 'value': 4000, 'unit': 'MiB'},
            {'name': 'swap_free', 'value': 3500 - i, 'unit': 'MiB'},
            {'name': 'gpu_freq', 'value': [300, 900, 600][i], 'unit': 'MHz'}]})
    (tmp_path / 'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    s = run_summary.summarize(tmp_path, write=False)
    assert s['nvme_gib'] == 2048 / 1024 ** 2
    assert s['avg_kib_per_io'] == 64
    assert s['mem_available_min_mib'] == 2998
    assert s['swap_max_mib'] == 502
    assert s['gpu_freq_median'] == 600


def test_cli_no_write(tmp_path):
    target = tmp_path / 'base'
    shutil.copytree(FIXTURE / 'base-a', target)
    (target / 'summary.json').unlink(missing_ok=True)
    result = subprocess.run([sys.executable, str(Path(run_summary.__file__)), str(target),
                             '--json', '--no-write'], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['eval_tps'] == 1.18
    assert not (target / 'summary.json').exists()


def runs(numbers, sha='aaa'):
    return [{'eval_tps': n, 'output_sha': sha, 'status': 'ok'} for n in numbers]


@pytest.mark.parametrize('b,expected', [(110, 'improved'), (102, 'null'), (90, 'regressed')])
def test_gate_verdict(b, expected):
    g = gate.evaluate({'base': runs([100, 100]), 'B': runs([b, b])})
    assert g['comparisons']['B']['verdict'] == expected
    assert g['comparisons']['B']['ratio'] == b / 100
    assert g['inputs']['B'] == [b, b]
    assert g['threshold'] == .03
    assert g['output']['class'] == 'identical'


def test_gate_noise_and_drift():
    g = gate.evaluate({'base': runs([90, 110]), 'B': runs([120, 120])})
    assert g['aa_gap'] == .2
    assert g['drift'] == .2
    assert g['drift_halves'] == [90, 110]
    assert g['threshold'] == .4
    assert g['comparisons']['B']['verdict'] == 'null'


def test_output_differs():
    g = gate.evaluate({'base': runs([1, 1]), 'B': runs([2, 2], 'bbb')})
    assert g['output']['class'] == 'differs'
    assert list(g['output']['differs']) == ['B']
    assert len(g['output']['differs']['B']) == 2


@pytest.mark.parametrize('bad', [None, 0, float('nan')])
def test_missing_gate_metric(bad):
    g = gate.evaluate({'base': runs([1, 1]), 'B': runs([bad, 2])})
    assert g['reasons']
    assert g['comparisons']['B']['verdict'] == 'null'


def test_preflight_reject_and_cache(tmp_path):
    calls = []
    def fake(image, tool):
        calls.append((image, tool))
        return HELP
    with pytest.raises(ValueError, match='unsupported flags: -c'):
        preflight.check('img', 'llama-bench', ['-c', '4096'], fake, tmp_path)
    preflight.check('img', 'llama-bench', ['-m', '/m', '-n', '32', '-ngl', '99', '--output=json'], fake, tmp_path)
    assert calls == [('img', 'llama-bench')]
    with pytest.raises(ValueError, match='--n-gpu'):
        preflight.check('img', 'llama-bench', ['--n-gpu'], fake, tmp_path)
    preflight.check('img2', 'llama-bench', ['-n', '-1'], fake, tmp_path)
    assert len(calls) == 2


def spec(tmp_path, count=2):
    arms = {'base': {'image': 'test', 'env': {}},
            'pf': {'image': 'test', 'env': {'SWITCH': '1'}, 'expect_switch': True}}
    if count == 3:
        arms['pfn'] = {'image': 'test', 'env': {'SWITCH': '2'}, 'expect_switch': True}
    return {'id': 'test', 'outdir_base': str(tmp_path / 'outputs'), 'runner': 'om_run',
            'tool': 'llama-bench', 'argv': ['-n', '32'], 'arms': arms,
            'switch_regex': ['switch active'], 'metric': 'eval_tps',
            'order': 'ABBA' if count == 2 else 'ABCCBA', 'rounds': 1}


@pytest.mark.parametrize('count,order', [(2, ['base', 'pf', 'pf', 'base']),
                                       (3, ['base', 'pf', 'pfn', 'pfn', 'pf', 'base'])])
def test_dry_run(tmp_path, count, order):
    s = spec(tmp_path, count)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(s))
    result = subprocess.run([sys.executable, str(Path(abrun.__file__)), str(path), '--dry-run'],
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[0] == 'Order: ' + ', '.join(order)
    assert sum(line.startswith('Run ') for line in result.stdout.splitlines()) == len(order)
    assert not (tmp_path / 'outputs').exists()


def test_rounds_explicit_and_mirrored(tmp_path):
    s = spec(tmp_path, 3)
    s.update(order='explicit', explicit_order=['base', 'pf', 'pfn'], rounds=2)
    assert abrun.planned_order(s) == ['base', 'pf', 'pfn'] * 2
    s['arms']['fourth'] = {'image': 'test', 'env': {}}
    s.update(order='mirrored', rounds=1)
    assert abrun.planned_order(s) == ['base', 'pf', 'pfn', 'fourth', 'fourth', 'pfn', 'pf', 'base']


@pytest.mark.parametrize('mode,expected_count,status', [('inert', 2, 'INVALID'),
                                                       ('leak', 1, 'INVALID'),
                                                       ('ok', 4, 'completed'),
                                                       ('fail', 1, 'INVALID')])
def test_fake_execution(tmp_path, mode, expected_count, status):
    s = spec(tmp_path)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(s))
    fake = tmp_path / 'runner.py'
    fake.write_text('''import json, os, pathlib, sys
out, image, env, tool, *argv = sys.argv[1:]
p = pathlib.Path(out); p.mkdir(parents=True)
mode = os.environ['TEST_MODE']
active = (mode == 'ok' and 'SWITCH' in env) or mode == 'leak'
(p/'stderr.txt').write_text(('switch active\\n' if active else '') + 'common_perf_print: eval time = 1000 ms / 10 runs (10 tokens per second)')
(p/'stdout.txt').write_text('same output')
(p/'result.json').write_text(json.dumps({'status':'failed' if mode=='fail' else 'ok', 'wall_s':1}))
sys.exit(1 if mode=='fail' else 0)
''')
    helper = tmp_path / 'help.py'
    helper.write_text('import pathlib,sys\np=pathlib.Path(__file__).with_suffix(".calls")\np.write_text(p.read_text()+"x" if p.exists() else "x")\nprint("-n --help")\n')
    env = dict(os.environ, ABRUN_RUNNER=f'{sys.executable} {fake}',
               ABRUN_PREFLIGHT_RUNNER=f'{sys.executable} {helper}',
               ABRUN_PREFLIGHT_CACHE_DIR=str(tmp_path / 'help-cache'), TEST_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(abrun.__file__)), str(path)],
                            capture_output=True, text=True, env=env)
    out = tmp_path / 'outputs/test'
    verdict = json.loads((out / 'verdict.json').read_text())
    assert result.returncode == (0 if status == 'completed' else 1), result.stderr
    assert verdict['status'] == status
    assert len(verdict['executed_order']) == expected_count
    assert len(list(out.glob('*/summary.json'))) == expected_count
    assert (out / 'REPORT.md').exists()
    assert helper.with_suffix('.calls').read_text() == 'x'
    if mode == 'inert':
        assert any('switch inert' in reason for reason in verdict['invalid_reasons'])
    if mode == 'leak':
        assert any('switch leaked' in reason for reason in verdict['invalid_reasons'])


@pytest.mark.parametrize('b', [97, 103])
def test_gate_strict_threshold(b):
    g = gate.evaluate({'base': runs([100, 100]), 'B': runs([b, b])})
    assert g['comparisons']['B']['verdict'] == 'null'


def test_real_om7_gate(tmp_path):
    grouped = {'base': [], 'pf': [], 'pfn': []}
    expected = {'base-a': 1.18, 'base-b': 1.22, 'pf-a': .72, 'pf-b': .76,
                'pfn-a': .64, 'pfn-b': .65}
    for name in ['base-a', 'pf-a', 'pfn-a', 'pfn-b', 'pf-b', 'base-b']:
        target = tmp_path / name
        shutil.copytree(FIXTURE / name, target)
        summary = run_summary.summarize(target, ['expert_prefetch:.*enabled'])
        assert summary['eval_tps'] == expected[name]
        arm = name.rsplit('-', 1)[0]
        assert bool(summary['switch_lines']) == (arm != 'base')
        grouped[arm].append(summary)
    verdict = gate.evaluate(grouped)
    assert verdict['means']['base'] == 1.2
    assert verdict['output']['class'] == 'identical'
    assert all(c['verdict'] == 'regressed' for c in verdict['comparisons'].values())


def test_preflight_failure_report(tmp_path, monkeypatch):
    s = spec(tmp_path)
    out, plan = abrun.plan(s)
    def reject(*args):
        raise ValueError('unsupported flags: -n')
    monkeypatch.setattr(preflight, 'check', reject)
    assert abrun.execute(s, out, plan) == 1
    verdict = json.loads((out / 'verdict.json').read_text())
    assert verdict['status'] == 'INVALID'
    assert verdict['executed_order'] == []
    assert 'unsupported flags: -n' in verdict['invalid_reasons']
    assert not list(out.glob('*/summary.json'))


def test_submit_is_one_gpu_job(tmp_path, monkeypatch):
    s = spec(tmp_path)
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(s))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    # The submission adapter needs only existence; supply a temporary queue path.
    fake_root = tmp_path / 'repo'
    (fake_root / 'bin').mkdir(parents=True)
    (fake_root / 'bin/jobq').touch()
    monkeypatch.setattr(abrun, 'ROOT', fake_root)
    monkeypatch.setattr(abrun.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['abrun.py', str(path), '--submit'])
    assert abrun.main() == 0
    assert len(calls) == 1
    assert calls[0][1:6] == ['submit', '--class', 'gpu', '--name', 'test']
    assert '--submit' not in calls[0]
    assert not (tmp_path / 'outputs').exists()


def test_per_arm_all_matches_and_legacy_or(tmp_path):
    s = spec(tmp_path)
    s['switch_regex'] = ['legacy active', 'legacy alternative']
    s['arms']['pf']['switch_regex'] = ['fill enabled', 'random enabled']
    abrun.validate(s)
    (tmp_path / 'stderr.txt').write_text('legacy active\nfill enabled\nfill enabled\n')
    summary = run_summary.summarize(tmp_path, abrun.switch_patterns(s), write=False)
    assert summary['switch_matches'] == {'legacy active': 1, 'legacy alternative': 0,
                                         'fill enabled': 2, 'random enabled': 0}
    errors = abrun.switch_errors(s, 'pf', summary)
    assert len(errors) == 1 and 'random enabled' in errors[0]
    with (tmp_path / 'stderr.txt').open('a') as stream:
        stream.write('random enabled\n')
    summary = run_summary.summarize(tmp_path, abrun.switch_patterns(s), write=False)
    assert abrun.switch_errors(s, 'pf', summary) == []
    del s['switch_regex']
    abrun.validate(s)  # Per-arm signatures alone suffice.
    assert abrun.switch_errors(s, 'pf', summary) == []


def test_cross_arm_leak_in_nonbase_control(tmp_path, monkeypatch):
    s = spec(tmp_path, 3)
    s.pop('switch_regex')
    s['arms']['pf'].update(expect_switch=False)
    s['arms']['pfn']['switch_regex'] = ['next prefetch enabled']
    abrun.validate(s)
    monkeypatch.setattr(preflight, 'check', lambda *args: None)
    def fake_run(command, **kwargs):
        directory = Path(command[-1])
        directory.mkdir()
        (directory/'stderr.txt').write_text(
            ('next prefetch enabled\n' if directory.name.endswith('-pf') else '') +
            'eval time = 1000 ms / 10 runs (10 tokens per second)\n')
        (directory/'stdout.txt').write_text('same')
        (directory/'result.json').write_text(json.dumps({'status': 'ok', 'wall_s': 1}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(abrun.subprocess, 'run', fake_run)
    out, planned = abrun.plan(s)
    for row in planned:
        row['command'] = ['fake-runner', row['dir']]
    assert abrun.execute(s, out, planned) == 1
    verdict = json.loads((out/'verdict.json').read_text())
    assert verdict['executed_order'] == ['base', 'pf']
    assert any('pf: switch leaked' in r for r in verdict['invalid_reasons'])
    report = (out/'REPORT.md').read_text()
    assert 'Switch matches' in report and 'next prefetch enabled' in report
    assert 'Secondary metrics' in report


def test_regate_exclude_and_preserve_data(tmp_path):
    from cc.abrun import regate
    s = spec(tmp_path)
    s.pop('switch_regex')
    s['arms']['pf']['expect_switch'] = False
    (tmp_path/'spec.json').write_text(json.dumps(s))
    for run_id, speed in [('001-base', 50), ('002-pf', 110), ('003-pf', 110), ('004-base', 100)]:
        directory = tmp_path/run_id
        directory.mkdir()
        summary = runs([speed])[0]
        summary.update(prompt_tps=20, nvme_gib_per_token=.5, switch_matches={}, switch_lines=[])
        (directory/'summary.json').write_text(json.dumps(summary))
    original = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert regate.recompute(tmp_path)['comparisons']['pf']['verdict'] == 'null'
    args = [sys.executable, regate.__file__, str(tmp_path), '--exclude', '001-base',
            '--reason', 'overlap', '--metric', 'eval_tps']
    result = subprocess.run([*args, '--no-write'], capture_output=True, text=True, check=True)
    verdict = json.loads(result.stdout)
    assert verdict['excluded_runs'] == ['001-base']
    assert verdict['reason'] == 'overlap'
    assert verdict['comparisons']['pf']['verdict'] == 'improved'
    assert verdict['secondary_metrics']['pf'] == dict(nvme_gib_per_token=.5, prompt_tps=20, eval_tps=110,
        eval_ms_per_token=None, nvme_gib_per_token_decode=None, gpu_util_median=None, energy_j_per_token=None,
        **{'threads.cpu_workers_r_fraction': None, 'threads.main_workers_d_fraction': None,
           'threads.majflt_per_token': None})
    assert not (tmp_path/'verdict-regate.json').exists()
    subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads((tmp_path/'verdict-regate.json').read_text()) == verdict
    assert all(p.read_bytes() == content for p, content in original.items())
    with pytest.raises(ValueError, match='unknown excluded'):
        regate.recompute(tmp_path, ['999-typo'])


def test_secondary_metrics_do_not_affect_gate():
    arms = {'base': runs([100, 100]), 'pf': runs([110, 110])}
    for arm in arms.values():
        arm[0].update(nvme_gib_per_token=2, prompt_tps=10)
        arm[1].update(nvme_gib_per_token=4, prompt_tps=None)
    verdict = gate.evaluate(arms)
    assert verdict['comparisons']['pf']['verdict'] == 'improved'
    assert verdict['secondary_metrics']['pf'] == dict(nvme_gib_per_token=3, prompt_tps=10, eval_tps=110,
        eval_ms_per_token=None, nvme_gib_per_token_decode=None, gpu_util_median=None, energy_j_per_token=None,
        **{'threads.cpu_workers_r_fraction': None, 'threads.main_workers_d_fraction': None,
           'threads.majflt_per_token': None})


CAPTURED = Path(__file__).parent / 'tests/fixtures/J-OM10c-003-pg8i8'


def test_captured_h5_metrics_and_provenance():
    source = ROOT / 'experiments/jetson/OM/J-OM10c/003-pg8i8'
    for name in ('stderr.txt', 'telemetry.jsonl'):
        original = set((source / name).read_text().splitlines())
        assert set((CAPTURED / name).read_text().splitlines()) <= original
    s = run_summary.summarize(CAPTURED, write=False)
    assert s['eval_ms_per_token'] == 716.54
    assert s['prompt_ms_per_token'] == 174.41
    assert s['eval_ms_total'] == 45141.77
    assert s['eval_runs'] == 63
    assert s['load_ms'] == 60357.82
    assert s['eval_tps'] == 1.4
    window = s['phase_windows']
    assert window['decode_end'] - window['decode_start'] == pytest.approx(45.14177)
    assert window['absolute_error_bound_s'] is None  # No absolute log origin was captured.
    assert s['nvme_gib_per_token_decode'] == s['nvme_gib_decode'] / s['eval_runs']
    assert 0 < s['nvme_gib_per_token_decode'] < s['nvme_gib_per_token']
    assert s['nvme_gib_load'] > 0
    assert s['gpu_util_median'] == 18
    assert s['gpu_util_p90'] == 36
    assert s['power_in_w_median'] == 7.158
    assert s['temp_tj_max'] == 52.093
    assert 0 < s['device_util_median'] <= 100
    assert s['energy_j_per_token'] == pytest.approx(5.130958057657607)
    # The trimmed capture preserves the same clock endpoints and phase results.
    full = run_summary.summarize(source, write=False)
    for field in ('nvme_gib_decode', 'nvme_gib_load', 'energy_j_per_token',
                  'gpu_util_median', 'device_util_median'):
        assert s[field] == full[field]


def test_captured_window_integration(tmp_path):
    # Use actual power and counter samples, truncating only stderr-free metadata.
    shutil.copytree(CAPTURED, tmp_path / 'run')
    s = run_summary.summarize(tmp_path / 'run', write=False)
    rows = [json.loads(line) for line in (CAPTURED / 'telemetry.jsonl').read_text().splitlines()]
    lo, hi = s['phase_windows']['decode_start'], s['phase_windows']['decode_end']
    def points(name):
        return sorted((r['mono_s'], m['value'] * ({'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3}[m['unit']]
                      if name == 'device_read' else 1))
                      for r in rows for m in r['metrics'] if m['name'] == name)
    def integral(samples):
        total = 0
        for (a, x), (b, y) in zip(samples, samples[1:]):
            left, right = max(a, lo), min(b, hi)
            if right > left:
                vl = x + (y-x) * (left-a) / (b-a)
                vr = x + (y-x) * (right-a) / (b-a)
                total += (right-left) * (vl+vr) / 2
        return total
    assert s['energy_j_per_token'] == pytest.approx(integral(points('power_VDD_IN_instant')) / 1000 / s['eval_runs'])
    reads = points('device_read')
    # Cumulative-counter delta equals integrated interval read rate.
    delta = sum((y-x) * max(0, min(b, hi)-max(a, lo)) / (b-a)
                for (a,x),(b,y) in zip(reads, reads[1:]))
    assert s['nvme_gib_decode'] == pytest.approx(delta / 1024**3)


def test_lower_gate_and_default_spec(tmp_path):
    s = spec(tmp_path)
    s.pop('metric')
    abrun.validate(s)
    s['direction'] = 'sideways'
    with pytest.raises(ValueError, match='direction'):
        abrun.validate(s)
    base = run_summary.summarize(ROOT / 'experiments/jetson/OM/J-OM10c/001-base', write=False)
    pg = run_summary.summarize(CAPTURED, write=False)
    verdict = gate.evaluate({'base': [base, base], 'pg': [pg, pg]}, 'eval_ms_per_token', direction='lower')
    c = verdict['comparisons']['pg']
    assert c['ratio'] == base['eval_ms_per_token'] / pg['eval_ms_per_token']
    assert c['verdict'] == 'improved'
    reversed_gate = gate.evaluate({'base': [pg, pg], 'pg': [base, base]}, 'eval_ms_per_token')
    assert reversed_gate['comparisons']['pg']['verdict'] == 'regressed'


def test_real_regate_dry_recomputes_without_writes(tmp_path):
    from cc.abrun import regate
    source = ROOT / 'experiments/jetson/OM/J-OM10c'
    original = {p: hashlib.sha256(p.read_bytes()).digest() for p in source.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, regate.__file__, str(source), '--dry'],
                            capture_output=True, text=True, check=True)
    assert 'ms/token' in result.stdout and '| base | 840.975 |' in result.stdout
    assert '| pg8i8 | 720.34 |' in result.stdout
    verdict = regate.recompute(source, metric='eval_ms_per_token')
    assert verdict['direction'] == 'lower'
    assert verdict['inputs']['pg8i8'][0] == 716.54
    assert all(c['verdict'] == 'improved' for c in verdict['comparisons'].values())
    assert original == {p: hashlib.sha256(p.read_bytes()).digest() for p in source.rglob('*') if p.is_file()}
    # A spec without a primary metric selects the new precise default.
    spec_data = json.loads((source / 'spec.json').read_text())
    spec_data.pop('metric')
    (tmp_path / 'spec.json').write_text(json.dumps(spec_data))
    shutil.copytree(CAPTURED, tmp_path / '001-pg8i8')
    summary = run_summary.summarize(tmp_path / '001-pg8i8')
    verdict = regate.recompute(tmp_path)
    assert verdict['metric'] == 'eval_ms_per_token'
