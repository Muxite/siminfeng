"""Synthetic inputs only; no lab services or external dependencies beyond pytest."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('pareto', Path(__file__).with_name('pareto.py'))
pareto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pareto)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=pareto.FIELDS)
        writer.writeheader()
        for overrides in rows:
            row = dict(config='base', desc='synthetic', **{'class': 'transferable'},
                       gpu_mib=600, host_mib=100, tg_ts=20, pp_ts=50,
                       kld=0.1, top1_pct=98, ppl_ratio=1.02, notes='')
            row.update(overrides)
            writer.writerow(row)
    return path


def load(tmp_path, overrides, budget=700):
    path = write_csv(tmp_path / 'configs.csv', overrides)
    return {r['config']: r for r in pareto.analyze(pareto.read_csv(path), budget)}


def test_dominated_and_tradeoff(tmp_path):
    rows = load(tmp_path, [dict(config='best'), dict(config='dominated', gpu_mib=650, tg_ts=19, kld=.2),
                           dict(config='fast', gpu_mib=800, tg_ts=30, kld=.2)])
    assert rows['best']['pareto_optimal']
    assert not rows['dominated']['pareto_optimal']
    assert rows['fast']['pareto_optimal']


def test_budget_boundary_and_unknown(tmp_path):
    rows = load(tmp_path, [dict(config='boundary', gpu_mib=700), dict(config='over', gpu_mib=700.1, tg_ts=30),
                           dict(config='unknown', gpu_mib='')])
    assert rows['boundary']['fits_budget'] is True
    assert rows['boundary']['budget_pareto_optimal']
    assert rows['over']['fits_budget'] is False
    assert rows['over']['pareto_optimal']
    assert not rows['over']['budget_pareto_optimal']
    assert rows['unknown']['fits_budget'] is None
    assert rows['unknown']['missing_objectives'] == ['gpu_mib']
    assert not rows['unknown']['pareto_optimal']


def test_missing_quality_never_zero_filled(tmp_path):
    rows = load(tmp_path, [dict(config='missing', gpu_mib=1, tg_ts=100, kld='', top1_pct='NA', ppl_ratio='nan'),
                           dict(config='complete'), dict(config='kld_only', top1_pct='', ppl_ratio='')])
    assert rows['missing']['missing_quality'] == ['kld', 'top1_pct', 'ppl_ratio']
    assert rows['missing']['fits_budget']
    assert not rows['missing']['pareto_optimal']
    assert not rows['missing']['budget_pareto_optimal']
    assert rows['complete']['pareto_optimal']
    assert rows['kld_only']['pareto_optimal']  # KLD alone supplies the quality objective.


def test_exact_ties_and_single_axis_domination(tmp_path):
    rows = load(tmp_path, [dict(config='tie1'), dict(config='tie2'), dict(config='worse', kld=.11)])
    assert rows['tie1']['pareto_optimal'] and rows['tie2']['pareto_optimal']
    assert rows['tie1']['budget_pareto_optimal'] and rows['tie2']['budget_pareto_optimal']
    assert not rows['worse']['pareto_optimal']


def test_classes_are_labelled_and_have_separate_frontiers(tmp_path):
    rows = load(tmp_path, [dict(config='transfer'), dict(config='discrete', gpu_mib=500, **{'class': 'discrete-only'})])
    assert rows['discrete']['pareto_optimal']
    assert not rows['transfer']['pareto_optimal']
    assert rows['transfer']['class_pareto_optimal']
    assert rows['discrete']['class'] == 'discrete-only'


def test_cli_outputs_sorted_markdown_json_and_scatter(tmp_path, capsys):
    source = write_csv(tmp_path / 'configs.csv', [dict(config='large', gpu_mib=900),
                                                dict(config='small|x', gpu_mib=0, kld=0),
                                                dict(config='unknown', gpu_mib='')])
    output = tmp_path / 'out'
    assert pareto.main([str(source), '--output-dir', str(output), '--budget-mib', '0']) == 0
    payload = json.loads((output / 'pareto.json').read_text())
    assert [r['config'] for r in payload['rows']] == ['small|x', 'large', 'unknown']
    assert payload['rows'][0]['fits_budget']
    assert payload['budget_mib'] == 0
    report = (output / 'pareto.md').read_text()
    assert capsys.readouterr().out == report
    assert 'small\\|x' in report
    assert 'GPU MiB' in report and '```text' in report
    assert 'statistical significance' in report
    assert 'Discrete-only' in report


def test_header_only_csv(tmp_path):
    path = write_csv(tmp_path / 'empty.csv', [])
    assert pareto.read_csv(path) == []
    assert 'No measurements' in pareto.markdown([], 700, str(path))
    assert 'No scatter points' in pareto.scatter([], 700)


@pytest.mark.parametrize('field,value', [('gpu_mib', '-1'), ('kld', 'oops'), ('class', 'bad'), ('top1_pct', '101')])
def test_invalid_csv_values_rejected(tmp_path, field, value):
    path = write_csv(tmp_path / 'bad.csv', [{field: value}])
    with pytest.raises(ValueError):
        pareto.read_csv(path)


@pytest.mark.parametrize('budget', ['-1', 'nan', 'inf'])
def test_invalid_budget_rejected(budget, tmp_path):
    with pytest.raises(SystemExit) as exc:
        pareto.main(['--budget-mib', budget, '--output-dir', str(tmp_path)])
    assert exc.value.code == 2


def test_explicit_missing_csv_does_not_silently_use_lab(tmp_path):
    with pytest.raises(SystemExit) as exc:
        pareto.main([str(tmp_path / 'missing.csv'), '--output-dir', str(tmp_path)])
    assert exc.value.code == 2


def test_csv_is_primary_even_if_lab_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(pareto, 'ROOT', tmp_path)
    source = tmp_path / 'experiments/memory/configs.csv'
    source.parent.mkdir(parents=True)
    write_csv(source, [dict(config='manual')])
    # An invalid lab file must not be read when the manual CSV is present.
    lab = tmp_path / 'lab/jobs/job-a/att-1'
    lab.mkdir(parents=True)
    (lab / 'result.json').write_text('not json')
    pareto.main(['--output-dir', str(tmp_path / 'out')])
    payload = json.loads((tmp_path / 'out/pareto.json').read_text())
    assert payload['source_kind'] == 'csv'
    assert payload['rows'][0]['config'] == 'manual'


def test_fallback_keeps_measurements_separate_and_skips_failures(tmp_path):
    lab = tmp_path / 'lab'
    attempt = lab / 'jobs/job-a/att-1'
    attempt.mkdir(parents=True)
    (lab / 'tests').mkdir()
    (lab / 'tests/mem.json').write_text(json.dumps({'mem': {'model_file': 'q4.gguf', 'args': ['-n', '128']}}))
    (attempt.parent / 'job.json').write_text(json.dumps({'spec': {'model': 'bf16.gguf', 'tests': ['mem', 'ppl']}}))
    (attempt / 'result.json').write_text(json.dumps({
        'execution': {'status': 'completed'}, 'quality': {'tests': {
            'mem#0': {'exit_code': 0, 'mem': {'gpu_peak_mib': 650, 'gpu_delta_mib': 600},
                      'samples': [{'n_prompt': 0, 'n_gen': 128, 'avg_ts': 42}]},
            'ppl#1': {'exit_code': 0, 'ppl': 7.2, 'ppl_stderr': .1},
            'bad#2': {'exit_code': 1}, 'partial#3': {'exit_code': 0, 'partial': True}}}}))
    interrupted = attempt.parent / 'att-2'
    interrupted.mkdir()
    (interrupted / 'result.json').write_text(json.dumps({'execution': {'status': 'interrupted'},
                                                       'quality': {'tests': {'bad': {'exit_code': 0}}}}))
    rows = pareto.read_lab(lab)
    assert len(rows) == 2
    assert rows[0]['desc'] == 'q4.gguf'
    assert rows[0]['gpu_mib'] == 650  # Peak, never delta.
    assert rows[0]['tg_ts'] == 42
    assert rows[0]['kld'] is None and rows[0]['ppl_ratio'] is None
    assert rows[1]['tg_ts'] is None
    assert 'ppl=7.2' in rows[1]['notes']
    assert all(r['class'] == 'unclassified' for r in rows)
    assert not any(r['pareto_optimal'] for r in pareto.analyze(rows, 700))
