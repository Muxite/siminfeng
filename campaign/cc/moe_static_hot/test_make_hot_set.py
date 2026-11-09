import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('make_hot_set', Path(__file__).with_name('make_hot_set.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def row(pos=0, layer=0, experts=None, phase='prefill', seq=0):
    return dict(seq=seq, pos=pos, layer=layer, experts=experts or list(range(8)),
                weights=[0.125]*8, phase=phase)


def choose(records, **kwargs):
    args = dict(budget=160, strides=[8, 8], alignment=16, layers=2,
                train_seqs={0}, train_phase='prefill', train_start=0, train_end=2)
    args.update(kwargs)
    return m.select_hot(records, **args)


def test_alignment_and_guard_slot():
    assert m.bank_bytes(8, [8, 8], 16) == 160
    assert m.bank_bytes(9, [8, 8], 16) == 160
    assert m.bank_bytes(10, [8, 8], 16) == 192
    _, report = choose([row()])
    assert report['capacity'] == 9
    assert report['selected_payload_bytes'] == 128
    assert report['engine_bank_bytes'] <= report['budget_bytes']


def test_global_ranking_and_ties():
    rows = [row(layer=1), row(1, layer=1), row(layer=0)]
    selected, _ = choose(rows)
    assert selected == [(0, 0)] + [(1, e) for e in range(8)]
    assert choose(list(reversed(rows)))[0] == selected


def test_train_excludes_phase_position_sequence():
    train = [row()]
    excluded = [row(phase='decode', experts=list(range(8, 16))),
                row(pos=2, experts=list(range(16, 24))),
                row(seq=1, experts=list(range(24, 32)))] * 20
    assert choose(train)[0] == choose(train + excluded)[0]
    assert choose(train + excluded)[1]['train_records'] == 1


def test_start_boundary():
    selected, _ = choose([row(0), row(1, experts=list(range(8, 16)))], train_start=1)
    assert selected == [(0, e) for e in range(8, 16)]


def test_full_capacity_and_no_zero_frequency_padding():
    selected, report = choose([row()], budget=10**8)
    assert report['capacity'] == 64
    assert len(selected) == 8


@pytest.mark.parametrize('kwargs', [dict(budget=159), dict(strides=[]), dict(strides=[0]),
    dict(alignment=0), dict(layers=0), dict(train_seqs=set()), dict(train_start=2),
    dict(train_phase='bad'), dict(train_seqs={-1})])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        choose([row()], **kwargs)


def test_empty_split_and_layer_range():
    with pytest.raises(ValueError, match='no records'):
        choose([row(phase='decode')])
    with pytest.raises(ValueError, match='layer outside'):
        choose([row(layer=2)])


@pytest.mark.parametrize('mutate', [lambda r: r.update(experts=[0]*8),
    lambda r: r.update(weights=[float('nan')]*8), lambda r: r.update(seq=True),
    lambda r: r.update(experts=list(range(7))), lambda r: r.update(phase='train')])
def test_trace_validation(tmp_path, mutate):
    r = row()
    mutate(r)
    path = tmp_path / 'bad.jsonl'
    path.write_text(json.dumps(r) + '\n')
    with pytest.raises(ValueError):
        m.read_trace(path)


def test_duplicate_validation(tmp_path):
    path = tmp_path / 'duplicate.jsonl'
    path.write_text((json.dumps(row()) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicate'):
        m.read_trace(path)


def test_cli(tmp_path):
    trace = tmp_path / 'trace.jsonl'
    trace.write_text(json.dumps(row()) + '\n')
    output, report = tmp_path / 'hot.txt', tmp_path / 'report.json'
    cmd = [sys.executable, str(Path(m.__file__)), str(trace), '--output', str(output),
           '--report', str(report), '--budget-bytes', '160', '--expert-bytes', '8', '8',
           '--alignment', '16', '--train-seq', '0', '--train-phase', 'prefill',
           '--train-pos-end', '1']
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(report.read_text())
    assert len(metadata['trace_sha256']) == 64
    assert metadata['train']['pos_end_exclusive'] == 1
    assert output.read_text().splitlines()[2:] == [f'0 {e}' for e in range(8)]
    first = output.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert output.read_bytes() == first
