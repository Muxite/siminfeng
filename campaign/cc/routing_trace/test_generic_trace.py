import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('generic_validator', ROOT / 'validate_generic_trace.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
spec = importlib.util.spec_from_file_location('expert_replay_generic_test', ROOT.parent / 'expert_replay/expert_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture():
    return [dict(type='metadata', arch='qwen35moe', n_layer=2, layers=2, experts=256,
                 n_expert=[256, 256], k=[2, 2], model_path='model.gguf', model_sha256=None,
                 shared_experts_present=[True, True], shared_experts_routed=False),
            *[dict(tok=t, layer=l, experts=[254, 255], weights=[.4, .6],
                   phase='prefill' if t == 0 else 'decode', seq=0)
              for t in range(2) for l in range(2)],
            dict(type='metadata', trace_complete=True, tokens=2, records=4)]


def write(tmp_path, rows):
    path = tmp_path / 'trace.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_valid_and_replay_compatible(tmp_path):
    path = write(tmp_path, fixture())
    assert validator.validate(path) == dict(tokens=2, records=4)
    rows, layers, experts, diagnostics = replay.read_trace(path)
    assert (len(rows), layers, experts, diagnostics['malformed_lines']) == (4, 2, 256, 0)


@pytest.mark.parametrize('change', ['footer', 'count', 'token_count', 'record_boundary', 'gap',
                                  'duplicate', 'bounds', 'weights', 'phase', 'after_footer', 'header'])
def test_reject(tmp_path, change):
    rows = copy.deepcopy(fixture())
    if change == 'footer': rows.pop()
    elif change == 'count': rows[-1]['records'] += 1
    elif change == 'token_count': rows[-1]['tokens'] += 1
    elif change == 'record_boundary': rows.pop(-2)
    elif change == 'gap': rows[3]['tok'] = 3
    elif change == 'duplicate': rows[2] = rows[1]
    elif change == 'bounds': rows[1]['experts'][0] = 256
    elif change == 'weights': rows[1]['weights'][0] = float('nan')
    elif change == 'phase': rows[2]['phase'] = 'decode'
    elif change == 'after_footer': rows.append(rows[1])
    elif change == 'header': rows[0]['k'] = [2]
    with pytest.raises(ValueError): validator.validate(write(tmp_path, rows))


def test_partial_json(tmp_path):
    path = write(tmp_path, fixture()[:-1])
    with path.open('a') as out: out.write('{"trace_complete":')
    with pytest.raises(ValueError): validator.validate(path)


def test_variable_k_and_dense_layer(tmp_path):
    rows = fixture()
    rows[0].update(n_expert=[0, 256], k=[0, 3], shared_experts_present=[False, True])
    rows = [r for r in rows if r.get('layer') != 0]
    for row in rows[1:-1]:
        row.update(experts=[1, 100, 255], weights=[.2, .3, .5])
    rows[-1]['records'] = 2
    assert validator.validate(write(tmp_path, rows))['records'] == 2


@pytest.mark.parametrize('k', [1, 8, 16])
def test_256_experts_variable_topk(tmp_path, k):
    rows = fixture()
    rows[0]['k'] = [k, k]
    for row in rows[1:-1]:
        row.update(experts=list(range(256-k, 256)), weights=[1/k]*k)
    path = write(tmp_path, rows)
    assert validator.validate(path)['tokens'] == 2
    assert replay.read_trace(path)[3]['malformed_lines'] == 0
