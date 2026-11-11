import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('kernel_screen', HERE / 'kernel_screen.py')
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)
FIXTURE = (HERE / 'fixtures/perf.txt').read_text()
SHAPE = dict(n_embd=1024, n_ff=512, n_expert=32, n_used=8, n_tokens=1)


def test_console_format_and_units():
    rows = ks.parse_output(FIXTURE.replace('83.89 GFLOPS', '\x1b[1;34m83.89 GFLOPS\x1b[0m'))
    assert len(rows) == 7
    assert rows[0]['us_run'] == 12.5
    assert rows[0]['gflops'] == 83.89
    assert rows[0]['gb_s'] is None
    assert rows[0]['fields']['bs'] == '[1,1]'
    assert rows[2]['gb_s'] == 20
    assert rows[2]['gflops'] is None
    assert rows[3]['op'] == 'SWIGLU'
    assert not rows[4]['supported']
    assert rows[5]['gflops'] == 4190
    assert rows[6]['gflops'] == .002
    assert ks.parse_output('noise\n') == []


def test_shape_filters_and_fusion_dimension_mapping():
    rows = ks.parse_output(FIXTURE)
    assert re.search(ks.param_regex(SHAPE, 'MUL_MAT', 'Q4_K'), rows[0]['params'])
    assert re.search(ks.param_regex(SHAPE, 'MUL_MAT_VEC_FUSION', 'Q4_K'), rows[2]['params'])
    assert not re.search(ks.param_regex(dict(SHAPE, n_tokens=8), 'MUL_MAT', 'Q4_K'), rows[0]['params'])
    assert not re.search(ks.param_regex(SHAPE, 'MUL_MAT', 'Q5_K'), rows[0]['params'])
    assert not re.search(ks.param_regex(dict(SHAPE, direction='down'), 'MUL_MAT', 'Q4_K'), rows[0]['params'])
    assert re.search(ks.param_regex(dict(SHAPE, n_tokens=8), 'SWIGLU', 'F32'), rows[3]['params'])


def args(tmp_path):
    return ks.parser().parse_args(['--binary', '/fake/backend', '--out', str(tmp_path / 'out'),
                                  '--n-embd', '1024', '--n-ff', '512', '--n-expert', '32',
                                  '--n-used', '8', '--n-tokens', '1', '--direction', 'up',
                                  '--ops', 'MUL_MAT', '--types', 'Q4_K', '--repeats', '2'])


def test_interleaving_reports_and_clean_env(tmp_path, monkeypatch):
    a = args(tmp_path)
    monkeypatch.setenv('GGML_CUDA_DISABLE_FUSION', 'inherited')
    calls = []
    def fake(cmd, **kw):
        calls.append(kw['env'].get('GGML_CUDA_DISABLE_FUSION'))
        return subprocess.CompletedProcess(cmd, 0, FIXTURE.splitlines()[1] + '\n')
    assert ks.execute(a, fake) == 0
    assert calls == [None, None, '1', '1', None, None]
    report = json.loads((Path(a.out) / 'results.json').read_text())
    assert report['summary'][0]['ratios_A_over_arm'] == {'A2': 1, 'B': 1}
    assert report['summary'][0]['arms']['A']['spread_pct'] == 0
    assert (Path(a.out) / 'results.md').exists()
    assert len(list(Path(a.out).glob('*.log'))) == 6


@pytest.mark.parametrize('output,code', [('nothing', 0), (FIXTURE.splitlines()[5], 0), ('oops', 9)])
def test_no_false_success(tmp_path, output, code):
    a = args(tmp_path)
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, code, output)
    assert ks.execute(a, fake) == 1
    assert json.loads((Path(a.out) / 'results.json').read_text())['errors']


def test_docker_prefix_and_unset(tmp_path, monkeypatch):
    a = args(tmp_path)
    a.binary = None
    a.image = 'image:local'
    monkeypatch.setenv('DOCKER_CMD_PREFIX', 'sg docker -c')
    cmd, env = ks.command(a, 'MUL_MAT_ID', 'type_a=q4_k', {'CUSTOM': 'a; literal', 'UNSET': None}, {'CUSTOM', 'UNSET'})
    import shlex
    assert cmd[:3] == ['sg', 'docker', '-c']
    inner = shlex.split(cmd[3])
    assert inner[:6] == ['docker', 'run', '--rm', '--runtime', 'nvidia', '--network=none']
    assert 'CUSTOM=a; literal' in inner
    assert '-u' in inner and 'UNSET' in inner
    assert env['CUSTOM'] == 'a; literal' and 'UNSET' not in env


def test_spread_and_ratio():
    def run(arm, us):
        return dict(arm=arm, rows=[dict(case='x', supported=True, us_run=us)])
    row = ks.summarize([run('A', 10), run('A', 12), run('B', 5)], ['A', 'B'])[0]
    assert row['ratios_A_over_arm']['B'] == 2.2
    assert row['arms']['A']['spread_pct'] == pytest.approx(20)


def test_qwen_preset_matches_layout():
    root = HERE.resolve().parents[1]
    layout = json.loads((root / 'experiments/jetson/OM/qwen36-layout.json').read_text())['kv']
    shapes = json.loads((HERE / 'presets/qwen36-35b-a3b.json').read_text())['shapes']
    assert {s['n_tokens'] for s in shapes} == {1, 8}
    for s in shapes:
        assert s['n_embd'] == layout['qwen35moe.embedding_length']
        assert s['n_ff'] == layout['qwen35moe.expert_feed_forward_length']
        assert s['n_expert'] == layout['qwen35moe.expert_count']
        assert s['n_used'] == layout['qwen35moe.expert_used_count']


def test_timeout_retains_partial_log(tmp_path):
    a = args(tmp_path)
    def fake(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1, output=b'partial output\n')
    assert ks.execute(a, fake) == 1
    assert (Path(a.out) / '00000.log').read_text() == 'partial output\n'


def test_additional_filter_is_anded(tmp_path):
    a = args(tmp_path)
    a.filter = 'bs=\\[1,1\\]'
    def fake(cmd, **kw):
        pattern = cmd[cmd.index('-p') + 1]
        assert re.search(pattern, ks.parse_output(FIXTURE)[0]['params'])
        assert not re.search(pattern, ks.parse_output(FIXTURE)[0]['params'].replace('bs=[1,1]', 'bs=[2,1]'))
        return subprocess.CompletedProcess(cmd, 0, FIXTURE.splitlines()[1] + '\n')
    assert ks.execute(a, fake) == 0


def test_partial_arm_suppresses_ratio(tmp_path):
    a = args(tmp_path)
    calls = []
    def fake(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '' if len(calls) == 3 else FIXTURE.splitlines()[1] + '\n')
    assert ks.execute(a, fake) == 1
    result = json.loads((Path(a.out) / 'results.json').read_text())
    assert result['summary'][0]['ratios_A_over_arm'] == {'A2': None, 'B': None}
