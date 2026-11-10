import csv
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

PROFILERS = Path(__file__).resolve().parents[1]
ROOT = PROFILERS.parents[1]
sys.path.insert(0, str(PROFILERS))
import bpf_split_timer as bpf
import capture
import ncu_summary as ncu


def test_real_bpf():
    result = bpf.parse((ROOT / 'artifacts/bpf1/bpf.txt').read_text(), 5, 23.1)
    assert result['splits'] == 5626
    assert result['splits_per_token'] == pytest.approx(48.70995671)
    assert result['cpu_time_frac'] == pytest.approx(.89034)
    assert result['mean_us'] == pytest.approx(791.2726626)
    assert next(b for b in result['hist'] if b['count'] == 5405) == dict(low_us=512, high_us=1024, count=5405)
    assert result['threads'] == {}  # dangling @s entry is not a completed call


def test_bpf_generated_format_and_threads():
    sample = 'PROFW window_s=2\n@graphs: 2\n@graph_tot_us: 1500\n@graph_us:\n[512, 1K) 2 |@@|\n@thread_graphs[42]: 2\n@thread_tot_us[42]: 1500\n'
    assert bpf.parse(sample, tokens_per_second=1)['threads'] == {'42': {'graphs': 2, 'tot_us': 1500}}
    script = bpf.script(123, '/proc/123/root/app/libggml-cpu.so.0.27.0', per_thread=True)
    assert 'uprobe:/proc/123/root/app/libggml-cpu.so.0.27.0:ggml_graph_compute' in script
    assert 'uretprobe:' in script and 'pid == 123 && @s[tid]' in script
    assert 'delete(@s[tid])' in script and 'clear(@s)' in script
    assert '@thread_graphs[tid] = count();' in script
    assert 'PROFW window_s=5.0\\n' in script
    assert '@thread_graphs' not in bpf.script(123, '/app/libggml-cpu.so.0')


@pytest.mark.parametrize('seconds,tps', [(0, 1), (-1, 1), (1, 0), (1, float('nan'))])
def test_bpf_invalid_window(seconds, tps):
    with pytest.raises(ValueError):
        bpf.parse((ROOT / 'artifacts/bpf1/bpf.txt').read_text(), seconds, tps)


def test_bpf_requires_complete_capture():
    source = (ROOT / 'artifacts/bpf1/bpf.txt').read_text()
    with pytest.raises(ValueError, match='explicit'):
        bpf.parse(source)
    with pytest.raises(ValueError, match='histogram'):
        bpf.parse(source.replace('5405', '5404'), 5)
    with pytest.raises(ValueError, match='aggregate'):
        bpf.parse(source + '\n@graphs: 5626\n', 5)
    with pytest.raises(ValueError):
        bpf.script(1, '/app/lib:bad')


def test_container_recipe_quoting():
    name = "engine; touch /tmp/nope ' quoted"
    recipe = bpf.container_command(name, per_thread=True)
    assert "docker inspect -f '{{.State.Pid}}' " + shlex.quote(name) in recipe
    assert 'for candidate in /proc/"$pid"/root/app/libggml-cpu.so.*' in recipe
    assert '[ ! -L "$candidate" ]' in recipe
    assert '--per-thread' in recipe
    subprocess.run(['bash', '-n'], input=recipe, text=True, check=True)


def test_real_ncu():
    result = ncu.summarize((ROOT / 'artifacts/ncu1/ncu.csv').read_text())
    assert result['kernels'] == 600
    assert result['time_s'] == pytest.approx(.00222688)
    assert result['effective_GB_s'] == pytest.approx(221.852565)
    assert result['byte_scope'] == 'read'
    assert result['launch_bound_count'] == 401
    assert result['launch_bound_time_share'] == pytest.approx(.457709441)
    assert sum(g['time_share'] for g in result['classes'].values()) == pytest.approx(1)
    q8 = result['classes']['quantize_q8_1']
    assert q8['count'] == 101 and q8['launch_bound'] is True
    assert q8['dram_pct'] < 1 and q8['sm_pct'] < 1
    assert result['classes']['mul_mat_vec_q']['launch_bound'] is False


def csv_sample(launches):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['ID', 'Process ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
    for i, (duration, unit, size, byte_unit, dram) in enumerate(launches):
        for metric, metric_unit, value in [(ncu.TIME, unit, duration), (ncu.READ, byte_unit, size),
                                          (ncu.DRAM, '%', dram), (ncu.SM, '%', 2)]:
            writer.writerow([i, 7, 'void example<int>(float *, int)', metric, metric_unit, value])
    return stream.getvalue()


@pytest.mark.parametrize('unit,duration', [('ns', 2000), ('us', 2), ('ms', .002)])
@pytest.mark.parametrize('byte_unit,size', [('byte', '2,000,000'), ('Kbyte', 2000), ('Mbyte', 2), ('Gbyte', .002)])
def test_ncu_units(unit, duration, byte_unit, size):
    result = ncu.summarize(csv_sample([(duration, unit, size, byte_unit, 1)]))
    assert result['effective_GB_s'] == pytest.approx(1000)
    assert result['classes']['example']['launch_bound']


def test_ncu_weighting_and_strict_thresholds():
    result = ncu.summarize(csv_sample([(4, 'us', 8, 'Kbyte', 4), (5, 'us', 8, 'Kbyte', 4),
                                     (4, 'us', 8, 'Kbyte', 5)]))
    group = result['classes']['example']
    assert group['dram_pct'] == pytest.approx(56 / 13)
    assert group['launch_bound_count'] == 1 and not group['launch_bound']
    assert group['GB_s'] == pytest.approx(24 / 13)
    assert result['launch_bound_time_share'] == pytest.approx(4 / 13)


def test_ncu_rejects_bad_or_incomplete_metrics():
    source = csv_sample([(2, 'us', 8, 'Kbyte', 1)])
    for bad in ('', source.replace('Kbyte', 'KiB'), source.replace(',2\r', ',NaN\r'),
                '\n'.join(source.splitlines()[:-1]), source + source.splitlines()[1] + '\n'):
        with pytest.raises(ValueError):
            ncu.summarize(bad)


def test_extract_does_not_start_engine():
    commands = capture.extract('engine@sha256:abc', '/tmp/app directory', 'profw-test')
    assert commands == [
        ['docker', 'create', '--name', 'profw-test', '--entrypoint', '/bin/true', 'engine@sha256:abc'],
        ['docker', 'cp', 'profw-test:/app/.', '/tmp/app directory'],
        ['docker', 'rm', 'profw-test']]


@pytest.mark.parametrize('tool', ['ncu', 'nsys'])
def test_capture_qualified_flags(tool):
    args = ['-m', '/m/model with spaces.gguf', '-p', '0', '-n', '32']
    argv = capture.capture(tool, '/tmp/app directory', '/tmp/models', '/tmp/output', args)
    assert 'campaign/cuda-tools:c1' in argv
    assert 'LD_LIBRARY_PATH=/app' in argv
    assert 'type=bind,src=/tmp/models,dst=/m,readonly' in argv
    assert 'type=bind,src=/tmp/app directory,dst=/app,readonly' in argv
    assert argv[-len(args):] == args
    assert argv[argv.index('campaign/cuda-tools:c1') + 1] == tool
    if tool == 'ncu':
        assert argv[argv.index('--cap-add') + 1] == 'SYS_ADMIN'
        assert argv[argv.index('--launch-skip') + 1] == '2000'
        assert argv[argv.index('--launch-count') + 1] == '600'
        assert argv[argv.index('--metrics') + 1] == ','.join(ncu.METRICS)
    else:
        assert '--cap-add' not in argv
        assert '--sample=none' in argv and '--cpuctxsw=none' in argv
        assert '--cuda-graph-trace=graph' in argv
        assert '--cuda-graph-trace=node' in capture.capture(tool, '/tmp/a', '/tmp/m', '/tmp/o', [], graph_trace='node')


def test_capture_validation():
    with pytest.raises(ValueError):
        capture.capture('ncu', '/a', '/m', '/o', [], launch_count=0)
    with pytest.raises(ValueError):
        capture.capture('ncu', '/a', '/m', '/o', [], launch_skip=-1)
    with pytest.raises(ValueError):
        capture.capture('ncu', '/a,readonly', '/m', '/o', [])


def test_cli():
    completed = subprocess.run([sys.executable, str(PROFILERS / 'capture.py'), 'ncu', '--app', '/a',
        '--models', '/m', '--output', '/o', '--launch-skip', '600', '--launch-count', '12',
        '--', '-m', '/m/model.gguf', '-n', '32'], text=True, capture_output=True, check=True)
    argv = json.loads(completed.stdout)
    assert argv[-4:] == ['-m', '/m/model.gguf', '-n', '32']
    assert argv[argv.index('--launch-skip') + 1] == '600'


def test_bpf_small_histogram_buckets():
    source = 'PROFW window_s=1\n@graphs: 2\n@graph_tot_us: 1\n@graph_us:\n[0] 1 |@|\n[1] 1 |@|\n'
    assert bpf.parse(source)['hist'] == [dict(low_us=0, high_us=1, count=1),
                                         dict(low_us=1, high_us=2, count=1)]
