import json
import os
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import io_bench as bench


def string(s):
    data = s.encode()
    return struct.pack('<Q', len(data)) + data


def synthetic_gguf(path):
    # A Q4_K expert tensor: nb[2] = 144 * 2 = 288, three experts.
    header = b'GGUF' + struct.pack('<IQQ', 3, 2, 2)
    header += string('general.alignment') + struct.pack('<II', 4, 64)
    header += string('test.array') + struct.pack('<IIQii', 9, 5, 2, 1, 2)
    header += string('blk.0.ffn_up_exps.weight') + struct.pack('<IQQQIQ', 3, 256, 2, 3, 12, 0)
    header += string('output.weight') + struct.pack('<IQQIQ', 2, 4, 4, 0, 896)
    base = (len(header) + 63) // 64 * 64
    path.write_bytes(header + bytes(base - len(header)) + bytes(960))
    return base


def test_extract_gguf(tmp_path):
    path = tmp_path / 'test.gguf'
    base = synthetic_gguf(path)
    result = bench.extract_ranges(path)
    assert result['slices'] == [[base, 288], [base + 288, 288], [base + 576, 288]]
    assert result['tensors'][0]['size'] == 864
    assert result['tensors'][0]['nb2'] == 288
    assert len(result['tensors']) == 1
    path.write_bytes(path.read_bytes()[:-200])
    with pytest.raises(ValueError, match='exceeds file'):
        bench.extract_ranges(path)


def test_truncated_gguf(tmp_path):
    path = tmp_path / 'bad.gguf'
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 100000000, 0))
    with pytest.raises(ValueError):
        bench.extract_ranges(path)


@pytest.mark.parametrize('value,expected', [('256K', 262144), ('1.83M', 1918894), ('2MiB', 2097152)])
def test_sizes(value, expected):
    assert bench.size_bytes(value) == expected


def test_plan_boundaries():
    spans = [[64, 9000], [20000, 7000]]
    plan = bench.make_plan(spans, 5000, 17000, 'random-expert')
    assert sum(n for _, n in plan) == 17000
    assert all(n <= 5000 and any(lo <= off and off + n <= lo + length for lo, length in spans)
               for off, n in plan)
    direct = bench.make_plan(spans, 5000, 17000, 'random-expert', direct=True)
    assert all(off % 4096 == n % 4096 == 0 for off, n in direct)
    assert sum(n for _, n in direct) <= 17000
    assert bench.make_plan([[0, 16384]], 4096, 8192, 'sequential') == [(0, 4096), (4096, 4096)]


def test_ranges_validation(tmp_path):
    path = tmp_path / 'ranges.json'
    path.write_text(json.dumps([[0, 20]]))
    assert bench.load_ranges(path, 20) == [[0, 20]]
    for bad in ([[0, 21]], [[-1, 3]], [[0, 0]], [[0, '3']], []):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            bench.load_ranges(path, 20)


def test_buffered_cold_read(tmp_path):
    path = tmp_path / 'data'
    path.write_bytes(os.urandom(128 * 1024))
    with path.open('rb') as file:
        os.fsync(file.fileno())
        plan = bench.make_plan([[0, path.stat().st_size]], 4096, 65536, 'sequential')
        bench.evict(file.fileno(), plan)
        before = bench.residency(file.fileno(), plan)
        result = bench.python_cell(file.fileno(), plan, 4)
        after = bench.residency(file.fileno(), plan)
    assert before['sample_pages'] > 0
    assert result['bytes_read'] == 65536
    assert result['mb_s'] > 0
    assert 0 < result['p50_ns'] <= result['p95_ns']
    assert after['resident_fraction'] == 1


def test_fio_replay(tmp_path):
    if not Path('/usr/bin/fio').exists():
        pytest.skip('fio unavailable')
    path = tmp_path / 'data'
    path.write_bytes(os.urandom(128 * 1024))
    plan = [(0, 4096), (65536, 8192), (32768, 4096)]
    result = bench.fio_cell(path, plan, 2, 'psync', False, tmp_path / 'jobs', 15)
    assert result['bytes_read'] == 16384
    assert 0 < result['p50_ns'] <= result['p95_ns']
    text = (tmp_path / 'jobs' / 'worker0.iolog').read_text()
    assert 'read 0 4096' in text and 'read 32768 4096' in text
    assert 'rw=read' in (tmp_path / 'jobs' / 'job.fio').read_text()
    # Reusing an output directory must not mix latency samples from old workers.
    again = bench.fio_cell(path, plan, 1, 'psync', False, tmp_path / 'jobs', 15)
    assert again['bytes_read'] == 16384


def test_cli_small(tmp_path, capsys):
    path = tmp_path / 'data'
    path.write_bytes(os.urandom(65536))
    out = tmp_path / 'out'
    assert bench.main([str(path), '--raw-file', '--quick', '--sizes', '4K', '--concurrency', '1,2',
                       '--engines', 'python', '--bytes-per-cell', '16K', '--output-dir', str(out)]) == 0
    cells = json.loads((out / 'results.json').read_text())['cells']
    assert len(cells) == 4
    assert all(c['status'] == 'ok' and c['bytes_read'] == 16384 for c in cells)
    assert 'ceiling at engine pattern' in capsys.readouterr().out
