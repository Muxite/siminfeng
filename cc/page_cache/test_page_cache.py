"""CPU-only tests; cache mutations are restricted to temporary files."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cc.page_cache import page_cache as cache, run_protocol


@pytest.fixture
def model(tmp_path):
    path = tmp_path / 'model file.gguf'
    with path.open('wb') as stream:
        stream.write(b'x' * (4 * 1024 ** 2 + 31))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def test_read_and_evict(model):
    model.read_bytes()
    before = cache.resident(model)
    assert before['resident_bytes'] > 0
    assert before['file_bytes'] == model.stat().st_size
    assert before['pages'] == (before['file_bytes'] + cache.PAGE_SIZE - 1) // cache.PAGE_SIZE
    assert before['elapsed_ms'] >= 0
    observed = cache.evict(model)
    if observed['after']['resident_bytes'] >= observed['before']['resident_bytes']:
        pytest.skip('filesystem/kernel did not evict clean temporary file pages with DONTNEED')
    assert observed['after']['resident_bytes'] < observed['before']['resident_bytes']


def test_windows_unaligned_ranges_and_tail(model, monkeypatch):
    model.read_bytes()
    monkeypatch.setattr(cache, 'WINDOW_BYTES', 2 * cache.PAGE_SIZE)
    size = model.stat().st_size
    ranges = [dict(name='unaligned', offset=13, size=3 * cache.PAGE_SIZE),
              dict(name='tail', offset=size - 31, size=31),
              dict(name='empty', offset=size, size=0)]
    result = cache.resident(model, ranges)
    assert result['resident_bytes'] == size
    assert result['resident_pct'] == 100
    assert [r['resident_bytes'] for r in result['ranges']] == [r['size'] for r in ranges]
    assert [r['pages'] for r in result['ranges']] == [4, 1, 0]


def test_warm_prefix_and_ranges(model):
    result = cache.warm(model, prefix_bytes=cache.PAGE_SIZE)
    assert result['read_bytes'] == cache.PAGE_SIZE
    assert result['after']['resident_bytes'] > 0
    ranges = [dict(name='b', offset=8193, size=123), dict(name='a', offset=3, size=61)]
    result = cache.warm(model, ranges=ranges)
    assert result['read_bytes'] == 184
    assert result['ranges'] == ranges
    assert cache.warm(model, prefix_bytes=model.stat().st_size + 1)['read_bytes'] == model.stat().st_size


def test_empty(tmp_path):
    path = tmp_path / 'empty'
    path.touch()
    assert cache.resident(path)['resident_bytes'] == 0
    assert cache.resident(path)['pages'] == 0
    assert cache.evict(path)['after']['resident_pct'] == 0
    assert cache.warm(path, prefix_bytes=1)['read_bytes'] == 0


@pytest.mark.parametrize('ranges', [{}, [None], [dict(name='bad', offset=-1, size=1)],
                                   [dict(name='bad', offset=0, size=True)],
                                   [dict(name='bad', offset=0, size=10 ** 12)]])
def test_invalid_ranges(model, ranges):
    with pytest.raises(ValueError):
        cache.resident(model, ranges)
    with pytest.raises(ValueError):
        cache.warm(model, ranges=ranges)


def test_warm_selection_required(model):
    with pytest.raises(ValueError):
        cache.warm(model)
    with pytest.raises(ValueError):
        cache.warm(model, prefix_bytes=-1)
    with pytest.raises(ValueError):
        cache.warm(model, ranges=[], prefix_bytes=0)


@pytest.mark.parametrize('action', ['resident', 'evict', 'warm'])
def test_cli(model, tmp_path, action):
    ranges_path = tmp_path / 'ranges.json'
    ranges_path.write_text(json.dumps([dict(name='head', offset=0, size=100)]))
    command = [sys.executable, cache.__file__, action, str(model), '--json']
    if action != 'evict':
        command += ['--ranges', str(ranges_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert isinstance(json.loads(result.stdout), dict)


@pytest.mark.parametrize('protocol', ['carryover', 'cold', 'warm'])
def test_protocol_record(model, tmp_path, monkeypatch, protocol):
    monkeypatch.setenv('OM_CACHE_PROTOCOL', protocol)
    monkeypatch.setenv('OM_CACHE_WARM_BYTES', '4096')
    monkeypatch.delenv('OM_CACHE_WARM_RANGES', raising=False)
    monkeypatch.delenv('OM_CACHE_MODEL', raising=False)
    argv = ['--model', '/om/' + model.name]
    assert run_protocol.record('start', tmp_path, model.parent, argv) == 0
    start = json.loads((tmp_path / 'page_cache.start.json').read_text())
    assert start['protocol'] == protocol
    assert 0 <= start['resident_pct'] <= 100
    if protocol != 'carryover':
        assert 'before' in start['eviction'] and 'after' in start['eviction']
    if protocol == 'warm':
        assert start['warming']['read_bytes'] == 4096
    assert run_protocol.record('end', tmp_path, model.parent, argv) == 0
    end = json.loads((tmp_path / 'page_cache.end.json').read_text())
    assert 'eviction' not in end and 'resident_pct' in end


def test_protocol_errors_and_host_override(model, tmp_path, monkeypatch):
    monkeypatch.delenv('OM_CACHE_MODEL', raising=False)
    monkeypatch.delenv('OM_CACHE_PROTOCOL', raising=False)
    assert run_protocol.record('start', tmp_path, model.parent, []) == 0
    assert 'error' in json.loads((tmp_path / 'page_cache.start.json').read_text())
    monkeypatch.setenv('OM_CACHE_PROTOCOL', 'cold')
    assert run_protocol.record('start', tmp_path, model.parent, []) == 1
    monkeypatch.setenv('OM_CACHE_MODEL', str(model))
    assert run_protocol.record('start', tmp_path, model.parent, []) == 0
    monkeypatch.setenv('OM_CACHE_PROTOCOL', 'warm')
    monkeypatch.delenv('OM_CACHE_WARM_BYTES', raising=False)
    monkeypatch.delenv('OM_CACHE_WARM_RANGES', raising=False)
    assert run_protocol.record('start', tmp_path, model.parent, []) == 1


def test_protocol_warm_ranges(model, tmp_path, monkeypatch):
    path = tmp_path / 'ranges.json'
    path.write_text(json.dumps([dict(name='expert', offset=17, size=100)]))
    monkeypatch.setenv('OM_CACHE_PROTOCOL', 'warm')
    monkeypatch.setenv('OM_CACHE_MODEL', str(model))
    monkeypatch.setenv('OM_CACHE_WARM_RANGES', str(path))
    assert run_protocol.record('start', tmp_path, model.parent, []) == 0
    record = json.loads((tmp_path / 'page_cache.start.json').read_text())
    assert record['warming']['read_bytes'] == 100


def test_carryover_only_measures_and_end_never_prepares(model, tmp_path, monkeypatch):
    monkeypatch.setenv('OM_CACHE_MODEL', str(model))
    monkeypatch.delenv('OM_CACHE_PROTOCOL', raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail('measurement must not evict or warm')
    monkeypatch.setattr(cache, 'evict', forbidden)
    monkeypatch.setattr(cache, 'warm', forbidden)
    assert run_protocol.record('start', tmp_path, model.parent, []) == 0
    monkeypatch.setenv('OM_CACHE_PROTOCOL', 'warm')
    assert run_protocol.record('end', tmp_path, model.parent, []) == 0


def test_invalid_warm_set_does_not_evict(model, tmp_path, monkeypatch):
    monkeypatch.setenv('OM_CACHE_MODEL', str(model))
    monkeypatch.setenv('OM_CACHE_PROTOCOL', 'warm')
    ranges = tmp_path / 'bad.json'
    ranges.write_text('[{"name":"invalid", "offset":-1, "size":1}]')
    monkeypatch.setenv('OM_CACHE_WARM_RANGES', str(ranges))
    def forbidden(*args):
        pytest.fail('invalid warm configuration must be rejected before eviction')
    monkeypatch.setattr(cache, 'evict', forbidden)
    assert run_protocol.record('start', tmp_path, model.parent, []) == 1
