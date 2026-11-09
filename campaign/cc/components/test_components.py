"""Offline regression tests. All fixture/scratch writes use pytest's basetemp."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abi_diff
import patch_stack

spec = importlib.util.spec_from_file_location(
    'n3_test_helpers', Path(__file__).resolve().parents[1] / 'build_engine/test_host_linkage.py')
n3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n3)


def repo(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    patch_stack.git(source, 'init', '-q')
    for name in ['a', 'b']:
        (source / name).write_text('one\ntwo\nthree\n')
    patch_stack.git(source, 'add', '-A')
    patch_stack.git(source, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid',
                    'commit', '-qm', 'fixture')
    return source


def patch(tmp_path, name, target, replacement):
    path = tmp_path / name
    path.write_text(f'--- a/{target}\n+++ b/{target}\n@@ -1,3 +1,3 @@\n one\n-two\n+{replacement}\n three\n')
    return path


def test_independent_stack_and_release_replay(tmp_path):
    source = repo(tmp_path)
    a, b = patch(tmp_path, 'a.patch', 'a', 'A'), patch(tmp_path, 'b.patch', 'b', 'B')
    release = tmp_path / 'release.patch'
    report = patch_stack.test_stack(source, 'HEAD', [a, b], tmp_path / 'scratch', release)
    assert report['order_independent'] and len(report['permutations']) == 2
    assert len(report['tree_hashes']) == 1
    assert report['release']['sha256'] == hashlib.sha256(release.read_bytes()).hexdigest()
    replay = patch_stack.test_stack(source, 'HEAD', [release], tmp_path / 'replay')
    assert replay['tree_hashes'] == report['tree_hashes']
    assert (source / 'a').read_text() == 'one\ntwo\nthree\n'
    assert patch_stack.git(source, 'status', '--porcelain').stdout == b''


def test_conflict_both_orders_and_no_release(tmp_path):
    source = repo(tmp_path)
    a, b = patch(tmp_path, 'a.patch', 'a', 'A'), patch(tmp_path, 'b.patch', 'a', 'B')
    report = patch_stack.test_stack(source, 'HEAD', [a, b], tmp_path / 'scratch')
    assert not report['order_independent']
    assert all(not p['applies'] and p['conflict']['stderr'] for p in report['permutations'])
    assert report['pairwise_overlap'][0]['shared_files'][0]['overlapping_hunks'] == [[0, 0]]
    with pytest.raises(ValueError, match='refusing release'):
        patch_stack.test_stack(source, 'HEAD', [a, b], tmp_path / 'scratch', tmp_path / 'bad.patch')
    assert not (tmp_path / 'bad.patch').exists()


def test_applies_but_order_changes_tree(tmp_path):
    source = repo(tmp_path)
    a = tmp_path / 'a.patch'
    b = tmp_path / 'b.patch'
    # Identical context appears twice after either patch; Git applies at another
    # location with an offset, making both orders valid but different.
    (source / 'a').write_text('prefix\nx\nx\nx\nseparator\nx\nx\nx\nsuffix\n')
    patch_stack.git(source, 'add', '-A')
    patch_stack.git(source, '-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid',
                    'commit', '-qm', 'repeated lines')
    for p, value in [(a, 'A'), (b, 'B')]:
        p.write_text(f'--- a/a\n+++ b/a\n@@ -2,3 +2,3 @@\n x\n-x\n+{value}\n x\n')
    report = patch_stack.test_stack(source, 'HEAD', [a, b], tmp_path / 'scratch')
    assert all(p['applies'] for p in report['permutations'])
    assert len(report['tree_hashes']) == 2 and not report['order_independent']


@pytest.mark.parametrize('path,expected', [
    ('ggml/src/ggml-cuda/mmvq.cu', 'backend-cuda'),
    ('ggml/src/ggml-cpu/ggml-cpu.c', 'backend-cpu'),
    ('ggml/src/ggml-backend.cpp', 'ggml-core'), ('ggml/src/ggml.c', 'ggml-core'),
    ('ggml/src/ggml-alloc.c', 'ggml-core'), ('src/models/granite.cpp', 'libllama'),
    ('common/arg.cpp', 'libllama-common'), ('tools/quantize/a.cpp', 'tool'),
    ('tests/CMakeLists.txt', 'build'), ('ggml/cmake/a.cmake', 'build'),
    ('docs/ops.md', 'unmapped'), ('tests/test.cpp', 'unmapped')])
def test_layer(path, expected):
    assert patch_stack.layer(path) == expected


def test_hunk_counts(tmp_path):
    p = tmp_path / 'patch'
    p.write_text('--- a/src/a\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n')
    entry = patch_stack.inspect_patch(p)['files'][0]
    assert entry['path'] == 'src/a'
    assert entry['hunks'][0]['old_count'] == 1
    assert entry['hunks'][0]['new_count'] == 0


@pytest.mark.parametrize('wide,endian', [(True, '<'), (True, '>'), (False, '<'), (False, '>')])
def test_n3_synthetic_elf(tmp_path, wide, endian):
    p = tmp_path / 'libfixture.so'
    p.write_bytes(n3.crafted_elf(wide=wide, endian=endian))
    info = abi_diff.inspect_library(p)
    assert info['dt_needed'] == ['libcudart.so.12']
    assert info['soname'] is None
    assert {s['name'] for s in info['exports']} == {'cudaFree', 'cudaMemGetInfo', 'cuMemCreate'}
    assert info['elf_class'] == (64 if wide else 32)


def versioned_fixture(version='VER_1', hidden=False, soname='fixture.so'):
    # Extend T-N3's portable ELF64 fixture with SONAME, versym and verdef.
    raw = n3.crafted_elf()
    shfmt = '<IIQQQQIIQQ'
    sections = [struct.unpack_from(shfmt, raw, 64 + i * 64) for i in range(4)]
    strings = raw[sections[1][4]:sections[1][4] + sections[1][5]]
    soname_offset = len(strings)
    strings += soname.encode() + b'\0'
    ver_offset = len(strings)
    strings += version.encode() + b'\0'
    symbols = raw[sections[2][4]:sections[2][4] + sections[2][5]]
    dynamic = struct.pack('<qQqQqQ', 1, 1, 14, soname_offset, 0, 0)
    versions = [1] * (len(symbols) // 24)
    versions[4] = 2 | (0x8000 if hidden else 0)
    versym = struct.pack('<' + 'H' * len(versions), *versions)
    verdef = struct.pack('<HHHHIII', 1, 0, 2, 1, 0, 20, 0) + struct.pack('<II', ver_offset, 0)
    payloads = [strings, symbols, dynamic, versym, verdef]
    attrs = [(3, 0, 0), (11, 1, 24), (6, 1, 16), (0x6fffffff, 2, 2), (0x6ffffffd, 1, 0)]
    table, offset = bytes(64), 64 + 6 * 64
    for data, (kind, link, stride) in zip(payloads, attrs):
        table += struct.pack(shfmt, 0, kind, 0, 0, offset, len(data), link, 0, 1, stride)
        offset += len(data)
    header = bytearray(raw[:64])
    struct.pack_into('<H', header, 60, 6)
    return bytes(header) + table + b''.join(payloads)


def compare_fixture(tmp_path, a, b):
    before, after = tmp_path / 'before', tmp_path / 'after'
    before.mkdir()
    after.mkdir()
    (before / 'lib.so.1').write_bytes(a)
    (after / 'lib.so.1').write_bytes(b)
    (before / 'lib.so').symlink_to('lib.so.1')
    (after / 'lib.so').symlink_to('lib.so.1')
    return abi_diff.compare_directories(before, after)


def test_versioned_export_and_soname(tmp_path):
    p = tmp_path / 'lib.so'
    p.write_bytes(versioned_fixture())
    info = abi_diff.inspect_library(p)
    assert info['soname'] == 'fixture.so'
    symbol = next(s for s in info['exports'] if s['name'] == 'cudaFree')
    assert symbol['version'] == 'VER_1' and symbol['default_version']


@pytest.mark.parametrize('kwargs,reason', [({'version': 'VER_2'}, 'exports removed or changed'),
                                         ({'hidden': True}, 'exports removed or changed'),
                                         ({'soname': 'other.so'}, 'soname changed')])
def test_incompatible_versions_and_soname(tmp_path, kwargs, reason):
    result = compare_fixture(tmp_path, versioned_fixture(), versioned_fixture(**kwargs))
    assert not result['compatible']
    assert reason in result['libraries'][0]['reasons']


def test_added_export_direction(tmp_path):
    result = compare_fixture(tmp_path, n3.crafted_elf(), n3.crafted_elf(import_malloc=False))
    assert result['compatible'] and result['libraries'][0]['bytes_changed']
    assert [s['name'] for s in result['libraries'][0]['exports_added']] == ['cudaMalloc']
    result = abi_diff.compare_directories(tmp_path / 'after', tmp_path / 'before')
    assert not result['compatible']


def test_bad_elf_and_missing_library(tmp_path):
    result = compare_fixture(tmp_path, b'not elf', n3.crafted_elf())
    assert not result['compatible'] and result['libraries'][0]['before']['error']
    (tmp_path / 'after/lib.so').unlink()
    (tmp_path / 'after/lib.so.1').unlink()
    result = abi_diff.compare_directories(tmp_path / 'before', tmp_path / 'after')
    assert not result['compatible'] and result['libraries'][0]['after'] is None


def test_escaping_symlink(tmp_path):
    (tmp_path / 'lib.so').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='escapes'):
        abi_diff.inventory(tmp_path)


def test_n3_compiler_fixture(tmp_path, monkeypatch):
    # T-N3 helper may compile; direct its temporary files into our allowed root.
    monkeypatch.setattr(n3.tempfile, 'tempdir', str(tmp_path))
    n3.fixture_elf.cache_clear()
    p = tmp_path / 'compiled.so'
    p.write_bytes(n3.fixture_elf())
    info = abi_diff.inspect_library(p)
    assert 'libcudart.so.12' in info['dt_needed']
    assert any(s['name'] == 'backend' for s in info['exports'])


def test_catalog_schema_and_component_prerequisites():
    import jsonschema
    root = Path(__file__).resolve().parent
    schema = json.loads((root / 'components.schema.json').read_text())
    catalog = json.loads((root / 'components.json').read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(catalog, schema)
    entries = {c['id']: c for c in catalog['components']}
    assert len(entries) == len(catalog['components']) == 8
    assert entries['hybrid_split']['status'] == entries['static_hot']['status'] == 'rejected'
    for item in entries.values():
        for prerequisite in item['prerequisites']:
            if prerequisite['kind'] == 'component':
                assert prerequisite['value'] in entries
    assert {'kind': 'component', 'value': 'kv_quant_fix'} in entries['kv-q8']['prerequisites']
    campaign = root.parents[1]
    for item in entries.values():
        for record in item['identities']:
            # Hash small authoring inputs; large model artifact identity is in
            # its provenance catalog and was recorded by make_manifests.py.
            if record['role'] in ('patch', 'recipe', 'source'):
                assert patch_stack.sha256(campaign / record['path']) == record['sha256']


def test_dependency_change_and_identical_bytes(tmp_path):
    baseline = versioned_fixture()
    result = compare_fixture(tmp_path, baseline, baseline)
    assert result['compatible'] and not result['libraries'][0]['bytes_changed']
    (tmp_path / 'after/lib.so.1').write_bytes(baseline.replace(b'libcudart.so.12', b'libcudart.so.13'))
    result = abi_diff.compare_directories(tmp_path / 'before', tmp_path / 'after')
    assert not result['compatible']
    assert 'dt_needed changed' in result['libraries'][0]['reasons']


def test_visibility_and_weak_exports(tmp_path):
    raw = bytearray(n3.crafted_elf())
    symbols = struct.unpack_from('<IIQQQQIIQQ', raw, 64 + 2 * 64)[4]
    raw[symbols + 4 * 24 + 4] = 0x22  # weak FUNC
    raw[symbols + 5 * 24 + 5] = 2     # hidden
    raw[symbols + 6 * 24 + 5] = 1     # internal
    p = tmp_path / 'lib.so'
    p.write_bytes(raw)
    exports = abi_diff.inspect_library(p)['exports']
    assert len(exports) == 1 and exports[0]['name'] == 'cudaFree'
    assert exports[0]['binding'] == 2
