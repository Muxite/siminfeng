import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess

import pytest

SPEC = importlib.util.spec_from_file_location('policy_replay', Path(__file__).with_name('replay.py'))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def test_classlru_matches_pinned_cpp_every_plan(tmp_path):
    """Compile the actual upstream policy, compare every slot, hit and admission."""
    compiler = shutil.which('g++')
    assert compiler, 'g++ required for upstream differential test'
    source = (r.ROOT / 'artifacts/codexPOL/source/classlru.cpp').read_text()
    body = source[source.index('struct moe_cache_lru {'):source.index('// gate, up, down')]
    cpp = r'''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)
#define GGML_ABORT(...) std::abort()
'''+body+r'''
int main() {
    int cap, q; std::cin >> cap >> q;
    moe_cache_lru cache; cache.init(4, 128, cap);
    std::vector<std::vector<int32_t>> maps(4, std::vector<int32_t>(128, cap));
    std::vector<uint32_t> carry(4, 0);
    for (int i=0;i<4;i++) cache.slot_map[i] = maps[i].data();
    int layer;
    while (std::cin >> layer) {
        int32_t ids[8]; for (auto &e : ids) std::cin >> e;
        std::vector<moe_cache_lru::fill> fills, cpu;
        size_t hits;
        uint32_t share = cache.n_filled == cap ? q : 0;
        auto ncpu = [&](size_t n) {
            uint64_t acc = n*uint64_t(share) + carry[layer];
            carry[layer] = acc & 65535; return size_t(acc >> 16);
        };
        cache.plan(layer, ids, 8, 1, 8, q ? 8 : 0, ncpu, fills, cpu, hits);
        std::cout << hits << ' ' << fills.size() << ' ' << cpu.size();
        for (int s=0;s<cap;s++) std::cout << ' ' << cache.key_of[s] << ' ' << int(cache.list_of[s]);
        std::cout << '\n';
    }
}
'''
    src = tmp_path / 'oracle.cpp'
    exe = tmp_path / 'oracle'
    src.write_text(cpp)
    subprocess.run([compiler, '-std=c++17', '-O2', str(src), '-o', str(exe)], check=True, capture_output=True)
    rng = random.Random(4189)
    events = [(rng.randrange(4), rng.sample(range(24 if i % 7 else 128), 8)) for i in range(240)]
    for cap in (8, 11, 32, 65):
        for share in (0, .37, 1):
            inp = f'{cap} {int(share * 65536 + .5)}\n' + ''.join(f'{l} ' + ' '.join(map(str, es)) + '\n' for l, es in events)
            output = subprocess.run([str(exe)], input=inp, text=True, check=True, capture_output=True).stdout.splitlines()
            cache = r.Cache(cap, 'ClassLRU', share=share)
            for (l, es), line in zip(events, output, strict=True):
                stat = cache.plan(l, es)
                expected = [*stat]
                for key, cls in zip(cache.keys, cache.classes):
                    expected.extend((-1 if key is None else key[0] * 128 + key[1], cls))
                assert expected == list(map(int, line.split()))


def test_lru_matches_historical_cpp_every_plan(tmp_path):
    source = (r.ROOT / 'artifacts/codexPOL/source/lru.cpp').read_text()
    body = source[source.index('struct moe_cache_lru {'):source.index('// gate, up, down')]
    cpp = '''#include <algorithm>\n#include <cstdint>\n#include <cstdlib>\n#include <iostream>\n#include <vector>\n#define GGML_ASSERT(x) do { if (!(x)) std::abort(); } while (0)\n'''+body+r'''
int main() {
    moe_cache_lru c; c.init(4, 128, 19);
    std::vector<std::vector<int32_t>> maps(4, std::vector<int32_t>(128, -1));
    for(int i=0;i<4;i++) c.slot_map[i] = maps[i].data();
    int l; while(std::cin >> l) {
        int32_t ids[8]; for(auto &e:ids) std::cin >> e;
        std::vector<moe_cache_lru::fill> fills; size_t hits;
        c.plan(l, ids, 8, fills, hits); std::cout << hits;
        for(auto k:c.key_of) std::cout << ' ' << k;
        std::cout << '\n';
    }
}
'''
    src, exe = tmp_path / 'lru.cpp', tmp_path / 'lru'
    src.write_text(cpp)
    subprocess.run(['g++', '-std=c++17', '-O2', str(src), '-o', str(exe)], check=True, capture_output=True)
    rng = random.Random(789)
    events = [(rng.randrange(4), rng.sample(range(128), 8)) for _ in range(200)]
    inp = ''.join(f'{l} ' + ' '.join(map(str, es)) + '\n' for l, es in events)
    lines = subprocess.run([str(exe)], input=inp, text=True, capture_output=True, check=True).stdout.splitlines()
    c = r.Cache(19, 'LRU')
    for (l, es), line in zip(events, lines, strict=True):
        h, _, _ = c.plan(l, es)
        assert [h] + [-1 if k is None else k[0] * 128 + k[1] for k in c.keys] == list(map(int, line.split()))


@pytest.mark.parametrize('policy', r.POLICIES[:-1])
def test_cpu_share_freezes_full_residency(policy):
    c = r.Cache(8, policy, share=1)
    assert c.plan(0, list(range(8))) == (0, 8, 0)
    initial = set(c.maps)
    assert c.plan(0, list(range(8, 16))) == (0, 0, 8)
    assert set(c.maps) == initial
    assert c.plan(0, list(range(8))) == (8, 0, 0)


def test_static_training_only_and_tie_order():
    groups = [{'layers': [0, 1]}]
    a = [{'slots': 8}]
    counts = r.Counter({(1, 7): 10, (0, 3): 4})
    hot = r.static_sets(groups, a, counts, 8)[0]
    assert hot[:2] == [(1, 7), (0, 3)]
    assert hot[2:] == [(0, e) for e in (0, 1, 2, 4, 5, 6)]
    for share in (0, 1):
        c = r.Cache(8, 'Static', hot, share)
        for _ in range(3):
            h, u, cpu = c.plan(1, list(range(8)))
            assert (h, u, cpu) == ((1, 0, 7) if share else (1, 7, 0))
        assert c.static == set(hot)


def test_budget_allocation_reserves_guard_and_zero_slots():
    groups = [{'layers': [0, 1], 'projection_bytes': [128, 256, 384], 'expert_bytes': 768, 'host_bytes': 768 * 256}]
    assert r.allocate(groups, 768 * 30)[0]['slots'] == 29
    assert r.allocate(groups, 768 * 30, share=1)[0]['slots'] == 21
    assert r.allocate(groups, 768 * 1000)[0]['slots'] == 256
    with pytest.raises(ValueError):
        r.allocate(groups, 100)


@pytest.mark.parametrize('damage', ['duplicate', 'expert', 'weight', 'missing_layer', 'gap'])
def test_trace_validation(tmp_path, damage):
    rows = [dict(seq=0, pos=p, layer=l, phase=ph, experts=list(range(8)), weights=[.125] * 8)
            for p, ph in ((0, 'prefill'), (1, 'decode'), (2, 'decode')) for l in range(2)]
    if damage == 'duplicate':
        rows.append(dict(rows[-1]))
    elif damage == 'expert':
        rows[0]['experts'][-1] = 128
    elif damage == 'weight':
        rows[0]['weights'][0] = float('nan')
    elif damage == 'missing_layer':
        rows.pop()
    elif damage == 'gap':
        rows = [row for row in rows if row['pos'] != 1]
        for row in rows:
            if row['pos'] == 2:
                row['pos'] = 3
        # Add a decode position before the gap.
        rows.extend(dict(rows[-1], pos=1, layer=l) for l in range(2))
    path = tmp_path / 'trace.jsonl'
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    with pytest.raises(ValueError):
        r.read_trace(path, 2, 128, 8)


def test_predict_units_and_fit():
    p = {'h2d_decimal_GB_s': 28.7, 'gpu_floor_seconds': .004,
         'cpu_full_layer_seconds': .001, 'cpu_boundary_seconds': .00003}
    s = {'upload_bytes_per_token': 28.7e6, 'cpu_bytes_per_token': 8e6,
         'cpu_active_layers_per_token': 1}
    assert r.predict(s, p, 1e6)['seconds'] == pytest.approx(.00603)
    pts = [{'upload_seconds': x, 'observed_tps': 1 / (.004 + 1.7*x)} for x in (.001, .004)]
    assert r.fit_multiplier(pts, .004) == pytest.approx(1.7)


def test_crossovers_keep_reversals():
    result = r.crossings([(1, -.001), (2, .001), (3, -.001)])
    assert [c['GB'] for c in result] == [1.5, 2.5]
    assert [c['direction'] for c in result] == ['cache wins', 'static wins']


def test_slru_reconstruction_demotes_protected_overflow():
    c = r.Cache(8, 'SLRU*')
    c.plan(0, list(range(8)))
    c.plan(0, list(range(8)))
    assert len(c.lists[1]) == 4
    assert {c.keys[s] for s in c.lists[1]} == {(0, e) for e in range(4, 8)}
    c.plan(1, list(range(8)))
    assert c.filled == 8
