import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_gguf import transform, read_hot_set, gguf


def tiny(path, quantized=False):
    w = gguf.GGUFWriter(path, 'granitemoe')
    w.add_uint32('granitemoe.block_count', 1)
    w.add_uint32('granitemoe.expert_count', 32)
    w.add_string('test.text', 'preserve me')
    w.add_array('test.ints', [2, 7, 9])
    router = np.arange(32 * 256, dtype=np.float32).reshape(32, 256) / 100
    w.add_tensor('blk.0.ffn_gate_inp.weight', router)
    banks = {}
    rng = np.random.default_rng(4)
    for bank in ('up', 'gate', 'down'):
        a = rng.normal(size=(32, 3, 256)).astype(np.float32)
        if quantized:
            # gguf-py has no Q4_K encoder. Construct finite legal packed blocks.
            a = rng.integers(0, 256, size=(32, 3, 144), dtype=np.uint8)
            a[..., :4] = np.asarray([.03, .01], dtype=np.float16).view(np.uint8)
            w.add_tensor(f'blk.0.ffn_{bank}_exps.weight', a, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        else:
            w.add_tensor(f'blk.0.ffn_{bank}_exps.weight', a)
        banks[bank] = a
    w.add_tensor('output_norm.weight', np.ones(256, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return router, banks


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('selected', [[2, 7, 19], [], list(range(31))])
def test_permutation_and_routing(tmp_path, quantized, selected):
    original, split, hot = tmp_path / 'tiny.gguf', tmp_path / 'split.gguf', tmp_path / 'hot.txt'
    router, banks = tiny(original, quantized)
    hot.write_text('# LLAMA_MOE_CACHE_STATIC v1\n' + ''.join(f'0 {e}\n' for e in selected))
    report = transform(original, split, hot)
    r = gguf.GGUFReader(split)
    ts = {t.name: t for t in r.tensors}
    perm = selected + [i for i in range(32) if i not in selected]
    nh = len(selected)
    assert report['new_to_original'] == [perm]
    assert r.fields['hybrid_split.permutation'].contents() == perm
    assert r.fields['test.text'].contents() == 'preserve me'
    assert r.fields['test.ints'].contents() == [2, 7, 9]
    np.testing.assert_array_equal(ts['blk.0.ffn_gate_inp.weight'].data, router[perm])
    ih = ts['blk.0.ffn_expert_ids.ids_hot'].data.ravel()
    ic = ts['blk.0.ffn_expert_ids.ids_cold'].data.ravel()
    inverse = ts['blk.0.ffn_expert_ids.ids_router'].data
    np.testing.assert_array_equal(ts['blk.0.ffn_gate_inp.weight'].data[inverse], router)
    rng = np.random.default_rng(2)
    x = rng.normal(size=256).astype(np.float32)
    before = router @ x
    after = ts['blk.0.ffn_gate_inp.weight'].data @ x
    np.testing.assert_allclose(after, before[perm], rtol=1e-6, atol=.002)
    old_top = np.argsort(-before)[:8]
    new_top = np.argsort(-after)[:8]
    assert [perm[i] for i in new_top] == old_top.tolist()
    for bank in ('up', 'gate', 'down'):
        h = ts[f'blk.0.ffn_{bank}_exps.weight.hot']
        c = ts[f'blk.0.ffn_{bank}_exps.weight.cold']
        np.testing.assert_array_equal(h.data[:-1], banks[bank][selected])
        np.testing.assert_array_equal(c.data, banks[bank][perm[nh:]])
        for idx, old in enumerate(perm):
            reconstructed = h.data[ih[old]] if ic[old] == -1 else c.data[ic[old]]
            np.testing.assert_array_equal(reconstructed, banks[bank][old])
        null = gguf.quants.dequantize(h.data[-1], h.tensor_type)
        np.testing.assert_array_equal(null, np.zeros_like(null))


def test_invalid_hot_sets(tmp_path):
    path = tmp_path / 'hot.txt'
    for content in ('0 1\n0 1\n', '1 0\n', '0 32\n', '0\n', ''.join(f'0 {i}\n' for i in range(32))):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_hot_set(path, 1, 32)


def test_preserve_input_and_refuse_overwrite(tmp_path):
    source = tmp_path / 'tiny.gguf'
    tiny(source)
    hot = tmp_path / 'hot.txt'
    hot.write_text('0 1\n')
    before = source.read_bytes()
    with pytest.raises(ValueError, match='differ'):
        transform(source, source, hot)
    output = tmp_path / 'out.gguf'
    output.write_bytes(b'exists')
    with pytest.raises(ValueError, match='overwrite'):
        transform(source, output, hot)
    assert source.read_bytes() == before
    assert output.read_bytes() == b'exists'
