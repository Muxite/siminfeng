#!/usr/bin/env python3
"""Losslessly partition Granite MoE experts using the pinned local gguf-py."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'ref/llama.cpp/gguf-py'))
import gguf

BANK = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
ROUTER = re.compile(r'blk\.(\d+)\.ffn_gate_inp\.weight$')
# Byte-zero blocks decode to exact zero for these types. Do not silently extend.
ZERO_TYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
              gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q8_0}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_hot_set(path, layers, experts):
    hot = [set() for _ in range(layers)]
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError('hot set rows must contain layer expert')
        layer, expert = map(int, fields)
        if not 0 <= layer < layers or not 0 <= expert < experts:
            raise ValueError('hot set index outside model')
        if expert in hot[layer]:
            raise ValueError('duplicate hot set entry')
        hot[layer].add(expert)
    if any(len(h) == experts for h in hot):
        raise ValueError('each layer needs at least one cold expert (zero-size GGUF banks unsupported)')
    return [sorted(h) for h in hot]


def transform(source, output, hot_set):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('input and output must differ')
    if output.exists():
        raise ValueError('refusing to overwrite output')
    reader = gguf.GGUFReader(source)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError('only little-endian GGUF supported')
    arch = reader.fields['general.architecture'].contents()
    if arch != 'granitemoe':
        raise ValueError('only granitemoe supported')
    if any(k.startswith('hybrid_split.') for k in reader.fields):
        raise ValueError('model is already split')
    layers = int(reader.fields[f'{arch}.block_count'].contents())
    experts = int(reader.fields[f'{arch}.expert_count'].contents())
    if not 1 <= layers <= 256 or experts != 32:
        raise ValueError('expected 32 experts and 1..256 layers')
    hot = read_hot_set(hot_set, layers, experts)
    perms = [h + [e for e in range(experts) if e not in h] for h in hot]
    tensors = {t.name: t for t in reader.tensors}
    for il in range(layers):
        for bank in ('gate', 'up', 'down'):
            t = tensors.get(f'blk.{il}.ffn_{bank}_exps.weight')
            if t is None or len(t.shape) != 3 or t.shape[2] != experts or t.tensor_type not in ZERO_TYPES:
                raise ValueError(f'unsupported or missing expert bank in layer {il}')
        router = tensors.get(f'blk.{il}.ffn_gate_inp.weight')
        if router is None or len(router.shape) != 2 or router.shape[1] != experts or router.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f'expected F32 router in layer {il}')
    # Reject unknown expert-indexed tensors: silently leaving them unpermuted is unsafe.
    for t in reader.tensors:
        if ('_exps.' in t.name and not BANK.fullmatch(t.name)) or '.ffn_gate_inp.' in t.name and not ROUTER.fullmatch(t.name):
            raise ValueError(f'unsupported expert/router companion tensor: {t.name}')

    writer = gguf.GGUFWriter(output, arch, endianess=reader.endianess)
    if 'general.alignment' in reader.fields:
        writer.data_alignment = int(reader.fields['general.alignment'].contents())
    for field in reader.fields.values():
        if field.name.startswith('GGUF.') or field.name == 'general.architecture':
            continue
        kind = field.types[0]
        writer.add_key_value(field.name, field.contents(), kind,
                             sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    writer.add_uint32('hybrid_split.version', 1)
    writer.add_string('hybrid_split.source_sha256', sha256(source))
    writer.add_string('hybrid_split.hot_set_sha256', sha256(hot_set))
    writer.add_array('hybrid_split.hot_counts', [len(h) for h in hot])
    # new index -> original index, flattened in layer order; routers use this order.
    writer.add_array('hybrid_split.permutation', [e for p in perms for e in p])
    writer.add_string('hybrid_split.null_expert', 'one byte-zero expert after real hot experts')

    # Keep only descriptors until streaming data, never a second model-sized copy.
    entries = []
    def add(name, t, part, il):
        if part in ('hot', 'cold'):
            shape = list(t.data.shape)
            shape[0] = len(hot[il]) + 1 if part == 'hot' else experts - len(hot[il])
            nbytes = int(np.prod(shape)) * t.data.dtype.itemsize
            writer.add_tensor_info(name, shape, t.data.dtype, nbytes, t.tensor_type)
        elif part == 'map':
            writer.add_tensor_info(name, (experts,) if name.endswith('ids_router') else (experts, 1), np.dtype('int32'), experts * 4, gguf.GGMLQuantizationType.I32)
        else:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        entries.append((name, t, part, il))

    for t in reader.tensors:
        match = BANK.fullmatch(t.name)
        router = ROUTER.fullmatch(t.name)
        if match:
            il = int(match[1])
            add(t.name + '.hot', t, 'hot', il)
            add(t.name + '.cold', t, 'cold', il)
        elif router:
            add(t.name, t, 'router', int(router[1]))
        else:
            add(t.name, t, 'copy', None)
    for il in range(layers):
        add(f'blk.{il}.ffn_expert_ids.ids_hot', None, 'map', il)
        add(f'blk.{il}.ffn_expert_ids.ids_cold', None, 'map', il)
        add(f'blk.{il}.ffn_expert_ids.ids_router', None, 'map', il)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, t, part, il in entries:
        if part == 'hot':
            data = np.zeros((len(hot[il]) + 1, *t.data.shape[1:]), dtype=t.data.dtype)
            data[:-1] = t.data[hot[il]]
        elif part == 'cold':
            data = t.data[perms[il][len(hot[il]):]]
        elif part == 'router':
            data = t.data[perms[il]]
        elif part == 'map':
            nh = len(hot[il])
            inverse = np.argsort(perms[il]).tolist()
            values = ([i if i < nh else nh for i in inverse] if name.endswith('ids_hot')
                      else [-1 if i < nh else i - nh for i in inverse] if name.endswith('ids_cold') else inverse)
            data = np.asarray(values, dtype=np.int32)
            if not name.endswith('ids_router'):
                data = data.reshape(experts, 1)
        else:
            data = t.data
        writer.write_tensor_data(np.ascontiguousarray(data), tensor_endianess=reader.endianess)
    writer.close()
    report = {'source': str(source), 'output': str(output), 'source_sha256': sha256(source),
              'output_sha256': sha256(output), 'hot_set_sha256': sha256(hot_set),
              'hot_counts': [len(h) for h in hot], 'new_to_original': perms,
              'hot_payload_bytes_including_nulls': sum(t.n_bytes // experts * (len(hot[int(BANK.fullmatch(t.name)[1])]) + 1)
                                                       for t in reader.tensors if BANK.fullmatch(t.name)),
              'null_payload_bytes': sum(t.n_bytes // experts for t in reader.tensors if BANK.fullmatch(t.name))}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--hot-set', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    a = p.parse_args()
    allowed = (ROOT / 'artifacts/codexR/models').resolve()
    if not a.output.resolve().is_relative_to(allowed):
        p.error('transformed GGUF outputs must be under artifacts/codexR/models/')
    try:
        report = transform(a.source, a.output, a.hot_set)
        a.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'new_to_original'}))
    except (OSError, ValueError, KeyError) as e:
        p.exit(2, f'error: {e}\n')


if __name__ == '__main__':
    main()
