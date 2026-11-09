#!/usr/bin/env python3
"""Read actual GGUF tensor types, assert the recipe, and emit the handoff inventory/tests."""
import collections
import hashlib
import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/codexPAP'
sys.path.insert(0, str(OUT / 'tree/gguf-py'))
import gguf
import numpy as np

SOURCE = Path.home() / 'models/campaign/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf'
BASELINE = SOURCE.with_name('granite-3.1-1b-a400m-instruct-mixE4K.gguf')
TRAIN = Path.home() / 'models/campaign/data/wikitext/wikitext-2-raw/wiki.train.raw'
TEST = TRAIN.with_name('wiki.test.raw')
EXPERT = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
VARIANTS = [(f'pap{k}_q{b}', k, b, True) for k in (12, 18, 24) for b in (3, 2)]
VARIANTS += [('pap24_q3_noimat', 24, 3, False), ('mix4_imat', 0, 4, True)]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def file_record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def inspect(path, k, bits):
    reader = gguf.GGUFReader(str(path))
    source = gguf.GGUFReader(str(SOURCE))
    assert len(reader.tensors) == len(source.tensors) == 242
    src = {t.name: t for t in source.tensors}
    layers = {str(i): {} for i in range(24)}
    tensors = []
    counts = collections.Counter()
    expert_bytes = collections.Counter()
    for t in reader.tensors:
        original = src[t.name]
        assert list(t.shape) == list(original.shape), t.name
        match = EXPERT.fullmatch(t.name)
        if match:
            layer, kind = int(match[1]), match[2]
            expected = f'Q{bits}_K' if layer < k else 'Q4_K'
            layers[str(layer)][kind] = t.tensor_type.name
            expert_bytes['cpu' if layer < k else 'gpu'] += int(t.n_bytes)
        else:
            # Norm vectors and the router matrix are excluded by upstream policy.
            expected = 'F32' if original.tensor_type.name == 'F32' else 'Q8_0'
        assert t.tensor_type.name == expected, (path.name, t.name, t.tensor_type.name, expected)
        counts[t.tensor_type.name] += 1
        tensors.append({'name': t.name, 'type': t.tensor_type.name,
                        'shape': list(map(int, t.shape)), 'bytes': int(t.n_bytes)})
    assert all(set(layer) == {'gate', 'up', 'down'} for layer in layers.values())
    return {'per_layer_expert_types': layers, 'tensor_type_counts': dict(counts),
            'expert_tensor_bytes_by_placement': dict(expert_bytes), 'tensors': tensors,
            'imatrix_metadata': {key: field.contents() for key, field in reader.fields.items()
                                 if key.startswith('quantize.imatrix.')}}


def tests():
    match = json.loads((ROOT / 'lab/tests/match1.json').read_text())
    kl = json.loads((ROOT / 'lab/tests/mem2.json').read_text())['kl_q8']
    result = {}
    def decode(name, filename, k):
        template = json.loads(json.dumps(match[f'm1_static_n{k}']))
        template['model_file'] = filename
        result[f'pap_decode_{name}_n{k}'] = template
    for name, k, bits, calibrated in VARIANTS:
        for placement in ([k] if k else [12, 18, 24]):
            decode(name, f'{name}.gguf', placement)
    for k in (12, 18, 24):
        decode('mixE4K', BASELINE.name, k)
    for name, filename in [(v[0], f'{v[0]}.gguf') for v in VARIANTS] + [('mixE4K', BASELINE.name)]:
        entry = json.loads(json.dumps(kl))
        entry['model_file'] = filename
        result[f'pap_kl_{name}'] = entry
    assert len(result) == 22
    for entry in result.values():
        assert Path(entry['model_file']).name == entry['model_file']
    (OUT / 'lab-tests-pap.json').write_text(json.dumps(result, indent=2) + '\n')
    return {'path': 'artifacts/codexPAP/lab-tests-pap.json', 'decode_count': 13, 'kl_count': 9,
            'execution_status': 'defined; Docker/GPU execution delegated to orchestrator',
            'names': list(result)}


def main():
    manifest = {'assignment': 'T-PAP', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_commit': '31288c9c66e59377f7788d920678a565bb89dd37',
                'source_model': file_record(SOURCE), 'baseline': file_record(BASELINE),
                'build': {'cuda': False, 'jobs': 4, 'nice': 10,
                          'targets': ['llama-quantize', 'llama-imatrix', 'llama-perplexity']},
                'imatrix': file_record(OUT / 'models/granite-wtrain.imatrix'),
                'models': [], 'lab_tests': tests()}
    imat = manifest['imatrix']
    imat.update({'train': file_record(TRAIN), 'evaluation_text': file_record(TEST),
                 'context': 512, 'requested_chunks': 100, 'threads': 6, 'batch_threads': 6,
                 'from_chunk': 0, 'calibration_tokens': 51200, 'batch_size': 2048, 'parallel_sequences': 4,
                 'gpu_layers': 0, 'format': 'gguf', 'no_ppl': True,
                 'log': 'artifacts/codexPAP/logs/imatrix.log'})
    imat['started_utc'] = (OUT / 'logs/calibrate.start').read_text().strip()
    imat['finished_utc'] = (OUT / 'logs/calibrate.end').read_text().strip()
    imat['calibration_executable_sha256'] = (OUT / 'logs/calibration-binary.sha256').read_text().split()[0]
    manifest['build']['source_archive_sha256'] = (OUT / 'logs/source-archive.sha256').read_text().split()[0]
    r = gguf.GGUFReader(str(OUT / 'models/granite-wtrain.imatrix'))
    imat['metadata'] = {key: field.contents() for key, field in r.fields.items()
                        if not key.startswith('GGUF.')}
    imat['tensor_count'] = len(r.tensors)
    chunks = r.get_field('imatrix.chunk_count')
    assert chunks is not None and chunks.contents() == 100
    imat['completed_chunks'] = chunks.contents()
    coverage = {}
    for t in r.tensors:
        if t.name.endswith('.counts') and EXPERT.fullmatch(t.name[:-7]):
            counts = np.asarray(t.data).reshape(-1)
            assert len(counts) == 32 and np.all(np.isfinite(counts)) and np.all(counts > 0), t.name
            coverage[t.name[:-7]] = {'experts_observed': len(counts),
                                    'min_observations': float(counts.min()),
                                    'max_observations': float(counts.max())}
    assert len(coverage) == 72, len(coverage)
    imat['expert_coverage'] = coverage
    for name, k, bits, calibrated in VARIANTS:
        p = OUT / 'models' / f'{name}.gguf'
        record = file_record(p)
        record.update({'name': name, 'cpu_layers': k, 'cpu_expert_type': f'Q{bits}_K',
                       'imatrix_used': calibrated, 'verification': inspect(p, k, bits)})
        metadata = record['verification']['imatrix_metadata']
        assert bool(metadata) == calibrated
        if calibrated:
            assert metadata['quantize.imatrix.chunks_count'] == 100
            assert metadata['quantize.imatrix.entries_count'] == 192
            assert metadata['quantize.imatrix.dataset'] == str(TRAIN)
        record['started_utc'] = (OUT / f'logs/{name}.start').read_text().strip()
        record['finished_utc'] = (OUT / f'logs/{name}.end').read_text().strip()
        manifest['models'].append(record)
        print(f'{p.name}: {record["bytes"]} bytes sha256={record["sha256"]}')
    baseline = manifest['baseline']
    baseline['verification'] = inspect(BASELINE, 0, 4)
    kl = ROOT / 'artifacts/kl/bf16.kl'
    with kl.open('rb') as f:
        magic, context, vocab, chunks = struct.unpack('<8sIii', f.read(20))
    manifest['kl_baseline'] = dict(file_record(kl), magic=magic.decode(), context=context,
                                   vocab=vocab, chunks=chunks,
                                   header_compatible=magic == b'_logits_' and context == 512 and vocab == 49155)
    sanity = OUT / 'cpu-sanity.json'
    manifest['cpu_sanity'] = json.loads(sanity.read_text()) if sanity.exists() else {'status': 'not run (optional)'}
    manifest['claims'] = {'verified': ['file sizes and SHA256', 'all per-layer tensor types',
                                       'TRAIN imatrix and disjoint TEST file provenance'],
                          'unmeasured': ['GPU decode throughput', 'GPU/host runtime memory',
                                         'quality of all variants other than any CPU sanity result']}
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
