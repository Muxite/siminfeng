"""Materialize content-addressed manifests from read-only campaign inputs."""
import hashlib
import json
from pathlib import Path

OUT = Path('cc/components')
BASE = '31288c9c66e59377f7788d920678a565bb89dd37'


def identity(path, role):
    return {'path': path, 'role': role,
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def binary(path, abi, observed=None):
    return {'path': path, 'abi_version': abi,
            'sha256': identity(observed, 'binary')['sha256'] if observed else None,
            'observed_path': observed}


def component(id, layers, kind, status, files, outputs, prereqs, tests, evidence, scope, config=None):
    return {'id': id, 'layers': layers, 'kind': kind, 'base_commit': BASE,
            'identities': files, 'outputs': outputs, 'ggml_abi_version': '0.27.0',
            'supported_models': ['Granite-3.1-1B-A400M-Instruct'],
            'supported_arch': ['linux-x86_64', 'CUDA-sm_120 / RTX 5070 Ti'],
            'prerequisites': prereqs, 'tests': tests, 'evidence': evidence,
            'status': status, 'scope': scope, 'config': config or {'argv': [], 'env': {}, 'build': []}}


def req(kind, value):
    return {'kind': kind, 'value': value}


def test(name, result, evidence):
    return {'name': name, 'result': result, 'evidence': evidence}


items = []
items.append(component('q8_reuse', ['backend-cuda'], 'source-patch', 'accepted',
    [identity('cc/q8_reuse/q8-activation-reuse.patch', 'patch')],
    [binary('libggml-cuda.so.0.27.0', '0.27.0', 'artifacts/x7-comp/reuse/libggml-cuda.so.0.27.0')],
    [req('runtime', 'GGML_CUDA_Q8_REUSE=1 (unset also enables); eligible Q8_0 batch-1 MMVQ path'),
     req('build', 'same pinned engine/compiler/CUDA settings for swaps')],
    [test('CUDA MUL_MAT', '2445/2445 passed', 'D029'),
     test('paired quality', 'greedy text and PPL identical OFF/ON', 'D029'),
     test('decode ABBA', '+2.06% Q8_0 llama-bench', 'D029'),
     test('backend swap', 'same greedy text; +2.0% decode', 'D036'),
     test('ELF surface', 'two weak C++ exports removed/two added; strict compatibility fails',
          'artifacts/codexCOMP/abi-diff.json')],
    ['D026', 'D029', 'D036'],
    'Qualified Granite batch-1 decode on RTX 5070 Ti. No memory-reduction claim. '
    'Recorded swap works but exported ELF surface is not identical; arbitrary consumers are unqualified.',
    {'argv': [], 'env': {'GGML_CUDA_Q8_REUSE': '1'}, 'build': []}))
items.append(component('kv_quant_fix', ['ggml-core'], 'source-patch', 'accepted',
    [identity('cc/kv_quant_fix/kv-quant-fix.patch', 'patch')],
    [binary('libggml-base.so.0.27.0', '0.27.0')],
    [req('source', 'apply to pinned scheduler; rebuild ggml-base')],
    [test('scheduler node+leaf capacity', 'CPU regression recorded', 'artifacts/codexKVB/ROOTCAUSE.md'),
     test('quantized KV CUDA load and quality', 'D033 job 15/15 clean; q8_0 PPL +0.01%', 'D033'),
     test('release composability', 'all 2 permutations, identical tree', 'artifacts/codexCOMP/release-stack.json')],
    ['D033'], 'Generic scheduler capacity fix; GPU qualification recorded for Granite. Correctness enabler, not a speed gain.'))
items.append(component('imatrix-quantization', ['model-artifact'], 'artifact', 'accepted',
    [identity('cc/pap/make_pap.sh', 'recipe'), identity('artifacts/codexPAP/MANIFEST.json', 'provenance')],
    [binary('artifacts/codexPAP/models/granite-wtrain.imatrix', None,
            'artifacts/codexPAP/models/granite-wtrain.imatrix'),
     binary('artifacts/codexPAP/models/mix4_imat.gguf', None,
            'artifacts/codexPAP/models/mix4_imat.gguf')],
    [req('artifact', 'BF16 GGUF + WikiText TRAIN SHA256s in T-PAP MANIFEST; TEST held out'),
     req('recipe', 'make_pap.sh calibrate: context 512, 100 chunks; quantize mix4_imat with --imatrix, experts Q4_K/base Q8_0'),
     req('quality', 'qualify each new model/corpus; PAP lower-bit variants have separate status')],
    [test('GGUF types and identities', 'verified by T-PAP', 'artifacts/codexPAP/MANIFEST.json'),
     test('held-out KLD', '4-bit 0.040->0.027 (-33%); 3-bit -13%', 'D035')],
    ['D032', 'D035'], 'Artifact recipe, no inference engine modification. Imatrix accepted default within measured scope; '
    'PAP12-Q3 speed is provisional and is not promoted here. Generalization beyond WikiText untested.'))
items.append(component('flash-attn', ['runtime-config'], 'runtime-flag', 'accepted', [], [],
    [req('runtime', 'CUDA FA support for model/head dimensions/KV types')],
    [test('long-context f16 comparison', '-984 MiB; decode +10%; prefill +19%; text differs', 'D028'),
     test('quality limitation', 'FA vs non-FA paired PPL not measured', 'D028')],
    ['D028'], 'Recommended for long context; reduction order changes output. Quality equivalence to non-FA not established.',
    {'argv': ['-fa', 'on'], 'env': {}, 'build': []}))
items.append(component('kv-q8', ['runtime-config'], 'runtime-flag', 'accepted', [], [],
    [req('component', 'kv_quant_fix'), req('component', 'flash-attn'),
     req('runtime', 'set both K and V cache types to q8_0; declare context')],
    [test('32k memory and speed', '-720 MiB (2762->2042); decode -10.5%', 'D033'),
     test('WikiText PPL 4k', '7.5385->7.5392 (+0.01%); passed +1% gate', 'D033')],
    ['D028', 'D033'], 'Representation trade-off measured on Granite mixE4K; q4_0 KV fails the same accuracy gate.',
    {'argv': ['-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0'], 'env': {}, 'build': []}))
items.append(component('hybrid_split', ['build', 'backend-cpu', 'libllama', 'unmapped'], 'source-patch', 'rejected',
    [identity('cc/hybrid_split/hybrid-split.patch', 'patch')],
    [binary('libggml-cpu.so.0.27.0', '0.27.0'), binary('libllama.so.0.7.0', '0.7.0')],
    [req('model', 'Granite expert split configuration; see D027; never auto-select rejected candidates')],
    [test('matched memory integrated decode', '-16%/-19%/-32% vs static offload', 'D030'),
     test('quality', 'gate not reached; CUDA unit mismatch open', 'D030'),
     test('three-patch composability', '6 orders clean and identical; no runtime implication', 'artifacts/codexCOMP/multi-layer-stack.json')],
    ['D027', 'D030', 'D031'], 'Rejected performance candidate retained as multi-layer test case. Unmapped files are docs and test source.'))
items.append(component('static_hot', ['libllama'], 'source-patch', 'rejected',
    [identity('cc/moe_static_hot/static-hot.patch', 'patch')],
    [binary('libllama.so.0.7.0', '0.7.0')],
    [req('runtime', 'LLAMA_MOE_CACHE_STATIC hot-set path; positive --moe-cache-mib; host expert weights'),
     req('policy', 'never auto-select rejected candidates')],
    [test('engagement fingerprint', 'failed: banks allocated but unused; CPU fingerprint', 'D024')],
    ['D024'], 'Mechanism never engages; static mode returns no slot map. Not an acceleration component.'))
items.append(component('gpu_budget', ['infrastructure'], 'binary', 'accepted',
    [identity('cc/gpu_budget/gpubudget.c', 'source'), identity('cc/gpu_budget/build.sh', 'recipe'),
     identity('artifacts/x3-shim/cuda/libgpubudget.so', 'binary')],
    [binary('libgpubudget.so', None, 'artifacts/x3-shim/cuda/libgpubudget.so')],
    [req('build', 'GGML_CUDA_NO_VMM=ON is compile-time; dynamic libcudart with verified imports'),
     req('runtime', 'LD_PRELOAD=libgpubudget.so; GPU_BUDGET_MIB and per-process log'),
     req('measurement', 'verify interception and measure uncounted context/driver overhead')],
    [test('real CUDA shim APIs', '39/39 passed', 'D021'),
     test('engine probe', '3 cudaMalloc intercepted; VMM temporaries uncovered', 'D021')],
    ['D021', 'D023'], 'Infrastructure only: cooperative per-process intercepted-allocation budget, bypassable; '
    'not a hard physical VRAM cap. Shim ABI unversioned; ggml ABI field describes the engine context.'))
(OUT / 'components.json').write_text(json.dumps({'schema_version': 1, 'components': items}, indent=2) + '\n')
