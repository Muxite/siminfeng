"""Active weight traffic model; bandwidths are effective GB/s, times are microseconds."""
from __future__ import annotations
import math

TYPE_BY_REP = {'mixE4K': 'Q4_K', 'mixE3K': 'Q3_K', 'mixE2K': 'Q2_K',
               'Q8_0': 'Q8_0', 'bf16': 'BF16', 'Q4_K_M': 'Q4_K'}


def traffic(stats, family, coord, coverage=None):
    layers, experts, used = (stats[k] for k in ('n_layer', 'n_expert', 'n_expert_used'))
    if family not in ('static', 'hybrid', 'lru'):
        raise ValueError('unknown placement family')
    if family == 'static' and (type(coord) is not int or not 0 <= coord <= layers):
        raise ValueError('CPU layers outside model')
    tensors = [t for t in stats.get('tensors', []) if t['role'] == 'expert_ffn']
    # Sibling representation enumeration can retain the input tensor headers.
    if tensors and not math.isclose(sum(t['bytes'] for t in tensors), stats['expert_bytes'], rel_tol=1e-6):
        tensors = []
    if not tensors:
        tensors = [dict(layer=l, ggml_type=TYPE_BY_REP.get(stats.get('representation'), 'Q4_K'),
                        bytes=stats['expert_bytes']/layers) for l in range(layers)]
    active = stats['expert_bytes'] * used/experts
    gpu = stats['nonexpert_bytes']
    cpu = {}
    resident = 0
    if family in ('hybrid', 'lru'):
        if coverage is None:
            coverage = min(1., coord*1048576/stats['expert_bytes']) if family == 'lru' else coord/(layers*experts)
        if not 0 <= coverage <= 1:
            raise ValueError('coverage must be in [0,1]')
    for tensor in tensors:
        fraction = float(tensor['layer'] < coord) if family == 'static' else 1-coverage
        size = tensor['bytes'] * used/experts
        kind = tensor['ggml_type']
        if family != 'lru':
            cpu[kind] = cpu.get(kind, 0.) + size*fraction
            gpu += size*(1-fraction)
        else:
            gpu += size  # cached experts still execute on GPU; misses transfer, not CPU GEMV
        resident += tensor['bytes']*(1-fraction)
    touched = coord if family == 'static' else 0
    # A split dispatches in every layer containing cold experts (D031).
    if family == 'hybrid':
        touched = layers if coverage < 1 else 0
    return dict(gpu_active_bytes=gpu, cpu_active_bytes=cpu, cpu_touching_layers=touched,
                resident_expert_bytes=resident, h2d_bytes=active*(1-coverage) if family == 'lru' else 0.,
                cache_misses=layers*used*(1-coverage) if family == 'lru' else 0.)


def predict(stats, family, coord, constants, coverage=None):
    parts = traffic(stats, family, coord, coverage)
    gpu_us = parts['gpu_active_bytes']/ (constants['bw_gpu_gbs']*1000)
    cpu_us, fallbacks = 0., []
    fitted = constants['bw_cpu_gbs']
    screened = constants.get('screen_bw_cpu_gbs', {})
    for kind, size in parts['cpu_active_bytes'].items():
        if not size:
            continue
        bw = fitted.get(kind)
        if bw is None:
            bw = screened.get(kind, fitted['Q4_K'])
            fallbacks.append(f'{kind}: indicative CPU screen' if kind in screened else f'{kind}: Q4_K bandwidth fallback')
        cpu_us += size/(bw*1000)
    handoff_us = parts['cpu_touching_layers']*constants['handoff_us']
    transfer_us = parts['h2d_bytes']/(constants.get('bw_h2d_gbs', 12.)*1000)
    cache_us = parts['cache_misses']*constants.get('cache_miss_us', 0.)
    total = gpu_us+cpu_us+handoff_us+transfer_us+cache_us
    if not math.isfinite(total) or total <= 0:
        raise ValueError('invalid physical decode prediction')
    return dict(decode_us=total, decode_tps=1e6/total, gpu_us=gpu_us, cpu_us=cpu_us,
                handoff_us=handoff_us, transfer_us=transfer_us, cache_us=cache_us,
                traffic=parts, fallbacks=fallbacks)
