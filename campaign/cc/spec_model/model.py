"""Trace expert unions and a static CPU-offload speculative bandwidth forecast.

No engine execution or planner mutation. Times are microseconds, bytes are bytes,
and bandwidths are decimal GB/s, matching moe_planner/decode_model.py.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path


def decode_runs(path, n_layer=48, n_expert=128, top_k=8):
    """Return complete consecutive decode runs; never bridge gaps or sequences.

    Ignore prefill entirely. Reject malformed decode records and incomplete tokens
    rather than silently estimating unions from partial layers.
    """
    tokens = defaultdict(dict)
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] != 'decode':
                continue
            seq, pos, layer = (row[k] for k in ('seq', 'pos', 'layer'))
            ids = row['experts']
            if (any(type(x) is not int or x < 0 for x in (seq, pos, layer))
                    or layer >= n_layer or len(ids) != top_k
                    or any(type(x) is not int or not 0 <= x < n_expert for x in ids)
                    or len(set(ids)) != top_k):
                raise ValueError('invalid decode routing record')
            token = tokens[seq, pos]
            if layer in token:
                raise ValueError('duplicate decode token/layer')
            token[layer] = frozenset(ids)
    if not tokens:
        raise ValueError('no decode tokens')
    runs, run, previous = [], [], None
    for key, layers in sorted(tokens.items()):
        if set(layers) != set(range(n_layer)):
            raise ValueError('incomplete decode token')
        if previous is not None and key != (previous[0], previous[1] + 1):
            runs.append(run)
            run = []
        run.append([layers[layer] for layer in range(n_layer)])
        previous = key
    runs.append(run)
    return runs


def union_curves(runs, max_d=16):
    """All overlapping length-d windows, averaged per layer with equal weight."""
    if type(max_d) is not int or max_d < 1:
        raise ValueError('max_d must be positive')
    n_layer = len(runs[0][0])
    curves = {}
    for d in range(1, max_d + 1):
        totals, windows = [0] * n_layer, 0
        for run in runs:
            for start in range(len(run) - d + 1):
                for layer in range(n_layer):
                    totals[layer] += len(set().union(*(t[layer] for t in run[start:start+d])))
                windows += 1
        if not windows:
            raise ValueError(f'no consecutive windows of length {d}')
        per_layer = [x / windows for x in totals]
        curves[d] = dict(windows=windows, per_layer=per_layer,
                         mean=sum(per_layer)/n_layer)
    return curves


def pool_curves(curves):
    """Pool windows without inventing adjacency across traces."""
    out = {}
    for d in curves[0]:
        windows = sum(c[d]['windows'] for c in curves)
        layers = [sum(c[d]['per_layer'][l]*c[d]['windows'] for c in curves)/windows
                  for l in range(len(curves[0][d]['per_layer']))]
        out[d] = dict(windows=windows, per_layer=layers, mean=sum(layers)/len(layers))
    return out


def predict(stats, constants, k, d, a, unions, *, draft_us=0.,
            extra_us_per_token=0., scale_nonexpert=False):
    """Forecast accepted drafts/s for one length-d verification batch.

    a is aggregate accepted_drafts / proposed_drafts, NOT conditional per-position
    survival. Reward is a*d; bonus/correction tokens are excluded. All d proposals
    incur traffic even on rejection. Nonexpert weights are read once per step;
    scale_nonexpert=True provides the literal whole-GPU-union sensitivity.
    """
    n_layer, n_expert, top_k = (stats[x] for x in ('n_layer', 'n_expert', 'n_expert_used'))
    if type(k) is not int or not 0 <= k <= n_layer:
        raise ValueError('CPU layer count outside model')
    if type(d) is not int or d < 1 or not math.isfinite(a) or not 0 <= a <= 1:
        raise ValueError('invalid draft length or acceptance')
    if (len(unions) != n_layer or any(not math.isfinite(u) or
            not top_k <= u <= min(n_expert, d*top_k) for u in unions)):
        raise ValueError('invalid expert union')
    if any(not math.isfinite(x) or x < 0 for x in (draft_us, extra_us_per_token)):
        raise ValueError('invalid additional time')
    tensors = [t for t in stats['tensors'] if t['role'] == 'expert_ffn']
    if not math.isclose(sum(t['bytes'] for t in tensors), stats['expert_bytes'], rel_tol=1e-9):
        raise ValueError('expert tensor bytes do not match model')
    cpu_bytes = defaultdict(float)
    gpu_expert_bytes = 0.
    for tensor in tensors:
        layer = tensor['layer']
        # PL4 single-token bytes * union_l(d)/top_k, preserving type and layer.
        size = tensor['bytes'] * top_k/n_expert * unions[layer]/top_k
        if layer < k:
            cpu_bytes[tensor['ggml_type']] += size
        else:
            gpu_expert_bytes += size
    nonexpert_bytes = stats['nonexpert_bytes']
    if scale_nonexpert:
        nonexpert_bytes *= sum(unions)/(n_layer*top_k)
    gpu_bytes = gpu_expert_bytes + nonexpert_bytes
    gpu_us = gpu_bytes/(constants['bw_gpu_gbs']*1000)
    cpu_us, fallbacks = 0., []
    for kind, size in cpu_bytes.items():
        bw = constants['bw_cpu_gbs'].get(kind)
        if bw is None:
            bw = constants.get('screen_bw_cpu_gbs', {}).get(kind, constants['bw_cpu_gbs']['Q4_K'])
            fallbacks.append(kind)
        cpu_us += size/(bw*1000)
    handoff_us = k*constants['handoff_us']
    additional_us = draft_us + extra_us_per_token*d
    step_us = gpu_us + cpu_us + handoff_us + additional_us
    return dict(k=k, d=d, a=a, accepted_tokens=a*d, step_us=step_us,
                tps=1e6*a*d/step_us, cpu_us=cpu_us, gpu_us=gpu_us,
                handoff_us=handoff_us, additional_us=additional_us,
                cpu_bytes=dict(cpu_bytes), gpu_bytes=gpu_bytes,
                gpu_expert_bytes=gpu_expert_bytes, nonexpert_bytes=nonexpert_bytes,
                fallbacks=fallbacks)


def implied_length(target_speedup, times, base_us):
    """Bracket a deterministic/effective d at a=1 by interpolating step TIMES.

    This is a mixture of adjacent batch lengths; interpolate reward and latency,
    not throughput. It cannot identify a real length distribution from one rate.
    """
    lengths = sorted(times)
    for lo, hi in zip(lengths, lengths[1:]):
        s0, s1 = lo*base_us/times[lo], hi*base_us/times[hi]
        if min(s0, s1) <= target_speedup <= max(s0, s1) and s0 != s1:
            f = (target_speedup*times[lo] - base_us*lo) / (
                base_us*(hi-lo) - target_speedup*(times[hi]-times[lo]))
            return dict(mean_d=lo+f*(hi-lo), lower_d=lo, upper_d=hi,
                        upper_fraction=f)
    return None


def recommend(stats, constants, k, curves, a, *, engaged_fraction=1.,
              draft_us=0., extra_us_per_token=0., fits=True, min_speedup=1.10,
              feasible_lengths=None):
    """Choose a max from 2..16 using the worst supplied domain prediction.

    a is acceptance on engaged steps. engaged_fraction is the fraction of output
    tokens produced on engaged steps, so combined speedup is a harmonic mixture.
    a and draft_us may be scalars or dictionaries indexed by d. feasible_lengths
    restricts the externally checked batch sizes. Memory fit is an external
    prerequisite, not inferred from a bandwidth model.
    """
    if not 0 <= engaged_fraction <= 1:
        raise ValueError('engaged fraction outside [0,1]')
    if not fits or engaged_fraction == 0:
        return dict(enable=False, draft_max=0, reason='no fit or no engagement')
    base = predict(stats, constants, k, 1, 1., curves[0][1]['per_layer'])['step_us']
    candidates = []
    for d in sorted(set.intersection(*(set(c) for c in curves))):
        if d < 2 or (feasible_lengths is not None and d not in feasible_lengths):
            continue
        acceptance = a[d] if isinstance(a, dict) else a
        cost = draft_us[d] if isinstance(draft_us, dict) else draft_us
        speedups = [predict(stats, constants, k, d, acceptance, c[d]['per_layer'],
                            draft_us=cost, extra_us_per_token=extra_us_per_token)['tps']*base/1e6
                    for c in curves]
        speedup = min(speedups)
        combined = 0. if speedup == 0 else 1/(1-engaged_fraction+engaged_fraction/speedup)
        candidates.append((combined, d))
    if not candidates:
        return dict(enable=False, draft_max=0, reason='no supported draft length')
    best, d = max(candidates)
    return dict(enable=best >= min_speedup, draft_max=d if best >= min_speedup else 0,
                predicted_speedup=best, required_speedup=min_speedup,
                reason='worst-domain forecast; validate with end-to-end timings')
