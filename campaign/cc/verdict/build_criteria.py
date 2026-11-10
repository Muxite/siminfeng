"""Materialize the reviewed protocol transcriptions; never copies or invents job data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'criteria'
C1 = 'job-34c09bec98'
KV = 'job-a945fa8656'


def ref(job, test, count):
    return dict(job=job, attempt='att-1', test=test, count=count)


def numeric(id, metric, candidate, baseline=None, mode='value', op='<=', threshold=0, **extra):
    return dict(id=id, metric=metric, candidate=candidate, **({'baseline': baseline} if baseline else {}),
                mode=mode, op=op, threshold=threshold, **extra)


def text(id, groups):
    return dict(id=id, metric='text_sha', groups=groups, op='==', threshold=True)


def sidecar(path):
    return dict(path=path, sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())


def write(name, scope, criteria, sources):
    (OUT / (name + '.json')).write_text(json.dumps(dict(schema='criteria/1', id=name, scope=scope,
                                                      sources=sources, criteria=criteria), indent=2) + '\n')


def main():
    acceptance = [numeric('Q8_0 decode gain', 'tg_ts', ref(C1, 'c1_q8_tg_r1', 4), ref(C1, 'c1_q8_tg_r0', 4),
                          'delta_pct', '>=', 2,
                          alternating=dict(group=['c1_q8_tg_r0', 'c1_q8_tg_r1'], rounds=2, repeat=2))]
    for model in ('q8', 'mix'):
        acceptance += [text(f'{model} OFF/ON greedy identity', [ref(C1, f'c1_{model}_gen_r{r}', 4) for r in (0, 1)]),
                       numeric(f'{model} printed PPL identity', 'ppl', ref(C1, f'c1_{model}_ppl_r1', 1),
                               ref(C1, f'c1_{model}_ppl_r0', 1), 'abs_delta')]
        label = 'Q8_0' if model == 'q8' else 'mixE4K'
        secondary = text(f'{model} secondary base/OFF/ON identity', [ref(C1, f'c1_{model}_gen_r0', 4)])
        secondary['files'] = [sidecar(f'artifacts/x4-c1/gen-{variant}-{label}.out')
                              for variant in ('base', 'off', 'on')]
        acceptance.append(secondary)
    acceptance += [dict(id='CUDA MUL_MAT correctness', metric='backend_ops', job=C1, attempt='att-1',
                        backend='CUDA0', operation='MUL_MAT', op='==', threshold=True,
                        **sidecar('artifacts/x4-c1/tbo-mulmat.out')),
                   dict(id='No compiler during timing', metric='host_quiet', job=C1, attempt='att-1',
                        counters={'cc': 0}, max_gap_s=15, op='==', threshold=True,
                        **sidecar('artifacts/x2/timed-hostload.log'))]
    write('D029-q8_reuse', 'D026 primary Q8_0 acceptance, with correctness for both model groups. '
          'Host quiet means the recorded cc counter is zero; this log does not measure downloads or CPU inference. '
          'The backend log is legacy external evidence, not a test inside the job.', acceptance, ['D026', 'D029'])
    # D029 explicitly reports mixE4K as a separate, sub-threshold track.
    mix = json.loads(json.dumps(acceptance))
    mix[0] = numeric('mixE4K decode gain', 'tg_ts', ref(C1, 'c1_mix_tg_r1', 4), ref(C1, 'c1_mix_tg_r0', 4),
                     'delta_pct', '>=', 2, alternating=dict(group=['c1_mix_tg_r0', 'c1_mix_tg_r1'], rounds=2, repeat=2))
    write('D029-mixE4K', 'D026 separate mixE4K track (recorded sub-threshold).', mix, ['D026', 'D029'])
    for quant in ('q8_0', 'q4_0'):
        quality = [numeric('PPL within one percent', 'ppl', ref(KV, f'kv_ppl4k_{quant}', 1),
                           ref(KV, 'kv_ppl4k_f16', 1), 'abs_delta_pct', '<=', 1)]
        write('D033-' + quant, 'D033 reported same-accuracy PPL gate only (±1%). '
              'Does not establish all D028 requirements: nonidentical greedy text leaves KLD pending.',
              quality, ['D028', 'D033'])
        write('D033-' + quant + '-frozen', 'D028 full same-accuracy gate: PPL within 1% AND greedy text identical. '
              'D028 leaves KLD pending when text differs and freezes no KLD threshold. '
              'This file fails closed on nonidentity; it does not invent a KLD waiver or threshold.',
              quality + [text('Greedy identity or unresolved KLD', [ref(KV, 'kv_gen32k_f16', 3),
                                                                  ref(KV, f'kv_gen32k_{quant}', 3)])], ['D028', 'D033'])


if __name__ == '__main__':
    main()
