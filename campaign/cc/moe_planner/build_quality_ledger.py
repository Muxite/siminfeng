#!/usr/bin/env python3
"""Rebuild the quality ledger from clean lab results and frozen test definitions."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = [('aa57700ee5', 'D013'), ('56115e2bf1', 'D034'),
           ('3d2fbbd061', 'D035'), ('a945fa8656', 'D033'), ('34c09bec98', 'D029')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def build(root=ROOT):
    entries, proofs, sources = [], [], []
    for job, decision in SOURCES:
        directory = root / f'lab/jobs/job-{job}'
        path = directory / 'att-1/result.json'
        result = read(path)
        assert result['execution']['status'] == 'completed' and result['quality']['all_ok']
        spec = read(directory / 'job.json')['spec']
        # D013 predates frozen definitions in the job envelope.
        defs = spec.get('test_definitions') or read(root / 'lab/tests/mem2.json')
        source = dict(job_id=result['job'], decision=decision, attempt=result['attempt'],
                      result_path=str(path.relative_to(root)), result_sha256=sha(path),
                      job_sha256=sha(directory / 'job.json'))
        if 'test_definitions' not in spec:
            source.update(definitions_path='lab/tests/mem2.json',
                          definitions_sha256=sha(root / 'lab/tests/mem2.json'))
        sources.append(source)
        for name, test in result['quality']['tests'].items():
            if not any(k in test for k in ('kld', 'ppl')):
                continue
            assert test['ok'] and not test.get('partial') and test['exit_code'] == 0
            td = defs[name.split('#')[0]]
            args = td['args']
            def arg(flag, default):
                return args[args.index(flag)+1] if flag in args else default
            filename = td['model_file']
            rep = filename.removesuffix('.gguf').removeprefix('granite-3.1-1b-a400m-instruct-')
            components = []
            if td.get('env', {}).get('GGML_CUDA_Q8_REUSE') == '1':
                components.append('q8_reuse')
            if decision == 'D033':
                components.append('kv_quant_fix')
            config = dict(representation=rep, placement=dict(family='static', params={'n_cpu_moe': int(arg('--n-cpu-moe', 0))}),
                          kv_type=arg('-ctk', 'f16'), fa=arg('-fa', 'off') == 'on', components=components,
                          ctx=int(arg('-c', 512)), n_chunks=int(arg('--chunks', 0)), source=dict(source, test=name))
            for field, metric in [('kld', 'kld'), ('top1_pct', 'top1'), ('ppl', 'ppl')]:
                if field in test:
                    entries.append(dict(config, metric=metric, value=test[field]))
        if decision == 'D029':
            for rep, prefix in [('Q8_0', 'c1_q8_'), ('mixE4K', 'c1_mix_')]:
                rows = [(n, t) for n, t in result['quality']['tests'].items() if n.startswith(prefix)]
                for kind in ('_gen_', '_ppl_'):
                    states = {defs[n.split('#')[0]].get('env', {}).get('GGML_CUDA_Q8_REUSE')
                              for n, t in rows if kind in n}
                    assert states == {'0', '1'}, 'proof requires both OFF and ON'
                texts = {result['evidence']['files'][n+'.stdout'] for n, t in rows if '_gen_' in n}
                ppls = {t['ppl'] for n, t in rows if 'ppl' in t}
                assert len(texts) == len(ppls) == 1
                proofs.append(dict(component='q8_reuse', representation=rep, text_identity=True,
                                   ppl_identity=True, text_sha256=next(iter(texts)),
                                   tests=[n for n, t in rows if '_gen_' in n or 'ppl' in t], source=source))
    # D036 is a manual binary-swap run, not a lab result.json job. It corroborates
    # D029 text identity, but does not supply the missing paired swap PPL.
    swap = root / 'artifacts/x7-comp/swap-gen.out'
    binary_evidence = dict(decision='D036', job_id=None, path=str(swap.relative_to(root)),
                           sha256=sha(swap), paired_ppl=False)
    assert binary_evidence['sha256'] == next(p['text_sha256'] for p in proofs if p['representation'] == 'Q8_0')
    return dict(schema='quality-ledger/1', sources=sources, entries=entries,
                computation_preserving=proofs, binary_evidence=binary_evidence)


if __name__ == '__main__':
    ledger = build()
    (HERE / 'quality_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(f"Wrote {len(ledger['entries'])} metric entries")
