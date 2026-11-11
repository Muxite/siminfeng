"""Effective-value assertions must work for disabled baseline configurations too."""
import copy
import json
from pathlib import Path
import re

from cc.abrun import abrun


def test_j_om11_effective_config_all_match():
    spec = json.loads((Path(__file__).parent / 'specs/J-OM11.json').read_text())
    abrun.validate(spec)
    assert spec['metric'] == 'eval_ms_per_token' and spec['direction'] == 'lower'
    assert spec['arms']['base']['env'] == {}
    assert spec['explicit_order'] == ['base', 'pg8', 'pg10', 'pg8la2', 'pg12la2',
                                      'pg12la2', 'pg8la2', 'pg10', 'pg8', 'base']
    for name, arm in spec['arms'].items():
        env = arm['env']
        k = env.get('LLAMA_EXPERT_PREGATE_K', '8')
        la = env.get('LLAMA_EXPERT_PREGATE_LOOKAHEAD', '1')
        line = ('expert_config: effective ' +
                ('fill=0 random=0 max_tokens=0 threads=8 chunk_kib=1024 pregate=0 '
                 if name == 'base' else
                 'fill=1 random=decode max_tokens=1 threads=16 chunk_kib=2048 pregate=1 ') +
                f'k={k} order=after max_inflight=4 lookahead={la} min_tokens=0 diag=0 '
                'populate=0 diag_csv=(unset) prefetch=0 prefetch_next=0')
        stderr = line
        if name != 'base':
            stderr += (f'\nexpert_fill: active\nexpert_fill: random advice applied at decode\n'
                       f'expert_pregate: active k={k} arch=qwen35moe\nexpert_pregate: stats')
        def summary(text):
            matches = {p: int(bool(re.search(p, text, re.MULTILINE)))
                       for p in abrun.switch_patterns(spec)}
            return dict(switch_matches=matches, switch_lines=[text])
        assert abrun.switch_errors(spec, name, summary(stderr)) == []
        assert abrun.switch_errors(spec, name, summary(stderr.replace(f'k={k} ', 'k=7 ')))
        assert abrun.switch_errors(spec, name, summary(stderr.replace(f'lookahead={la} ', 'lookahead=9 ')))
        assert abrun.switch_errors(spec, name, summary(stderr.replace('max_inflight=4 ', 'max_inflight=8 ')))
        for pattern in arm['switch_regex']:
            observed = summary(stderr)
            observed['switch_matches'][pattern] = 0
            assert abrun.switch_errors(spec, name, observed)  # ALL, never OR
    base = spec['arms']['base']
    observed = dict(switch_matches={p: 1 for p in abrun.switch_patterns(spec)}, switch_lines=['active'])
    assert any('leaked' in e for e in abrun.switch_errors(spec, 'base', observed))
    invalid = copy.deepcopy(spec)
    invalid['arms']['base']['switch_regex_is_config'] = 'true'
    try:
        abrun.validate(invalid)
        assert False, 'nonboolean config marker accepted'
    except ValueError:
        pass
    assert 'k=8 ' in base['switch_regex'][0]
