"""Cache protocol planning, observations, and descriptive confound reporting."""
import json

import pytest

from cc.abrun import abrun, gate, regate, run_summary


def spec(tmp_path, **kwargs):
    return dict(id='cache', outdir_base=str(tmp_path), runner='om_run', tool='llama-bench',
                argv=['-m', '/om/model.gguf'], arms={'base': {'image': 'test'},
                'other': {'image': 'test'}}, order='ABBA', **kwargs)


@pytest.mark.parametrize('protocol', [None, 'carryover', 'cold', 'warm'])
def test_protocol_plan(tmp_path, monkeypatch, protocol):
    monkeypatch.setenv('ABRUN_RUNNER', '/fake/runner')
    s = spec(tmp_path)
    if protocol is not None:
        s['cache_protocol'] = protocol
    if protocol == 'warm':
        s['cache_warm_bytes'] = 4096
    abrun.validate(s)
    _, planned = abrun.plan(s)
    for row in planned:
        assert f'OM_CACHE_PROTOCOL={protocol or "carryover"}' in row['command']
        if protocol == 'warm':
            assert 'OM_CACHE_WARM_BYTES=4096' in row['command']


@pytest.mark.parametrize('extra', [dict(cache_protocol='hot'), dict(cache_protocol=[]),
                                  dict(cache_warm_bytes=-1), dict(cache_warm_bytes=True),
                                  dict(cache_warm_ranges=''),
                                  dict(cache_warm_bytes=1, cache_warm_ranges='r.json')])
def test_invalid_cache_spec(tmp_path, extra):
    with pytest.raises(ValueError):
        abrun.validate(spec(tmp_path, **extra))


@pytest.mark.parametrize('value,expected', [(23.5, 23.5), (0, 0), (None, None),
                                         (101, None), (True, None)])
def test_cache_summary(tmp_path, value, expected):
    (tmp_path / 'page_cache.start.json').write_text(json.dumps(dict(resident_pct=value)))
    (tmp_path / 'page_cache.end.json').write_text('{broken')
    summary = run_summary.summarize(tmp_path, write=False)
    assert summary['cache_start_resident_pct'] == expected
    assert summary['cache_end_resident_pct'] is None
    assert 'cache_end_resident_pct' in summary['reasons']


@pytest.mark.parametrize('gap,flag', [(10, False), (10.01, True), (0, False)])
def test_cache_confound_does_not_change_gate(gap, flag):
    arms = {'base': [dict(eval_tps=100, cache_start_resident_pct=p) for p in (10, 20)],
            'other': [dict(eval_tps=120, cache_start_resident_pct=p + gap) for p in (10, 20)]}
    plain = gate.evaluate({a: [dict(eval_tps=r['eval_tps']) for r in rs] for a, rs in arms.items()})
    verdict = gate.evaluate(arms)
    assert bool(verdict['cache_confound']) == flag
    assert {k: v for k, v in verdict.items() if k != 'cache_confound'} == {
        k: v for k, v in plain.items() if k != 'cache_confound'}


def test_report_and_regate_protocol(tmp_path):
    s = spec(tmp_path, cache_protocol='cold')
    (tmp_path / 'spec.json').write_text(json.dumps(s))
    for arm, pct in [('base', 0), ('other', 30)]:
        directory = tmp_path / f'001-{arm}'
        directory.mkdir()
        (directory / 'summary.json').write_text(json.dumps(dict(
            status='ok', wall_s=1, output_sha='same', eval_tps=100,
            eval_ms_per_token=10, cache_start_resident_pct=pct)))
    verdict = regate.recompute(tmp_path)
    assert verdict['cache_protocol'] == 'cold'
    assert verdict['cache_confound']
    abrun.write_report(tmp_path, s, [dict(index=1, arm='base', summary=dict(
        status='ok', wall_s=1, output_sha='same', cache_start_resident_pct=0,
        cache_end_resident_pct=30))], verdict)
    report = (tmp_path / 'REPORT.md').read_text()
    assert 'Page-cache protocol: cold' in report
    assert 'Cache start resident %' in report and 'Cache end resident %' in report
    assert 'cache_confound' in report
