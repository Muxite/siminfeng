"""Real-record regressions and corruption tests. No invented measurements or jobs."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import verdict

C1 = 'job-34c09bec98'
KV = 'job-a945fa8656'


def protocol(name):
    return verdict.read_json(HERE / 'criteria' / (name + '.json'))


def run(value, lab=None):
    return verdict.evaluate(value, lab or ROOT / 'lab', ROOT)


class RealRecords(unittest.TestCase):
    def test_recorded_verdicts(self):
        expected = {'D029-q8_reuse': 'PASS', 'D029-mixE4K': 'FAIL',
                    'D033-q8_0': 'PASS', 'D033-q4_0': 'FAIL',
                    'D033-q8_0-frozen': 'FAIL', 'D033-q4_0-frozen': 'FAIL'}
        for name, status in expected.items():
            with self.subTest(name=name):
                result = run(protocol(name))
                self.assertEqual(result['status'], status, result)
                self.assertFalse(any('error' in c for c in result['criteria']), result)
                self.assertEqual(result['job_ids'], [C1 if name.startswith('D029') else KV])

    def test_D029_values_and_full_hash(self):
        result = run(protocol('D029-q8_reuse'))
        speed = result['criteria'][0]
        self.assertAlmostEqual(speed['value'], 2.0677823745951507)
        self.assertAlmostEqual(speed['evidence'][0]['aggregate'], 807.629565)
        self.assertAlmostEqual(speed['evidence'][1]['aggregate'], 791.2678675)
        text = result['criteria'][1]['evidence'][0]['samples'][0]['sha256']
        self.assertEqual(len(text), 64)
        self.assertTrue(text.startswith('65894776'))
        backend = next(c for c in result['criteria'] if c['metric'] == 'backend_ops')['evidence'][0]
        self.assertEqual((backend['passed'], backend['total']), (2445, 2445))
        host = next(c for c in result['criteria'] if c['metric'] == 'host_quiet')['evidence'][0]
        self.assertEqual(host['counter_maxima'], {'cc': 0})
        self.assertEqual(host['sample_count'], 24)

    def test_D033_quality_and_pending_KLD(self):
        q8 = run(protocol('D033-q8_0'))['criteria'][0]
        q4 = run(protocol('D033-q4_0'))['criteria'][0]
        self.assertEqual([e['aggregate'] for e in q8['evidence']], [7.5392, 7.5385])
        self.assertAlmostEqual(q8['value'], 0.00928566690987509)
        self.assertAlmostEqual(q4['value'], 7.936592160244071)
        result = run(protocol('D033-q8_0-frozen'))
        self.assertEqual([c['status'] for c in result['criteria']], ['PASS', 'FAIL'])
        hashes = [e['samples'][0]['sha256'][:8] for e in result['criteria'][1]['evidence']]
        self.assertEqual(hashes, ['96310158', '4e8a7ed7'])

    def test_aggregation_matches_original_script(self):
        source = ROOT.parent / '.claude/skills/frozen-experiment/scripts/agg.py'
        self.assertTrue(source.is_file(), f'missing referenced aggregation script: {source}')
        spec = importlib.util.spec_from_file_location('original_agg', source)
        agg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg)
        records = verdict.Records(ROOT / 'lab', ROOT)
        for job in (C1, KV):
            result = verdict.read_json(ROOT / 'lab/jobs' / job / 'att-1/result.json')
            expected = agg.summarize(agg.group(result))
            for name, summary in expected.items():
                for metric, alias in agg.METRICS:
                    if alias in summary:
                        value, _ = records.metric(dict(job=job, attempt='att-1', test=name, count=summary['n']), metric)
                        self.assertEqual(value, summary[alias], (name, metric))

    def test_D035_D040_quality_speed_and_host(self):
        # Actual noisy D035 window versus quiet D040 rerun. No synthesized records.
        kl = dict(id='PAP quality', metric='kld', op='<=', threshold=0.08,
                  candidate=dict(job='job-3d2fbbd061', attempt='att-1', test='pap_kl_pap12_q3', count=1))
        for job, rounds, expected in [('job-3d2fbbd061', 3, ['PASS', 'PASS', 'FAIL']),
                                       ('job-d2e342e743', 5, ['PASS', 'FAIL', 'FAIL'])]:
            log = ROOT / ('artifacts/x2/pap1-hostload.log' if rounds == 3 else 'artifacts/x2/chain4-hostload.log')
            speed = dict(id='PAP speed', metric='decode_tps', mode='delta_pct', op='>=', threshold=5,
                         candidate=dict(job=job, attempt='att-1', test='pap_decode_pap12_q3_n12', count=rounds),
                         baseline=dict(job=job, attempt='att-1', test='pap_decode_mixE4K_n12', count=rounds))
            host = dict(id='Quiet', metric='host_quiet', op='==', threshold=True, job=job, attempt='att-1',
                        path=str(log), sha256=verdict.digest(log.read_bytes()), counters={'cc': 0, 'dl': 0}, max_gap_s=15)
            result = run(dict(schema='criteria/1', criteria=[kl, speed, host]))
            self.assertEqual([c['status'] for c in result['criteria']], expected, result)
            # Both monitors stop before the last test finishes. A quiet-host
            # claim cannot pass our full-interval coverage requirement.
            self.assertIn('does not cover full run interval', result['criteria'][2]['error'])
            self.assertFalse(any('error' in c for c in result['criteria'][:2]), result)

    def test_CLI_exit_and_json(self):
        for name, code in [('D029-q8_reuse', 0), ('D033-q4_0', 1)]:
            result = subprocess.run([sys.executable, '-B', str(HERE / 'verdict.py'),
                                     str(HERE / 'criteria' / (name + '.json')), '--lab', str(ROOT / 'lab')],
                                    cwd=ROOT / 'artifacts/codexVER', capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertEqual(json.loads(result.stdout)['schema'], 'verdict/1')


class FailClosed(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='corrupted-real-', dir=ROOT / 'artifacts/codexVER')
        self.addCleanup(self.temp.cleanup)
        self.lab = Path(self.temp.name)
        shutil.copytree(ROOT / 'lab/jobs' / C1, self.lab / 'jobs' / C1)
        self.path = self.lab / 'jobs' / C1 / 'att-1'
        self.value = protocol('D029-q8_reuse')
        self.value['criteria'] = [self.value['criteria'][0]]

    def mutate_result(self, func):
        path = self.path / 'result.json'
        result = verdict.read_json(path)
        func(result)
        path.write_text(json.dumps(result))

    def assert_failed(self, phrase=None):
        result = run(self.value, self.lab)
        self.assertEqual(result['status'], 'FAIL', result)
        if phrase:
            self.assertIn(phrase, result['criteria'][0]['error'])
        return result

    def test_missing_job(self):
        shutil.rmtree(self.lab / 'jobs' / C1)
        self.assert_failed('No such file')

    def test_incomplete_coverage(self):
        self.mutate_result(lambda r: r['evidence']['coverage'].update(tests_completed_cleanly=35))
        self.assert_failed('coverage')

    def test_interrupted(self):
        self.mutate_result(lambda r: r['execution'].update(status='interrupted'))
        self.assert_failed('incomplete')

    def test_missing_metric_on_one_sample(self):
        self.mutate_result(lambda r: r['quality']['tests']['c1_q8_tg_r0#0'].pop('samples'))
        self.assert_failed('numeric')

    def test_nonfinite_metric(self):
        self.mutate_result(lambda r: r['quality']['tests']['c1_q8_tg_r0#0']['samples'][0].update(avg_ts=float('nan')))
        self.assert_failed('non-finite')

    def test_missing_evidence(self):
        (self.path / 'c1_q8_tg_r0#0.stdout').unlink()
        self.assert_failed('No such file')

    def test_changed_evidence(self):
        with (self.path / 'c1_q8_tg_r0#0.stdout').open('ab') as stream:
            stream.write(b'corruption')
        self.assert_failed('hash mismatch')

    def test_bad_sample(self):
        self.mutate_result(lambda r: r['quality']['tests']['c1_q8_tg_r0#0'].update(ok=False))
        self.assert_failed('unclean')

    def test_missing_sample(self):
        self.mutate_result(lambda r: r['quality']['tests'].pop('c1_q8_tg_r0#0'))
        self.assert_failed('submitted order')

    def test_identity_mismatch(self):
        self.mutate_result(lambda r: r.update(job='job-wrong'))
        self.assert_failed('identity')

    def test_wrong_sample_count(self):
        self.value['criteria'][0]['candidate']['count'] = 3
        self.assert_failed('expected 3')

    def test_wrong_alternating_order(self):
        self.value['criteria'][0]['alternating']['group'].reverse()
        self.assert_failed('order mismatch')

    def test_unknown_metric_and_operator(self):
        for key, value in [('metric', 'typo'), ('op', 'typo'), ('mode', 'typo')]:
            with self.subTest(key=key):
                old = self.value['criteria'][0].get(key)
                self.value['criteria'][0][key] = value
                self.assert_failed('unknown')
                self.value['criteria'][0][key] = old

    def test_zero_baseline(self):
        def zero(r):
            for k, v in r['quality']['tests'].items():
                if k.startswith('c1_q8_tg_r0#'):
                    v['samples'][0]['avg_ts'] = 0
        self.mutate_result(zero)
        self.assert_failed('baseline')

    def test_missing_KLD_fails(self):
        self.value['criteria'][0].update(metric='kld', mode='value')
        self.value['criteria'][0].pop('alternating')
        self.assert_failed('numeric')

    def test_missing_host_counter(self):
        self.value['criteria'] = [copy.deepcopy(protocol('D029-q8_reuse')['criteria'][-1])]
        self.value['criteria'][0]['counters']['dl'] = 0
        self.assert_failed('missing counter')

    def test_loadmon_coverage_gap(self):
        self.value['criteria'] = [copy.deepcopy(protocol('D029-q8_reuse')['criteria'][-1])]
        self.value['criteria'][0]['max_gap_s'] = 1
        self.assert_failed('sampling gap')

    def test_missing_sidecar(self):
        self.value['criteria'] = [copy.deepcopy(protocol('D029-q8_reuse')['criteria'][-2])]
        self.value['criteria'][0]['path'] = str(self.lab / 'missing-backend.log')
        self.assert_failed('No such file')

    def test_wrong_backend(self):
        self.value['criteria'] = [copy.deepcopy(protocol('D029-q8_reuse')['criteria'][-2])]
        self.value['criteria'][0]['backend'] = 'CUDA99'
        self.assert_failed('absent')

    def test_empty_and_duplicate_criteria(self):
        for value in [dict(schema='criteria/1', criteria=[]),
                      dict(schema='criteria/1', criteria=self.value['criteria'] * 2),
                      dict(schema='wrong', criteria=self.value['criteria'])]:
            self.assertEqual(run(value, self.lab)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
