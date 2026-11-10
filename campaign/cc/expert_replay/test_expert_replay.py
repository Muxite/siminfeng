import contextlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from expert_replay import main, make_profile, partition, read_trace, replay


def rows(ids, phase='decode'):
    return [dict(seq=0, tok=i, layer=0, experts=[e], phase=phase) for i, e in enumerate(ids)]


class ReplayTests(unittest.TestCase):
    def test_hand_counts(self):
        trace = rows([0, 1, 0, 2, 0, 1])
        profile = make_profile(rows([0, 0, 1]), 1, 3, 'raw')
        for policy, expected in [('lru', 2), ('lfu', 2), ('lfu-decay:2', 2),
                                 ('belady', 3), ('static-profile', 3)]:
            with self.subTest(policy=policy):
                self.assertEqual(replay(trace, 1, 3, 2, policy, profile=profile)['overall']['hits'], expected)

    def test_decay_changes_victim(self):
        trace = rows([0]*8 + [1, 2, 1, 2, 1, 2])
        self.assertEqual(replay(trace, 1, 3, 2, 'lfu')['overall']['hits'], 7)
        self.assertEqual(replay(trace, 1, 3, 2, 'lfu-decay:1')['overall']['hits'], 11)

    def test_oracle_dominates_exhaustive(self):
        for seq in itertools.product(range(3), repeat=6):
            trace = rows(seq)
            profile = make_profile(trace, 1, 3, 'raw')
            for cap in (0, 1, 2, 3):
                best = replay(trace, 1, 3, cap, 'belady')['overall']['hits']
                for policy in ('lru', 'lfu', 'lfu-decay:1', 'static-profile'):
                    self.assertGreaterEqual(best, replay(trace, 1, 3, cap, policy, profile=profile)['overall']['hits'])

    def test_partition_and_warmup(self):
        trace = rows([0, 0, 1, 1])
        train, test = partition(trace, '1:1')
        self.assertEqual([r['tok'] for r in train], [0, 1])
        self.assertEqual(replay(test, 1, 2, 1)['overall']['hits'], 1)
        result = replay(test, 1, 2, 1, warmup=1)
        self.assertEqual(result['overall']['requests'], 1)
        self.assertEqual(result['overall']['hits'], 1)
        multilayer = [dict(r, layer=l) for r in trace for l in (0, 1)]
        a, b = partition(multilayer, '1:1')
        self.assertEqual(len(a), 4)
        self.assertFalse({r['tok'] for r in a} & {r['tok'] for r in b})

    def test_bytes_prediction_and_layers(self):
        trace = rows([0, 0], 'prefill') + [dict(r, tok=r['tok']+2, layer=1) for r in rows([0, 0])]
        result = replay(trace, 2, 2, 1, per_layer=True, layer_bytes=[100, 200], bw_gbs=2, compute_ms=3)
        self.assertEqual(result['overall']['hits'], 2)
        self.assertEqual(result['decode_only']['bytes_moved_per_token'], 100)
        self.assertAlmostEqual(result['prediction']['decode_seconds_per_token'], .003 + 100/2e9)
        self.assertEqual(result['per_layer']['0']['hit_rate'], .5)

    def test_scope_and_rank_order(self):
        trace = [dict(seq=0, tok=t, layer=l, experts=[0], phase='decode')
                 for t in range(2) for l in range(2)]
        self.assertEqual(replay(trace, 2, 1, 1)['overall']['hits'], 0)
        self.assertEqual(replay(trace, 2, 1, 1, per_layer=True)['overall']['hits'], 2)
        trace = [dict(seq=0, tok=t, layer=0, experts=[0, 1], phase='decode') for t in range(2)]
        self.assertEqual(replay(trace, 1, 2, 1)['overall']['hits'], 0)

    def test_weighting(self):
        trace = rows([0]*10) + [dict(r, seq=1) for r in rows([1, 1])]
        raw = make_profile(trace, 1, 2, 'raw')
        equal = make_profile(trace, 1, 2, 'per-request')
        self.assertEqual(raw['scores'], [10, 2])
        self.assertAlmostEqual(equal['scores'][0], equal['scores'][1])

    def test_parser_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)/'trace.jsonl'
            profile = Path(directory)/'profile.json'
            trace.write_text(json.dumps(dict(type='metadata', layers=1, experts=3))+'\n' +
                '\n'.join(json.dumps(dict(tok=r['tok'], layer=0, experts=r['experts'], phase='decode')) for r in rows([0, 0, 1, 1]))+'\n')
            parsed, l, e, diag = read_trace(trace)
            self.assertEqual((l, e, diag['malformed_lines']), (1, 3, 0))
            main(['profile', '--trace', str(trace), '--split', '1:1', '--out', str(profile)])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(['simulate', '--trace', str(trace), '--split', '1:1', '--capacity-experts', '1', '--policy', f'static-profile:{profile}'])
            self.assertEqual(json.loads(output.getvalue())['overall']['hits'], 0)
            main(['profile', '--trace', str(trace), '--out', str(profile)])
            with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                main(['simulate', '--trace', str(trace), '--split', '1:1', '--capacity-experts', '1', '--policy', f'static-profile:{profile}'])
            self.assertIn('leakage', error.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(['sweep', '--trace', str(trace), '--capacities', '0,1,3', '--policy', 'belady'])
            self.assertEqual(len(output.getvalue().splitlines()), 4)
            self.assertIn('reads-the-future upper bound', output.getvalue())
            with trace.open('a') as out:
                out.write('{}\n{"tok":')
            _, _, _, diag = read_trace(trace)
            self.assertEqual(diag['malformed_lines'], 2)
            self.assertEqual(diag['possible_truncated_tail'], 1)
            self.assertEqual(diag['valid_records'], 4)

    def test_primary_format_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace'
            path.write_text(json.dumps(dict(seq=0, pos=0, layer=0, experts=[0, 1], weights=[.5, .5], phase='prefill'))+'\n')
            self.assertEqual(read_trace(path, 2, 2)[0][0]['tok'], 0)
            with self.assertRaises(ValueError):
                read_trace(path, 2, 1)
            with self.assertRaises(ValueError):
                read_trace(path)

    def test_invalid_configuration_and_empty_decode(self):
        self.assertIsNone(replay(rows([0], 'prefill'), 1, 1, 0)['prediction']['decode_seconds_per_token'])
        for kw in ({'capacity': -1}, {'capacity': 1, 'policy': 'lfu-decay:0'},
                   {'capacity': 1, 'bw_gbs': 0}, {'capacity': 1, 'layer_bytes': []}):
            with self.assertRaises(ValueError):
                replay(rows([0]), 1, 1, **kw)


if __name__ == '__main__':
    unittest.main()
