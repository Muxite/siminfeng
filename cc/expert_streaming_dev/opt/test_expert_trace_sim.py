#!/usr/bin/env python3
import itertools
import unittest
from pathlib import Path

from expert_trace_sim import Record, analyze, cache_report, predictor_report, read_trace, reuse_report, union_ranked


class TraceTests(unittest.TestCase):
    def test_read_prompt_and_append(self):
        # Use an artifact-local file so the lab stays inside its workspace.
        path = Path(__file__).with_name('synthetic-trace.csv')
        try:
            path.write_text('run,token_idx,layer,phase,selected,predicted\n'
                            'a,0,1,prompt,1 2,\na,0,0,prompt,0 3,\n'
                            'a,1,0,decode,0 1,0 2\nb,0,0,decode,2 3,\n')
            rows = read_trace(path)
            self.assertEqual([(r.run, r.token, r.layer) for r in rows],
                             [('a', 0, 0), ('a', 0, 1), ('a', 1, 0), ('b', 0, 0)])
            result = analyze(rows, 2, [0], 64)
            self.assertEqual(result['decode_tokens'], 2)
            self.assertEqual(result['caches']['0']['lru']['misses'], 4)
            path.write_text('token_idx,layer,phase,selected,predicted\n0,0,bogus,1,\n')
            with self.assertRaises(ValueError):
                read_trace(path)
        finally:
            path.unlink(missing_ok=True)

    def test_predictors_do_not_see_future(self):
        rows = [Record('a', 0, 0, 'decode', (1, 2), (1, 3)),
                Record('a', 1, 0, 'decode', (1, 3), (1, 3))]
        result = predictor_report(rows, 2)['overall']
        self.assertEqual(result['router'], 0.75)
        self.assertEqual(result['previous_token'], 0.25)
        self.assertEqual(result['frequency'], 0.25)
        self.assertEqual(union_ranked(((1, 2), (3, 4)), 2), [1, 3])

    def test_cache_known_sequence_and_identity(self):
        rows = [Record('a', i, 0, 'decode', (e,)) for i, e in enumerate((1, 2, 3, 1, 2, 3))]
        self.assertEqual(cache_report(rows, 2, 100, 'lru')['misses'], 6)
        result = cache_report(rows, 2, 100, 'belady')
        self.assertEqual(result['misses'], 4)
        self.assertAlmostEqual(result['bytes_per_decode_token'], 400 / 6)
        for policy in ('lru', 'lfu', 'lru_frequency', 'belady'):
            self.assertEqual(cache_report(rows, 3, 100, policy)['misses'], 3)
        rows = [Record('a', 0, 0, 'prompt', (1,)), Record('a', 1, 1, 'decode', (1,)),
                Record('b', 0, 0, 'decode', (1,))]
        self.assertEqual(cache_report(rows, 4, 100, 'lru')['misses'], 2)

    def test_belady_matches_exhaustive_optimum(self):
        def optimum(sequence, resident, capacity):
            if not sequence:
                return 0
            expert, *tail = sequence
            if expert in resident:
                return optimum(tail, resident, capacity)
            if len(resident) < capacity:
                return 1 + optimum(tail, resident | {expert}, capacity)
            return 1 + min(optimum(tail, resident - {victim} | {expert}, capacity) for victim in resident)
        for sequence in itertools.product(range(3), repeat=6):
            rows = [Record('a', i, 0, 'decode', (e,)) for i, e in enumerate(sequence)]
            optimal = cache_report(rows, 2, 1, 'belady')['misses']
            self.assertEqual(optimal, optimum(list(sequence), set(), 2))
            for policy in ('lru', 'lfu', 'lru_frequency'):
                self.assertGreaterEqual(cache_report(rows, 2, 1, policy)['misses'], optimal)

    def test_reuse_distance(self):
        rows = [Record('a', i, 0, 'decode', (e,)) for i, e in enumerate((1, 2, 1, 1))]
        result = reuse_report(rows)
        self.assertEqual(result['token_gap'], {'cold': 2, '2-3': 1, '1': 1})
        self.assertEqual(result['distinct_experts_between_uses'], {'cold': 2, '1': 1, '0': 1})


if __name__ == '__main__':
    unittest.main()
