import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from mahjong_sandbox.engine import Engine, IllegalAction
from mahjong_sandbox.fast import FastEngine, FastRiichiEngine, RuntimeOptions, TileList
from mahjong_sandbox.model import Phase, RuleConfig
from mahjong_sandbox.riichi_model import RiichiConfig
from mahjong_sandbox.riichi_engine import RiichiEngine
from mahjong_sandbox.bots import make_bot
from mahjong_sandbox.replay import export_replay, replay
from mahjong_sandbox.operators import native_available, numpy, shanten_batch, waits_batch, discard_ukeire_batch


class FastTests(unittest.TestCase):
    def test_counter_mutations(self):
        from mahjong_sandbox.tiles import counts_of
        a = TileList([0, 1, 8])
        operations = [lambda: a.append(12), lambda: a.remove(1), lambda: a.extend([16, 20]),
                      lambda: a.pop(), lambda: a.insert(1, 24), lambda: a.__setitem__(slice(0, 2), iter([28, 29])),
                      lambda: a.__delitem__(0), lambda: a.reverse(), lambda: a.sort(), lambda: a.clear()]
        for operation in operations:
            operation()
            self.assertEqual(tuple(a.counts), counts_of(a))

    def differential(self, backend):
        configurations = [RuleConfig(), RuleConfig(exchange_three=False, seven_pairs_quads=False,
                           kong_discard_transfer=True, force_win_last_four=True), RiichiConfig(),
                           RiichiConfig(red_fives=(), open_tanyao=False, multi_ron=False)]
        for cfg in configurations:
            riichi = isinstance(cfg, RiichiConfig)
            variant = 'riichi' if riichi else 'sichuan'
            for seed in range(2):
                ref = (RiichiEngine if riichi else Engine)(cfg, seed=seed)
                fast = (FastRiichiEngine if riichi else FastEngine)(cfg, seed=seed,
                           runtime=RuntimeOptions(record=True, validate_every=1, operator_backend=backend))
                original = [make_bot('greedy', variant=variant, seed=seed * 4 + p, reference=True) for p in range(4)]
                accelerated = [make_bot('greedy', variant=variant, seed=seed * 4 + p, backend=backend) for p in range(4)]
                while ref.state.phase != Phase.TERMINAL:
                    p = ref.next_player()
                    self.assertEqual(ref.legal_actions(p), fast.legal_actions(p))
                    a, b = original[p].act(ref.observe(p)), accelerated[p].act(fast.observe(p))
                    self.assertEqual(a, b, (variant, seed, ref.state.revision))
                    ref.submit(p, a, ref.state.decision_id)
                    fast.submit(p, b, fast.state.decision_id)
                    self.assertEqual(ref.state_hash(), fast.state_hash())
                self.assertEqual(ref.event_hash(), fast.event_hash())
                self.assertEqual(replay(export_replay(fast)).state_hash(), ref.state_hash())

    def test_python_differential(self):
        self.differential('python')

    @unittest.skipUnless(native_available(), 'optional numba extra not installed')
    def test_native_differential(self):
        self.differential('numba')

    def test_compact_isolation_and_trace_controls(self):
        e = FastRiichiEngine(runtime=RuntimeOptions(operator_backend='python'))
        o = e.observe(0)
        self.assertNotIn('concealed', o['players'][1])
        o['players'][0]['concealed'].clear()
        o['events'][0]['data']['rules']['dealer'] = 2
        self.assertEqual(len(e.observe(0)['players'][0]['concealed']), 14)
        self.assertEqual(e.config.dealer, 0)
        self.assertFalse(e.events)
        with self.assertRaises(ValueError):
            export_replay(e)
        before = e.state_hash()
        with self.assertRaises(IllegalAction):
            e.submit(0, e.legal_actions(0)[0], -1)
        self.assertEqual(before, e.state_hash())

    @unittest.skipUnless(native_available(), 'optional numba extra not installed')
    def test_operators_match_reference_and_are_reentrant(self):
        np, rng = numpy(), random.Random(478)
        for variant in ('riichi', 'sichuan'):
            rows, melds = [], []
            for _ in range(1000):
                m = rng.randrange(5)
                c = [0] * 34
                for t in rng.sample(range(136 if variant == 'riichi' else 108), 13 - m * 3 + rng.randrange(2)):
                    c[t // 4] += 1
                rows.append(c)
                melds.append(m)
            counts, m = np.array(rows), np.array(melds)
            saved = counts.copy()
            for fn in (shanten_batch, waits_batch):
                expected = fn(counts, m, variant=variant, backend='python')
                with ThreadPoolExecutor(2) as pool:
                    tasks = [pool.submit(fn, counts, m, variant=variant, backend='numba') for _ in range(2)]
                    for task in tasks:
                        np.testing.assert_array_equal(task.result(), expected)
                np.testing.assert_array_equal(counts, saved)
            for quads in (False, True):
                args = dict(variant=variant, seven_pairs_quads=quads)
                expected = discard_ukeire_batch(counts[:20], counts[:20], m[:20], backend='python', **args)
                actual = discard_ukeire_batch(counts[:20], counts[:20], m[:20], backend='numba', **args)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
                allowed = counts[:20] > 0
                allowed[:, ::2] = False
                partial = discard_ukeire_batch(counts[:20], counts[:20], m[:20], backend='numba',
                                              best_only=True, candidate_mask=allowed, **args)
                for row in range(20):
                    best = min((expected[0][row, k] for k in range(34) if allowed[row, k]), default=127)
                    for k in range(34):
                        self.assertEqual(partial[1][row,k], expected[1][row,k] if allowed[row,k] and expected[0][row,k]==best else 0)

    def test_operator_validation(self):
        try:
            np = numpy()
        except ImportError:
            self.skipTest('numpy extra not installed')
        for counts in (np.zeros((2, 33), dtype=int), np.full((2, 34), 5), np.zeros((2, 34))):
            with self.assertRaises(ValueError):
                shanten_batch(counts, backend='python')

    @unittest.skipUnless(native_available(), 'optional numba extra not installed')
    def test_native_special_and_exposed_waits(self):
        from mahjong_sandbox.tiles import parse_hand
        from mahjong_sandbox.operators import structural_waits, shanten
        for text, melds, variant in [('19m19p19s1234567z', 0, 'riichi'),
                                     ('1112223334445z', 0, 'riichi'), ('7z', 4, 'riichi'),
                                     ('1122334455667m', 0, 'riichi'),
                                     ('1111223344556m', 0, 'sichuan'),
                                     ('9999m112233445p', 0, 'sichuan')]:
            c = list(parse_hand(text))
            for quads in (False, True):
                for fn in (structural_waits, shanten):
                    self.assertEqual(fn(c, melds, variant=variant, seven_pairs_quads=quads, backend='python'),
                                     fn(c, melds, variant=variant, seven_pairs_quads=quads, backend='numba'))
        self.assertEqual(len(structural_waits(parse_hand('19m19p19s1234567z'), backend='numba')), 13)


if __name__ == '__main__':
    unittest.main()
