from pathlib import Path
import unittest
from mahjong_sandbox.batch import BatchConfig, SyncBatchEnv, META_FIELDS
from mahjong_sandbox.operators import numpy
try:
    numpy()
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, 'optional numpy extra not installed')
class BatchTests(unittest.TestCase):
    def choose(self, b):
        np = numpy()
        return np.where(b['terminated'], -1, b['action_mask'].argmax(axis=1))

    def compare(self, a, b):
        for k in a:
            numpy().testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_reject_whole_batch_and_stale_tokens(self):
        with SyncBatchEnv(4, BatchConfig(variant='sichuan', backend='python')) as e:
            b = e.observation()
            hashes = [x.state_hash() for x in e.engines]
            actions = self.choose(b)
            actions[-1] = 511
            with self.assertRaises(ValueError):
                e.step(actions, b['tokens'])
            self.assertEqual(hashes, [x.state_hash() for x in e.engines])
            updated = e.step(self.choose(b), b['tokens'])
            with self.assertRaises(ValueError):
                e.step(self.choose(updated), b['tokens'])
            self.assertFalse(b['scores'].flags.writeable)
            numpy().testing.assert_array_equal(b['tokens'][:, 1], 0)

    def test_visible_projection(self):
        np = numpy()
        for variant in ('sichuan', 'riichi'):
            with SyncBatchEnv(4, BatchConfig(variant=variant, backend='python')) as e:
                b = e.observation()
                self.assertTrue(np.all(b['revealed'][:, 1:] == -1))
                for i, engine in enumerate(e.engines):
                    np.testing.assert_array_equal(b['own_counts'][i], engine.state.players[0].concealed.counts)
                    self.assertLessEqual(int(b['known_counts'][i].sum()), 15)
                forbidden = {'wall', 'head', 'tail', 'dead_wall', 'replies'}
                self.assertFalse(set(b) & forbidden)

    def test_announced_robbed_tile_and_ura_are_counted_once(self):
        from mahjong_sandbox.bots import known_counts
        obs = {'players': [{'concealed': [45], 'river': [], 'melds': []}],
               'dora_indicators': [], 'ura_indicators': [46], 'pending_tile': 44,
               'wins': [{'tile': 44}, {'tile': 44}]}
        counts = known_counts(obs, announcements=True)
        self.assertEqual(counts[11], 3)
        self.assertEqual(sum(counts), 3)

    def test_episode_reward_and_reset_contract(self):
        np = numpy()
        for variant in ('sichuan', 'riichi'):
            cfg = BatchConfig(variant=variant, opponent_policy='win-first', backend='python', seed=9)
            with SyncBatchEnv(8, cfg) as e, SyncBatchEnv(8, cfg) as mirror:
                b = e.observation()
                initial = b['scores'].copy()
                rewards = np.zeros((8, 4))
                for _ in range(400):
                    a = self.choose(b)
                    b = e.step(a, b['tokens'])
                    rewards += b['rewards']
                    if b['terminated'].all():
                        break
                self.assertTrue(b['terminated'].all())
                self.assertFalse(b['truncated'].any())
                np.testing.assert_array_equal(rewards, b['scores'] - initial)
                closed = e.step(np.full(8, -1), b['tokens'])
                self.assertFalse(closed['rewards'].any())
                b = e.reset_done()
                self.assertTrue(np.all(b['tokens'][:, 0] == 1))
                self.compare(b, mirror.reset())

    def test_self_play_sealed_decisions_have_unique_tokens(self):
        with SyncBatchEnv(1, BatchConfig(variant='sichuan', controlled_player=None, backend='python')) as e:
            a = e.observation()
            b = e.step(self.choose(a), a['tokens'])
            self.assertEqual(int(a['decision_ids'][0]), int(b['decision_ids'][0]))
            self.assertNotEqual(int(a['meta'][0, 1]), int(b['meta'][0, 1]))
            with self.assertRaises(ValueError):
                e.step(self.choose(b), a['tokens'])
            self.assertEqual(int(b['exchange_offsets'][0]), -1)
            for _ in range(3):
                b = e.step(self.choose(b), b['tokens'])
            self.assertIn(int(b['exchange_offsets'][0]), (1, 2, 3))

    def test_process_shards_equal_sync_and_cleanup(self):
        from mahjong_sandbox.parallel import ProcessBatchEnv
        cfg = BatchConfig(variant='sichuan', controlled_player=None, backend='python', seed=33)
        with SyncBatchEnv(8, cfg) as sync:
            for workers in (1, 2, 4):
                with ProcessBatchEnv(8, workers, cfg) as process:
                    path = process.path
                    b = process.observation()
                    # Fresh sync with identical lane/episode streams.
                    with SyncBatchEnv(8, cfg) as mirror:
                        self.compare(b, mirror.observation())
                        for _ in range(24):
                            actions = self.choose(b)
                            b = process.step(actions, b['tokens'])
                            self.compare(b, mirror.step(actions, mirror.buffers['tokens'].copy()))
                        self.compare(process.reset([1, 5]), mirror.reset([1, 5]))
                self.assertFalse(Path(path).exists())
                self.assertTrue(all(not p.is_alive() for p in process.processes))

    def test_pending_timeout_and_worker_failure(self):
        from mahjong_sandbox.parallel import ProcessBatchEnv
        cfg = BatchConfig(variant='riichi', backend='python')
        with ProcessBatchEnv(4, 2, cfg) as e:
            b = e.observation()
            e.step_async(self.choose(b), b['tokens'])
            with self.assertRaises(RuntimeError):
                e.step_async(self.choose(b), b['tokens'])
            try:
                e.step_wait(0)
            except TimeoutError:
                with self.assertRaises(RuntimeError):
                    e.observation()
                e.step_wait(30)
            e.processes[0].terminate()
            e.processes[0].join(5)
            b = e.observation()
            with self.assertRaises(RuntimeError):
                e.step(self.choose(b), b['tokens'])

    def test_training_performs_real_reproducible_updates(self):
        from mahjong_sandbox.training import training_smoke
        a = training_smoke(n=8, steps=140, backend='python')
        b = training_smoke(n=8, steps=140, backend='python')
        self.assertGreater(a['episodes'], 0)
        self.assertGreater(a['weight_norm'], 0)
        self.assertEqual(a['weights'], b['weights'])

    def test_evaluation_is_independent_of_worker_count(self):
        from mahjong_sandbox.evaluation import evaluate
        args = dict(blocks=2, policies=('win-first', 'greedy'), backend='python', variant='riichi')
        a, b = evaluate(workers=1, **args), evaluate(workers=2, **args)
        self.assertEqual(a['policies'], b['policies'])
        self.assertEqual(a['commands'], b['commands'])
        self.assertEqual(a['end_reasons'], b['end_reasons'])

    def test_zero_observations_do_not_give_zero_width_probability_interval(self):
        from mahjong_sandbox.evaluation import _interval
        np = numpy()
        interval = _interval([0] * 50, np.random.default_rng(0), bounded=True)
        self.assertEqual(interval[0], 0)
        self.assertGreater(interval[1], .05)


if __name__ == '__main__':
    unittest.main()
