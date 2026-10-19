"""Frozen-opponent isolation, worker parity and bounded restart recovery."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_RL = all(importlib.util.find_spec(n) is not None for n in ('numpy', 'torch', 'sb3_contrib'))


@unittest.skipUnless(HAS_RL, 'optional RL dependencies not installed')
class SelfPlayTests(unittest.TestCase):
    def fixture(self, root):
        import torch
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.env import SichuanVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.checkpoint import save_checkpoint
        torch.set_num_threads(1)
        env = SichuanVecEnv(2, backend='python', hints=False)
        self.addCleanup(env.close)
        model = MaskablePPO(MahjongPolicy, env, n_steps=8, batch_size=8, n_epochs=1, device='cpu', seed=9)
        path = Path(root) / 'initial.zip'
        save_checkpoint(model, path, hints=False)
        return model, path

    def test_snapshots_freeze_inflight_agents_and_verify_contract(self):
        import torch
        from mahjong_sandbox.model import RuleConfig
        from mahjong_sandbox.rl.selfplay import publish_snapshot, SnapshotPool
        with tempfile.TemporaryDirectory() as root:
            model, _ = self.fixture(root)
            directory = Path(root) / 'pool'
            def publish():
                return publish_snapshot(model, directory, rules=RuleConfig(), hints=False, training={}, pool_size=1)
            publish()
            pool = SnapshotPool(directory / 'opponents.json', rules=RuleConfig(), backend='python', probability=1)
            pool.refresh()
            old = pool.opponent(0, None)
            original = next(old.model.policy.parameters()).detach().clone()
            with torch.no_grad():
                next(model.policy.parameters()).add_(1)
            model.num_timesteps = 16
            publish()
            pool.refresh()
            new = pool.opponent(0, None)
            self.assertIsNot(old, new)
            torch.testing.assert_close(next(old.model.policy.parameters()), original)
            self.assertFalse(torch.equal(next(new.model.policy.parameters()), original))
            pool.probability = 0
            self.assertEqual(pool.opponent(0, 'bot'), 'bot')
            pool.probability = 1
            pool.entries[0]['sha256'] = 'wrong'
            pool.cache.clear()
            with self.assertRaises(ValueError):
                pool.opponent(0, None)

    def test_frozen_selfplay_has_same_trajectory_across_shards(self):
        import numpy as np
        from mahjong_sandbox.batch import BatchConfig, SyncBatchEnv
        from mahjong_sandbox.parallel import ProcessBatchEnv
        from mahjong_sandbox.model import RuleConfig
        from mahjong_sandbox.rl.selfplay import publish_snapshot
        with tempfile.TemporaryDirectory() as root:
            model, _ = self.fixture(root)
            publish_snapshot(model, root, rules=RuleConfig(), hints=False, training={})
            config = BatchConfig(variant='sichuan', seed=17, backend='python', rotate_seats=True,
                opponent_manifest=str(Path(root) / 'opponents.json'), self_play_probability=1)
            with SyncBatchEnv(2, config) as sync, ProcessBatchEnv(2, 1, config) as process:
                for _ in range(12):
                    a, b = sync.observation(), process.observation()
                    for key in a:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                    actions = a['action_mask'].argmax(axis=1)
                    actions[a['terminated']] = -1
                    sync.step(actions, a['tokens'])
                    process.step(actions, b['tokens'])

    def test_longrun_is_bounded_and_recovers_torn_checkpoint(self):
        from mahjong_sandbox.rl.train import TrainConfig
        from mahjong_sandbox.rl.longrun import run, newest_checkpoint
        with tempfile.TemporaryDirectory() as root:
            _, initial = self.fixture(root)
            output = Path(root) / 'run'
            config = TrainConfig(total_timesteps=32, n_envs=2, workers=0, rollout_steps=8, batch_size=8,
                epochs=1, bc_samples=0, validation_blocks=0, hints=False, backend='python',
                self_play_probability=1, checkpoint_every=1, snapshot_every=1, snapshot_pool_size=2)
            self.assertEqual(run(config, output, initial, additional_steps=32, evaluate=False), 0)
            latest = output / 'latest.json'
            self.assertEqual(json.loads(latest.read_text())['timesteps'], 32)
            self.assertEqual(len(json.loads((output / 'selfplay/opponents.json').read_text())['snapshots']), 2)
            self.assertEqual(run(config, output, initial, additional_steps=32, evaluate=False), 0)
            self.assertFalse((output / 'training_before_32.json').exists())
            data = json.loads(latest.read_text())
            data['model_sha256'] = 'torn'
            latest.write_text(json.dumps(data))
            path, meta = newest_checkpoint(output, initial, rules=config.rules)
            self.assertEqual(meta['timesteps'], 32)
            self.assertIn('checkpoints', str(path))
            with self.assertRaises(ValueError):
                run(replace(config, self_play_probability=.5), output, initial, additional_steps=32, evaluate=False)

    def test_stop_waits_for_completed_update_and_bad_config_rejected(self):
        from types import SimpleNamespace
        from mahjong_sandbox.rl.train import TrainConfig, TrainingCallback, TrainingInterrupted
        for changes in ({'self_play_probability': float('nan')}, {'self_play_probability': 1.01},
                        {'snapshot_every': 0}, {'snapshot_pool_size': 0}):
            with self.assertRaises(ValueError):
                TrainConfig(**changes)
        with tempfile.TemporaryDirectory() as root:
            callback = TrainingCallback(root, TrainConfig(validation_blocks=0), {})
            callback.model = SimpleNamespace(_n_updates=0)
            callback.stop_requested = True
            with self.assertRaises(TrainingInterrupted):
                callback._on_rollout_start()


if __name__ == '__main__':
    unittest.main()
