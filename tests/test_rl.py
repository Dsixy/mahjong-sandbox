"""Behavioral checks of RL information boundaries, masks, transitions and persistence."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_RL = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'sb3_contrib'))


@unittest.skipUnless(HAS_RL, 'optional RL dependencies not installed')
class RLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def make_model(self, env):
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.policy import MahjongPolicy
        return MaskablePPO(MahjongPolicy, env, n_steps=8, batch_size=8, n_epochs=2,
                           seed=71, gamma=1, device='cpu', verbose=0)

    def test_public_batch_and_single_observation_features_equal(self):
        import numpy as np
        from mahjong_sandbox.batch import SyncBatchEnv, BatchConfig
        from mahjong_sandbox.rl.features import encode_batch, observation_batch
        with SyncBatchEnv(4, BatchConfig(variant='sichuan', seed=3, rotate_seats=True, backend='python')) as env:
            for _ in range(14):
                raw = env.observation()
                features = encode_batch(raw, backend='python')
                for i in range(4):
                    single = encode_batch(observation_batch(env.engines[i].observe(env.players[i])), backend='python')
                    for key in features:
                        np.testing.assert_array_equal(features[key][i], single[key][0], err_msg=key)
                a = features['mask'].argmax(axis=1)
                a[raw['terminated']] = -1
                env.step(a, raw['tokens'])

    def test_semantic_dedup_keeps_public_drawn_cut_distinction(self):
        from mahjong_sandbox.batch import allocate, ACTION_KINDS
        from mahjong_sandbox.rl.features import canonical_mask
        b = allocate(1)
        b['actions'].fill(-1)
        b['action_mask'].fill(False)
        for j, tile in enumerate((104, 105, 106)):
            b['actions'][0, j, :3] = ACTION_KINDS.index('discard'), 1, tile
            b['action_mask'][0, j] = True
        b['meta'][0, 10] = 105
        self.assertEqual(canonical_mask(b)[0, :3].tolist(), [True, True, False])

    def test_hidden_state_and_physical_copy_identity_do_not_enter_state(self):
        import copy
        import numpy as np
        from mahjong_sandbox.batch import SyncBatchEnv, BatchConfig
        from mahjong_sandbox.rl.features import encode_batch
        with SyncBatchEnv(1, BatchConfig(variant='sichuan', backend='python')) as env:
            raw = env.observation()
            changed = {k: v.copy() for k, v in raw.items()}
            changed['tokens'] += 100
            changed['decision_ids'] += 777
            # These engine identities are deliberately outside neural observations.
            a, b = encode_batch(raw, backend='python'), encode_batch(changed, backend='python')
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            obs = env.engines[0].observe(0)
            self.assertNotIn('wall', obs)
            self.assertNotIn('concealed', obs['players'][1])
            from mahjong_sandbox.rl.features import observation_batch
            altered = copy.deepcopy(obs)
            altered['events'].append({'kind': 'unrelated', 'data': {'wall': [1, 2, 3]}})
            a, b = encode_batch(observation_batch(obs), backend='python'), encode_batch(observation_batch(altered), backend='python')
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_policy_is_menu_permutation_equivariant_and_masks_exactly(self):
        import numpy as np
        import torch
        from mahjong_sandbox.rl.env import SichuanVecEnv
        env = SichuanVecEnv(2, backend='python', hints=False)
        try:
            model = self.make_model(env)
            obs = env.reset()
            tensor, _ = model.policy.obs_to_tensor(obs)
            dist = model.policy.get_distribution(tensor, action_masks=obs['mask'])
            probs = dist.distribution.probs.detach().numpy()
            self.assertTrue(np.all(probs[~obs['mask']] == 0))
            permutation = np.random.default_rng(8).permutation(512)
            changed = {k: (v[:, permutation] if k != 'state' else v.copy()) for k, v in obs.items()}
            tensor_b, _ = model.policy.obs_to_tensor(changed)
            dist_b = model.policy.get_distribution(tensor_b, action_masks=changed['mask'])
            np.testing.assert_allclose(probs[:, permutation], dist_b.distribution.probs.detach().numpy(), atol=1e-6)
            torch.testing.assert_close(model.policy.predict_values(tensor), model.policy.predict_values(tensor_b))
            for _ in range(20):
                a, _ = model.predict(obs, action_masks=obs['mask'])
                self.assertTrue(obs['mask'][np.arange(2), a].all())
        finally:
            env.close()

    def test_vecenv_terminal_autoreset_preserves_rewards_and_rotates_seats(self):
        import numpy as np
        from mahjong_sandbox.batch import ACTION_KINDS
        from mahjong_sandbox.rl.env import SichuanVecEnv
        env = SichuanVecEnv(4, opponents=('win-first',), backend='python', hints=False)
        try:
            obs = env.reset()
            totals = np.zeros(4)
            first_done = set()
            for _ in range(400):
                seats = env.raw['meta'][:, 1].copy()
                a = obs['mask'].argmax(axis=1)
                win = obs['mask'] & np.isin(obs['candidates'][:, :, 0], [ACTION_KINDS.index('ron'), ACTION_KINDS.index('tsumo')])
                a[win.any(axis=1)] = win[win.any(axis=1)].argmax(axis=1)
                obs, r, done, infos = env.step(a)
                totals += r
                for i in np.flatnonzero(done):
                    self.assertEqual(totals[i] * 32, infos[i]['metrics']['net_score'])
                    self.assertEqual(infos[i]['metrics']['seat'], seats[i])
                    self.assertEqual(infos[i]['terminal_observation']['state'].shape, (1549,))
                    self.assertFalse(infos[i]['terminal_observation']['mask'].any())
                    self.assertTrue(obs['mask'][i].any())
                    self.assertFalse(infos[i]['TimeLimit.truncated'])
                    self.assertEqual(env.raw['meta'][i, 1], (seats[i] + 1) % 4)
                    totals[i] = 0
                    first_done.add(i)
                if len(first_done) == 4:
                    break
            self.assertEqual(len(first_done), 4)
        finally:
            env.close()

    def test_process_env_matches_sync_with_rotating_seats_and_opponent_pool(self):
        import numpy as np
        from mahjong_sandbox.rl.env import SichuanVecEnv
        a = SichuanVecEnv(4, workers=0, seed=19, backend='python', hints=False, opponents=('greedy', 'defensive'))
        b = SichuanVecEnv(4, workers=2, seed=19, backend='python', hints=False, opponents=('greedy', 'defensive'))
        try:
            x, y = a.reset(), b.reset()
            for _ in range(14):
                for key in x:
                    np.testing.assert_array_equal(x[key], y[key], err_msg=key)
                actions = x['mask'].argmax(axis=1)
                x, r, d, _ = a.step(actions)
                y, s, e, _ = b.step(actions)
                np.testing.assert_array_equal(r, s)
                np.testing.assert_array_equal(d, e)
        finally:
            a.close()
            b.close()

    def test_update_checkpoint_optimizer_and_resume(self):
        import numpy as np
        import torch
        from mahjong_sandbox.model import RuleConfig
        from mahjong_sandbox.rl.env import SichuanVecEnv
        from mahjong_sandbox.rl.agent import RLAgent
        from mahjong_sandbox.rl.checkpoint import save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            env = SichuanVecEnv(2, backend='python', hints=False)
            try:
                model = self.make_model(env)
                before = {k: v.clone() for k, v in model.policy.state_dict().items()}
                model.learn(16)
                self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items()))
                self.assertTrue(all(torch.isfinite(v).all() for v in model.policy.parameters()))
                path = Path(directory) / 'model.zip'
                save_checkpoint(model, path, hints=False)
                loaded, metadata = load_checkpoint(path)
                obs = env.reset()
                a, _ = model.predict(obs, deterministic=True, action_masks=obs['mask'])
                b, _ = loaded.predict(obs, deterministic=True, action_masks=obs['mask'])
                np.testing.assert_array_equal(a, b)
                optimizer = model.policy.optimizer.state_dict()
                restored = loaded.policy.optimizer.state_dict()
                self.assertTrue(optimizer['state'])
                for key in optimizer['state']:
                    for field, value in optimizer['state'][key].items():
                        torch.testing.assert_close(value, restored['state'][key][field])
                raw_obs = env.core.engines[0].observe(env.core.players[0])
                agent = RLAgent(model=loaded, metadata=metadata, backend='python')
                chosen = agent.act(raw_obs)
                self.assertEqual(chosen, env.core.menus[0][a[0]])
                with self.assertRaises(ValueError):
                    load_checkpoint(path, expected_rules=RuleConfig(exchange_three=False))
                loaded.set_env(env)
                loaded.learn(16, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 32)
                data = json.loads(path.with_suffix('.json').read_text())
                data['feature_version'] = 'wrong'
                path.with_suffix('.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_checkpoint(path)
            finally:
                env.close()

    def test_cli_training_resume_evaluation_and_overlap_rejection(self):
        from mahjong_sandbox.rl.train import TrainConfig, train
        from mahjong_sandbox.rl.evaluate import evaluate_models
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig(total_timesteps=32, n_envs=2, workers=0, rollout_steps=8, batch_size=8,
                epochs=1, bc_samples=0, validation_blocks=0, hints=False, backend='python')
            first = train(directory, config)
            self.assertEqual(first['final_timesteps'], 32)
            second = train(directory, config, resume=Path(directory) / 'latest.zip')
            self.assertEqual(second['start_timesteps'], 32)
            self.assertEqual(second['final_timesteps'], 64)
            args = dict(model=Path(directory) / 'latest.zip', blocks=1, baselines=('win-first',), backend='python')
            a, b = evaluate_models(workers=1, **args), evaluate_models(workers=2, **args)
            self.assertEqual(a['matchups'], b['matchups'])
            self.assertEqual(a['game_records'], b['game_records'])
            path = Path(directory) / 'latest.json'
            meta = json.loads(path.read_text())
            meta['training'].update(validation_seed=200000, validation_blocks=8)
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                evaluate_models(workers=1, **args)

    def test_checkpoint_schedule_counts_rollouts_after_early_kl_stop(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from mahjong_sandbox.rl.train import TrainConfig, TrainingCallback
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig(checkpoint_every=2, validation_blocks=0)
            callback = TrainingCallback(directory, config, {})
            callback.model = SimpleNamespace(num_timesteps=100, _n_updates=4, device='cpu',
                logger=SimpleNamespace(name_to_value={'train/loss': .1, 'train/explained_variance': float('nan')}))
            callback._on_training_start()
            with patch('mahjong_sandbox.rl.train.save_checkpoint') as save:
                # One epoch versus four: either case is exactly one rollout update.
                callback.model.num_timesteps = 200
                callback.model._n_updates = 5
                callback.finish_iteration()
                self.assertEqual(save.call_count, 0)
                callback.model.num_timesteps = 300
                callback.model._n_updates = 9
                callback.finish_iteration()
                self.assertEqual(save.call_count, 1)
                self.assertEqual(callback.rollout_updates, 2)
                self.assertIsNone(callback.history[-1]['train/explained_variance'])
                callback.finish_iteration(final=True)
                self.assertEqual(save.call_count, 1)

    def test_imitation_and_configuration_guards(self):
        import torch
        from mahjong_sandbox.rl.train import TrainConfig
        from mahjong_sandbox.rl.checkpoint import resolve_device
        from mahjong_sandbox.rl.env import SichuanVecEnv
        from mahjong_sandbox.rl.imitation import pretrain
        for kwargs in ({'n_envs': 0}, {'workers': 999}, {'gamma': 1.1}, {'batch_size': 13},
                       {'reward_scale': 0}, {'opponents': ['unknown']}, {'seed': -1}):
            with self.assertRaises(ValueError):
                TrainConfig(**kwargs)
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device('cuda')
        env = SichuanVecEnv(2, backend='python', hints=False)
        try:
            model = self.make_model(env)
            result = pretrain(model, samples=32, epochs=3, backend='python', hints=False, batch_size=16)
            self.assertEqual(result['samples'], 32)
            self.assertLess(result['training_cross_entropy'][-1], result['training_cross_entropy'][0])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
