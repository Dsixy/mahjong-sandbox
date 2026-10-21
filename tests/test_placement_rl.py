"""Whole-hanchan rank rewards and objective-aware model selection."""
from dataclasses import asdict
import importlib.util
import tempfile
from pathlib import Path
import unittest

HAS_RL = all(importlib.util.find_spec(x) for x in ('torch', 'numpy', 'sb3_contrib'))


@unittest.skipUnless(HAS_RL, 'install RL dependencies')
class PlacementRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_rank_reward_is_terminal_only_over_full_south_match(self):
        import numpy as np
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.riichi_features import canonical_index
        from mahjong_sandbox.bots import make_bot
        env = RiichiVecEnv(1, seed=991, match_length='south', reward_mode='rank', backend='numba', hints=False)
        bot = make_bot('greedy', variant='riichi', seed=66, backend='numba')
        try:
            env.reset(); total = 0.; observed_point_change = False
            for _ in range(3000):
                core = env.core
                action = bot.act(core.engines[0].observe(core.players[0]))
                previous = env.raw['scores'].copy()
                obs, rewards, done, infos = env.step(np.array([canonical_index(env.raw, action)]))
                total += float(rewards[0])
                if done[0]:
                    metrics = infos[0]['metrics']
                    self.assertEqual(total, 2.5 - metrics['rank'])
                    self.assertEqual(float(rewards[0]), total)
                    self.assertEqual(infos[0]['episode']['r'], total)
                    self.assertTrue(obs['mask'][0].any())
                    self.assertFalse(infos[0]['terminal_observation']['mask'].any())
                    self.assertTrue(observed_point_change)
                    break
                self.assertEqual(float(rewards[0]), 0.)
                observed_point_change |= bool(np.any(previous != env.raw['scores']))
            else: self.fail('full hanchan did not terminate')
        finally: env.close()

    def test_rank_contract_rejects_discount_and_point_scaling(self):
        from mahjong_sandbox.rl.train import TrainConfig
        for values in ({'gamma':.99}, {'reward_scale':10000}, {'match_length':'hand'}, {'rank_rewards':[1,1,0,-1]}):
            kwargs=dict(variant='riichi', match_length='south', reward_mode='rank');kwargs.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):TrainConfig(**kwargs)
        self.assertEqual(TrainConfig(variant='riichi', match_length='south', reward_mode='rank').reward_scale,1.)

    def test_validation_and_recovery_select_rank_even_when_points_disagree(self):
        import json
        from unittest.mock import patch
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.train import TrainConfig, TrainingCallback
        from mahjong_sandbox.rl.checkpoint import read_metadata, save_checkpoint
        from mahjong_sandbox.rl.longrun import recover_best
        from mahjong_sandbox.riichi_model import RiichiConfig
        config=TrainConfig(variant='riichi', match_length='south', reward_mode='rank', validation_blocks=1)
        context={'config':asdict(config),'validation_seed':config.validation_seed,'validation_blocks':1}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);env=RiichiVecEnv(2,hints=False)
            try:
                model=MaskablePPO(MahjongPolicy,env,n_steps=8,batch_size=8,n_epochs=1,policy_kwargs={'variant':'riichi','hidden':32})
                callback=TrainingCallback(output,config,context);callback.init_callback(model)
                model.num_timesteps=16
                with patch('mahjong_sandbox.rl.train.validate_model',return_value={'rank_utility':.5,'mean_rank':2.,'mean_net_score':-5000}):callback.validate()
                model.num_timesteps=32
                save_checkpoint(model,output/'checkpoints/step_32.zip',rules=RiichiConfig(),training=context)
                with patch('mahjong_sandbox.rl.train.validate_model',return_value={'rank_utility':0.,'mean_rank':2.5,'mean_net_score':5000}):callback.validate()
                self.assertEqual(read_metadata(output/'best.zip')['timesteps'],16)
                # Recover from immutable evaluated updates, using the same objective.
                model.num_timesteps=16
                save_checkpoint(model,output/'checkpoints/step_16.zip',rules=RiichiConfig(),training=context)
                (output/'best.zip').unlink();(output/'best.json').unlink()
                recover_best(output,config)
                restored=read_metadata(output/'best.zip')
                self.assertEqual(restored['timesteps'],16)
                self.assertEqual(restored['training']['validation_metric'],'rank_utility')
            finally:env.close()

    def test_parallel_validation_and_context_counters_have_same_results(self):
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.riichi_evaluate import validate_model, _context_choice, _context
        from mahjong_sandbox.rl.train import TrainConfig
        from mahjong_sandbox.model import Action, ActionKind
        from collections import Counter
        config=TrainConfig(variant='riichi', match_length='south', reward_mode='rank')
        env=RiichiVecEnv(2,hints=False)
        try:
            model=MaskablePPO(MahjongPolicy,env,n_steps=8,batch_size=8,n_epochs=1,policy_kwargs={'variant':'riichi','hidden':32})
            metadata={'rules':asdict(config.rules),'hints':False,'training':{'config':asdict(config)}}
            a=validate_model(model,metadata,blocks=1,seed=199991,backend='numba',workers=1)
            b=validate_model(model,metadata,blocks=1,seed=199991,backend='numba',workers=2)
            self.assertEqual(a,b);self.assertAlmostEqual(a['rank_utility'],2.5-a['mean_rank'])
            obs={'round_wind':28,'hand_number':4,'player':0,'scores':[31000,29000,21000,19000], 'legal_actions':[{'kind':'riichi'},{'kind':'discard'}]}
            c=Counter();_context_choice(obs,Action(ActionKind.DISCARD,(0,)),c)
            self.assertEqual(c['late_lead_riichi_opportunities'],1)
            self.assertEqual(c['late_lead_riichi_choices'],0)
            obs['scores']=[10000,30000,30000,30000];self.assertEqual(_context(obs),'late_last')
            obs['legal_actions']=[{'kind':'ron'},{'kind':'pass'}]
            _context_choice(obs,Action(ActionKind.PASS),c)
            self.assertEqual(c['late_last_win_opportunities'],1)
            self.assertEqual(c['late_last_win_choices'],0)
        finally:env.close()
