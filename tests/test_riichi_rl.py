"""Riichi public information, red actions, whole-match accounting and persistence."""
from copy import deepcopy
from dataclasses import replace,asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_RL=all(importlib.util.find_spec(p) is not None for p in ('numpy','torch','sb3_contrib'))

@unittest.skipUnless(HAS_RL,'install RL dependencies')
class RiichiRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_public_projection_parity_and_hidden_wall_invariance(self):
        import numpy as np
        from mahjong_sandbox.riichi_engine import RiichiEngine
        from mahjong_sandbox.fast import FastRiichiEngine
        from mahjong_sandbox.bots import make_bot
        from mahjong_sandbox.rl.features import observation_batch,encode_batch
        from mahjong_sandbox.riichi_model import RiichiConfig
        rules=RiichiConfig();reference,fast=RiichiEngine(rules,seed=9),FastRiichiEngine(rules,seed=9)
        bots=[make_bot('greedy',variant='riichi',seed=p) for p in range(4)]
        for _ in range(20):
            p=reference.next_player();obs=reference.observe(p)
            a=encode_batch(observation_batch(obs),rules=rules)
            b=encode_batch(observation_batch(fast.observe(p)),rules=rules)
            for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
            action=bots[p].act(obs)
            reference.submit(p,action,reference.state.decision_id);fast.submit(p,action,fast.state.decision_id)
        a=encode_batch(observation_batch(fast.observe(fast.next_player())),rules=rules)
        changed=deepcopy(fast);head=changed.state.head
        changed.state.wall[head],changed.state.wall[head+1]=changed.state.wall[head+1],changed.state.wall[head]
        b=encode_batch(observation_batch(changed.observe(changed.next_player())),rules=rules)
        for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
        self.assertEqual(a['state'].shape,(1,2350))

    def test_red_and_drawn_tile_actions_remain_distinct(self):
        from mahjong_sandbox.batch import allocate,ACTION_KINDS
        from mahjong_sandbox.rl.riichi_features import canonical_mask,canonical_index
        from mahjong_sandbox.model import Action,ActionKind
        b=allocate(1);b['actions'].fill(-1);b['meta'][0,10]=-1
        for i,t in enumerate([16,17,18]):
            b['actions'][0,i,:3]=[ACTION_KINDS.index('discard'),1,t];b['action_mask'][0,i]=True
        b['red_mask'][0,16]=True
        self.assertEqual(canonical_mask(b)[0,:3].tolist(),[True,True,False])
        self.assertEqual(canonical_index(b,Action(ActionKind.DISCARD,(18,))),1)
        b['meta'][0,10]=18
        self.assertEqual(canonical_mask(b)[0,:3].tolist(),[True,True,True])

    def test_full_match_rewards_include_deposits_and_final_bank(self):
        import numpy as np
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.riichi_features import canonical_index
        from mahjong_sandbox.bots import make_bot
        env=RiichiVecEnv(1,seed=18,opponents=('greedy',),hints=False,match_length='east')
        try:
            obs=env.reset();total=0.;bot=make_bot('greedy',variant='riichi',seed=100)
            for _ in range(2000):
                core=env.core;action=bot.act(core.engines[0].observe(core.players[0]))
                index=canonical_index(env.raw,action)
                obs,reward,done,infos=env.step(np.array([index]));total+=float(reward[0])
                if done[0]:
                    self.assertAlmostEqual(total*10000,infos[0]['metrics']['net_score'],places=2)
                    self.assertIn(infos[0]['metrics']['rank'],range(1,5))
                    self.assertFalse(infos[0]['terminal_observation']['mask'].any())
                    self.assertTrue(obs['mask'].any())
                    break
            else:self.fail('no full match completed')
        finally:env.close()

    def test_ppo_checkpoint_resume_and_dynamic_round_rules(self):
        import numpy as np
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.checkpoint import save_checkpoint,load_checkpoint
        from mahjong_sandbox.rl.agent import RLAgent
        from mahjong_sandbox.riichi_model import RiichiConfig
        from mahjong_sandbox.fast import FastRiichiEngine
        from mahjong_sandbox.model import RuleConfig
        with tempfile.TemporaryDirectory() as directory:
            env=RiichiVecEnv(2,hints=False)
            try:
                model=MaskablePPO(MahjongPolicy,env,n_steps=8,batch_size=8,n_epochs=1,device='cpu',policy_kwargs={'variant':'riichi','hidden':32},seed=71)
                model.learn(16);path=Path(directory)/'model.zip'
                save_checkpoint(model,path,rules=RiichiConfig(),hints=False)
                restored,metadata=load_checkpoint(path,device='cpu')
                self.assertEqual(metadata['feature_version'],'riichi_visible_v1')
                self.assertEqual(restored._n_updates,model._n_updates)
                obs=env.reset()
                a,_=model.predict(obs,action_masks=obs['mask'],deterministic=True)
                b,_=restored.predict(obs,action_masks=obs['mask'],deterministic=True)
                np.testing.assert_array_equal(a,b)
                restored.set_env(env);restored.learn(16,reset_num_timesteps=False)
                self.assertEqual(restored.num_timesteps,32)
                rules=replace(RiichiConfig(),dealer=2,round_wind=28,hand_number=3,honba=2,riichi_sticks=2,initial_scores=(31000,23000,23000,21000))
                engine=FastRiichiEngine(rules,seed=5);p=engine.next_player()
                agent=RLAgent(model=model,metadata=metadata)
                self.assertIn(agent.act(engine.observe(p)),engine.legal_actions(p))
                wrong=FastRiichiEngine(replace(rules,red_fives=()),seed=5)
                with self.assertRaises(ValueError):agent.act(wrong.observe(wrong.next_player()))
                with self.assertRaises(ValueError):load_checkpoint(path,expected_rules=RuleConfig())
                with self.assertRaises(ValueError):save_checkpoint(model,Path(directory)/'wrong.zip',rules=RuleConfig())
            finally:env.close()

    def test_process_match_trajectory_and_frozen_riichi_opponent(self):
        import numpy as np
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.selfplay import publish_snapshot
        from mahjong_sandbox.riichi_model import RiichiConfig
        with tempfile.TemporaryDirectory() as directory:
            base=RiichiVecEnv(2,hints=False)
            try:
                model=MaskablePPO(MahjongPolicy,base,n_steps=8,batch_size=8,n_epochs=1,device='cpu',policy_kwargs={'variant':'riichi','hidden':32})
                publish_snapshot(model,directory,rules=RiichiConfig(),hints=False,training={})
            finally:base.close()
            args=dict(seed=19,hints=False,match_length='east',opponent_manifest=Path(directory)/'opponents.json',self_play_probability=1)
            left,right=RiichiVecEnv(2,workers=0,**args),RiichiVecEnv(2,workers=1,**args)
            try:
                a,b=left.reset(),right.reset()
                for _ in range(12):
                    for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
                    actions=a['mask'].argmax(axis=1)
                    a,r,d,_=left.step(actions);b,s,e,_=right.step(actions)
                    np.testing.assert_array_equal(r,s);np.testing.assert_array_equal(d,e)
            finally:left.close();right.close()

    def test_style_summary_uses_hand_counts_and_seed_blocks(self):
        from mahjong_sandbox.rl.riichi_evaluate import summarize
        rows=[]
        for seed,hands,wins in [(1,2,1),(2,8,0)]:
            rows.append({'seed':seed,'hands':hands,'win':wins,'deal_in':0,'riichi':0,'open':0,'kongs':0,'self_draw':0,'win_han':wins*2,'win_income':wins*3900,'net_score':100,'rank':2,'style_counters':{}})
        r=summarize(rows)
        self.assertEqual(r['win_rate'],.1)
        self.assertEqual(r['hands'],10)
        self.assertEqual(r['mean_net_score'],100)
        self.assertEqual(r['style']['win_income'],3900)

    def test_bounded_training_from_scratch_and_evaluation_dispatch(self):
        from mahjong_sandbox.rl.train import TrainConfig
        from mahjong_sandbox.rl.longrun import run
        from mahjong_sandbox.rl.evaluate import evaluate_models
        config=TrainConfig(variant='riichi',total_timesteps=16,n_envs=2,workers=0,rollout_steps=8,
            batch_size=8,epochs=1,bc_samples=0,validation_blocks=0,hints=False,backend='numba')
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run(config,directory,None,additional_steps=16,evaluate=False),0)
            latest=Path(directory)/'latest.zip'
            self.assertEqual(run(config,directory,None,additional_steps=16,evaluate=False),0)
            self.assertFalse((Path(directory)/'training_before_16.json').exists())
            args=dict(model=latest,blocks=1,baselines=('greedy',),backend='numba',seed=200000)
            a=evaluate_models(workers=1,**args);b=evaluate_models(workers=2,**args)
            self.assertEqual(a['variant'],'riichi')
            self.assertEqual(a['game_records'],b['game_records'])
            self.assertEqual(a['matchups'],b['matchups'])
