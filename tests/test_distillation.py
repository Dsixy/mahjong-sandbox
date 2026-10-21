"""Teacher ties, whole-match split boundaries and anchored PPO updates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_RL=all(importlib.util.find_spec(name) is not None for name in ('numpy','torch','sb3_contrib'))


@unittest.skipUnless(HAS_RL,'optional RL dependencies not installed')
class DistillationTests(unittest.TestCase):
    def test_checkpoint_selection_protects_common_actions_and_critical_gates(self):
        from mahjong_sandbox.rl.distillation import fidelity_key,FIDELITY_THRESHOLDS
        good={name:{'teacher_set_accuracy':max(threshold,.99),'set_cross_entropy':.05}
              for name,threshold in FIDELITY_THRESHOLDS.items()}
        good['kong']={'teacher_set_accuracy':.5,'set_cross_entropy':30.}
        bad={name:dict(value) for name,value in good.items()}
        bad['all']['teacher_set_accuracy']=.92;bad['discard']['teacher_set_accuracy']=.90
        bad['kong']={'teacher_set_accuracy':.99,'set_cross_entropy':.01}
        self.assertLess(fidelity_key(good),fidelity_key(bad))
        unsafe={name:dict(value) for name,value in good.items()}
        unsafe['all']['teacher_set_accuracy']=.999;unsafe['win']['teacher_set_accuracy']=.98
        self.assertLess(fidelity_key(good),fidelity_key(unsafe))

    def test_set_loss_respects_teacher_ties_and_legal_distribution(self):
        import torch
        from mahjong_sandbox.rl.distillation import set_loss
        logits=torch.tensor([[1.,1.,-50.],[3.,-50.,2.]],requires_grad=True)
        distribution=torch.distributions.Categorical(logits=logits)
        acceptable=[[True,True,False],[True,False,False]]
        loss=set_loss(distribution,acceptable)
        self.assertAlmostEqual(float(loss.detach()),float((-distribution.log_prob(torch.tensor([0,0]))[1]/2).detach()),places=6)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):set_loss(distribution,[[False]*3,[True,False,False]])

    def test_collected_targets_match_teacher_and_both_versions(self):
        import numpy as np
        from mahjong_sandbox.batch import allocate,encode_observation
        from mahjong_sandbox.bots import make_bot
        from mahjong_sandbox.fast import FastRiichiEngine,RuntimeOptions
        from mahjong_sandbox.rl.distillation import ChoiceRecorder,teacher_targets,collect_dataset,read_shard
        engine=FastRiichiEngine(seed=12,runtime=RuntimeOptions(operator_backend='numba'))
        p=engine.next_player();obs=engine.observe(p);raw=allocate(1,public_history=True)
        encode_observation(obs,engine.legal_actions(p),raw)
        teacher=make_bot('greedy',variant='riichi',seed=1,backend='numba');teacher.rng=ChoiceRecorder(1)
        action,label,targets=teacher_targets(teacher,obs,raw)
        self.assertTrue(targets[label]);self.assertIn(action.to_dict(),obs['legal_actions'])
        with tempfile.TemporaryDirectory() as directory:
            result=collect_dataset(directory,16,seed=600,workers=1,shard_size=8)
            self.assertEqual(sum(x['samples'] for x in result['shards']),16)
            a,labels=read_shard(Path(directory)/result['shards'][0]['path'],'riichi_visible_v2')
            b,other=read_shard(Path(directory)/result['shards'][0]['path'],'riichi_visible_v1')
            self.assertIn('public_history',a);self.assertNotIn('public_history',b)
            for name in ('mask','candidates','hints'):np.testing.assert_array_equal(a[name],b[name])
            self.assertTrue((labels['acceptable'] & ~a['mask']).sum()==0)
            self.assertTrue(labels['acceptable'][np.arange(8),labels['labels']].all())
            self.assertFalse(set(result['shards'][0]['match_ids']) & set(result['shards'][1]['match_ids']))
            self.assertEqual(collect_dataset(directory,16,seed=600,workers=1,shard_size=8),result)
            with self.assertRaises(ValueError):collect_dataset(directory,17,seed=600,workers=1,shard_size=8)

    def test_offline_fit_split_guard_saved_model_and_real_anchor_update(self):
        import torch
        from gymnasium import spaces
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        from mahjong_sandbox.rl.distillation import collect_dataset,fit_policy,audit_policy,TeacherAnchor
        from mahjong_sandbox.rl.checkpoint import load_checkpoint
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,seed in [('train',600),('validation',900)]:
                collect_dataset(root/name,32,seed=seed,workers=1,shard_size=32)
            env=RiichiVecEnv(2,workers=0,feature_version='riichi_visible_v2',backend='numba')
            try:
                model=MaskablePPO(MahjongPolicy,env,n_steps=8,batch_size=8,n_epochs=1,
                    policy_kwargs={'hidden':32,'variant':'riichi'},seed=1)
                with self.assertRaises(ValueError):fit_policy(model,[root/'train'],root/'train',root/'invalid',epochs=1)
                report=fit_policy(model,[root/'train'],root/'validation',root/'fit',epochs=2,batch_size=16)
                self.assertEqual(report['epochs'],2)
                restored,metadata=load_checkpoint(root/'fit/warmstart.zip')
                self.assertEqual(metadata['timesteps'],0)
                self.assertEqual(audit_policy(restored,[root/'validation'])['all']['samples'],32)
                before=[parameter.detach().clone() for parameter in restored.policy.parameters()]
                anchor=TeacherAnchor([root/'train'],feature_version='riichi_visible_v2',coefficient=.05,steps=2,batch_size=16)
                result=anchor.update(restored)
                self.assertEqual(result['anchor/optimizer_steps'],2)
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,restored.policy.parameters())))
                # Collection actually executes student choices and records the
                # teacher labels on those new positions, preserving provenance.
                dagger=collect_dataset(root/'dagger',16,seed=1200,workers=1,shard_size=16,
                    learner=root/'fit/warmstart.zip',teacher_mix=.25)
                self.assertEqual(dagger['plan']['learner_sha256'],metadata['model_sha256'])
                from mahjong_sandbox.rl.train import train,TrainConfig
                config=TrainConfig(variant='riichi',feature_version='riichi_visible_v2',match_length='south',
                    reward_mode='rank',total_timesteps=16,n_envs=2,workers=0,rollout_steps=8,batch_size=8,
                    epochs=1,bc_samples=0,validation_blocks=0,backend='numba',self_play_probability=.5,
                    snapshot_every=1,anchor_directories=(str(root/'train'),),anchor_coefficient=.05,
                    anchor_batch_size=16,anchor_steps=1)
                trained=train(root/'ppo',config,resume=root/'fit/warmstart.zip')
                self.assertEqual(trained['final_timesteps'],16)
                event=next(x for x in map(json.loads,(root/'ppo/metrics.jsonl').read_text().splitlines()) if x['event']=='ppo_update')
                self.assertEqual(event['anchor/optimizer_steps'],1)
                self.assertEqual(trained['illegal_actions'],0)
            finally:env.close()

    def test_teacher_anchor_configuration_and_finished_update_order(self):
        from mahjong_sandbox.rl.train import TrainConfig,TrainingCallback
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        for changes in ({'anchor_coefficient':.1},{'anchor_directories':['missing']},
                        {'anchor_steps':0},{'anchor_coefficient':float('nan')},
                        {'anchor_batch_size':0}):
            with self.assertRaises(ValueError):TrainConfig(variant='riichi',**changes)
        config=TrainConfig(variant='riichi',validation_blocks=0)
        with tempfile.TemporaryDirectory() as directory:
            callback=TrainingCallback(directory,config,{})
            callback.model=SimpleNamespace(num_timesteps=16,_n_updates=1,logger=SimpleNamespace(name_to_value={}))
            callback.anchor=Mock();callback.anchor.update.return_value={'anchor/set_cross_entropy':.2}
            with patch('mahjong_sandbox.rl.train.save_checkpoint') as save:
                callback.finish_iteration(final=True)
                callback.anchor.update.assert_called_once_with(callback.model)
                self.assertEqual(callback.history[0]['anchor/set_cross_entropy'],.2)
                save.assert_called_once()
                callback.finish_iteration(final=True)
                callback.anchor.update.assert_called_once()


if __name__=='__main__':unittest.main()
