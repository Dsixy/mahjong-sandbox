"""Global-step selection survives resume offsets and artifact corruption."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HAS_RL=all(importlib.util.find_spec(name) is not None for name in ('numpy','torch','sb3_contrib'))


@unittest.skipUnless(HAS_RL,'optional RL dependencies not installed')
class CurriculumSelectionTests(unittest.TestCase):
    def test_final_boundary_recovery_keeps_model_bytes_and_checks_optimizer(self):
        from scripts.riichi_runtime import configuration,recover_finished_training
        from mahjong_sandbox.rl.checkpoint import save_checkpoint,read_metadata,source_fingerprint
        from mahjong_sandbox.riichi_model import RiichiConfig
        model,_=self.model('v2');model.num_timesteps=128
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);label='v2_91'
            (root/'plan.json').write_text(json.dumps({'source_sha256':source_fingerprint(),'train_seeds':[91],
                'ppo_steps_per_run':128,'n_envs':2,'workers':1,'match_config':{'length':'south'},
                'checkpoint_validation_seed':1500000,'checkpoint_validation_blocks':1}))
            config=configuration(root,label);path=root/label/'checkpoints/step_128.zip'
            save_checkpoint(model,path,rules=RiichiConfig(),training={'config':asdict(config)})
            (root/label/'metrics.jsonl').write_text('')
            with self.assertRaises(ValueError):recover_finished_training(root,label,config)
            model.learn(total_timesteps=16);model.num_timesteps=128;model._n_updates=8
            save_checkpoint(model,path,rules=RiichiConfig(),training={'config':asdict(config)})
            result=recover_finished_training(root,label,config)
            self.assertEqual(result['optimizer_epochs'],8);self.assertTrue(result['all_finite'])
            self.assertEqual((root/label/'latest.zip').read_bytes(),path.read_bytes())
            self.assertEqual(read_metadata(root/label/'latest.zip')['timesteps'],128)
            self.assertIsNone(json.loads((root/label/'training.json').read_text())['elapsed_seconds'])
            self.assertIsNone(recover_finished_training(root,label,config))

    def test_match_cache_freezes_task_and_detects_corruption(self):
        from scripts.riichi_evaluation import cached_block,digest
        def play(seed, seat, candidate, opponent, **kwargs):
            return {'seed':seed,'seat':seat,'policy':candidate,'opponent':opponent,'net_score':seat*100}
        with tempfile.TemporaryDirectory() as directory:
            args=(1800000,'rl','greedy',{},'numba',True,'south',True,{'length':'south'})
            identity={'models':{'rl':'original'},'source':'frozen'}
            with patch('mahjong_sandbox.rl.riichi_evaluate.play_match',side_effect=play) as match:
                original=cached_block(args,cache_root=directory,identity=identity)
                self.assertEqual(match.call_count,4)
                self.assertEqual(cached_block(args,cache_root=directory,identity=identity),original)
                self.assertEqual(match.call_count,4)
                cached_block(args,cache_root=directory,identity={**identity,'models':{'rl':'new'}})
                self.assertEqual(match.call_count,8)
                path=Path(directory)/(digest({'identity':identity,'task':args})+'.json');receipt=json.loads(path.read_text())
                receipt['rows'][0]['net_score']=999;path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    cached_block(args,cache_root=directory,identity=identity)
                with self.assertRaises(ValueError):cached_block((*args[:5],False,*args[6:]),cache_root=directory,identity=identity)

    def test_resource_migration_preserves_learning_settings_and_target(self):
        from scripts.riichi_runtime import configuration
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan={'train_seeds':[91],'ppo_steps_per_run':1024,'n_envs':16,'workers':12,
                'match_config':{'length':'south'},'checkpoint_validation_seed':1500000,'checkpoint_validation_blocks':1}
            (root/'plan.json').write_text(json.dumps(plan));label='v1_91'
            base=configuration(root,label);folder=root/label;folder.mkdir()
            manifest={'config':json.loads(json.dumps(asdict(base))),'target_timesteps':1024,'source_sha256':'frozen'}
            (folder/'experiment.json').write_text(json.dumps(manifest))
            (root/'runtime-overrides.json').write_text(json.dumps({label:{'workers':8,'validation_workers':4}}))
            actual=configuration(root,label,migrate=True)
            self.assertEqual(actual.workers,8);self.assertEqual(actual.n_envs,16)
            updated=json.loads((folder/'experiment.json').read_text())
            self.assertEqual(updated['target_timesteps'],1024)
            self.assertEqual(updated['source_sha256'],'frozen')
            self.assertEqual(updated['config']['learning_rate'],manifest['config']['learning_rate'])
            self.assertTrue(list(folder.glob('experiment.before_resources.*.json')))
            (root/'runtime-overrides.json').write_text(json.dumps({label:{'learning_rate':.1}}))
            with self.assertRaises(ValueError):configuration(root,label,migrate=True)

    def model(self,version):
        import torch
        from sb3_contrib import MaskablePPO
        from mahjong_sandbox.rl.env import RiichiVecEnv
        from mahjong_sandbox.rl.policy import MahjongPolicy
        torch.set_num_threads(1)
        env=RiichiVecEnv(2,workers=0,backend='numba',feature_version='riichi_visible_'+version)
        self.addCleanup(env.close)
        model=MaskablePPO(MahjongPolicy,env,n_steps=8,batch_size=8,n_epochs=1,seed=4,
            policy_kwargs={'hidden':32,'variant':'riichi'})
        return model,env

    def test_inference_buffer_override_preserves_parameters_probabilities_and_adam(self):
        import torch
        from mahjong_sandbox.rl.checkpoint import save_checkpoint,load_checkpoint
        from mahjong_sandbox.riichi_model import RiichiConfig
        model,env=self.model('v2');model.learn(total_timesteps=16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.zip';save_checkpoint(model,path,rules=RiichiConfig())
            original,_=load_checkpoint(path,device='cpu')
            inference,_=load_checkpoint(path,device='cpu',n_steps=1,n_envs=1)
            self.assertEqual(inference.rollout_buffer.action_masks.shape,(1,1,512))
            self.assertLess(inference.rollout_buffer.action_masks.nbytes,original.rollout_buffer.action_masks.nbytes)
            for a,b in zip(original.policy.parameters(),inference.policy.parameters()):self.assertTrue(torch.equal(a,b))
            self.assertTrue(original.policy.optimizer.state)
            for a,b in zip(original.policy.optimizer.state.values(),inference.policy.optimizer.state.values()):
                for key,value in a.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,b[key]))
            obs=env.reset()
            with torch.no_grad():
                values=[]
                for candidate in [original,inference]:
                    tensors,_=candidate.policy.obs_to_tensor(obs)
                    values.append(candidate.policy.get_distribution(tensors,action_masks=obs['mask']).distribution.probs)
                self.assertTrue(torch.equal(*values))

    def test_fixed_global_candidates_selection_and_exact_best_repair(self):
        from scripts import run_riichi_curriculum as curriculum
        from scripts.select_riichi_checkpoints import select
        from mahjong_sandbox.rl.checkpoint import save_checkpoint,read_metadata,source_fingerprint
        from mahjong_sandbox.riichi_model import RiichiConfig
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan={'source_sha256':source_fingerprint(),'train_seeds':[91],'ppo_steps_per_run':1024,
                'n_envs':2,'workers':1,'match_config':{'length':'south'},'checkpoint_validation_seed':1500000,
                'checkpoint_validation_blocks':1}
            (root/'plan.json').write_text(json.dumps(plan))
            for version in ['v1','v2']:
                model,_=self.model(version);label=version+'_91';config=curriculum.configuration(root,version,91)
                for step in [512,1024]:
                    model.num_timesteps=step
                    path=root/label/'checkpoints'/f'step_{step}.zip'
                    save_checkpoint(model,path,rules=RiichiConfig(),training={'config':asdict(config),
                        'validation_seed':config.validation_seed,'validation_blocks':config.validation_blocks})
                for suffix in ['zip','json']:shutil.copyfile(path.with_suffix('.'+suffix),root/label/f'latest.{suffix}')
            def validation(model,metadata,**kwargs):
                self.assertEqual(model.n_envs,1);self.assertEqual(model.n_steps,1)
                utility=.2 if model.num_timesteps==512 else -.1
                return {'rank_utility':utility,'mean_rank':2.5-utility,'mean_net_score':utility*1000}
            with patch('mahjong_sandbox.rl.riichi_evaluate.validate_model',side_effect=validation) as validate:
                report=select(root,1)
                self.assertEqual(validate.call_count,4)
                self.assertEqual(json.loads((root/'artifact-model.json').read_text())['label'],'v1_91')
                for label in ['v1_91','v2_91']:
                    self.assertEqual(report['runs'][label]['selected']['step'],512)
                    self.assertEqual(read_metadata(root/label/'best.zip')['timesteps'],512)
                (root/'v2_91/best.zip').write_bytes(b'interrupted replacement')
                select(root,1)
                self.assertEqual(validate.call_count,4)
                self.assertEqual(read_metadata(root/'v2_91/best.zip')['model_sha256'],report['runs']['v2_91']['selected']['sha256'])
                (root/'v1_91/checkpoints/step_512.zip').write_bytes(b'changed candidate')
                with self.assertRaises(ValueError):select(root,1)


if __name__=='__main__':unittest.main()
