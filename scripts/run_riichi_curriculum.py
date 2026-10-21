"""Bounded V1/V2 teacher distillation, DAgger, mixed self-play and held-out tests."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def prepare(root, *, samples=32768, validation_samples=4096, dagger_samples=8192,
            epochs=20, dagger_epochs=8, steps=524288, workers=12, n_envs=512,
            eval_blocks=256, seeds=(20261021,20261022),teacher_dataset=None,validation_dataset=None):
    from mahjong_sandbox.rl.checkpoint import source_fingerprint,read_metadata
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    legacy=Path(__file__).resolve().parents[1]/'mahjong_sandbox/models/riichi-high/best.zip'
    plan={'format':'riichi_curriculum_v1','source_sha256':source_fingerprint(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'teacher_samples':samples,'validation_samples':validation_samples,'dagger_samples':dagger_samples,
        'epochs':epochs,'dagger_epochs':dagger_epochs,'ppo_steps_per_run':steps,
        'workers':workers,'n_envs':n_envs,'eval_blocks':eval_blocks,'train_seeds':list(seeds),
        'teacher_seed':1100000,'validation_seed':1200000,'dagger_seed':1300000,
        'checkpoint_validation_seed':1500000,'checkpoint_validation_blocks':32,'test_seed':1800000,
        'probe_seed':1900000,'probe_blocks':16,'legacy_sha256':read_metadata(legacy)['model_sha256'],
        'match_config':{'length':'south','target':30000,'bankruptcy':True,'agari_yame':True,
            'tenpai_yame':False,'west_extension':True},
        'reward':'final match placement only: [1.5,0.5,-0.5,-1.5]',
        'opponents':'each opponent seat: 50% greedy, 50% frozen historical self-play; selected for a whole match',
        'teacher_anchor':{'coefficient':.05,'steps_per_ppo_update':4,'batch_size':256},
        'coldstart_thresholds':{'all':.94,'discard':.94,'call':.90,'riichi':.98,'win':.995,'threat':.90},
        'comparison':'V1 published baseline plus a V1 control with matched new training; V2 BC and two independent V2 RL seeds. Test walls never select a checkpoint.',
        'release_gate':'Independent test upper 95% CI of rank difference vs greedy and vs BC must both be below zero. A failed gate retains experimental artifacts without replacing the game model.'}
    plan['coldstart_selection']='Fidelity gates first, then overall teacher-set accuracy, then overall held-out teacher-set loss; rare-action losses are reported separately.'
    plan['imported_data']={}
    for name,directory,count,seed in [('teacher',teacher_dataset,samples,plan['teacher_seed']),
                                      ('validation',validation_dataset,validation_samples,plan['validation_seed'])]:
        if directory:
            directory=Path(directory).resolve();manifest=directory/'manifest.json'
            saved=json.loads(manifest.read_text())
            from mahjong_sandbox.riichi_model import RiichiConfig
            from mahjong_sandbox.riichi_match import MatchConfig
            if (saved['plan']['format']!='riichi_distillation_dataset_v1' or saved['plan']['samples']!=count or
                    saved['plan']['seed']!=seed or saved['plan']['learner_sha256'] is not None or
                    saved['plan']['rules']!=json.loads(json.dumps(asdict(RiichiConfig()))) or
                    saved['plan']['match_config']!=asdict(MatchConfig(**plan['match_config']))):
                raise ValueError('imported immutable dataset does not match the planned rules, size or seed')
            for shard in saved['shards']:
                if hashlib.sha256((directory/shard['path']).read_bytes()).hexdigest()!=shard['sha256']:
                    raise ValueError('imported dataset checksum mismatch')
            destination=root/'data'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists() and destination.resolve()!=directory:raise ValueError('dataset import destination already exists')
            if not destination.exists():destination.symlink_to(directory,target_is_directory=True)
            plan['imported_data'][name]={'directory':str(directory),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'producer_source_sha256':saved['plan']['source_sha256'],
                'note':'Immutable public feature tensors reused with original producer provenance; no source hash is relabeled.'}
    destination=root/'plan.json'
    if destination.exists() and json.loads(destination.read_text())!=plan:raise ValueError('frozen experiment plan changed')
    write_json(destination,plan)
    return plan


def configuration(root,version,seed):
    from mahjong_sandbox.rl.train import TrainConfig
    from mahjong_sandbox.riichi_match import MatchConfig
    p=json.loads((root/'plan.json').read_text())
    return TrainConfig(variant='riichi',feature_version='riichi_visible_'+version,match_length='south',
        match_config=MatchConfig(**p['match_config']),total_timesteps=p['ppo_steps_per_run'],
        n_envs=p['n_envs'],workers=p['workers'],rollout_steps=32,batch_size=min(1024,p['n_envs']*32),
        epochs=4,learning_rate=5e-5,gamma=1.,gae_lambda=1.,entropy_coef=.003,
        reward_mode='rank',reward_scale=1.,seed=seed,backend='numba',torch_threads=2,
        opponents=('greedy',),bc_samples=0,bc_epochs=0,checkpoint_every=4,validation_every=8,
        validation_blocks=p['checkpoint_validation_blocks'],validation_seed=p['checkpoint_validation_seed'],
        validation_workers=min(8,p['workers']),self_play_probability=.5,snapshot_every=4,snapshot_pool_size=8,
        anchor_directories=(str(root/'data/teacher'),str(root/f'data/dagger_{version}')),
        anchor_coefficient=.05,anchor_steps=4,anchor_batch_size=256)


def collect(root,split,version,workers):
    from mahjong_sandbox.rl.distillation import collect_dataset
    from mahjong_sandbox.riichi_match import MatchConfig
    p=json.loads((root/'plan.json').read_text())
    kwargs={'workers':workers,'match_config':MatchConfig(**p['match_config'])}
    if split=='dagger':
        kwargs.update(learner=root/f'bc_{version}_initial/warmstart.zip',teacher_mix=.25)
        count=p['dagger_samples'];seed=p['dagger_seed']+(10000 if version=='v2' else 0);name='dagger_'+version
    else:
        count=p['teacher_samples' if split=='teacher' else 'validation_samples']
        seed=p['teacher_seed' if split=='teacher' else 'validation_seed'];name=split
    if name in p.get('imported_data',{}):
        manifest=root/'data'/name/'manifest.json'
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=p['imported_data'][name]['manifest_sha256']:
            raise ValueError('imported dataset manifest changed')
        saved=json.loads(manifest.read_text())
        for shard in saved['shards']:
            if hashlib.sha256((manifest.parent/shard['path']).read_bytes()).hexdigest()!=shard['sha256']:
                raise ValueError('imported dataset checksum mismatch')
        return saved
    return collect_dataset(root/'data'/name,count,seed=seed,**kwargs)


def fit(root,version,dagger):
    import torch
    from sb3_contrib import MaskablePPO
    from mahjong_sandbox.rl.env import observation_space
    from mahjong_sandbox.rl.policy import MahjongPolicy
    from mahjong_sandbox.rl.checkpoint import load_checkpoint,read_metadata
    from mahjong_sandbox.rl.distillation import fit_policy,audit_policy
    from mahjong_sandbox.riichi_model import RiichiConfig
    from gymnasium import spaces
    import gymnasium as gym
    p=json.loads((root/'plan.json').read_text());torch.set_num_threads(2)
    name=f'bc_{version}'+('' if dagger else '_initial');output=root/name
    if (output/'imitation.json').exists() and (output/'acceptance.json').exists():
        read_metadata(output/'warmstart.zip');return json.loads((output/'imitation.json').read_text())
    config=configuration(root,version,p['train_seeds'][0])
    model_space=observation_space(RiichiConfig(),'riichi_visible_'+version)
    class UnusedEnvironment(gym.Env):
        observation_space=model_space
        action_space=spaces.Discrete(512)
        def reset(self,**kwargs):raise RuntimeError('offline distillation never resets an environment')
        def step(self,action):raise RuntimeError('offline distillation never steps an environment')
    if dagger:
        model,_=load_checkpoint(root/f'bc_{version}_initial/warmstart.zip')
    else:
        model=MaskablePPO(MahjongPolicy,UnusedEnvironment(),seed=p['train_seeds'][0],device='cpu',
            policy_kwargs={'hidden':128,'variant':'riichi'},n_steps=8,batch_size=8,n_epochs=1)
    datasets=[root/'data/teacher']+([root/f'data/dagger_{version}'] if dagger else [])
    report=fit_policy(model,datasets,root/'data/validation',output,epochs=p['dagger_epochs' if dagger else 'epochs'],
        learning_rate=1e-4 if dagger else 3e-4,seed=p['train_seeds'][0],training={
            'config':asdict(config),'validation_seed':p['checkpoint_validation_seed'],
            'validation_blocks':p['checkpoint_validation_blocks']})
    model,metadata=load_checkpoint(output/'warmstart.zip')
    audit=audit_policy(model,[root/'data/validation'])
    gate={name:bool(name in audit and audit[name]['teacher_set_accuracy']>=threshold)
        for name,threshold in p['coldstart_thresholds'].items()}
    write_json(output/'acceptance.json',{'passed':all(gate.values()),'checks':gate,'thresholds':p['coldstart_thresholds'],
        'audit':audit,'model_sha256':metadata['model_sha256'],'note':'Teacher fidelity only; playing strength requires whole-match evaluation.'})
    return report


def train_run(root,label):
    from mahjong_sandbox.rl.longrun import run
    from mahjong_sandbox.rl.checkpoint import source_fingerprint
    p=json.loads((root/'plan.json').read_text())
    if source_fingerprint()!=p['source_sha256']:raise ValueError('source differs from frozen experiment')
    version,seed=label.split('_');seed=int(seed)
    if version not in ('v1','v2') or seed not in p['train_seeds']:raise ValueError('unplanned training run')
    acceptance=json.loads((root/f'bc_{version}/acceptance.json').read_text())
    if not acceptance['passed']:raise ValueError('coldstart fidelity gate failed; inspect acceptance.json')
    config=configuration(root,version,seed)
    write_json(root/f'{label}.json',asdict(config))
    return run(config,root/label,root/f'bc_{version}/warmstart.zip',additional_steps=p['ppo_steps_per_run'],evaluate=False)


def evaluate(root,workers):
    from mahjong_sandbox.rl.riichi_evaluate import evaluate_models
    from mahjong_sandbox.rl.checkpoint import read_metadata
    p=json.loads((root/'plan.json').read_text());seeds=p['train_seeds']
    labels=['v1_'+str(seeds[0]),*['v2_'+str(seed) for seed in seeds]]
    models={label:root/label/'best.zip' for label in labels}
    for label in labels:
        if read_metadata(root/label/'latest.zip')['timesteps']!=p['ppo_steps_per_run']:
            raise ValueError('planned training budget not complete')
    selected=max([x for x in labels if x.startswith('v2')],key=lambda x:read_metadata(models[x])['training']['validation_score'])
    models.update(v2_bc=root/'bc_v2/warmstart.zip',v1_bc=root/'bc_v1/warmstart.zip',
        v1_published=Path(__file__).resolve().parents[1]/'mahjong_sandbox/models/riichi-high/best.zip')
    frozen={key:{'path':str(path),'sha256':read_metadata(path)['model_sha256']} for key,path in models.items()}
    receipt=root/'test-models.json'
    if receipt.exists() and json.loads(receipt.read_text())!=frozen:raise ValueError('test model set changed')
    write_json(receipt,frozen)
    result=evaluate_models(models[selected],additional_models={k:v for k,v in models.items() if k!=selected},
        baselines=('greedy','defensive'),blocks=p['eval_blocks'],seed=p['test_seed'],workers=workers,
        backend='numba',match_length='south',match_config=p['match_config'])
    result['selected_label']=selected;result['experiment']=p
    paired=result['paired_comparisons']['greedy']
    gate={key:paired[key]['rank']['ci95'][1]<0 for key in ('greedy','v2_bc')}
    result['release_gate']={'passed':all(gate.values()),'checks':gate,'criterion':p['release_gate']}
    write_json(root/'evaluation.json',result)
    return {k:v for k,v in result.items() if k not in ('game_records','matchups')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('stage',choices=['prepare','teacher','validation','fit','dagger','refit','train','evaluate'])
    parser.add_argument('--version',choices=['v1','v2'],default='v2');parser.add_argument('--label')
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--n-envs',type=int,default=512)
    parser.add_argument('--samples',type=int,default=32768);parser.add_argument('--validation-samples',type=int,default=4096)
    parser.add_argument('--dagger-samples',type=int,default=8192);parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--dagger-epochs',type=int,default=8);parser.add_argument('--steps',type=int,default=524288)
    parser.add_argument('--eval-blocks',type=int,default=256)
    parser.add_argument('--teacher-dataset',type=Path);parser.add_argument('--validation-dataset',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.stage=='prepare':result=prepare(root,samples=args.samples,validation_samples=args.validation_samples,
        dagger_samples=args.dagger_samples,epochs=args.epochs,dagger_epochs=args.dagger_epochs,steps=args.steps,
        workers=args.workers,n_envs=args.n_envs,eval_blocks=args.eval_blocks,
        teacher_dataset=args.teacher_dataset,validation_dataset=args.validation_dataset)
    elif args.stage in ('teacher','validation','dagger'):result=collect(root,args.stage,args.version,args.workers)
    elif args.stage in ('fit','refit'):result=fit(root,args.version,args.stage=='refit')
    elif args.stage=='train':result=train_run(root,args.label)
    else:result=evaluate(root,args.workers)
    if isinstance(result,dict):print(json.dumps({k:v for k,v in result.items() if k not in ('history','shards')},indent=2))
    return result if isinstance(result,int) else 0

if __name__=='__main__':raise SystemExit(main())
