"""Audited process-count overrides; learning settings and targets stay frozen."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_riichi_curriculum as curriculum

RESOURCE_FIELDS=frozenset(('workers','validation_workers','torch_threads'))


def recover_finished_training(root,label,config):
    """Recover a saved final optimizer boundary if final validation was killed.

    No learning is repeated. Validation selection remains a separate later phase.
    Unavailable wall time and episode totals are explicitly left unknown.
    """
    import torch
    from mahjong_sandbox.rl.checkpoint import read_metadata,load_checkpoint
    root=Path(root);folder=root/label;plan=json.loads((root/'plan.json').read_text())
    target=plan['ppo_steps_per_run'];checkpoint=folder/'checkpoints'/f'step_{target}.zip'
    latest=folder/'latest.zip'
    if not checkpoint.exists():return None
    report_path=folder/'training.json'
    try:
        if (read_metadata(latest)['timesteps']==target and
                json.loads(report_path.read_text())['final_timesteps']==target):return None
    except (OSError,ValueError,KeyError):
        # A process can stop between the two atomic checkpoint replacements,
        # or after replacing them but before writing its final report.
        pass
    metadata=read_metadata(checkpoint,expected_rules=config.rules)
    if metadata['timesteps']!=target or metadata['source_sha256']!=plan['source_sha256']:
        raise ValueError('final recovery checkpoint differs from frozen target or source')
    model,_=load_checkpoint(checkpoint,device='cpu',n_steps=1,n_envs=1)
    expected_updates=target//(config.n_envs*config.rollout_steps)*config.epochs
    if target%(config.n_envs*config.rollout_steps) or model._n_updates!=expected_updates:
        raise ValueError('final recovery checkpoint has an incomplete PPO optimizer boundary')
    parameters=list(model.policy.parameters())
    adam=[value for state in model.policy.optimizer.state.values() for value in state.values() if isinstance(value,torch.Tensor)]
    if not adam or any(not torch.isfinite(value).all() for value in [*parameters,*adam]):
        raise ValueError('final recovery policy or Adam state is incomplete/nonfinite')
    events=[]
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    recovery={'format':'riichi_finished_optimizer_recovery_v1','checkpoint_sha256':metadata['model_sha256'],
        'source_sha256':metadata['source_sha256'],'target_timesteps':target,'optimizer_epochs':model._n_updates,
        'parameter_tensors':len(parameters),'adam_tensors':len(adam),'all_finite':True,
        'note':'Final optimizer boundary was saved before interrupted validation; original checkpoint bytes restored without further learning. Global candidates are validated separately.'}
    for suffix in ['zip','json']:
        temporary=latest.with_suffix('.recovery.'+suffix)
        shutil.copyfile(checkpoint.with_suffix('.'+suffix),temporary)
        temporary.replace(latest.with_suffix('.'+suffix))
    if report_path.exists():shutil.copyfile(report_path,folder/'training.before_target_recovery.json')
    report={'algorithm':'sb3-contrib MaskablePPO','config':asdict(config),'actual_device':'cpu',
        'final_timesteps':target,'optimizer_epochs':model._n_updates,'interrupted':False,
        'validations':[row for row in events if row.get('event')=='validation'],
        'elapsed_seconds':None,'completed_episodes_this_run':None,
        'latest_model_sha256':metadata['model_sha256'],'source_sha256':metadata['source_sha256'],
        'latest':str(latest),'best':str(folder/'best.zip'),'target_recovery':recovery}
    curriculum.write_json(report_path,report);curriculum.write_json(folder/'target-recovery.json',recovery)
    return recovery


def configuration(root,label,*,migrate=False):
    root=Path(root);version,seed=label.split('_');seed=int(seed)
    plan=json.loads((root/'plan.json').read_text())
    if version not in ('v1','v2') or seed not in plan['train_seeds'] or (version=='v1' and seed!=plan['train_seeds'][0]):
        raise ValueError('runtime override must belong to a planned training run')
    config=curriculum.configuration(root,version,seed)
    path=root/'runtime-overrides.json';overrides=json.loads(path.read_text()).get(label,{}) if path.exists() else {}
    if set(overrides)-RESOURCE_FIELDS:raise ValueError('runtime overrides cannot change learning settings')
    config=replace(config,**overrides)
    manifest=root/label/'experiment.json'
    if migrate and manifest.exists():
        original=manifest.read_bytes();previous=json.loads(original)
        actual=json.loads(json.dumps(asdict(config)))
        changes={key:{'before':previous['config'].get(key),'after':value} for key,value in actual.items() if previous['config'].get(key)!=value}
        if set(changes)-RESOURCE_FIELDS:raise ValueError('resume migration would change learning settings')
        if changes:
            old_sha=hashlib.sha256(original).hexdigest()
            archive=manifest.with_name(f'experiment.before_resources.{old_sha[:12]}.json')
            if not archive.exists():archive.write_bytes(original)
            previous['config']=actual;curriculum.write_json(manifest,previous)
            record={'utc':datetime.now(timezone.utc).isoformat(),'changes':changes,'before_sha256':old_sha,
                'after_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'target_timesteps':previous['target_timesteps'],
                'note':'Only process/thread counts changed; environment count, rollout, rewards, optimizer and cumulative target are preserved.'}
            with manifest.with_name('resource-migrations.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    return config


def train_run(root,label):
    from mahjong_sandbox.rl.checkpoint import source_fingerprint
    from mahjong_sandbox.rl.longrun import run
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    if plan['source_sha256']!=source_fingerprint() or plan['runner_sha256']!=hashlib.sha256(Path(curriculum.__file__).read_bytes()).hexdigest():
        raise ValueError('runtime training source differs from frozen plan')
    version=label.split('_')[0]
    if not json.loads((root/f'bc_{version}/acceptance.json').read_text())['passed']:
        raise ValueError('coldstart fidelity gate failed')
    config=configuration(root,label,migrate=True)
    curriculum.write_json(root/f'{label}.json',asdict(config))
    recover_finished_training(root,label,config)
    return run(config,root/label,root/f'bc_{version}/warmstart.zip',additional_steps=plan['ppo_steps_per_run'],evaluate=False)
