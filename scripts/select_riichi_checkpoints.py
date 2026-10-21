"""Select the predeclared global-step candidates, including after interruption.

Live callback counters can restart after a resumed rollout. Re-evaluate the same
planned four global checkpoints for every run before freezing the test models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def _publish(checkpoint,destination,stats,candidates_sha):
    metadata=json.loads(checkpoint.with_suffix('.json').read_text())
    metadata['training']={**metadata['training'],'validation_score':stats['rank_utility'],
        'validation_metric':'rank_utility','global_validation_candidates_sha256':candidates_sha}
    temporary=destination.with_name('best.selection.zip')
    temporary_json=temporary.with_suffix('.json')
    shutil.copyfile(checkpoint,temporary);temporary_json.write_text(json.dumps(metadata,indent=2)+'\n')
    os.replace(temporary,destination);os.replace(temporary_json,destination.with_suffix('.json'))


def _artifact_choice(root,result):
    from scripts import run_riichi_curriculum as curriculum
    label=max(result['runs'],key=lambda name:result['runs'][name]['selected']['statistics']['rank_utility'])
    entry=result['runs'][label]['selected']
    choice={'format':'riichi_validation_best_artifact_v1','label':label,'checkpoint':entry['path'],
        'sha256':entry['sha256'],'timesteps':entry['step'],'validation_rank_utility':entry['statistics']['rank_utility'],
        'selection':'Highest validation rank utility among all fitted RL seeds; earlier listed label wins a tie.',
        'validation_candidates_sha256':result['candidates_sha256'],
        'note':'Artifact selection only; match strength and the V2 release gate are assessed independently.'}
    path=root/'artifact-model.json'
    if path.exists():
        if json.loads(path.read_text())!=choice:raise ValueError('validation-best artifact changed')
    else:
        if (root/'test-models.json').exists():raise ValueError('cannot first choose an artifact after opening the test set')
        curriculum.write_json(path,choice)


def select(root,workers=8):
    import torch
    from scripts import run_riichi_curriculum as curriculum
    from scripts.riichi_runtime import configuration
    from mahjong_sandbox.rl.checkpoint import read_metadata,load_checkpoint,source_fingerprint
    from mahjong_sandbox.rl.riichi_evaluate import validate_model
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    if plan['source_sha256']!=source_fingerprint():raise ValueError('selection source differs from frozen plan')
    labels=['v1_'+str(plan['train_seeds'][0]),*['v2_'+str(s) for s in plan['train_seeds']]]
    frozen={};configs={}
    for label in labels:
        config=configuration(root,label);configs[label]=config
        if read_metadata(root/label/'latest.zip')['timesteps']!=plan['ppo_steps_per_run']:
            raise ValueError('all training targets must finish before global checkpoint selection')
        interval=config.n_envs*config.rollout_steps*config.validation_every
        steps=sorted(set([*range(interval,plan['ppo_steps_per_run']+1,interval),plan['ppo_steps_per_run']]))
        entries=[]
        for step in steps:
            path=root/label/'checkpoints'/f'step_{step}.zip';metadata=read_metadata(path,expected_rules=config.rules)
            if metadata['timesteps']!=step:raise ValueError('checkpoint global step differs from its file name')
            entries.append({'step':step,'path':str(path),'sha256':metadata['model_sha256']})
        frozen[label]=entries
    candidates={'format':'riichi_global_validation_candidates_v1','validation_seed':plan['checkpoint_validation_seed'],
        'validation_blocks':plan['checkpoint_validation_blocks'],'runs':frozen,
        'criterion':'same predeclared global steps for every seed; maximize rank utility; earlier global step wins a tie'}
    receipt=root/'validation-candidates.json'
    if receipt.exists() and json.loads(receipt.read_text())!=candidates:raise ValueError('validation candidate set changed')
    curriculum.write_json(receipt,candidates);digest=hashlib.sha256(receipt.read_bytes()).hexdigest()
    destination=root/'validation-selection.json'
    if destination.exists():
        result=json.loads(destination.read_text())
        if result['candidates_sha256']!=digest:raise ValueError('validation selection receipt changed')
        for label,details in result['runs'].items():
            _publish(Path(details['selected']['path']),root/label/'best.zip',details['selected']['statistics'],digest)
        _artifact_choice(root,result)
        return result
    torch.set_num_threads(2);result={'format':'riichi_global_validation_selection_v1','candidates_sha256':digest,'runs':{}}
    started=time.perf_counter()
    for label,entries in frozen.items():
        scores=[];config=configs[label]
        for entry in entries:
            # Inference needs no N-environment training buffer. This changes
            # neither parameters nor optimizer state; original files are copied
            # unchanged when publishing the selected model.
            model,metadata=load_checkpoint(entry['path'],device='cpu',n_steps=1,n_envs=1)
            stats=validate_model(model,metadata,blocks=config.validation_blocks,seed=config.validation_seed,
                backend=config.backend,workers=min(workers,8))
            scores.append({**entry,'statistics':stats})
            print(json.dumps({'event':'global_checkpoint_validation','label':label,'step':entry['step'],
                'mean_rank':stats['mean_rank'],'mean_net_score':stats['mean_net_score']}),flush=True)
            del model
        selected=max(scores,key=lambda x:x['statistics']['rank_utility'])
        _publish(Path(selected['path']),root/label/'best.zip',selected['statistics'],digest)
        result['runs'][label]={'candidates':scores,'selected':selected}
    result['elapsed_seconds']=time.perf_counter()-started
    curriculum.write_json(destination,result);_artifact_choice(root,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();select(args.root,args.workers)

if __name__=='__main__':main()
