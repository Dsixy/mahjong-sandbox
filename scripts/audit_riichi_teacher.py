"""Audit fitted policies on one fixed, held-out greedy decision dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def error_types(model,dataset,feature_version):
    from collections import Counter
    import numpy as np
    import torch
    from mahjong_sandbox.batch import ACTION_KINDS
    from mahjong_sandbox.rl.distillation import read_shard,dataset_paths,GROUPS
    counts=Counter();by_group=Counter();kind_changes=Counter()
    discards=[ACTION_KINDS.index(name) for name in ('discard','riichi')]
    paths,_=dataset_paths([dataset])
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for path in paths:
            features,metadata=read_shard(path,feature_version)
            for start in range(0,len(metadata['labels']),256):
                ids=np.arange(start,min(start+256,len(metadata['labels'])))
                observation,_=model.policy.obs_to_tensor({k:v[ids] for k,v in features.items()})
                distribution=model.policy.get_distribution(observation,action_masks=features['mask'][ids]).distribution
                chosen=distribution.probs.argmax(-1).cpu().numpy();rows=np.arange(len(ids))
                targets=metadata['acceptable'][ids];wrong=~targets[rows,chosen]
                codes=features['candidates'][ids,:,0]
                chosen_codes=codes[rows,chosen];teacher_codes=codes[rows,metadata['labels'][ids]]
                hints=features['hints'][ids]
                for row in rows[wrong]:
                    counts['teacher_set_disagreements']+=1;by_group[GROUPS[int(metadata['groups'][ids[row]])]]+=1
                    kind_changes[ACTION_KINDS[int(teacher_codes[row])]+' -> '+ACTION_KINDS[int(chosen_codes[row])]]+=1
                compatible=np.isin(chosen_codes,discards)&np.isin(teacher_codes,discards)
                counts['discard_decisions_with_discard_teacher']+=int(compatible.sum())
                target_distance=np.where(targets,hints[:,:,0],np.inf).min(1)
                chosen_distance=hints[rows,chosen,0]
                distance_backoff=compatible&(chosen_distance>target_distance+1e-6)
                counts['discard_worsens_shanten_vs_teacher']+=int(distance_backoff.sum())
                same_distance=compatible&np.isclose(chosen_distance,target_distance,atol=1e-6)
                target_ukeire=np.where(targets,hints[:,:,1],-np.inf).max(1)
                counts['lower_ukeire_at_equal_shanten']+=int((same_distance&(hints[rows,chosen,1]+1e-6<target_ukeire)).sum())
                counts['shanten_backoff_with_more_genbutsu_than_teacher']+=int((distance_backoff&
                    (hints[rows,chosen,9]>hints[rows,metadata['labels'][ids],9]+1e-6)).sum())
                counts['missed_legal_win']+=int((wrong&(metadata['groups'][ids]==GROUPS.index('win'))).sum())
                counts['samples']+=len(ids)
    return {'counts':dict(counts),'disagreements_by_group':dict(by_group),'action_kind_changes':dict(kind_changes),
        'note':'Differences from the heuristic teacher, not certified tactical mistakes. After RL, shanten backoff can be deliberate defense; genbutsu is only a limited safety proxy.'}


def main():
    import torch
    from mahjong_sandbox.rl.checkpoint import load_checkpoint
    from mahjong_sandbox.rl.distillation import audit_policy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path);parser.add_argument('--models',required=True,nargs='+')
    parser.add_argument('--error-analysis',action='store_true')
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    manifest=args.dataset/'manifest.json';expected=json.loads(manifest.read_text())['plan']['rules']
    result={'format':'riichi_teacher_policy_audit_v1','dataset_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'models':{}}
    for value in args.models:
        label,path=value.split('=',1);model,metadata=load_checkpoint(path,device='cpu',n_steps=1,n_envs=1)
        if metadata['rules']!=expected:raise ValueError('teacher dataset and model rules differ')
        audit=audit_policy(model,[args.dataset])
        result['models'][label]={'model_sha256':metadata['model_sha256'],'feature_version':metadata['feature_version'],
            'timesteps':metadata['timesteps'],'audit':audit}
        if args.error_analysis:result['models'][label]['error_analysis']=error_types(model,args.dataset,metadata['feature_version'])
        print(json.dumps({'label':label,'teacher_set_accuracy':{k:v['teacher_set_accuracy'] for k,v in audit.items()}}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
