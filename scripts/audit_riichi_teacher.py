"""Audit fitted policies on one fixed, held-out greedy decision dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from mahjong_sandbox.rl.checkpoint import load_checkpoint
    from mahjong_sandbox.rl.distillation import audit_policy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path);parser.add_argument('--models',required=True,nargs='+')
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    manifest=args.dataset/'manifest.json';expected=json.loads(manifest.read_text())['plan']['rules']
    result={'format':'riichi_teacher_policy_audit_v1','dataset_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'models':{}}
    for value in args.models:
        label,path=value.split('=',1);model,metadata=load_checkpoint(path,device='cpu')
        if metadata['rules']!=expected:raise ValueError('teacher dataset and model rules differ')
        audit=audit_policy(model,[args.dataset])
        result['models'][label]={'model_sha256':metadata['model_sha256'],'feature_version':metadata['feature_version'],
            'timesteps':metadata['timesteps'],'audit':audit}
        print(json.dumps({'label':label,'teacher_set_accuracy':{k:v['teacher_set_accuracy'] for k,v in audit.items()}}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
