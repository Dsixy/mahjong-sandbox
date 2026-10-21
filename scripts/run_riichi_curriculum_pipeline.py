"""Run or resume the complete bounded distillation/self-play experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import run_riichi_curriculum as curriculum


def pipeline(root,workers):
    from mahjong_sandbox.rl.checkpoint import source_fingerprint,read_metadata
    root=Path(root).resolve()
    plan=json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else curriculum.prepare(root,workers=workers)
    if (plan['source_sha256']!=source_fingerprint() or
            plan['runner_sha256']!=hashlib.sha256(Path(curriculum.__file__).read_bytes()).hexdigest()):
        raise ValueError('source or experiment runner differs from frozen plan')
    curriculum.collect(root,'teacher','v2',workers)
    curriculum.collect(root,'validation','v2',min(workers,8))
    for version in ['v1','v2']:
        curriculum.fit(root,version,False)
        curriculum.collect(root,'dagger',version,min(workers,8))
        curriculum.fit(root,version,True)
    for label in ['v1_'+str(plan['train_seeds'][0]),*['v2_'+str(seed) for seed in plan['train_seeds']]]:
        status=curriculum.train_run(root,label)
        if status:return status
    if not (root/'evaluation.json').exists():curriculum.evaluate(root,min(workers,16))
    result=json.loads((root/'evaluation.json').read_text())
    frozen=json.loads((root/'test-models.json').read_text())
    for entry in frozen.values():
        if read_metadata(entry['path'])['model_sha256']!=entry['sha256']:raise ValueError('tested checkpoint changed')
    curriculum.write_json(root/'complete.json',{'format':'riichi_curriculum_complete_v1',
        'source_sha256':source_fingerprint(),'selected_label':result['selected_label'],
        'games':result['games'],'hands':result['hands'],'release_gate':result['release_gate'],
        'evaluation_sha256':hashlib.sha256((root/'evaluation.json').read_bytes()).hexdigest()})
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--workers',default=12,type=int)
    args=parser.parse_args();return pipeline(args.root,args.workers)

if __name__=='__main__':raise SystemExit(main())
