"""Bounded pure-rank versus net-points experiment, with two paired seeds.

Keep top-level imports light: spawned environment workers do not import Torch.
"""
import argparse
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(config_path, output, initial):
    from mahjong_sandbox.rl.train import TrainConfig
    from mahjong_sandbox.rl.longrun import run
    from mahjong_sandbox.rl.checkpoint import read_metadata, source_fingerprint
    plan=json.loads((Path(output).parent/'plan.json').read_text())
    if source_fingerprint()!=plan['source_sha256'] or read_metadata(initial)['model_sha256']!=plan['initial_sha256']:
        raise ValueError('training source or warmstart differs from the frozen plan')
    entry=next(x for x in plan['runs'] if x['label']==Path(output).name)
    if hashlib.sha256(Path(config_path).read_bytes()).hexdigest()!=entry['config_sha256']:
        raise ValueError('worker config differs from the frozen experiment plan')
    config = TrainConfig(**json.loads(Path(config_path).read_text()))
    return run(config, output, initial, additional_steps=config.total_timesteps, evaluate=False)


def evaluate(root, *, blocks, workers):
    from mahjong_sandbox.rl.riichi_evaluate import evaluate_models, summarize
    from mahjong_sandbox.rl.checkpoint import read_metadata
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    runs = {x['label']:root/x['label'] for x in plan['runs']}
    paths = {label:directory/'latest.zip' for label,directory in runs.items()}
    for run in plan['runs']:
        if read_metadata(paths[run['label']])['timesteps'] != run['target_steps']:
            raise ValueError('all equal-budget runs must finish before testing')
    rank_best = max((d/'best.zip' for k,d in runs.items() if k.startswith('rank_')),
                    key=lambda p:read_metadata(p)['training']['validation_score'])
    score_best = max((d/'best.zip' for k,d in runs.items() if k.startswith('score_')),
                     key=lambda p:read_metadata(p)['training']['validation_score'])
    paths.update(rank_selected=rank_best, score_selected=score_best, legacy_points=root/'legacy.zip')
    # Plan, model files and selected validation winners are fixed before testing.
    frozen = {label:{'path':str(p),'sha256':read_metadata(p)['model_sha256']} for label,p in paths.items()}
    receipt=root/'test-models.json'
    if receipt.exists() and json.loads(receipt.read_text()) != frozen:
        raise ValueError('test model set changed')
    receipt.write_text(json.dumps(frozen,indent=2)+'\n')
    primary='rank_'+str(plan['seeds'][0])
    additional={k:v for k,v in paths.items() if k!=primary}
    result=evaluate_models(paths[primary], baseline_model=root/'initial.zip', baseline_label='warmstart',
        additional_models=additional, baselines=('greedy','defensive'), blocks=blocks,
        seed=plan['test_seed'], workers=workers, backend='numba', match_length='south')
    # Preserve descriptive seed means; intervals condition on these fitted seeds.
    rows=result['game_records']
    for row in rows:
        if row['policy']=='rl':row['policy']=primary
    matchup=result['matchups']['greedy'];matchup[primary]=matchup.pop('rl')
    aggregates={}
    for mode in ('rank','score'):
        group=[r for r in rows if r['policy'] in [mode+'_'+str(s) for s in plan['seeds']]]
        aggregates[mode]=summarize(group,seed=plan['test_seed'])
    result['models'][primary]=result['models'].pop('rl')
    import numpy as np
    from mahjong_sandbox.rl.riichi_evaluate import _totals, METRICS
    seeds=sorted({x['seed'] for x in rows});indices=np.random.default_rng(plan['test_seed']).integers(0,len(seeds),(2000,len(seeds)))
    grouped={}
    for mode in ('rank','score'):
        labels=[mode+'_'+str(s) for s in plan['seeds']]
        grouped[mode]=[_totals([x for x in rows if x['seed']==seed and x['policy'] in labels]) for seed in seeds]
    paired={}
    for metric,(num,den) in METRICS.items():
        an=np.array([x[num] for x in grouped['rank']]);ad=np.array([x[den] for x in grouped['rank']])
        bn=np.array([x[num] for x in grouped['score']]);bd=np.array([x[den] for x in grouped['score']])
        if not ad.sum() or not bd.sum():continue
        aden=ad[indices].sum(axis=1);bden=bd[indices].sum(axis=1);valid=(aden>0)&(bden>0)
        samples=an[indices].sum(axis=1)[valid]/aden[valid]-bn[indices].sum(axis=1)[valid]/bden[valid]
        paired[metric]={'rank_minus_score':float(an.sum()/ad.sum()-bn.sum()/bd.sum()),'ci95':np.quantile(samples,[.025,.975]).tolist()}
    result['pooled_paired_comparison']=paired
    result['experiment']=plan
    result['pooled_final_models']=aggregates
    result['pooled_interval_note']='Test-wall bootstrap conditional on the two fitted training seeds; not a population CI over retraining.'
    destination=root/'evaluation.json'
    temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(destination)
    print(json.dumps({'evaluation':str(destination),'games':result['games'],'hands':result['hands'],
        'pooled_final_models':aggregates},indent=2))
    return 0


def prepare(root, initial, legacy, *, seeds, steps, workers, n_envs):
    from mahjong_sandbox.rl.train import TrainConfig
    from mahjong_sandbox.rl.checkpoint import read_metadata,source_fingerprint
    import shutil
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    initial=Path(initial).resolve();legacy=Path(legacy).resolve()
    if read_metadata(initial)['timesteps'] != 0:
        raise ValueError('paired experiment starts from a zero-step BC warmstart')
    for src,name in [(initial,'initial.zip'),(legacy,'legacy.zip')]:
        dst=root/name
        if dst.exists() and read_metadata(dst)['model_sha256']!=read_metadata(src)['model_sha256']:
            raise ValueError('initial model changed')
        if src!=dst:
            shutil.copyfile(src,dst);shutil.copyfile(src.with_suffix('.json'),dst.with_suffix('.json'))
    base=Path(__file__).resolve().parents[1]/'configs/rl'
    runs=[];configs={}
    for seed in seeds:
        for mode in ('score','rank'):
            values=json.loads((base/f'riichi-south-{mode}.json').read_text())
            values.update(seed=seed,total_timesteps=steps,workers=workers,n_envs=n_envs)
            config=TrainConfig(**values);label=f'{mode}_{seed}'
            raw=json.dumps(values,indent=2)+'\n';configs[label+'.json']=raw
            runs.append({'label':label,'reward_mode':mode,'seed':seed,'config':label+'.json','target_steps':config.total_timesteps, 'config_sha256':hashlib.sha256(raw.encode()).hexdigest()})
    plan={'format':'riichi_placement_ablation_v1','seeds':seeds,'runs':runs,
        'initial_sha256':read_metadata(root/'initial.zip')['model_sha256'],
        'source_sha256':source_fingerprint(),'match_length':'south',
        'reward_profiles':{'rank':[1.5,.5,-.5,-1.5],'score':'point delta /10000'},
        'main_analysis':'equal-budget final checkpoints, two paired training seeds',
        'secondary_analysis':'best checkpoints chosen on validation only',
        'validation_seeds':[120000,120015],'test_seed':600000,
        'opponents':'fixed greedy/defensive/value mixture; no self-play in this controlled ablation'}
    p=root/'plan.json'
    if p.exists() and json.loads(p.read_text())!=plan:raise ValueError('experiment plan changed')
    for name,raw in configs.items():(root/name).write_text(raw)
    p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan,indent=2))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--initial');p.add_argument('--legacy');p.add_argument('--worker-config')
    p.add_argument('--output');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--steps',type=int,default=524288);p.add_argument('--seeds',nargs=2,type=int,default=[20261,20262])
    p.add_argument('--workers',type=int,default=16);p.add_argument('--n-envs',type=int,default=512)
    p.add_argument('--eval-blocks',type=int,default=256);p.add_argument('--eval-workers',type=int,default=16)
    args=p.parse_args()
    if args.worker_config:return worker(args.worker_config,args.output,args.initial)
    if args.evaluate:return evaluate(args.root,blocks=args.eval_blocks,workers=args.eval_workers)
    return prepare(args.root,args.initial,args.legacy,seeds=args.seeds,steps=args.steps,workers=args.workers,n_envs=args.n_envs)

if __name__=='__main__':raise SystemExit(main())
