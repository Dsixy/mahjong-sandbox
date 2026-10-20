"""Compare frozen policies on the same held-out, public decision positions.

A stratified diagnostic bank measures decisions, not game win/placement rates.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gzip,hashlib,json,random,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def opportunities(obs):
    kinds={x['kind'] for x in obs['legal_actions']};result=[]
    if kinds.intersection(('ron','tsumo')):result.append('win')
    if 'riichi' in kinds:result.append('riichi')
    if kinds.intersection(('chi','peng','open_kong')):result.append('call')
    if not obs['players'][obs['player']]['riichi']:
        discards={x['tiles'][0]//4 for x in obs['legal_actions'] if x['kind'] in ('discard','riichi')}
        threats=[{x['tile']//4 for x in p['river']} for p in obs['players'] if p['player']!=obs['player'] and p['riichi']]
        if len(discards)>=2 and threats and discards.intersection(set.intersection(*threats)):result.append('threat')
    return result


def collect(output,blocks,seed,cap):
    from functools import partial
    from mahjong_sandbox.riichi_match import RiichiMatch,MatchConfig
    from mahjong_sandbox.fast import FastRiichiEngine,RuntimeOptions
    from mahjong_sandbox.bots import make_bot
    from mahjong_sandbox.model import Phase
    from mahjong_sandbox.rl.riichi_evaluate import _context
    from mahjong_sandbox.rl.checkpoint import source_fingerprint
    reservoir=defaultdict(list);seen=Counter();rng=random.Random(seed)
    for logical in range(seed,seed+blocks):
        match=RiichiMatch(MatchConfig(length='south'),seed=logical*256,record=False,
            engine_factory=partial(FastRiichiEngine,runtime=RuntimeOptions(operator_backend='numba')))
        bots=[make_bot('greedy',variant='riichi',seed=logical*4+p,backend='numba') for p in range(4)]
        hands=0
        while True:
            engine=match.engine
            while engine.state.phase!=Phase.TERMINAL:
                if engine.command_count>=2000:raise RuntimeError('nonterminating probe hand')
                p=engine.next_player();obs=engine.observe(p);tags=opportunities(obs)
                if tags and len(obs['legal_actions'])>1:
                    assert all('concealed' not in x for x in obs['players'] if x['player']!=p)
                    # Only the announced rule event is read by the feature contract.
                    obs['events']=[e for e in obs['events'] if e['kind']=='round_started']
                    context=_context(obs)
                    item={'id':f'{logical}:{hands}:{obs["decision_id"]}:{p}', 'seed':logical,'context':context,'opportunities':tags,'observation':obs}
                    for tag in tags:
                        key=context+'_'+tag;seen[key]+=1;pool=reservoir[key]
                        if len(pool)<cap:pool.append(item)
                        else:
                            index=rng.randrange(seen[key])
                            if index<cap:pool[index]=item
                action=bots[p].act(obs);engine.submit(p,action,engine.state.decision_id)
            hands+=1;match.complete_hand()
            if match.ended:break
            if hands>=128:raise RuntimeError('probe match too long')
            match.next_hand()
        print(json.dumps({'collected_block':logical,'hands':hands}),flush=True)
    unique={x['id']:x for pool in reservoir.values() for x in pool}
    result={'format':'riichi_public_decision_bank_v1','seed_range':[seed,seed+blocks-1],'source_sha256':source_fingerprint(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'selection':'reservoir per context/opportunity; overlapping positions deduplicated; diagnostic frequencies are not game frequencies',
        'cap_per_stratum':cap,'eligible_counts':dict(seen),'selected_counts':{k:len(v) for k,v in reservoir.items()},'records':list(unique.values())}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as file:
        with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as archive:archive.write(json.dumps(result,separators=(',',':')).encode())
    manifest={k:v for k,v in result.items() if k!='records'};manifest.update(positions=len(unique),bank_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_name(output.name.removesuffix('.json.gz')+'-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def evaluate(bank,models,output):
    import numpy as np
    import torch
    from mahjong_sandbox.rl.agent import RLAgent
    from mahjong_sandbox.rl.features import observation_batch,encode_batch
    from mahjong_sandbox.rl.riichi_evaluate import _context_choice,_threat_choice
    from mahjong_sandbox.model import Action
    torch.set_num_threads(1)
    with gzip.open(bank,'rt') as file:data=json.load(file)
    positions=data['records'];agents={label:RLAgent(path,backend='numba',device='cpu') for label,path in models.items()}
    choices={k:[] for k in agents};counts={k:Counter() for k in agents};rows={k:[] for k in agents}
    probabilities={k:defaultdict(lambda:[0,0.]) for k in agents}
    first=next(iter(agents.values()));rules=first.rules
    if any(a.rules!=rules or a.metadata['hints']!=first.metadata['hints'] for a in agents.values()):
        raise ValueError('shared probe requires matching rules and feature settings')
    for start in range(0,len(positions),64):
        batch=positions[start:start+64];singles=[observation_batch(x['observation']) for x in batch]
        features=encode_batch({k:np.concatenate([x[k] for x in singles]) for k in singles[0]},rules=rules,backend='numba',hints=first.metadata['hints'])
        for label,agent in agents.items():
            indices,_=agent.model.predict(features,action_masks=features['mask'],deterministic=True)
            with torch.no_grad():
                tensors,_=agent.model.policy.obs_to_tensor(features)
                probs=agent.model.policy.get_distribution(tensors,action_masks=features['mask']).distribution.probs.cpu().numpy()
            assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1,atol=1e-5)
            for item,index,distribution in zip(batch,indices,probs):
                obs=item['observation'];action=Action.from_dict(obs['legal_actions'][int(index)])
                own=Counter();_context_choice(obs,action,own);_threat_choice(obs,action,own,'numba')
                soft={}
                for group,kinds in [('riichi',('riichi',)),('win',('ron','tsumo')),('call',('chi','peng','open_kong'))]:
                    columns=[j for j,a in enumerate(obs['legal_actions']) if a['kind'] in kinds]
                    if columns:
                        probability=float(distribution[columns].sum());soft[group]=probability
                        accumulator=probabilities[label][item['context']+'_'+group]
                        accumulator[0]+=1;accumulator[1]+=probability
                if start==0 and len(choices[label])<4 and agent.act(obs).to_dict()!=action.to_dict():
                    raise AssertionError('batch probe and single-agent inference disagree')
                counts[label].update(own);rows[label].append({'id':item['id'],'seed':item['seed'],'counts':dict(own),'action':action.to_dict(),'choice_probabilities':soft})
                choices[label].append(action.to_dict())
    comparison={}
    labels=list(agents)
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            different=[x!=y for x,y in zip(choices[a],choices[b])]
            comparison[a+' vs '+b]={'different_actions':sum(different),'positions':len(different),
                'by_context':{c:{'different':sum(flag for x,flag in zip(positions,different) if x['context']==c),'positions':sum(x['context']==c for x in positions)} for c in sorted({x['context'] for x in positions})}}
    result={'format':'riichi_shared_position_probe_v1','bank_sha256':hashlib.sha256(Path(bank).read_bytes()).hexdigest(),
        'positions':len(positions),'seed_range':data['seed_range'],'selection':data['selection'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'batch_single_agent_checks_per_model':4,
        'probability_note':'Post-hoc diagnostic on the fixed bank: mean masked policy mass for each available action group, not match action rates.',
        'choice_probabilities':{label:{key:{'positions':v[0],'mean_probability':v[1]/v[0]} for key,v in groups.items()} for label,groups in probabilities.items()},
        'models':{k:{'sha256':a.metadata['model_sha256'],'timesteps':a.metadata['timesteps']} for k,a in agents.items()},
        'counts':{k:dict(v) for k,v in counts.items()},'action_disagreement':comparison,'decision_records':rows}
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='decision_records'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect',action='store_true');p.add_argument('--bank',required=True);p.add_argument('--output');p.add_argument('--blocks',type=int,default=32);p.add_argument('--seed',type=int,default=700000);p.add_argument('--cap',type=int,default=200);p.add_argument('--models',nargs='+')
    a=p.parse_args()
    if a.collect:return collect(a.bank,a.blocks,a.seed,a.cap)
    if not a.models or not a.output:p.error('evaluation needs --models label=path ... and --output')
    return evaluate(a.bank,dict(x.split('=',1) for x in a.models),a.output)
if __name__=='__main__':main()
