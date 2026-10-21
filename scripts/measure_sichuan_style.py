"""Replay the promoted Sichuan checkpoint and audit its observable playing style."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import torch
from mahjong_sandbox.bots import make_bot
from mahjong_sandbox.fast import FastEngine,RuntimeOptions
from mahjong_sandbox.model import Phase,RuleConfig
from mahjong_sandbox.rl.agent import RLAgent
from mahjong_sandbox.rl.checkpoint import read_metadata

_AGENT=None
_EXPECTED={}

def initialize(model,expected):
    global _AGENT,_EXPECTED
    torch.set_num_threads(1);_AGENT=RLAgent(model,backend='numba');_EXPECTED=expected


def play(work):
    seed,seat,policy=work
    engine=FastEngine(RuleConfig(),seed=seed,runtime=RuntimeOptions(operator_backend='numba'))
    bots=[make_bot('greedy',variant='sichuan',seed=seed*4+p,backend='numba') for p in range(4)]
    if policy=='rl':bots[seat]=_AGENT
    while engine.state.phase!=Phase.TERMINAL:
        p=engine.next_player();engine.submit(p,bots[p].act(engine.observe(p)),engine.state.decision_id)
    engine.assert_invariants(force=True)
    wins=[w for w in engine.state.wins if w.player==seat];own=engine.state.players[seat]
    row={'seed':seed,'seat':seat,'policy':policy,'net_score':engine.state.scores[seat],
         'win':int(bool(wins)),'deal_in':int(any(w.source_player==seat for w in engine.state.wins)),
         'open':int(any(m.kind!='concealed_kong' for m in own.melds)),
         'kongs':sum(m.kind in ('open_kong','concealed_kong','added_kong') for m in own.melds),
         'self_draw':sum(w.self_draw for w in wins),'win_fan':sum(w.value.fan for w in wins),
         'win_income':sum(t.amount for t in engine.state.transfers if t.receiver==seat and t.reason in ('ron','tsumo')),
         'pure_suit':sum(any(i.name=='清一色' for i in w.value.items) for w in wins)}
    key=f'{seed}:{seat}:{policy}'
    if key in _EXPECTED:
        original=_EXPECTED[key]
        assert all(row[k]==original[k] for k in ('win','deal_in','net_score')),(key,row,original)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--previous-evaluation');p.add_argument('--blocks',type=int,default=100);p.add_argument('--seed',type=int,default=300000);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    expected={}
    if a.previous_evaluation:
        r=json.loads(Path(a.previous_evaluation).read_text())
        for x in r['game_records']:
            if x['policy'] not in ('previous_best','greedy'):continue
            label='rl' if x['policy']=='previous_best' else 'greedy'
            expected[f'{x["seed"]}:{x["seat"]}:{label}']=x
    work=[(s,seat,policy) for policy in ('rl','greedy') for s in range(a.seed,a.seed+a.blocks) for seat in range(4)]
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(a.model,expected)) as pool:rows=list(pool.map(play,work,chunksize=4))
    summary={}
    for policy in ('rl','greedy'):
        rr=[r for r in rows if r['policy']==policy];n=len(rr);wins=sum(r['win'] for r in rr)
        summary[policy]={'games':n,'wins':wins,'win_rate':wins/n,'deal_in_rate':sum(r['deal_in'] for r in rr)/n,'mean_net_score':sum(r['net_score'] for r in rr)/n,'open_rate':sum(r['open'] for r in rr)/n,'kongs_per_hand':sum(r['kongs'] for r in rr)/n,'self_draw_share':sum(r['self_draw'] for r in rr)/wins if wins else None,'mean_win_fan':sum(r['win_fan'] for r in rr)/wins if wins else None,'mean_win_income':sum(r['win_income'] for r in rr)/wins if wins else None,'pure_suit_share':sum(r['pure_suit'] for r in rr)/wins if wins else None}
    metadata=read_metadata(a.model)
    result={'variant':'sichuan','model_sha256':metadata['model_sha256'],'timesteps':metadata['timesteps'],'seed':[a.seed,a.seed+a.blocks-1],'reproduced_records':sum(f'{r["seed"]}:{r["seat"]}:{r["policy"]}' in expected for r in rows),'summary':summary,'game_records':rows}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
