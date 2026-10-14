"""Serial property checks across independent seeded riichi rounds and profiles."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mahjong_sandbox import RiichiEngine,RiichiConfig,RiichiAgent,RandomAgent,Phase,export_replay,replay


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--games',type=int,default=1000);parser.add_argument('--report',type=Path,default=Path('docs/riichi-validation.json'));args=parser.parse_args()
    if args.games<1: parser.error('--games must be positive')
    profiles=[RiichiConfig(),RiichiConfig(red_fives=(),open_tanyao=False),
              RiichiConfig(multi_ron=False,triple_ron_abort=False,abortive_draws=False),
              RiichiConfig(kiriage_mangan=True,double_yakuman=False,counted_yakuman=False,pao=False,delayed_open_kan_dora=False)]
    start=time.perf_counter();reasons=Counter();events=Counter();policies=Counter();replays=0;winners=0
    for seed in range(args.games):
        cfg=replace(profiles[seed%len(profiles)],honba=seed%4,riichi_sticks=seed%3)
        e=RiichiEngine(cfg,seed);policy=('greedy','win-first','random')[seed%3];policies[policy]+=1
        agents=[RiichiAgent(seed*4+p) if policy=='greedy' else RandomAgent(seed*4+p,prefer_win=policy=='win-first') for p in range(4)]
        for _ in range(1000):
            if e.state.phase==Phase.TERMINAL: break
            p=e.next_player();obs=e.observe(p);e.submit(p,agents[p].act(obs),obs['decision_id'])
        else: raise AssertionError(f'round {seed} did not terminate')
        e.assert_invariants();assert sum(e.state.scores)+e.state.riichi_sticks*1000==100000+cfg.riichi_sticks*1000
        reasons[e.state.end_reason]+=1;events.update(ev.kind for ev in e.events);winners+=len(e.state.wins)
        if seed%50==0:
            assert replay(export_replay(e)).state_hash()==e.state_hash();replays+=1
        if (seed+1)%100==0:print(f'{seed+1}/{args.games} verified',flush=True)
    report={'games':args.games,'profiles':len(profiles),'policies':dict(policies),'end_reasons':dict(reasons),
            'winners':winners,'events':dict(events),'replays_verified':replays,'elapsed_seconds':round(time.perf_counter()-start,3),
            'inventory_and_points_conserved':True,'terminated':True,'execution':'serial CPU reference engine'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
