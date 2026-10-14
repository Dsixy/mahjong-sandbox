"""Seeded observation-only strategies and accelerated tile-efficiency baselines."""
from .agents import RandomAgent
from .greedy import GreedyAgent
from .riichi_agent import RiichiAgent
from .model import Action
from .operators import shanten,discard_ukeire_batch,native_available

POLICIES=('random','win-first','greedy','defensive','value')


def known_counts(observation, *, announcements=False):
    """Classic policy inventory, optionally including all announced public tiles.

    The reference heuristic uses hands/rivers/melds/dora. Training projections
    also account for pending kan tiles, robbed winning tiles and revealed ura.
    """
    seen=set()
    for p in observation['players']:
        seen.update(p.get('concealed',()))
        seen.update(d['tile'] for d in p['river'])
        for m in p['melds']:seen.update(m['tiles'])
    seen.update(observation.get('dora_indicators',()))
    if announcements:
        seen.update(observation.get('ura_indicators',()))
        seen.update(w['tile'] for w in observation.get('wins',()) if w['tile'] is not None)
        tile=observation.get('pending_tile')
        if tile is not None:seen.add(tile)
    result=[0]*34
    for t in seen:result[t//4]+=1
    return result


class _Accelerated:
    def __init__(self,seed=0,*,backend='auto'):
        super().__init__(seed);self.backend=backend
    def distance(self,counts,meld_count=0,*,seven_pairs_quads=True):
        values=tuple(counts)+((0,)*(34-len(counts)))
        return shanten(values,meld_count,variant=self.variant,seven_pairs_quads=seven_pairs_quads,backend=self.backend)

    def act(self,obs):
        if obs['phase']!='self_turn' or self.backend=='python' or not native_available():return super().act(obs)
        actions=obs['legal_actions']
        if not actions:raise ValueError('agent called without a decision')
        wins=[a for a in actions if a['kind'] in ('ron','tsumo')]
        if wins:return Action.from_dict(wins[0])
        own=obs['players'][obs['player']];counts=[0]*34
        for t in own['concealed']:counts[t//4]+=1
        rules=next(e['data']['rules'] for e in obs['events'] if e['kind']=='round_started')
        quads=rules.get('seven_pairs_quads',True);melds=len(own['melds']);known=known_counts(obs)
        discards=[a for a in actions if a['kind']=='discard']
        if not discards:return super().act(obs)
        import numpy as np
        allowed=np.zeros((1,34),dtype=bool)
        for a in discards:allowed[0,a['tiles'][0]//4]=True
        distances,ukeire=discard_ukeire_batch(np.array([counts]),np.array([known]),melds,variant=self.variant,seven_pairs_quads=quads,backend=self.backend,best_only=True,candidate_mask=allowed)
        distances,ukeire=distances[0],ukeire[0]
        baseline=min(int(distances[a['tiles'][0]//4]) for a in discards)
        kongs=[]
        for a in actions:
            permitted=('concealed_kong','added_kong') if self.variant=='riichi' else ('concealed_kong','added_kong','open_kong')
            if a['kind'] in permitted:
                after=self._remove(counts,a['tiles']);exposed=melds+(a['kind']!='added_kong')
                if self.distance(after,exposed,seven_pairs_quads=quads)<=baseline:kongs.append(a)
        if kongs:return Action.from_dict(self.rng.choice(kongs))
        candidates=[a for a in actions if a['kind']=='riichi'] if self.variant=='riichi' else []
        candidates=candidates or discards
        if self.variant=='sichuan':
            unique={}
            for a in candidates:unique.setdefault(a['tiles'][0]//4,a)
            candidates=list(unique.values())
        best=min(int(distances[a['tiles'][0]//4]) for a in candidates)
        candidates=[a for a in candidates if int(distances[a['tiles'][0]//4])==best]
        red=set(obs.get('red_fives',()))
        def score(a):
            kind=a['tiles'][0]//4
            return (int(ukeire[kind]),a['tiles'][0] not in red) if self.variant=='riichi' else (int(ukeire[kind]),known[kind],int(kind%9 in (0,8)))
        return self._best(candidates,score)


class FastGreedyAgent(_Accelerated,GreedyAgent):variant='sichuan'
class FastRiichiAgent(_Accelerated,RiichiAgent):variant='riichi'


class StrategicAgent:
    """Small distinct baselines: visible safety, or dora/closed-hand value bias."""
    def __init__(self,variant='riichi',seed=0,*,style='defensive',backend='auto'):
        if variant not in ('riichi','sichuan') or style not in ('defensive','value'):raise ValueError('unknown variant or strategy')
        self.base=(FastRiichiAgent if variant=='riichi' else FastGreedyAgent)(seed,backend=backend)
        self.variant,self.style=variant,style
    def act(self,obs):
        actions=obs['legal_actions'];own=obs['players'][obs['player']]
        for a in actions:
            if a['kind'] in ('ron','tsumo'):return Action.from_dict(a)
        threats=[p for p in obs['players'] if p['player']!=obs['player'] and p['can_act'] and
                 (p.get('riichi') if self.variant=='riichi' else len(p['melds'])>=2)]
        if self.style=='defensive' and threats and obs['phase']=='self_turn':
            discards=[a for a in actions if a['kind']=='discard']
            if discards:
                counts=[0]*34
                for t in own['concealed']:counts[t//4]+=1
                known=known_counts(obs)
                def score(a):
                    k=a['tiles'][0]//4
                    # Genbutsu is safe against that riichi player; other guesses are only heuristics.
                    safe=sum(k in {d['tile']//4 for d in p['river']} if self.variant=='riichi' else k//9==p['missing_suit'] for p in threats)
                    depleted=int(known[k]==4)
                    distance=self.base.distance(self.base._remove(counts,a['tiles']),len(own['melds']))
                    return safe,depleted,-distance,int(k>=27)
                return self.base._best(discards,score)
        if self.style=='value' and obs['phase']=='self_turn':
            discards=[a for a in actions if a['kind']=='discard']
            if discards:
                counts=[0]*34
                for t in own['concealed']:counts[t//4]+=1
                melds=len(own['melds']);rules=next(e['data']['rules'] for e in obs['events'] if e['kind']=='round_started')
                quads=rules.get('seven_pairs_quads',True)
                ds={a['tiles'][0]:self.base.distance(self.base._remove(counts,a['tiles']),melds,seven_pairs_quads=quads) for a in discards}
                best=min(ds.values());options=[a for a in actions if a['kind']=='riichi'] or [a for a in discards if ds[a['tiles'][0]]<=best]
                red=set(obs.get('red_fives',()))
                from .riichi_scoring import dora_kind
                dora={dora_kind(t) for t in obs.get('dora_indicators',())}
                known=known_counts(obs)
                def score(a):
                    t=a['tiles'][0];k=t//4
                    value=int(t in red)+int(k in dora)
                    return -ds[t],-value,int(k>=27 and counts[k]==1),known[k]
                return self.base._best(options,score)
        return self.base.act(obs)


def make_bot(policy='greedy',*,variant='riichi',seed=0,backend='auto',reference=False):
    if policy not in POLICIES or variant not in ('sichuan','riichi'):raise ValueError('unknown policy or variant')
    if policy in ('random','win-first'):return RandomAgent(seed,prefer_win=policy=='win-first')
    if policy=='greedy':
        if reference:return (RiichiAgent if variant=='riichi' else GreedyAgent)(seed)
        return (FastRiichiAgent if variant=='riichi' else FastGreedyAgent)(seed,backend=backend)
    return StrategicAgent(variant,seed,style=policy,backend=backend)
