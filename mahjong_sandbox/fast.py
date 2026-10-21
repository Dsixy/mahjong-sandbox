"""Lean rule-engine runtime: incremental counts, decision caches and trace controls.

The transition and scoring rules are inherited from the reference engines.
Record mode is replay-compatible; rollout mode drops private replay/command
history while riichi retains its compact public-information journal.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict,dataclass
from .engine import Engine,IllegalAction
from .riichi_engine import RiichiEngine
from .model import Action,Phase,Transition
from .tiles import counts_of,kind_of
from .operators import structural_waits,native_available


class TileList(list):
    """Physical-tile sequence with a maintained 34-kind count vector."""
    def __init__(self,tiles=()):
        values=list(tiles);super().__init__(values);self.counts=list(counts_of(values))
    def append(self,tile):
        kind=kind_of(tile);super().append(tile);self.counts[kind]+=1
    def extend(self,tiles):
        values=list(tiles)
        for tile in values:self.append(tile)
    def remove(self,tile):
        super().remove(tile);self.counts[kind_of(tile)]-=1
    def pop(self,index=-1):
        tile=super().pop(index);self.counts[tile//4]-=1;return tile
    def insert(self,index,tile):
        kind=kind_of(tile);super().insert(index,tile);self.counts[kind]+=1
    def clear(self):
        super().clear();self.counts=[0]*34
    def __setitem__(self,index,value):
        if isinstance(index,slice):value=list(value)
        new=list(self);new[index]=value;counts=counts_of(new)
        super().__setitem__(index,value);self.counts=list(counts)
    def __delitem__(self,index):
        super().__delitem__(index);self.counts=list(counts_of(self))
    def __iadd__(self,values):self.extend(values);return self
    def __imul__(self,n):
        new=list(self)*n;counts=counts_of(new);super().__imul__(n);self.counts=list(counts);return self
    def __deepcopy__(self,memo):
        result=type(self)(self);memo[id(self)]=result;return result


@dataclass(frozen=True)
class RuntimeOptions:
    record:bool=False
    validate_every:int=0
    operator_backend:str='auto'
    def __post_init__(self):
        if type(self.record) is not bool or type(self.validate_every) is not int or self.validate_every<0:
            raise ValueError('invalid runtime controls')
        if self.operator_backend not in ('auto','python','numba'):raise ValueError('unknown operator backend')


class _LeanRuntime:
    def __init__(self,*args,runtime=None,**kwargs):
        self.runtime=runtime or RuntimeOptions()
        self.event_counts=Counter();self.command_count=0
        self._legal_cache={};self._cache_key=None
        self._start_event=None
        super().__init__(*args,**kwargs)
        for hand in self.state.players:hand.concealed=TileList(hand.concealed)

    def _emit(self,kind,data,*,private=None,visible_to=None):
        self.event_counts[kind]+=1
        if kind=='round_started':self._start_event={'seq':0,'kind':kind,'data':deepcopy(data)}
        if self.runtime.record:super()._emit(kind,data,private=private,visible_to=visible_to)
        elif isinstance(self,RiichiEngine):
            from .public_history import remember_public
            remember_public(self,kind,data)

    def legal_actions(self,player):
        if type(player) is not int or not 0<=player<4:raise ValueError('invalid player')
        key=(self.state.revision,self.state.decision_id)
        if key!=self._cache_key:self._legal_cache={};self._cache_key=key
        if player not in self._legal_cache:self._legal_cache[player]=super().legal_actions(player)
        return self._legal_cache[player]

    def assert_invariants(self,*,force=False):
        every=self.runtime.validate_every
        if force or self.state.revision==0 or self.state.phase==Phase.TERMINAL or every and self.state.revision%every==0:
            super().assert_invariants()
            for hand in self.state.players:
                if isinstance(hand.concealed,TileList) and tuple(hand.concealed.counts)!=counts_of(hand.concealed):
                    raise AssertionError('incremental counts disagree with physical tiles')

    def submit(self,player,action,decision_id):
        state=self.state
        if type(player) is not int or not 0<=player<4 or type(decision_id) is not int or decision_id!=state.decision_id:
            raise IllegalAction('invalid player or stale decision_id')
        if not isinstance(action,Action) or action not in self.legal_actions(player):raise IllegalAction('action is not legal')
        before,scores=len(self.events),list(state.scores)
        if self.runtime.record:self.commands.append({'player':player,'decision_id':decision_id,'action':action.to_dict()})
        self.command_count+=1;state.revision+=1;state.replies[player]=action
        if state.phase==Phase.SELF_TURN:self._resolve_self(action)
        elif len(state.replies)==len(state.responders):
            if state.phase==Phase.EXCHANGE:self._resolve_exchange()
            elif state.phase==Phase.DECLARE_SUIT:
                for seat,reply in state.replies.items():state.players[seat].missing_suit=reply.suit
                self._emit('missing_suits_declared',{'suits':[p.missing_suit for p in state.players]})
                self._window(Phase.SELF_TURN,self.config.dealer,(self.config.dealer,))
            else:self._resolve_responses()
        self.assert_invariants()
        return Transition(tuple(deepcopy(self.events[before:])),tuple(a-b for a,b in zip(state.scores,scores)),state.decision_id,state.phase==Phase.TERMINAL)

    def observe(self,player,*,history=False):
        if history:
            if not self.runtime.record:raise ValueError('history requires record runtime')
            return super().observe(player)
        if type(player) is not int or not 0<=player<4:raise ValueError('invalid player')
        state=self.state;players=[]
        for seat,hand in enumerate(state.players):
            item={'player':seat,'concealed_count':len(hand.concealed),
                  'melds':[{'kind':m.kind,'tiles':m.tiles,'source_player':m.source_player,'called_tile':m.called_tile,'added_tile':m.added_tile} for m in hand.melds],
                  'river':[{'tile':d.tile,'tsumogiri':d.tsumogiri,'called_by':d.called_by,'winner_refs':list(d.winner_refs)} for d in hand.river],
                  'missing_suit':hand.missing_suit,'can_act':hand.can_act}
            if seat==player or not hand.can_act:item['concealed']=list(hand.concealed)
            if seat==player:item.update(last_drawn=hand.last_drawn,passed_ron_fan=hand.passed_ron_fan)
            players.append(item)
        obs={'rule_version':self.config.rule_version,'player':player,'phase':state.phase.value,'actor':state.actor,
             'decision_id':state.decision_id,'remaining_tiles':state.remaining_tiles,'scores':list(state.scores),
             'players':players,'wins':[asdict(w) for w in state.wins],
             'legal_actions':[a.to_dict() for a in self.legal_actions(player)],
             'events':[deepcopy(self._start_event)],'end_reason':state.end_reason,'pending_tile':state.pending_tile}
        if isinstance(self,RiichiEngine):
            for seat,item in enumerate(players):
                hand=state.players[seat]
                item.update(riichi=hand.riichi,double_riichi=hand.double_riichi,riichi_discard=hand.riichi_discard,seat_wind=27+(seat-self.config.dealer)%4)
                if seat==player:item.update(furiten=self._furiten(player),temporary_furiten=hand.temporary_furiten,riichi_furiten=hand.riichi_furiten,ippatsu=hand.ippatsu)
            obs.update(variant='riichi',dealer=self.config.dealer,round_wind=self.config.round_wind,hand_number=self.config.hand_number,
                       honba=self.config.honba,riichi_sticks=state.riichi_sticks,dora_indicators=self.indicators(),red_fives=list(self.config.red_fives),
                       ura_indicators=self.indicators(True) if state.phase==Phase.TERMINAL and any(p.riichi and not p.can_act for p in state.players) else [])
            from .public_history import observation_context
            obs.update(deepcopy(observation_context(self)))
        else:obs.update(variant='sichuan',dealer=self.config.dealer,
                        exchange_offset=self.exchange_offset if self.event_counts['exchange_completed'] else None)
        return obs

    @staticmethod
    def _counts(tiles):
        return tuple(tiles.counts) if isinstance(tiles,TileList) else counts_of(tiles)


class FastEngine(_LeanRuntime,Engine):
    pass


class FastRiichiEngine(_LeanRuntime,RiichiEngine):
    def _waits_for(self,player,tiles=None,melds=None):
        hand=self.state.players[player]
        tiles=hand.concealed if tiles is None else tiles;melds=hand.melds if melds is None else melds
        counts=self._counts(tiles);owned=list(counts)
        for meld in melds:
            for t in meld.tiles:owned[t//4]+=1
        return tuple(k for k in structural_waits(counts,len(melds),backend=self.runtime.operator_backend) if owned[k]<4)

    def _possible_win(self,counts,melds):
        if self.runtime.operator_backend=='numba' or self.runtime.operator_backend=='auto' and native_available():
            from ._kernels import is_win
            from .operators import numpy
            return bool(is_win(numpy().array(counts,dtype='int16'),melds,True,False))
        from .riichi_scoring import shapes
        return bool(shapes(counts,melds))

    def _self_value(self,player):
        hand=self.state.players[player]
        if not self._possible_win(self._counts(hand.concealed),len(hand.melds)):return None
        return super()._self_value(player)

    def _ron_value(self,player):
        hand=self.state.players[player];tile=self.state.pending_tile
        if tile is None:return None
        counts=list(self._counts(hand.concealed));counts[tile//4]+=1
        if counts[tile//4]>4 or not self._possible_win(tuple(counts),len(hand.melds)):return None
        return super()._ron_value(player)
