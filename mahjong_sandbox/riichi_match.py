"""East / South match controller. Rounds remain independently replayable."""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from .model import Phase
from .riichi_engine import RiichiEngine
from .riichi_model import RiichiConfig
from .replay import export_replay, replay


@dataclass(frozen=True)
class MatchConfig:
    rule_version: str = 'riichi_match_v1'
    length: str = 'south'
    target: int = 30000
    bankruptcy: bool = True
    agari_yame: bool = True
    tenpai_yame: bool = False
    west_extension: bool = True

    def __post_init__(self):
        if self.rule_version != 'riichi_match_v1' or self.length not in ('east', 'south'):
            raise ValueError('unsupported match profile')
        if type(self.target) is not int or self.target <= 0 or self.target % 100:
            raise ValueError('target must be a positive hundred-point score')
        for key in ('bankruptcy', 'agari_yame', 'tenpai_yame', 'west_extension'):
            if type(getattr(self, key)) is not bool: raise ValueError(f'{key} must be boolean')


class RiichiMatch:
    def __init__(self, config=None, rules=None, seed=0, *, engine_factory=RiichiEngine, record=True):
        self.config, self.rules, self.seed = config or MatchConfig(), rules or RiichiConfig(), seed
        if not isinstance(self.config, MatchConfig) or not isinstance(self.rules, RiichiConfig):
            raise ValueError('match config and riichi rules required')
        if self.rules.round_wind != 27 or self.rules.hand_number != 1:
            raise ValueError('a match must start at East 1')
        self.engine_factory, self.record = engine_factory, record
        self.engine = self.engine_factory(self.rules, seed)
        self.rounds = []
        self.ended = False
        self.final_scores = None
        self.end_reason = None
        self._resolved = False

    def summary(self):
        return deepcopy({'length': self.config.length, 'hands_played': len(self.rounds), 'ended': self.ended,
                'final_scores': self.final_scores, 'end_reason': self.end_reason,
                'ranking': sorted(range(4), key=lambda p: (-(self.final_scores or self.engine.state.scores)[p], (p-self.rules.dealer)%4))})

    def complete_hand(self):
        if self.engine.state.phase != Phase.TERMINAL:
            raise ValueError('current hand has not ended')
        if self._resolved: return self.summary()
        self.rounds.append(export_replay(self.engine) if self.record else None)
        self._resolved = True
        s, r, c = self.engine.state, self.engine.config, self.config
        leader = self.summary()['ranking'][0]
        final_wind = 27 if c.length == 'east' else 28
        scheduled_last = r.round_wind == final_wind and r.hand_number == 4
        extended = r.round_wind > final_wind
        reached = max(s.scores) >= c.target
        continuation = s.settlement['dealer_continues']
        win = s.end_reason in ('ron', 'tsumo')
        if c.bankruptcy and min(s.scores) < 0:
            self.end_reason = 'bankruptcy'
        elif scheduled_last and reached and leader == r.dealer and continuation and ((win and c.agari_yame) or (s.end_reason == 'wall_exhausted' and c.tenpai_yame)):
            self.end_reason = 'dealer_finish'
        elif extended and reached:
            self.end_reason = 'extension_target'
        elif not continuation and scheduled_last and (reached or not c.west_extension):
            self.end_reason = 'scheduled_end'
        elif not continuation and extended and r.hand_number == 4:
            self.end_reason = 'extension_end'
        if self.end_reason:
            self.ended = True
            self.final_scores = list(s.scores)
            self.final_scores[leader] += s.riichi_sticks * 1000
        return self.summary()

    def next_hand(self):
        self.complete_hand()
        if self.ended: raise ValueError('match has ended')
        s, r = self.engine.state, self.engine.config
        continuation = s.settlement['dealer_continues']
        wind, number = r.round_wind, r.hand_number
        if not continuation:
            number += 1
            if number == 5: wind, number = wind + 1, 1
        win = s.end_reason in ('ron', 'tsumo')
        rules = replace(r, dealer=r.dealer if continuation else (r.dealer+1)%4,
                        round_wind=wind, hand_number=number, honba=r.honba+1 if continuation or not win else 0,
                        riichi_sticks=s.riichi_sticks, initial_scores=tuple(s.scores))
        self.engine = self.engine_factory(rules, self.seed + len(self.rounds))
        self._resolved = False
        return self.engine

    def export(self):
        if not self.record:
            raise ValueError('match replay requires record mode')
        if not self.ended: raise ValueError('finish the match before exporting it')
        return deepcopy({'format_version': 2, 'kind': 'riichi_match', 'config': asdict(self.config),
                'rules': asdict(self.rules), 'seed': self.seed, 'rounds': self.rounds,
                'expected': self.summary()})


def replay_match(data):
    if not isinstance(data, dict):
        raise ValueError('match replay must be an object')
    if set(data) != {'format_version', 'kind', 'config', 'rules', 'seed', 'rounds', 'expected'} or data['format_version'] != 2 or data['kind'] != 'riichi_match':
        raise ValueError('invalid match replay schema')
    if not isinstance(data['config'], dict) or not isinstance(data['rules'], dict) or set(data['config']) != set(asdict(MatchConfig())) or set(data['rules']) != set(asdict(RiichiConfig())):
        raise ValueError('match requires complete configs')
    if not isinstance(data['rounds'], list) or not 1 <= len(data['rounds']) <= 128:
        raise ValueError('invalid match round count')
    match = RiichiMatch(MatchConfig(**data['config']), RiichiConfig(**data['rules']), data['seed'])
    for index, record in enumerate(data['rounds']):
        engine = replay(record)
        # Check not only each round, but the progression and score/deposit carry.
        if match.engine.config != engine.config or record['seed'] != match.seed + index:
            raise ValueError(f'altered match progression at hand {index}')
        match.engine = engine
        match.complete_hand()
        if index < len(data['rounds'])-1:
            match.next_hand()
    if not match.ended or match.summary() != data['expected']:
        raise ValueError('match replay verification failed')
    return match
