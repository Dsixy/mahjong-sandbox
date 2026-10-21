"""Typed decision batches with one learner seat or four-seat self play.

Rules remain authoritative engines. Buffers are a struct of arrays, never a
projection of the wall or opponents' private state. No implicit autoreset.
"""
from dataclasses import dataclass
from functools import partial
import hashlib
from .bots import make_bot, known_counts
from .fast import FastEngine, FastRiichiEngine, RuntimeOptions
from .model import ActionKind, Phase, RuleConfig
from .riichi_model import RiichiConfig
from .public_history import MAX_PUBLIC_EVENTS, HISTORY_FIELDS, HISTORY_VERSION, RULE_FLAGS
from .operators import numpy

MAX_ACTIONS = 512  # exchange candidates <= C(14,3) = 364
ACTION_KINDS = tuple(a.value for a in ActionKind)
PHASES = tuple(p.value for p in Phase)
MELD_KINDS = ('chi', 'peng', 'open_kong', 'concealed_kong', 'added_kong')
META_FIELDS = ('phase', 'player', 'actor', 'remaining_tiles', 'dealer',
               'round_wind', 'hand_number', 'honba', 'riichi_sticks',
               'pending_tile', 'last_drawn', 'furiten', 'temporary_furiten',
               'riichi_furiten', 'passed_ron_fan', 'ippatsu')
SEAT_FIELDS = ('concealed_count', 'missing_suit', 'can_act', 'riichi',
               'double_riichi', 'riichi_discard', 'seat_wind')


def derived_seed(seed, lane, episode, stream=0):
    """Stable streams independent of sharding, scheduling and process RNGs."""
    words = f'{seed}:{lane}:{episode}:{stream}'.encode('ascii')
    return int.from_bytes(hashlib.blake2b(words, digest_size=8).digest(), 'little')


def array_specs(n, *, public_history=False):
    specs = {
        'own_counts': ((n, 34), 'uint8'),
        'known_counts': ((n, 34), 'uint8'),
        'concealed': ((n, 14), 'int16'),
        'revealed': ((n, 4, 14), 'int16'),
        'seats': ((n, 4, len(SEAT_FIELDS)), 'int32'),
        'rivers': ((n, 4, 136), 'int16'),
        'river_flags': ((n, 4, 136), 'uint8'),
        'river_lengths': ((n, 4), 'uint8'),
        'melds': ((n, 4, 4, 9), 'int16'),
        'meta': ((n, len(META_FIELDS)), 'int32'),
        'scores': ((n, 4), 'int64'),
        'exchange_offsets': ((n,), 'int8'),
        'wins': ((n, 4, 8), 'int32'),
        'dora': ((n, 5), 'int16'),
        'ura': ((n, 5), 'int16'),
        'red_mask': ((n, 136), 'bool'),
        'actions': ((n, MAX_ACTIONS, 8), 'int16'),
        'action_mask': ((n, MAX_ACTIONS), 'bool'),
        'tokens': ((n, 2), 'uint64'),
        'decision_ids': ((n,), 'int64'),
        'rewards': ((n, 4), 'float32'),
        'terminated': ((n,), 'bool'),
        'truncated': ((n,), 'bool'),
    }
    if public_history:
        specs.update({
            'river_callers': ((n,4,136),'int8'), 'river_winners': ((n,4,136),'uint8'),
            'public_history': ((n,MAX_PUBLIC_EVENTS,len(HISTORY_FIELDS)),'int32'),
            'history_lengths': ((n,),'int16'), 'history_complete': ((n,),'bool'),
            'match_context': ((n,12),'int64'), 'rule_options': ((n,len(RULE_FLAGS)+4),'int16')})
    return specs


def allocate(n, *, public_history=False):
    np = numpy()
    return {k: np.zeros(shape, dtype=dtype) for k, (shape, dtype) in array_specs(n,public_history=public_history).items()}


def snapshot(buffers):
    result = {k: v.copy() for k, v in buffers.items()}
    for v in result.values():
        v.flags.writeable = False
    return result


def validate_actions(indices, tokens, batch):
    """Validate the entire submission before changing any lane."""
    np = numpy()
    a, t = np.asarray(indices), np.asarray(tokens)
    n = len(batch['terminated'])
    if a.shape != (n,) or a.dtype.kind not in 'iu':
        raise ValueError('action indices must be an integer [N] array')
    if t.shape != (n, 2) or t.dtype.kind not in 'iu' or not np.array_equal(t, batch['tokens']):
        raise ValueError('stale or invalid episode/decision tokens')
    done = batch['terminated'] | batch['truncated']
    if np.any(a[done] != -1):
        raise ValueError('closed lanes require action -1; reset_done to reuse them')
    active = np.flatnonzero(~done)
    if np.any(a[active] < 0) or np.any(a[active] >= MAX_ACTIONS):
        raise ValueError('action index out of bounds')
    if not np.all(batch['action_mask'][active, a[active]]):
        raise ValueError('masked illegal action')
    return np.asarray(a, dtype=np.int64).copy(), np.asarray(t, dtype=np.uint64).copy()


def encode_observation(obs, actions, b, i=0):
    """Project a player's public observation; never reads an engine/private state."""
    p = obs['player']
    own = obs['players'][p]
    for k in ('concealed', 'revealed', 'rivers', 'melds', 'dora', 'ura', 'actions', 'wins'):
        b[k][i].fill(-1)
    for k in ('own_counts', 'river_flags', 'red_mask', 'action_mask'):
        b[k][i].fill(0)
    extended='public_history' in b
    if extended:
        b['river_callers'][i].fill(-1)
        b['river_winners'][i].fill(0)
        b['public_history'][i].fill(-1)
        b['history_lengths'][i] = 0
        b['history_complete'][i] = False
        b['match_context'][i].fill(0)
        b['rule_options'][i].fill(0)
    tiles = own['concealed']
    b['concealed'][i, :len(tiles)] = tiles
    for t in tiles:
        b['own_counts'][i, t // 4] += 1
    b['known_counts'][i] = known_counts(obs, announcements=True)
    for seat, hand in enumerate(obs['players']):
        b['seats'][i, seat] = [int(hand.get(k, -1)) if hand.get(k) is not None else -1 for k in SEAT_FIELDS]
        revealed = hand.get('concealed', ())
        b['revealed'][i, seat, :len(revealed)] = revealed
        b['river_lengths'][i, seat] = len(hand['river'])
        for j, d in enumerate(hand['river']):
            b['rivers'][i, seat, j] = d['tile']
            b['river_flags'][i, seat, j] = int(d['tsumogiri']) | (int(d['called_by'] is not None) << 1) | (int(bool(d['winner_refs'])) << 2) | (int(d['tile'] == hand.get('riichi_discard')) << 3)
            if extended:
                b['river_callers'][i, seat, j] = d['called_by'] if d['called_by'] is not None else -1
                b['river_winners'][i, seat, j] = sum(1 << winner for winner in d['winner_refs'])
        for j, m in enumerate(hand['melds']):
            row = b['melds'][i, seat, j]
            row[0], row[8] = MELD_KINDS.index(m['kind']), len(m['tiles'])
            row[1:1 + len(m['tiles'])] = m['tiles']
            row[5:8] = [m[k] if m[k] is not None else -1 for k in ('source_player', 'called_tile', 'added_tile')]
    values = {**obs, **{k: own.get(k) for k in META_FIELDS if k in own}}
    values['phase'] = PHASES.index(obs['phase'])
    b['meta'][i] = [int(values.get(k, -1)) if values.get(k) is not None else -1 for k in META_FIELDS]
    b['scores'][i] = obs['scores']
    b['exchange_offsets'][i] = obs.get('exchange_offset') if obs.get('exchange_offset') is not None else -1
    for j, w in enumerate(obs['wins']):
        v = w['value']
        b['wins'][i,j] = [w['player'], w['source_player'] if w['source_player'] is not None else -1,
                          w['tile'] if w['tile'] is not None else -1, int(w['self_draw']),
                          v['fan'], v.get('fu', 0), v['payment'], v.get('yakuman', 0)]
    for field, name in (('dora', 'dora_indicators'), ('ura', 'ura_indicators')):
        indicators = obs.get(name, ())
        b[field][i, :len(indicators)] = indicators
    b['red_mask'][i, obs.get('red_fives', ())] = True
    if extended and obs.get('public_history_version') == HISTORY_VERSION and 'match_context' in obs:
        history = obs['public_history']
        if len(history) > MAX_PUBLIC_EVENTS or any(len(row) != len(HISTORY_FIELDS) for row in history):
            raise ValueError('invalid or overflowing public history')
        b['public_history'][i, :len(history)] = history
        b['history_lengths'][i] = len(history)
        context = obs['match_context'];config = context['config']
        b['match_context'][i] = [0 if config is None else ('hand','east','south').index(config['length']),
            config['target'] if config else 0,
            *[int(config[key]) if config else 0 for key in ('bankruptcy','agari_yame','tenpai_yame','west_extension')],
            context['initial_dealer'], *context['initial_scores'], context['hands_played']]
        announced = next(e['data']['rules'] for e in obs['events'] if e['kind'] == 'round_started')
        red = announced['red_fives']
        b['rule_options'][i] = [*[int(announced[key]) for key in RULE_FLAGS], announced['riichi_min_live'],
            *[sum(t//4 == kind for t in red) for kind in (4,13,22)]]
        b['history_complete'][i] = True
    if len(actions) > MAX_ACTIONS:
        raise AssertionError('action capacity exceeded')
    for j, a in enumerate(actions):
        row = b['actions'][i, j]
        row[0:2] = ACTION_KINDS.index(a.kind.value), len(a.tiles)
        row[2:2 + len(a.tiles)] = a.tiles
        row[6:8] = a.suit if a.suit is not None else -1, a.meld_index if a.meld_index is not None else -1
    b['action_mask'][i, :len(actions)] = True


@dataclass(frozen=True)
class BatchConfig:
    variant: str = 'riichi'
    seed: int = 0
    controlled_player: int | None = 0
    opponent_policy: str = 'greedy'
    backend: str = 'auto'
    validate_every: int = 0
    rules: RuleConfig | RiichiConfig | None = None
    rotate_seats: bool = False
    opponent_pool: tuple[str, ...] = ()
    opponent_manifest: str | None = None
    self_play_probability: float = 0.
    match_length: str = 'hand'
    match_config: object = None
    public_history: bool = False
    skip_forced_actions: bool = False

    def __post_init__(self):
        if self.variant not in ('sichuan', 'riichi') or type(self.seed) is not int:
            raise ValueError('invalid variant or seed')
        if self.match_length not in ('hand', 'east', 'south') or (self.variant != 'riichi' and self.match_length != 'hand'):
            raise ValueError('match episodes require riichi and hand/east/south length')
        if self.match_config is not None:
            from .riichi_match import MatchConfig
            if isinstance(self.match_config, dict):
                object.__setattr__(self, 'match_config', MatchConfig(**self.match_config))
            if not isinstance(self.match_config, MatchConfig) or self.match_config.length != self.match_length:
                raise ValueError('match config must agree with match_length')
        if type(self.skip_forced_actions) is not bool:
            raise ValueError('skip_forced_actions must be boolean')
        if type(self.public_history) is not bool or self.public_history and self.variant!='riichi':
            raise ValueError('public history buffers require riichi')
        if self.controlled_player is not None and (type(self.controlled_player) is not int or not 0 <= self.controlled_player < 4):
            raise ValueError('controlled_player must be a seat or None for self play')
        from .bots import POLICIES
        if self.opponent_policy not in POLICIES:
            raise ValueError('unknown opponent policy')
        if type(self.rotate_seats) is not bool or self.rotate_seats and self.controlled_player is None:
            raise ValueError('seat rotation requires one controlled player')
        if not isinstance(self.opponent_pool, tuple) or any(p not in POLICIES for p in self.opponent_pool):
            raise ValueError('opponent_pool must be a tuple of known policies')
        if not 0 <= self.self_play_probability <= 1:
            raise ValueError('self_play_probability must be in [0,1]')
        if self.opponent_manifest is not None and (not isinstance(self.opponent_manifest, str) or
                self.controlled_player is None):
            raise ValueError('snapshot opponents require a learner and a manifest path')
        RuntimeOptions(validate_every=self.validate_every, operator_backend=self.backend)
        expected = RiichiConfig if self.variant == 'riichi' else RuleConfig
        if self.rules is not None and not isinstance(self.rules, expected):
            raise ValueError('rules do not match variant')


class SyncBatchEnv:
    """N live boards. One step advances each board to its next learner decision.

    Returned snapshots remain valid after later steps. rewards contains raw
    score deltas for all four seats, including opponents' intervening actions.
    A Sichuan learner who has won receives the remaining settlement on the same
    transition. terminated means the entire hand ended, never just a seat won.
    """
    def __init__(self, n=1, config=None, *, lane_offset=0, buffers=None):
        if type(n) is not int or n <= 0 or type(lane_offset) is not int or lane_offset < 0:
            raise ValueError('n and lane_offset must be positive/nonnegative integers')
        self.n, self.config, self.lane_offset = n, config or BatchConfig(), lane_offset
        self.buffers = allocate(n,public_history=self.config.public_history) if buffers is None else buffers
        self.engines = [None] * n
        self.matches = [None] * n
        self.agents = [None] * n
        self.menus = [()] * n
        self.players = [0] * n
        self.learners = [self.config.controlled_player] * n
        self.episodes = [-1] * n
        self.tickets = [0] * n
        self.closed = False
        self.snapshot_pool = None
        if self.config.opponent_manifest:
            from .rl.selfplay import SnapshotPool
            default_rules = RiichiConfig() if self.config.variant == 'riichi' else RuleConfig()
            self.snapshot_pool = SnapshotPool(self.config.opponent_manifest, rules=self.config.rules or default_rules,
                backend=self.config.backend, probability=self.config.self_play_probability)
        self._reset(range(n))

    def _check_open(self):
        if self.closed:
            raise RuntimeError('environment is closed')

    def _advance(self, i):
        while True:
            engine = self.engines[i]
            while engine.state.phase != Phase.TERMINAL:
                p = engine.next_player()
                if self.learners[i] is None or p == self.learners[i]:
                    forced = engine.legal_actions(p) if self.config.skip_forced_actions else ()
                    if len(forced) != 1:
                        self.players[i] = p
                        return
                    action = forced[0]
                else:
                    action = self.agents[i][p].act(engine.observe(p))
                if engine.command_count >= 2000:
                    raise RuntimeError('nonterminating rules loop')
                engine.submit(p, action, engine.state.decision_id)
            match = self.matches[i]
            if match is None:
                break
            match.complete_hand()
            if match.ended:
                break
            if len(match.rounds) >= 128:
                raise RuntimeError('match exceeded 128 hands')
            self.engines[i] = match.next_hand()
        self.players[i] = self.learners[i] if self.learners[i] is not None else self.players[i]

    def _reset(self, indices):
        cls = FastRiichiEngine if self.config.variant == 'riichi' else FastEngine
        if self.snapshot_pool:
            self.snapshot_pool.refresh()
        for i in indices:
            self.episodes[i] += 1
            self.tickets[i] = 0
            lane = self.lane_offset + i
            seed = derived_seed(self.config.seed, lane, self.episodes[i])
            if self.config.rotate_seats:
                self.learners[i] = (self.config.controlled_player + lane + self.episodes[i]) % 4
            factory = partial(cls, runtime=RuntimeOptions(
                validate_every=self.config.validate_every, operator_backend=self.config.backend))
            if self.config.match_length != 'hand':
                from .riichi_match import RiichiMatch, MatchConfig
                self.matches[i] = RiichiMatch(self.config.match_config or MatchConfig(length=self.config.match_length),
                    self.config.rules, seed, engine_factory=factory, record=False)
                self.engines[i] = self.matches[i].engine
            else:
                self.engines[i] = factory(self.config.rules, seed=seed)
            self.agents[i] = [make_bot(
                self.config.opponent_pool[derived_seed(self.config.seed, lane, self.episodes[i], p + 11) % len(self.config.opponent_pool)]
                if self.config.opponent_pool else self.config.opponent_policy, variant=self.config.variant,
                                     seed=derived_seed(self.config.seed, lane, self.episodes[i], p + 1),
                                     backend=self.config.backend) for p in range(4)]
            if self.snapshot_pool:
                for p in range(4):
                    if p != self.learners[i]:
                        self.agents[i][p] = self.snapshot_pool.opponent(
                            derived_seed(self.config.seed, lane, self.episodes[i], p + 31), self.agents[i][p])
            self._advance(i)
            self._encode(i)
            self.buffers['rewards'][i] = 0

    def _encode(self, i):
        b, engine, p = self.buffers, self.engines[i], self.players[i]
        obs = engine.observe(p)
        self.menus[i] = engine.legal_actions(p)
        encode_observation(obs, self.menus[i], b, i)
        if self.matches[i] is not None and self.matches[i].ended:
            b['scores'][i] = self.matches[i].final_scores
        b['tokens'][i] = self.episodes[i], self.tickets[i]
        b['decision_ids'][i] = engine.state.decision_id
        b['terminated'][i] = engine.state.phase == Phase.TERMINAL
        b['truncated'][i] = False

    def observation(self):
        self._check_open()
        return snapshot(self.buffers)

    def _step(self, indices, tokens):
        a, _ = validate_actions(indices, tokens, self.buffers)
        for i in range(self.n):
            self.buffers['rewards'][i] = 0
            if a[i] == -1:
                continue
            engine, p = self.engines[i], self.players[i]
            if engine.command_count >= 2000:
                raise RuntimeError('nonterminating rules loop')
            before = tuple(engine.state.scores)
            engine.submit(p, self.menus[i][a[i]], engine.state.decision_id)
            self._advance(i)
            self.tickets[i] += 1
            self._encode(i)
            self.buffers['rewards'][i] = [now - old for now, old in zip(self.buffers['scores'][i], before)]

    def step(self, indices, tokens):
        self._check_open()
        self._step(indices, tokens)
        return self.observation()

    def reset(self, indices=None):
        self._check_open()
        selected = self._indices(indices)
        self.buffers['rewards'].fill(0)
        self._reset(selected)
        return self.observation()

    def _indices(self, indices):
        np = numpy()
        if indices is None:
            return list(range(self.n))
        a = np.asarray(indices)
        if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= self.n) or len(set(map(int, a))) != len(a):
            raise ValueError('reset indices must be unique valid lanes')
        return list(map(int, a))

    def reset_done(self):
        selected = numpy().flatnonzero(self.buffers['terminated'] | self.buffers['truncated'])
        return self.reset(selected)

    def close(self):
        self.closed = True
        self.engines.clear()
        self.matches.clear()
        self.agents.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
