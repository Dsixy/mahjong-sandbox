"""Versioned, player-relative Sichuan observations and semantic action menus."""
import numpy as np
from ..batch import (ACTION_KINDS, PHASES, META_FIELDS, MAX_ACTIONS, allocate,
                     encode_observation)
from ..model import Action, RuleConfig
from ..operators import shanten_batch, discard_ukeire_batch

FEATURE_VERSION = 'sichuan_visible_v1'
STATE_DIM = 1549
HINT_DIM = 7
RECENT_DISCARDS = 8


def semantic_key(row, last_drawn=-1):
    """Physical copies are equivalent except public drawn/hand-cut identity."""
    code, size = map(int, row[:2])
    kinds = tuple(sorted(int(t) // 4 for t in row[2:2 + size]))
    drawn = code == ACTION_KINDS.index('discard') and size == 1 and int(row[2]) == last_drawn
    return code, kinds, int(row[6]), int(row[7]), bool(drawn)


def canonical_mask(batch):
    mask = batch['action_mask'].copy()
    for i in range(len(mask)):
        seen = set()
        drawn = int(batch['meta'][i, META_FIELDS.index('last_drawn')])
        for j in np.flatnonzero(mask[i]):
            key = semantic_key(batch['actions'][i, j], drawn)
            if key in seen:
                mask[i, j] = False
            seen.add(key)
    return mask


def _hist(tiles):
    """Histogram last tile axis without conflating padding with tile kind zero."""
    kinds = np.where(tiles >= 0, tiles // 4, 27)
    return np.eye(28, dtype=np.float32)[kinds].sum(axis=-2)[..., :27]


def encode_batch(batch, *, rules=None, backend='auto', hints=True):
    from ..riichi_model import RiichiConfig
    if isinstance(rules, RiichiConfig):
        from .riichi_features import encode_batch as encode_riichi
        return encode_riichi(batch, rules=rules, backend=backend, hints=hints)
    rules = rules or RuleConfig()
    n = len(batch['meta'])
    players = batch['meta'][:, 1].astype(np.int64)
    relative = (players[:, None] + np.arange(4)) % 4
    rows = np.arange(n)[:, None]
    seats = batch['seats'][rows, relative]
    melds = batch['melds'][rows, relative]
    rivers = batch['rivers'][rows, relative]
    lengths = batch['river_lengths'][rows, relative]
    known = batch['known_counts'][:, :27]
    own = batch['own_counts'][:, :27]
    thresholds = np.arange(1, 5)
    parts = [(own[:, :, None] >= thresholds).reshape(n, -1),
             (known[:, :, None] >= thresholds).reshape(n, -1),
             (_hist(rivers) / 4).reshape(n, -1),
             (_hist(melds[..., 1:5].reshape(n, 4, -1)) / 4).reshape(n, -1),
             (_hist(batch['revealed'][rows, relative]) / 4).reshape(n, -1)]
    positions = lengths[:, :, None].astype(int) - RECENT_DISCARDS + np.arange(RECENT_DISCARDS)
    recent = np.take_along_axis(rivers, np.maximum(positions, 0), axis=2)
    recent = np.where(positions >= 0, recent // 4, 27)
    parts.append(np.eye(28, dtype=np.float32)[recent].reshape(n, -1))
    missing = np.where(seats[:, :, 1] >= 0, seats[:, :, 1], 3)
    parts.append(np.eye(4, dtype=np.float32)[missing].reshape(n, -1))
    meld_count = (melds[..., 8] > 0).sum(axis=2)
    scores = batch['scores'][rows, relative]
    parts.append(np.stack((seats[:, :, 0] / 14, seats[:, :, 2], meld_count / 4,
                           (scores - scores.mean(axis=1, keepdims=True)) / (96 * rules.base_points)), axis=2).reshape(n, -1))
    won = np.any(batch['wins'][:, :, 0, None] == relative[:, None, :], axis=1)
    parts.append(won)
    meta = batch['meta']
    actor = np.where(meta[:, 2] < 0, 4, (meta[:, 2] - players) % 4)
    parts.extend((np.eye(len(PHASES))[meta[:, 0]], np.eye(5)[actor],
                  np.eye(4)[(meta[:, 4] - players) % 4], meta[:, 3:4] / 55,
                  np.eye(4)[np.maximum(batch['exchange_offsets'], 0)],
                  np.clip(meta[:, 14:15], -1, rules.fan_cap) / max(1, rules.fan_cap)))
    for field in (9, 10):
        kind = np.where(meta[:, field] < 0, 27, meta[:, field] // 4)
        parts.append(np.eye(28)[kind])
    state = np.concatenate(parts, axis=1).astype(np.float32)
    if state.shape != (n, STATE_DIM):
        raise AssertionError(f'feature schema changed: {state.shape}')
    mask = canonical_mask(batch)
    candidate_hints = np.zeros((n, MAX_ACTIONS, HINT_DIM), np.float32)
    if hints:
        candidate_hints = _candidate_hints(batch, mask, rules, backend)
    # Drawn/hand cut is action identity, retained even in the no-domain-hints ablation.
    candidate_hints[:, :, 6] = (mask & (batch['actions'][:, :, 0] == ACTION_KINDS.index('discard'))
        & (batch['actions'][:, :, 2] == batch['meta'][:, 10, None]))
    return {'state': state, 'candidates': batch['actions'].copy(),
            'hints': candidate_hints, 'mask': mask}


def _candidate_hints(batch, mask, rules, backend):
    n = len(mask)
    result = np.zeros((n, MAX_ACTIONS, HINT_DIM), np.float32)
    own = batch['own_counts'].astype(np.int16)
    meld_counts = (batch['melds'][np.arange(n), batch['meta'][:, 1], :, 8] > 0).sum(axis=1)
    ii, jj = np.nonzero(mask)
    if not len(ii):
        return result
    after = own[ii].copy()
    exposed = meld_counts[ii].copy()
    codes = batch['actions'][ii, jj, 0]
    for k, (i, j) in enumerate(zip(ii, jj)):
        a = batch['actions'][i, j]
        tiles = a[2:2 + a[1]]
        kinds = tiles // 4
        for kind in kinds:
            after[k, kind] -= 1
        if a[0] == ACTION_KINDS.index('declare_suit'):
            after[k, a[6] * 9:(a[6] + 1) * 9] = 0
        if a[0] in [ACTION_KINDS.index(x) for x in ('peng', 'open_kong', 'concealed_kong')]:
            exposed[k] += 1
        suit = a[6] if a[6] >= 0 else (kinds[0] // 9 if len(kinds) else -1)
        if suit >= 0:
            result[i, j, 2] = own[i, suit * 9:(suit + 1) * 9].sum() / 14
        missing = batch['seats'][i, batch['meta'][i, 1], 1]
        result[i, j, 3] = np.count_nonzero(kinds // 9 == missing) / 4
        result[i, j, 4] = batch['known_counts'][i, kinds].sum() / 16
        result[i, j, 6] = (a[0] == ACTION_KINDS.index('discard') and a[1] == 1
                            and a[2] == batch['meta'][i, 10])
    distances = shanten_batch(after, exposed, variant='sichuan',
                             seven_pairs_quads=rules.seven_pairs_quads, backend=backend)
    result[ii, jj, 0] = -distances / 8
    for i in range(n):
        selected = ii == i
        if selected.any():
            result[i, jj[selected], 5] = distances[selected] == distances[selected].min()
    discard = mask & (batch['actions'][:, :, 0] == ACTION_KINDS.index('discard'))
    active = np.flatnonzero(discard.any(axis=1))
    if len(active):
        allowed = np.zeros((len(active), 34), bool)
        for row, i in enumerate(active):
            allowed[row, batch['actions'][i, discard[i], 2] // 4] = True
        _, ukeire = discard_ukeire_batch(own[active], batch['known_counts'][active], meld_counts[active],
            variant='sichuan', seven_pairs_quads=rules.seven_pairs_quads, backend=backend,
            best_only=True, candidate_mask=allowed)
        for row, i in enumerate(active):
            positions = np.flatnonzero(discard[i])
            result[i, positions, 1] = ukeire[row, batch['actions'][i, positions, 2] // 4] / 108
    return result


def observation_batch(observation):
    if observation.get('variant') not in ('sichuan', 'riichi'):
        raise ValueError('unknown RL observation variant')
    b = allocate(1)
    actions = tuple(Action.from_dict(a) for a in observation['legal_actions'])
    encode_observation(observation, actions, b)
    return b


def canonical_index(batch, action, row=0):
    """Map a teacher's concrete Action back to an unmasked semantic candidate."""
    a = np.full(8, -1, np.int16)
    a[:2] = ACTION_KINDS.index(action.kind.value), len(action.tiles)
    a[2:2 + len(action.tiles)] = action.tiles
    a[6:] = action.suit if action.suit is not None else -1, action.meld_index if action.meld_index is not None else -1
    key = semantic_key(a, int(batch['meta'][row, 10]))
    for j in np.flatnonzero(canonical_mask(batch)[row]):
        if semantic_key(batch['actions'][row, j], int(batch['meta'][row, 10])) == key:
            return int(j)
    raise ValueError('action does not match a legal semantic candidate')
