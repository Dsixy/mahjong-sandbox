"""Visible 34-kind riichi state and red-sensitive semantic action candidates."""
import numpy as np
from ..batch import ACTION_KINDS, PHASES, MAX_ACTIONS
from ..operators import shanten_batch, discard_ukeire_batch
from ..riichi_model import RiichiConfig
from ..riichi_scoring import dora_kind

FEATURE_VERSION = 'riichi_visible_v1'
STATE_DIM = 2350
HINT_DIM = 12
RECENT = 8
_DORA = np.array([dora_kind(k * 4) for k in range(34)])


def _hist(tiles):
    kinds = np.where(tiles >= 0, tiles // 4, 34)
    return np.eye(35, dtype=np.float32)[kinds].sum(axis=-2)[..., :34]


def semantic_key(row, last_drawn, red_mask):
    code, size = int(row[0]), int(row[1])
    tiles = row[2:2 + size]
    kinds = tuple(sorted(int(t) // 4 for t in tiles))
    red = tuple(sorted(int(t) // 4 for t in tiles if red_mask[int(t)]))
    drawn = code in (ACTION_KINDS.index('discard'), ACTION_KINDS.index('riichi')) and size == 1 and int(tiles[0]) == last_drawn
    return code, kinds, red, int(row[6]), int(row[7]), bool(drawn)


def canonical_mask(batch):
    mask = batch['action_mask'].copy()
    for i in range(len(mask)):
        seen = set()
        for j in np.flatnonzero(mask[i]):
            key = semantic_key(batch['actions'][i, j], int(batch['meta'][i, 10]), batch['red_mask'][i])
            if key in seen:
                mask[i, j] = False
            seen.add(key)
    return mask


def canonical_index(batch, action, row=0):
    encoded = np.full(8, -1, np.int16)
    encoded[:2] = ACTION_KINDS.index(action.kind.value), len(action.tiles)
    encoded[2:2 + len(action.tiles)] = action.tiles
    encoded[6:] = action.suit if action.suit is not None else -1, action.meld_index if action.meld_index is not None else -1
    target = semantic_key(encoded, int(batch['meta'][row, 10]), batch['red_mask'][row])
    for j in np.flatnonzero(canonical_mask(batch)[row]):
        if semantic_key(batch['actions'][row, j], int(batch['meta'][row, 10]), batch['red_mask'][row]) == target:
            return int(j)
    raise ValueError('action does not match a legal riichi candidate')


def encode_batch(batch, *, rules=None, backend='auto', hints=True):
    rules = rules or RiichiConfig()
    n = len(batch['meta'])
    players = batch['meta'][:, 1].astype(int)
    relative = (players[:, None] + np.arange(4)) % 4
    rows = np.arange(n)[:, None]
    seats, melds = batch['seats'][rows, relative], batch['melds'][rows, relative]
    rivers, lengths = batch['rivers'][rows, relative], batch['river_lengths'][rows, relative]
    own, known = batch['own_counts'], batch['known_counts']
    parts = [(own[:, :, None] >= np.arange(1, 5)).reshape(n, -1),
             (known[:, :, None] >= np.arange(1, 5)).reshape(n, -1),
             (_hist(rivers) / 4).reshape(n, -1),
             (_hist(melds[..., 1:5].reshape(n, 4, -1)) / 4).reshape(n, -1),
             (_hist(batch['revealed'][rows, relative]) / 4).reshape(n, -1)]
    positions = lengths[:, :, None].astype(int) - RECENT + np.arange(RECENT)
    recent = np.take_along_axis(rivers, np.maximum(positions, 0), axis=2)
    recent = np.where(positions >= 0, recent // 4, 34)
    parts.append(np.eye(35)[recent].reshape(n, -1))
    flags = np.take_along_axis(batch['river_flags'][rows, relative], np.maximum(positions, 0), axis=2)
    flags = np.where(positions >= 0, flags, 0)
    parts.append(((flags[..., None] & (1 << np.arange(4))) != 0).reshape(n, -1))
    meld_count = (melds[..., 8] > 0).sum(axis=2)
    scores = batch['scores'][rows, relative]
    parts.append(np.stack((seats[:, :, 0] / 14, seats[:, :, 2], meld_count / 4,
                           np.maximum(seats[:, :, 3], 0), np.maximum(seats[:, :, 4], 0),
                           scores / 50000, (scores - scores.mean(axis=1, keepdims=True)) / 32000), axis=2).reshape(n, -1))
    parts.append(np.eye(4)[(seats[:, :, 6] - 27).clip(0, 3)].reshape(n, -1))
    parts.append(np.eye(35)[np.where(seats[:, :, 5] >= 0, seats[:, :, 5] // 4, 34)].reshape(n, -1))
    meta = batch['meta']
    actor = np.where(meta[:, 2] < 0, 4, (meta[:, 2] - players) % 4)
    parts.extend((np.eye(len(PHASES))[meta[:, 0]], np.eye(5)[actor],
                  np.eye(4)[(meta[:, 4] - players) % 4], np.eye(4)[(meta[:, 5] - 27).clip(0, 3)],
                  np.eye(4)[(meta[:, 6] - 1).clip(0, 3)],
                  np.stack((meta[:, 7] / 10, meta[:, 8] / 10, meta[:, 3] / 70), axis=1),
                  np.maximum(meta[:, [11, 12, 13, 15]], 0)))
    for field in (9, 10):
        parts.append(np.eye(35)[np.where(meta[:, field] >= 0, meta[:, field] // 4, 34)])
    indicators = _hist(batch['dora'])
    targets = np.zeros_like(indicators)
    targets[:, _DORA] = indicators
    parts.extend((indicators / 5, targets / 5))
    concealed = batch['concealed']
    own_red = (concealed >= 0) & batch['red_mask'][rows, np.maximum(concealed, 0)]
    parts.append(_hist(np.where(own_red, concealed, -1)) / 4)
    visible = np.concatenate((concealed, batch['revealed'].reshape(n, -1), batch['rivers'].reshape(n, -1),
                              batch['melds'][..., 1:5].reshape(n, -1), batch['dora'], meta[:, 9:10],
                              batch['wins'][:, :, 2]), axis=1)
    present = np.zeros((n, 137), bool)
    present[rows, np.where(visible >= 0, visible, 136)] = True
    parts.append((present[:, :136] & batch['red_mask']).reshape(n, 34, 4).sum(axis=2) / 4)
    parts.append(np.tile([float(rules.open_tanyao), float(bool(rules.red_fives))], (n, 1)))
    state = np.concatenate(parts, axis=1).astype(np.float32)
    if state.shape != (n, STATE_DIM):
        raise AssertionError(f'riichi feature schema changed: {state.shape}')
    mask = canonical_mask(batch)
    candidate_hints = _candidate_hints(batch, mask, targets, backend) if hints else np.zeros((n, MAX_ACTIONS, HINT_DIM), np.float32)
    for i in range(n):
        for j in np.flatnonzero(mask[i]):
            action = batch['actions'][i, j]
            code, size = int(action[0]), int(action[1])
            candidate_hints[i, j, 6] = code in (ACTION_KINDS.index('discard'), ACTION_KINDS.index('riichi')) and size == 1 and action[2] == meta[i, 10]
            candidate_hints[i, j, 7] = sum(batch['red_mask'][i, int(t)] for t in action[2:2 + size]) / 3
    return {'state': state, 'candidates': batch['actions'].copy(), 'hints': candidate_hints, 'mask': mask}


def _candidate_hints(batch, mask, targets, backend):
    n = len(mask)
    result = np.zeros((n, MAX_ACTIONS, HINT_DIM), np.float32)
    counts, meld_counts, locations = [], [], []
    discard_codes = (ACTION_KINDS.index('discard'), ACTION_KINDS.index('riichi'))
    call_codes = tuple(ACTION_KINDS.index(k) for k in ('chi', 'peng', 'open_kong'))
    kong_codes = tuple(ACTION_KINDS.index(k) for k in ('open_kong', 'added_kong', 'concealed_kong'))
    allowed = np.zeros((n, 34), bool)
    own_melds = (batch['melds'][..., 8] > 0).sum(axis=2)
    for i in range(n):
        player = int(batch['meta'][i, 1])
        threats = [p for p in range(4) if p != player and batch['seats'][i, p, 3] > 0]
        safe = [{int(t) // 4 for t in batch['rivers'][i, p] if t >= 0} for p in threats]
        for j in np.flatnonzero(mask[i]):
            action = batch['actions'][i, j]
            code, size = int(action[0]), int(action[1])
            result[i, j, 2:6] = [code in (ACTION_KINDS.index('ron'), ACTION_KINDS.index('tsumo')),
                                 code == ACTION_KINDS.index('riichi'), code in call_codes, code in kong_codes]
            after = batch['own_counts'][i].astype(np.int16).copy()
            for tile in action[2:2 + size]:
                after[int(tile) // 4] -= 1
            counts.append(after)
            meld_counts.append(own_melds[i, player] + int(code in call_codes or code == ACTION_KINDS.index('concealed_kong')))
            locations.append((i, j))
            if code in discard_codes:
                kind = int(action[2]) // 4
                allowed[i, kind] = True
                result[i, j, 8] = targets[i, kind] / 5
                result[i, j, 9] = sum(kind in river for river in safe) / len(safe) if safe else 0
                result[i, j, 10] = batch['known_counts'][i, kind] / 4
                result[i, j, 11] = kind in (31, 32, 33, int(batch['meta'][i, 5]), int(batch['seats'][i, player, 6]))
    if counts:
        distances = shanten_batch(np.asarray(counts), np.asarray(meld_counts), variant='riichi', backend=backend)
        for (i, j), distance in zip(locations, distances):
            result[i, j, 0] = distance / 8
    active = np.flatnonzero(allowed.any(axis=1))
    if len(active):
        _, ukeire = discard_ukeire_batch(batch['own_counts'][active], batch['known_counts'][active],
            own_melds[active, batch['meta'][active, 1]], variant='riichi', backend=backend,
            best_only=True, candidate_mask=allowed[active])
        for row, i in enumerate(active):
            for j in np.flatnonzero(mask[i]):
                if batch['actions'][i, j, 0] in discard_codes:
                    result[i, j, 1] = ukeire[row, batch['actions'][i, j, 2] // 4] / 136
    return result
