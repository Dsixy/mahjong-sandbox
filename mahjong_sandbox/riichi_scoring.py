"""Riichi hand shapes, yaku, wait-sensitive fu, and rounded point payments."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from .hands import HandShape, winning_shapes
from .model import FanItem
from .riichi_model import RiichiConfig
from .tiles import counts_of, validate_counts

ORPHANS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
GREEN_TILES = frozenset((19, 20, 21, 23, 25, 32))


def terminal(kind):
    return kind < 27 and kind % 9 in (0, 8)


def outside(kind):
    return kind >= 27 or terminal(kind)


@lru_cache(maxsize=100_000)
def shapes(counts, meld_count=0):
    result = winning_shapes(counts, meld_count, seven_pairs_quads=False)
    if not meld_count and sum(counts) == 14 and all(counts[k] for k in ORPHANS) and sum(counts[k] for k in ORPHANS) == 14:
        result += (HandShape('thirteen_orphans', next(k for k in ORPHANS if counts[k] == 2)),)
    return result


@lru_cache(maxsize=100_000)
def riichi_waits(counts, meld_count=0):
    counts = validate_counts(counts)
    if sum(counts) != 13 - 3 * meld_count:
        return ()
    result = []
    for kind in range(34):
        if counts[kind] < 4:
            candidate = list(counts)
            candidate[kind] += 1
            if shapes(tuple(candidate), meld_count):
                result.append(kind)
    return tuple(result)


def dora_kind(indicator):
    kind = indicator // 4
    if kind < 27:
        return kind // 9 * 9 + (kind % 9 + 1) % 9
    if kind < 31:
        return 27 + (kind - 27 + 1) % 4
    return 31 + (kind - 31 + 1) % 3


@dataclass(frozen=True)
class RiichiContext:
    self_draw: bool = False
    seat_wind: int = 27
    round_wind: int = 27
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    rinshan: bool = False
    chankan: bool = False
    haitei: bool = False
    houtei: bool = False
    tenhou: bool = False
    chiihou: bool = False


@dataclass(frozen=True)
class RiichiValue:
    family: str
    items: tuple[FanItem, ...]
    raw_fan: int
    fan: int
    payment: int
    han: int
    fu: int
    base_points: int
    yakuman: int
    limit: str | None
    tsumo_dealer: int
    tsumo_other: int


def round100(points):
    return (points + 99) // 100 * 100


def _value(family, items, fu, context, config, yakuman=0):
    han = sum(item.fan for item in items)
    limit = None
    if yakuman:
        base, limit = 8000 * yakuman, '役满' if yakuman == 1 else f'{yakuman}倍役满'
    elif han >= 13 and config.counted_yakuman:
        base, limit = 8000, '累计役满'
    elif han >= 11:
        base, limit = 6000, '三倍满'
    elif han >= 8:
        base, limit = 4000, '倍满'
    elif han >= 6:
        base, limit = 3000, '跳满'
    else:
        base = fu * (1 << (han + 2))
        if han >= 5 or base >= 2000 or config.kiriage_mangan and (han, fu) in ((4, 30), (3, 60)):
            base, limit = 2000, '满贯'
    dealer = context.seat_wind == 27
    return RiichiValue(family, tuple(items), han, han, round100(base * (6 if dealer else 4)),
                       han, fu, base, yakuman, limit, round100(base * 2), round100(base * (2 if dealer else 1)))


def _wait_type(group, kind):
    if group[0] == group[1]:
        return 'shanpon'
    start = group[0] % 9
    if kind == group[1]:
        return 'kanchan'
    if start == 0 and kind == group[2] or start == 6 and kind == group[0]:
        return 'penchan'
    return 'ryanmen'


def score_hand(concealed, melds, winning_tile, *, context=None, config=None, indicators=(), ura_indicators=()):
    """concealed includes the winning tile exactly once; melds retain physical quads."""
    config, context = config or RiichiConfig(), context or RiichiContext()
    if winning_tile not in concealed:
        raise ValueError('winning tile must be in concealed')
    counts = counts_of(concealed)
    all_tiles = list(concealed) + [tile for meld in melds for tile in meld.tiles]
    counts_of(all_tiles)
    closed = all(m.kind == 'concealed_kong' for m in melds)
    kind = winning_tile // 4
    candidates = []
    for shape in shapes(counts, len(melds)):
        placements = [None] if shape.family != 'standard' else ([-1] if shape.pair == kind else []) + [i for i, g in enumerate(shape.groups) if kind in g]
        for placement in placements:
            value = _score_shape(shape, counts, all_tiles, melds, kind, placement, closed, context, config,
                                 indicators, ura_indicators)
            if value:
                candidates.append(value)
    return max(candidates, key=lambda v: (v.payment if not context.self_draw else (3 * v.tsumo_other if context.seat_wind == 27 else v.tsumo_dealer + 2 * v.tsumo_other),
                                         v.han, v.fu), default=None)


def _score_shape(shape, counts, all_tiles, melds, winning_kind, placement, closed, ctx, cfg, indicators, ura):
    kinds = [tile // 4 for tile in all_tiles]
    unique = set(kinds)
    groups = list(shape.groups) + [tuple(t // 4 for t in m.tiles) for m in melds]
    is_open = [not ctx.self_draw and i == placement and g[0] == g[1] for i, g in enumerate(shape.groups)] + [m.kind != 'concealed_kong' for m in melds]
    sequences = [g for g in groups if g[0] != g[1]]
    triplets = [g for g in groups if g[0] == g[1]]
    triplet_kinds = {g[0] for g in triplets}
    kongs = sum(len(g) == 4 for g in groups)
    concealed_triplets = sum(g[0] == g[1] and not opened for g, opened in zip(groups, is_open))
    pair = shape.pair
    wait_type = 'tanki' if placement == -1 else _wait_type(shape.groups[placement], winning_kind) if shape.family == 'standard' else 'special'
    yakuman = []

    def add_yakuman(name, double=False):
        yakuman.append(FanItem(name, 26 if double and cfg.double_yakuman else 13))

    if ctx.tenhou:
        add_yakuman('天和')
    if ctx.chiihou:
        add_yakuman('地和')
    if shape.family == 'thirteen_orphans':
        before = list(counts)
        before[winning_kind] -= 1
        thirteen = all(before[k] == 1 for k in ORPHANS)
        add_yakuman('国士无双十三面' if thirteen else '国士无双', thirteen)
    if {31, 32, 33} <= triplet_kinds:
        add_yakuman('大三元')
    wind_groups = len(triplet_kinds & {27, 28, 29, 30})
    if wind_groups == 4:
        add_yakuman('大四喜', True)
    elif wind_groups == 3 and pair in range(27, 31):
        add_yakuman('小四喜')
    if all(k >= 27 for k in kinds):
        add_yakuman('字一色')
    if unique <= GREEN_TILES:
        add_yakuman('绿一色')
    if all(terminal(k) for k in kinds):
        add_yakuman('清老头')
    if concealed_triplets == 4:
        tanki = wait_type == 'tanki'
        add_yakuman('四暗刻单骑' if tanki else '四暗刻', tanki)
    if kongs == 4:
        add_yakuman('四杠子')
    suits = {k // 9 for k in kinds if k < 27}
    honors = any(k >= 27 for k in kinds)
    if closed and not melds and len(suits) == 1 and not honors:
        offset = next(iter(suits)) * 9
        pattern = (3, 1, 1, 1, 1, 1, 1, 1, 3)
        if all(counts[offset+i] >= n for i, n in enumerate(pattern)):
            before = list(counts[offset:offset+9])
            before[winning_kind % 9] -= 1
            pure = tuple(before) == pattern
            add_yakuman('纯正九莲宝灯' if pure else '九莲宝灯', pure)
    if yakuman:
        return _value(shape.family, yakuman, 0, ctx, cfg, sum(i.fan for i in yakuman) // 13)

    items = []

    def add(name, han):
        items.append(FanItem(name, han))

    if ctx.double_riichi and closed:
        add('两立直', 2)
    elif ctx.riichi and closed:
        add('立直', 1)
    if ctx.ippatsu and ctx.riichi and closed:
        add('一发', 1)
    if ctx.self_draw and closed:
        add('门前清自摸和', 1)
    if ctx.rinshan:
        add('岭上开花', 1)
    if ctx.chankan:
        add('抢杠', 1)
    if ctx.haitei:
        add('海底摸月', 1)
    if ctx.houtei:
        add('河底捞鱼', 1)
    if all(not outside(k) for k in kinds) and (closed or cfg.open_tanyao):
        add('断幺九', 1)
    if shape.family == 'seven_pairs':
        add('七对子', 2)
    for dragon, name in ((31, '白'), (32, '发'), (33, '中')):
        if dragon in triplet_kinds:
            add(f'役牌·{name}', 1)
    if ctx.seat_wind in triplet_kinds:
        add('役牌·自风', 1)
    if ctx.round_wind in triplet_kinds:
        add('役牌·场风', 1)
    pair_fu = (2 * int(pair in (31, 32, 33)) + 2 * int(pair == ctx.seat_wind) + 2 * int(pair == ctx.round_wind))
    pinfu = shape.family == 'standard' and closed and len(sequences) == 4 and pair_fu == 0 and wait_type == 'ryanmen'
    if pinfu:
        add('平和', 1)
    if closed:
        seq_counts = Counter(sequences)
        pairs = sum(n // 2 for n in seq_counts.values())
        if pairs == 2:
            add('二杯口', 3)
        elif pairs:
            add('一杯口', 1)
    starts = {g[0] for g in sequences}
    if any({s * 9, s * 9 + 3, s * 9 + 6} <= starts for s in range(3)):
        add('一气通贯', 2 if closed else 1)
    if any({rank, rank + 9, rank + 18} <= starts for rank in range(7)):
        add('三色同顺', 2 if closed else 1)
    if any({rank, rank + 9, rank + 18} <= triplet_kinds for rank in range(9)):
        add('三色同刻', 2)
    if shape.family == 'standard' and not sequences:
        add('对对和', 2)
    if concealed_triplets >= 3:
        add('三暗刻', 2)
    if kongs == 3:
        add('三杠子', 2)
    if len(triplet_kinds & {31, 32, 33}) == 2 and pair in (31, 32, 33):
        add('小三元', 2)
    if all(outside(k) for k in kinds):
        add('混老头', 2)
    elif shape.family == 'standard' and sequences and outside(pair) and all(any(outside(k) for k in g) for g in groups):
        add('混全带幺九' if honors else '纯全带幺九', (2 if closed else 1) if honors else (3 if closed else 2))
    if len(suits) == 1:
        add('混一色' if honors else '清一色', (3 if closed else 2) if honors else (6 if closed else 5))
    if not items:
        return None
    if shape.family == 'seven_pairs':
        fu = 25
    elif pinfu:
        fu = 20 if ctx.self_draw else 30
    else:
        fu = 20 + pair_fu + (10 if closed and not ctx.self_draw else 0) + (2 if ctx.self_draw else 0)
        fu += 2 * int(wait_type in ('tanki', 'kanchan', 'penchan'))
        for group, opened in zip(groups, is_open):
            if group[0] == group[1]:
                fu += (8 if len(group) == 4 else 2) * (2 if not opened else 1) * (2 if outside(group[0]) else 1)
        fu = max(30, (fu + 9) // 10 * 10)
    dora_counts = Counter(dora_kind(t) for t in indicators)
    dora = sum(dora_counts[k] for k in kinds)
    red = sum(t in cfg.red_fives for t in all_tiles)
    ura_counts = Counter(dora_kind(t) for t in ura) if ctx.riichi else Counter()
    ura_count = sum(ura_counts[k] for k in kinds)
    for name, amount in (('宝牌', dora), ('赤宝牌', red), ('里宝牌', ura_count)):
        if amount:
            add(name, amount)
    return _value(shape.family, items, fu, ctx, cfg)
