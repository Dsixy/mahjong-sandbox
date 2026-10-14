"""The explicitly versioned v1 scoring profile; not a universal Sichuan rulebook."""
from dataclasses import dataclass

from .hands import winning_shapes, waits
from .model import FanItem, HandValue, PlayerState, RuleConfig
from .tiles import counts_of, kind_of


@dataclass(frozen=True)
class WinContext:
    self_draw: bool = False
    rob_kong: bool = False
    kong_draw: bool = False
    kong_discard: bool = False
    last_tile: bool = False


def evaluate_hand(
    player: PlayerState, config: RuleConfig, *, tile_kind: int | None = None,
    context: WinContext = WinContext(),
) -> HandValue | None:
    if player.missing_suit is None:
        return None
    counts = list(counts_of(player.concealed))
    if tile_kind is not None:
        if type(tile_kind) is not int or not 0 <= tile_kind < 27 or counts[tile_kind] >= 4:
            return None
        counts[tile_kind] += 1
    total = counts.copy()
    for meld in player.melds:
        for tile in meld.tiles:
            total[kind_of(tile)] += 1
    if any(total[27:]) or any(n > 4 for n in total) or any(total[player.missing_suit * 9:(player.missing_suit + 1) * 9]):
        return None
    shapes = winning_shapes(counts, len(player.melds), seven_pairs_quads=config.seven_pairs_quads)
    candidates = []
    for shape in shapes:
        items = []
        if shape.family == "seven_pairs":
            items.append(FanItem("七对", 2))
        else:
            triplets = all(group[0] == group[2] for group in shape.groups) and all(m.kind != "chi" for m in player.melds)
            jiang = triplets and all(kind % 9 in (1, 4, 7) for kind, n in enumerate(total) if n)
            yaojiu = (shape.pair % 9 in (0, 8)
                      and all(any(kind % 9 in (0, 8) for kind in group) for group in shape.groups)
                      and all(any(kind_of(t) % 9 in (0, 8) for t in meld.tiles) for meld in player.melds))
            # Base shape categories compete; they do not multiply each other.
            if jiang:
                items.append(FanItem("将对", 3))
            elif yaojiu:
                items.append(FanItem("带幺九", 2))
            elif triplets:
                items.append(FanItem("对对胡", 1))
                if len(player.melds) == 4:
                    items.append(FanItem("金钩钓加番", 1))
        if len({kind // 9 for kind, n in enumerate(total) if n}) == 1:
            items.append(FanItem("清一色", 2))
        roots = sum(n == 4 for n in total)
        if roots:
            items.append(FanItem("龙七对加番" if shape.family == "seven_pairs" else "根", roots))
        if context.rob_kong:
            items.append(FanItem("抢杠胡", 1))
        if context.kong_draw:
            items.append(FanItem("杠上花", 1))
        if context.kong_discard:
            items.append(FanItem("杠上炮", 1))
        if context.last_tile and config.last_tile_bonus:
            items.append(FanItem("海底自摸" if context.self_draw else "海底炮", 1))
        if context.self_draw and config.self_draw_bonus == "fan":
            items.append(FanItem("自摸加番", 1))
        if not items:
            items.append(FanItem("平胡", 0))
        raw_fan = sum(item.fan for item in items)
        fan = min(raw_fan, config.fan_cap)
        extra_base = int(context.self_draw and config.self_draw_bonus == "base")
        candidates.append(HandValue(shape.family, tuple(items), raw_fan, fan, config.base_points * ((1 << fan) + extra_base)))
    return max(candidates, key=lambda value: (value.payment, value.raw_fan, value.family), default=None)


def ready_values(player: PlayerState, config: RuleConfig) -> dict[int, HandValue]:
    """Structural waits, ignoring wall availability and passed-ron lock for 查叫."""
    if player.has_missing_tiles() or player.missing_suit is None:
        return {}
    counts = counts_of(player.concealed)
    total = list(counts)
    for meld in player.melds:
        for tile in meld.tiles:
            total[kind_of(tile)] += 1
    result = {}
    for kind in waits(counts, len(player.melds), seven_pairs_quads=config.seven_pairs_quads):
        if total[kind] < 4:
            value = evaluate_hand(player, config, tile_kind=kind)
            if value:
                result[kind] = value
    return result
