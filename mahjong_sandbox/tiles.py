"""Stable tile-kind numbering and physical tile identity; no game rules here."""
from collections.abc import Iterable

KIND_COUNT = 34
SICHUAN_KINDS = tuple(range(27))
SICHUAN_TILES = tuple(range(108))
SUITS = ("m", "p", "s")
HONORS = ("E", "S", "W", "N", "P", "F", "C")


def kind_of(tile_id: int) -> int:
    if type(tile_id) is not int or not 0 <= tile_id < 136:
        raise ValueError("tile_id must be an integer in [0, 135]")
    return tile_id // 4


def counts_of(tile_ids: Iterable[int]) -> tuple[int, ...]:
    result = [0] * KIND_COUNT
    seen = set()
    for tile in tile_ids:
        if tile in seen:
            raise ValueError("duplicate physical tile")
        seen.add(tile)
        kind = kind_of(tile)
        result[kind] += 1
    return tuple(result)


def validate_counts(counts: Iterable[int]) -> tuple[int, ...]:
    values = tuple(counts)
    if len(values) != KIND_COUNT or any(type(n) is not int or not 0 <= n <= 4 for n in values):
        raise ValueError("counts must contain 34 integers in [0, 4]")
    return values


def kind_name(kind: int) -> str:
    if type(kind) is not int or not 0 <= kind < KIND_COUNT:
        raise ValueError("invalid tile kind")
    return f"{kind % 9 + 1}{SUITS[kind // 9]}" if kind < 27 else HONORS[kind - 27]


def tile_name(tile_id: int) -> str:
    return kind_name(kind_of(tile_id))


def parse_hand(text: str) -> tuple[int, ...]:
    """Read compact notation, e.g. 123m456p789s11122z, as 34 counts."""
    counts = [0] * KIND_COUNT
    digits = []
    for char in text.replace(" ", ""):
        if char.isascii() and char.isdigit():
            digits.append(int(char))
        elif char in "mpsz" and digits:
            offset = "mpsz".index(char) * 9
            limit = 7 if char == "z" else 9
            for rank in digits:
                if not 1 <= rank <= limit:
                    raise ValueError("invalid tile rank")
                counts[offset + rank - 1] += 1
            digits.clear()
        else:
            raise ValueError("invalid hand notation")
    if digits:
        raise ValueError("tile digits need a suit suffix")
    return validate_counts(counts)
