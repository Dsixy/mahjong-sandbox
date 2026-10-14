"""Exhaustive winning decompositions and structural waits.

Counts contain only concealed tiles; a quad meld is one structural meld.
Seven-pair quad semantics are explicit rather than inherited from riichi.
"""
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Iterable

from .tiles import SICHUAN_KINDS, validate_counts


@dataclass(frozen=True)
class HandShape:
    family: str
    pair: int | None = None
    groups: tuple[tuple[int, int, int], ...] = ()


@lru_cache(maxsize=100_000)
def _groups(counts: tuple[int, ...]) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    first = next((i for i, count in enumerate(counts) if count), None)
    if first is None:
        return ((),)
    result = []
    remaining = list(counts)
    if remaining[first] >= 3:
        remaining[first] -= 3
        result.extend(((first, first, first),) + rest for rest in _groups(tuple(remaining)))
        remaining[first] += 3
    if first < 27 and first % 9 <= 6 and remaining[first + 1] and remaining[first + 2]:
        for kind in (first, first + 1, first + 2):
            remaining[kind] -= 1
        result.extend(((first, first + 1, first + 2),) + rest for rest in _groups(tuple(remaining)))
    return tuple(result)


def winning_shapes(
    counts: Iterable[int], meld_count: int = 0, *, seven_pairs_quads: bool = True,
) -> tuple[HandShape, ...]:
    counts = validate_counts(counts)
    if type(meld_count) is not int or not 0 <= meld_count <= 4:
        raise ValueError("meld_count must be in [0, 4]")
    if sum(counts) != 3 * (4 - meld_count) + 2:
        return ()
    result = []
    if meld_count == 0 and all(n % 2 == 0 for n in counts):
        if seven_pairs_quads or sum(n == 2 for n in counts) == 7:
            result.append(HandShape("seven_pairs"))
    for pair, count in enumerate(counts):
        if count >= 2:
            remaining = list(counts)
            remaining[pair] -= 2
            for groups in _groups(tuple(remaining)):
                result.append(HandShape("standard", pair, groups))
    return tuple(result)


def waits(
    counts: Iterable[int], meld_count: int = 0, *,
    enabled_kinds: Iterable[int] = SICHUAN_KINDS, seven_pairs_quads: bool = True,
) -> tuple[int, ...]:
    counts = validate_counts(counts)
    if sum(counts) != 3 * (4 - meld_count) + 1:
        return ()
    result = []
    for kind in enabled_kinds:
        if type(kind) is not int or not 0 <= kind < 34:
            raise ValueError("invalid enabled tile kind")
        if counts[kind] < 4:
            candidate = list(counts)
            candidate[kind] += 1
            if winning_shapes(candidate, meld_count, seven_pairs_quads=seven_pairs_quads):
                result.append(kind)
    return tuple(result)
