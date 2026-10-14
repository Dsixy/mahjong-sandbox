"""Visible-information tile-efficiency baseline; no learned policy or hidden state."""
from functools import lru_cache
import random

from .model import Action


@lru_cache(maxsize=150_000)
def _regular_distance(counts: tuple[int, ...], complete: int, partial: int, head: int) -> int:
    first = next((kind for kind, n in enumerate(counts) if n), None)
    if first is None:
        return 8 - 2 * complete - min(partial, 4 - complete) - head
    remaining = list(counts)
    remaining[first] -= 1
    best = _regular_distance(tuple(remaining), complete, partial, head)
    remaining[first] += 1

    def consume(kinds, completed, taatsu, pair):
        for kind in kinds:
            remaining[kind] -= 1
        value = _regular_distance(tuple(remaining), completed, taatsu, pair)
        for kind in kinds:
            remaining[kind] += 1
        return value

    if complete < 4:
        if remaining[first] >= 3:
            best = min(best, consume((first,) * 3, complete + 1, partial, head))
        if first < 27 and first % 9 <= 6 and remaining[first + 1] and remaining[first + 2]:
            best = min(best, consume((first, first + 1, first + 2), complete + 1, partial, head))
    if remaining[first] >= 2:
        if not head:
            best = min(best, consume((first, first), complete, partial, 1))
        if partial < 4 - complete:
            best = min(best, consume((first, first), complete, partial + 1, head))
    if partial < 4 - complete:
        for gap in (1, 2):
            if first < 27 and first % 9 + gap < 9 and remaining[first + gap]:
                best = min(best, consume((first, first + gap), complete, partial + 1, head))
    return best


def shape_distance(counts, meld_count=0, *, seven_pairs_quads=True):
    """Structural distance for 27 suited kinds, including the v1 seven-pair form."""
    values = tuple(counts[:27])
    normal = _regular_distance(values, meld_count, 0, 0)
    if meld_count:
        return normal
    if seven_pairs_quads:
        seven = 6 - sum(n // 2 for n in values)
    else:
        seven = 6 - sum(n >= 2 for n in values) + max(0, 7 - sum(n > 0 for n in values))
    return min(normal, seven)


class GreedyAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def distance(self,counts,meld_count=0,*,seven_pairs_quads=True):
        return shape_distance(counts,meld_count,seven_pairs_quads=seven_pairs_quads)

    def act(self, observation):
        actions = observation["legal_actions"]
        if not actions:
            raise ValueError("agent called without a decision")
        own = observation["players"][observation["player"]]
        counts = [0] * 27
        for tile in own["concealed"]:
            counts[tile // 4] += 1
        meld_count = len(own["melds"])
        rules = next((e["data"]["rules"] for e in observation["events"] if e["kind"] == "round_started"), {})
        quads = rules.get("seven_pairs_quads", True)
        distance = lambda c, m=meld_count: self.distance(c, m, seven_pairs_quads=quads)
        winners = [a for a in actions if a["kind"] in ("tsumo", "ron")]
        if winners:
            return Action.from_dict(winners[0])
        if actions[0]["kind"] == "declare_suit":
            def suit_score(action):
                suit = action["suit"]
                reduced = counts.copy()
                size = sum(reduced[suit * 9:suit * 9 + 9])
                reduced[suit * 9:suit * 9 + 9] = [0] * 9
                return (-size, -distance(reduced))
            return self._best(actions, suit_score)
        if actions[0]["kind"] == "exchange":
            def exchange_score(action):
                suit = action["tiles"][0] // 4 // 9
                reduced = self._remove(counts, action["tiles"])
                return (-sum(counts[suit * 9:suit * 9 + 9]), -distance(reduced))
            return self._best(actions, exchange_score)
        discards = [a for a in actions if a["kind"] == "discard"]
        baseline = min((distance(self._remove(counts, a["tiles"])) for a in discards), default=distance(counts))
        kongs = []
        for action in actions:
            if action["kind"] in ("concealed_kong", "open_kong", "added_kong"):
                after = self._remove(counts, action["tiles"])
                exposed = meld_count + int(action["kind"] != "added_kong")
                if distance(after, exposed) <= baseline:
                    kongs.append(action)
        if kongs:
            return Action.from_dict(self.rng.choice(kongs))
        if not discards:
            pengs = []
            for action in actions:
                if action["kind"] == "peng":
                    after = self._remove(counts, action["tiles"])
                    best = min(distance(self._remove(after, (4 * k,)), meld_count + 1)
                               for k, n in enumerate(after) if n)
                    if best < baseline:
                        pengs.append(action)
            if pengs:
                return Action.from_dict(self.rng.choice(pengs))
            return Action.from_dict(next(a for a in actions if a["kind"] == "pass"))
        # Dedupe physical copies; drawn/hand cut is still a valid concrete action.
        candidates = {}
        for action in discards:
            candidates.setdefault(action["tiles"][0] // 4, action)
        best_distance = min(distance(self._remove(counts, a["tiles"])) for a in candidates.values())
        candidates = [a for a in candidates.values() if distance(self._remove(counts, a["tiles"])) == best_distance]
        known_ids = set()
        for player in observation["players"]:
            known_ids.update(player.get("concealed", ()))
            known_ids.update(d["tile"] for d in player["river"])
            for meld in player["melds"]:
                known_ids.update(meld["tiles"])
        known = [0] * 27
        for tile in known_ids:
            known[tile // 4] += 1

        def discard_score(action):
            kind = action["tiles"][0] // 4
            after = self._remove(counts, action["tiles"])
            useful = 0
            for incoming in range(27):
                if known[incoming] < 4:
                    after[incoming] += 1
                    if distance(after) < best_distance:
                        useful += 4 - known[incoming]
                    after[incoming] -= 1
            return (useful, known[kind], int(kind % 9 in (0, 8)))
        return self._best(candidates, discard_score)

    def _best(self, actions, scorer):
        scores = [scorer(a) for a in actions]
        best = max(scores)
        return Action.from_dict(self.rng.choice([a for a, score in zip(actions, scores) if score == best]))

    @staticmethod
    def _remove(counts, tiles):
        result = counts.copy()
        for tile in tiles:
            result[tile // 4] -= 1
        return result
