"""Own-observation riichi tile-efficiency baseline, without trained weights."""
from .greedy import GreedyAgent, _regular_distance
from .model import Action
from .riichi_scoring import ORPHANS


def riichi_shanten(counts, meld_count=0):
    values = tuple(counts)
    if len(values) != 34 or any(n < 0 or n > 4 for n in values):
        raise ValueError('34 counts in 0..4 required')
    normal = _regular_distance(values, meld_count, 0, 0)
    if meld_count:
        return normal
    seven = 6 - sum(n >= 2 for n in values) + max(0, 7 - sum(n > 0 for n in values))
    orphans = 13 - sum(values[k] > 0 for k in ORPHANS) - int(any(values[k] > 1 for k in ORPHANS))
    return min(normal, seven, orphans)


class RiichiAgent(GreedyAgent):
    def distance(self,counts,meld_count=0,*,seven_pairs_quads=False):
        return riichi_shanten(counts,meld_count)

    def act(self, observation):
        actions = observation['legal_actions']
        if not actions:
            raise ValueError('agent called without a decision')
        own = observation['players'][observation['player']]
        counts = [0] * 34
        for t in own['concealed']:
            counts[t // 4] += 1
        meld_count = len(own['melds'])
        for a in actions:
            if a['kind'] in ('ron', 'tsumo'):
                return Action.from_dict(a)
        discards = [a for a in actions if a['kind'] == 'discard']
        distance = lambda c, m=meld_count: self.distance(c, m)
        baseline = min((distance(self._remove(counts, a['tiles'])) for a in discards), default=distance(counts))
        kongs = [a for a in actions if a['kind'] in ('concealed_kong', 'added_kong') and
                 distance(self._remove(counts, a['tiles']), meld_count + (a['kind'] != 'added_kong')) <= baseline]
        if kongs:
            return Action.from_dict(self.rng.choice(kongs))
        # Preserve menzen until a known yakuhai triplet gives an open hand a yaku.
        if not discards:
            value_honors = {31, 32, 33, own['seat_wind'], observation['round_wind']}
            has_yaku = any(m['kind'] != 'chi' and m['tiles'][0] // 4 in value_honors for m in own['melds'])
            calls = []
            for a in actions:
                if a['kind'] not in ('chi', 'peng', 'open_kong'):
                    continue
                if not has_yaku and not (a['kind'] in ('peng', 'open_kong') and a['tiles'][0] // 4 in value_honors):
                    continue
                after = self._remove(counts, a['tiles'])
                if a['kind'] == 'open_kong':
                    best = distance(after, meld_count + 1)
                else:
                    called = observation['pending_tile'] // 4
                    forbidden = {called}
                    if a['kind'] == 'chi':
                        start = min(called, *(t // 4 for t in a['tiles']))
                        if called == start and start % 9 < 6: forbidden.add(start + 3)
                        if called == start + 2 and start % 9 > 0: forbidden.add(start - 1)
                    best = min((distance(self._remove(after, (4*k,)), meld_count + 1)
                                for k, n in enumerate(after) if n and k not in forbidden), default=9)
                if best < baseline:
                    calls.append(a)
            return Action.from_dict(self.rng.choice(calls) if calls else next(a for a in actions if a['kind'] == 'pass'))
        riichi = [a for a in actions if a['kind'] == 'riichi']
        candidates = riichi or discards
        best_distance = min(distance(self._remove(counts, a['tiles'])) for a in candidates)
        candidates = [a for a in candidates if distance(self._remove(counts, a['tiles'])) == best_distance]
        known_ids = set()
        for p in observation['players']:
            known_ids.update(p.get('concealed', ()))
            known_ids.update(d['tile'] for d in p['river'])
            for m in p['melds']: known_ids.update(m['tiles'])
        known_ids.update(observation['dora_indicators'])
        known = [0] * 34
        for t in known_ids: known[t // 4] += 1
        red = set(observation['red_fives'])
        def score(a):
            after = self._remove(counts, a['tiles'])
            useful = 0
            for k in range(34):
                if known[k] == 4: continue
                after[k] += 1
                if distance(after) < best_distance: useful += 4 - known[k]
                after[k] -= 1
            return useful, a['tiles'][0] not in red
        return self._best(candidates, score)
