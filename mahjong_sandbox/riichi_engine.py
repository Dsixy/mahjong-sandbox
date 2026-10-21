"""Deterministic four-player riichi engine with sealed response arbitration."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations, product
import random

from .engine import Engine, IllegalAction
from .model import Action, ActionKind as K, Discard, Meld, Phase, Transfer, Transition, Win
from .riichi_model import RiichiConfig, RiichiPlayer, RiichiState
from .riichi_scoring import ORPHANS, RiichiContext, outside, riichi_waits, round100, score_hand, shapes
from .tiles import counts_of


class RiichiEngine(Engine):
    def __init__(self, config=None, seed=0, *, wall=None):
        self.config = config or RiichiConfig()
        if not isinstance(self.config, RiichiConfig) or type(seed) is not int:
            raise ValueError('riichi config and integer seed required')
        self.seed = seed
        deck = list(range(136)) if wall is None else list(wall)
        if any(type(t) is not int for t in deck) or sorted(deck) != list(range(136)):
            raise ValueError('wall must be a permutation of 136 physical tiles')
        if wall is None:
            random.Random(seed).shuffle(deck)
        self.initial_wall = tuple(deck)
        self.events, self.commands = [], []
        self.public_history, self._announced_indicators = [], {}
        players = [RiichiPlayer() for _ in range(4)]
        self.state = RiichiState(players, deck[:-14], 0, 122, Phase.SELF_TURN, self.config.dealer,
                                 scores=list(self.config.initial_scores), dead_wall=deck[-14:],
                                 riichi_sticks=self.config.riichi_sticks, initial_bank=self.config.riichi_sticks * 1000)
        self._emit('round_started', {'dealer': self.config.dealer, 'rules': asdict(self.config), 'variant': 'riichi'})
        for _ in range(13):
            for offset in range(4):
                player = (self.config.dealer + offset) % 4
                players[player].concealed.append(deck[self.state.head])
                self.state.head += 1
        for player, hand in enumerate(players):
            hand.concealed.sort()
            self._emit('dealt', {'player': player, 'count': 13}, private={player: {'tiles': list(hand.concealed)}})
        self._emit('dora_revealed', {'indicators': self.indicators()})
        self._draw(self.config.dealer, 'initial')
        self.assert_invariants()

    def _emit(self, kind, data, *, private=None, visible_to=None):
        from .public_history import remember_public
        remember_public(self, kind, data)
        super()._emit(kind, data, private=private, visible_to=visible_to)

    def indicators(self, ura=False):
        return [self.state.dead_wall[4 + 2 * i + int(ura)] for i in range(self.state.indicator_count)]

    def _waits_for(self, player, tiles=None, melds=None):
        hand = self.state.players[player]
        tiles, melds = hand.concealed if tiles is None else tiles, hand.melds if melds is None else melds
        owned = Counter(t // 4 for t in tiles + [t for m in melds for t in m.tiles])
        return tuple(k for k in riichi_waits(counts_of(tiles), len(melds)) if owned[k] < 4)

    def _discard_furiten(self, player):
        hand = self.state.players[player]
        tiles = list(hand.concealed)
        if len(tiles) + 3 * len(hand.melds) == 14 and hand.last_drawn in tiles:
            tiles.remove(hand.last_drawn)
        waits = self._waits_for(player, tiles)
        return bool(set(waits) & {d.tile // 4 for d in hand.river})

    def _furiten(self, player):
        hand = self.state.players[player]
        return hand.temporary_furiten or hand.riichi_furiten or self._discard_furiten(player)

    def _context(self, player, self_draw=False):
        hand, state = self.state.players[player], self.state
        first = hand.discard_count == 0 and not state.any_call
        return RiichiContext(self_draw=self_draw,
                            seat_wind=27 + (player - self.config.dealer) % 4, round_wind=self.config.round_wind,
                            riichi=hand.riichi, double_riichi=hand.double_riichi, ippatsu=hand.ippatsu,
                            rinshan=self_draw and hand.draw_source == 'kong',
                            chankan=not self_draw and state.phase == Phase.ROB_KONG_RESPONSE and state.pending_kong_kind == 'added_kong',
                            haitei=self_draw and state.remaining_tiles == 0 and hand.draw_source != 'kong',
                            houtei=not self_draw and state.phase == Phase.DISCARD_RESPONSE and state.remaining_tiles == 0 and state.discard_kong_id is None,
                            tenhou=self_draw and first and player == self.config.dealer and hand.draw_source == 'initial',
                            chiihou=self_draw and first and player != self.config.dealer and hand.draw_source == 'wall')

    def _self_value(self, player):
        hand = self.state.players[player]
        if hand.last_drawn is None:
            return None
        return score_hand(hand.concealed, hand.melds, hand.last_drawn, context=self._context(player, True), config=self.config,
                          indicators=self.indicators(), ura_indicators=self.indicators(True))

    def _ron_value(self, player):
        state, hand = self.state, self.state.players[player]
        if state.pending_tile is None or self._furiten(player):
            return None
        if state.pending_tile // 4 not in self._waits_for(player):
            return None
        if state.phase == Phase.ROB_KONG_RESPONSE and state.pending_kong_kind == 'concealed_kong':
            if not any(s.family == 'thirteen_orphans' for s in shapes(counts_of(hand.concealed + [state.pending_tile]), len(hand.melds))):
                return None
        return score_hand(hand.concealed + [state.pending_tile], hand.melds, state.pending_tile,
                          context=self._context(player), config=self.config,
                          indicators=self.indicators(), ura_indicators=self.indicators(True))

    def _riichi_kan_ok(self, player, quad):
        hand = self.state.players[player]
        if hand.last_drawn not in quad:
            return False
        before = list(hand.concealed)
        before.remove(hand.last_drawn)
        after = [t for t in hand.concealed if t not in quad]
        after_melds = hand.melds + [Meld('concealed_kong', quad)]
        waits = self._waits_for(player, before)
        if waits != self._waits_for(player, after, after_melds):
            return False
        quad_kind = quad[0] // 4
        for incoming in waits:
            old, new = list(counts_of(before)), list(counts_of(after))
            old[incoming] += 1
            new[incoming] += 1
            signatures = set()
            for shape in shapes(tuple(old), len(hand.melds)):
                if shape.family != 'standard' or (quad_kind,) * 3 not in shape.groups:
                    return False
                groups = list(shape.groups)
                groups.remove((quad_kind,) * 3)
                signatures.add((shape.pair, tuple(sorted(groups))))
            if signatures != {(s.pair, tuple(sorted(s.groups))) for s in shapes(tuple(new), len(after_melds)) if s.family == 'standard'}:
                return False
        return True

    def _call_forbidden(self, action, tile):
        kind = tile // 4
        forbidden = {kind}
        if action.kind == K.CHI:
            kinds = sorted([kind] + [t // 4 for t in action.tiles])
            start = kinds[0]
            if kind == start and start % 9 < 6:
                forbidden.add(start + 3)
            if kind == start + 2 and start % 9 > 0:
                forbidden.add(start - 1)
        return tuple(sorted(forbidden)) if self.config.kuikae_forbidden else ()

    def legal_actions(self, player):
        state = self.state
        if type(player) is not int or not 0 <= player < 4:
            raise ValueError('invalid player')
        if player not in state.responders or player in state.replies or state.phase == Phase.TERMINAL:
            return ()
        hand, tiles = state.players[player], sorted(state.players[player].concealed)
        if state.phase == Phase.SELF_TURN:
            discards = [t for t in tiles if t // 4 not in hand.forbidden_discards and (not hand.riichi or t == hand.last_drawn)]
            result = [Action(K.DISCARD, (t,)) for t in discards]
            if self._self_value(player):
                result.append(Action(K.TSUMO))
            if hand.closed and not hand.riichi and state.scores[player] >= 1000 and state.remaining_tiles >= self.config.riichi_min_live:
                result.extend(Action(K.RIICHI, (tile,)) for tile in discards if self._waits_for(player, [t for t in tiles if t != tile]))
            if hand.last_drawn is not None and state.remaining_tiles and state.rinshan_count < 4:
                for kind in sorted({t // 4 for t in tiles}):
                    same = tuple(t for t in tiles if t // 4 == kind)
                    if len(same) == 4 and (not hand.riichi or self._riichi_kan_ok(player, same)):
                        result.append(Action(K.CONCEALED_KONG, same))
                if not hand.riichi:
                    for i, meld in enumerate(hand.melds):
                        if meld.kind == 'peng':
                            result.extend(Action(K.ADDED_KONG, (t,), meld_index=i) for t in tiles if t // 4 == meld.tiles[0] // 4)
            if self.config.abortive_draws and hand.discard_count == 0 and not state.any_call and len({t // 4 for t in tiles} & set(ORPHANS)) >= 9:
                result.append(Action(K.ABORT_NINE))
            return tuple(result)
        result = [Action(K.PASS)]
        if self._ron_value(player):
            result.append(Action(K.RON))
        if state.phase != Phase.DISCARD_RESPONSE or hand.riichi or not state.remaining_tiles or state.four_kans_pending:
            return tuple(result)
        kind = state.pending_tile // 4
        same = [t for t in tiles if t // 4 == kind]
        calls = [Action(K.PENG, pair) for pair in combinations(same, 2)]
        if len(same) == 3 and state.rinshan_count < 4:
            calls.append(Action(K.OPEN_KONG, tuple(same)))
        if player == (state.actor + 1) % 4 and kind < 27:
            for start in range(max(kind // 9 * 9, kind - 2), min(kind, kind // 9 * 9 + 6) + 1):
                needed = [k for k in (start, start + 1, start + 2) if k != kind]
                for pair in product(*[[t for t in tiles if t // 4 == k] for k in needed]):
                    calls.append(Action(K.CHI, tuple(sorted(pair))))
        for action in calls:
            forbidden = self._call_forbidden(action, state.pending_tile)
            if action.kind == K.OPEN_KONG or any(t not in action.tiles and t // 4 not in forbidden for t in tiles):
                result.append(action)
        return tuple(result)

    def submit(self, player, action, decision_id):
        state = self.state
        if type(player) is not int or not 0 <= player < 4 or type(decision_id) is not int or decision_id != state.decision_id:
            raise IllegalAction('invalid player or stale decision_id')
        if not isinstance(action, Action) or action not in self.legal_actions(player):
            raise IllegalAction('action is not legal for this player and decision')
        before, scores = len(self.events), list(state.scores)
        self.commands.append({'player': player, 'decision_id': decision_id, 'action': action.to_dict()})
        state.revision += 1
        state.replies[player] = action
        if state.phase == Phase.SELF_TURN:
            self._resolve_self(action)
        elif len(state.replies) == len(state.responders):
            self._resolve_responses()
        self.assert_invariants()
        return Transition(tuple(deepcopy(self.events[before:])), tuple(a - b for a, b in zip(state.scores, scores)), state.decision_id, state.phase == Phase.TERMINAL)

    def _interrupt(self):
        self.state.any_call = True
        for hand in self.state.players:
            hand.ippatsu = False

    def _reveal_pending(self):
        if self.state.delayed_indicators:
            self.state.indicator_count += self.state.delayed_indicators
            self.state.delayed_indicators = 0
            self._emit('dora_revealed', {'indicators': self.indicators()})

    def _resolve_self(self, action):
        state, player = self.state, self.state.actor
        hand = state.players[player]
        if action.kind in (K.DISCARD, K.RIICHI):
            if hand.riichi:
                hand.ippatsu = False
            tile = action.tiles[0]
            if action.kind == K.RIICHI:
                state.pending_riichi = player
                self._emit('riichi_declared', {'player': player, 'tile': tile})
            hand.concealed.remove(tile)
            hand.river.append(Discard(tile, tile == hand.last_drawn))
            hand.discard_count += 1
            hand.forbidden_discards = ()
            state.discard_kong_id = state.kong_count if hand.draw_source == 'kong' else None
            hand.last_drawn, hand.draw_source = None, None
            self._reveal_pending()
            self._emit('discard', {'player': player, 'tile': tile, 'tsumogiri': hand.river[-1].tsumogiri})
            self._window(Phase.DISCARD_RESPONSE, player, tuple((player + i) % 4 for i in range(1, 4)), tile=tile)
        elif action.kind == K.TSUMO:
            self._win([player], self_draw=True)
        elif action.kind == K.ABORT_NINE:
            self._finish('nine_terminals')
        else:
            self._interrupt()
            state.pending_kong_tiles = action.tiles
            state.pending_kong_kind = action.kind.value
            state.pending_meld = action.meld_index
            tile = hand.last_drawn if hand.last_drawn in action.tiles else action.tiles[0]
            self._emit('kong_proposed', {'player': player, 'tile': tile, 'kind': action.kind.value, 'meld_index': action.meld_index})
            self._window(Phase.ROB_KONG_RESPONSE, player, tuple((player + i) % 4 for i in range(1, 4)), tile=tile, meld=action.meld_index)

    def _accept_riichi(self):
        player = self.state.pending_riichi
        if player is None:
            return
        hand = self.state.players[player]
        hand.riichi, hand.ippatsu = True, True
        hand.double_riichi = hand.discard_count == 1 and not self.state.any_call
        hand.riichi_discard = len(hand.river) - 1
        hand.riichi_waits = self._waits_for(player)
        self._transfer(player, 4, 1000, 'riichi_deposit')
        self._emit('riichi_accepted', {'player': player, 'double': hand.double_riichi, 'discard_index': hand.riichi_discard})
        self.state.pending_riichi = None

    def _resolve_responses(self):
        state, source = self.state, self.state.actor
        chosen = [p for p in state.responders if state.replies[p].kind == K.RON]
        for player in state.responders:
            rob_concealed = state.phase == Phase.ROB_KONG_RESPONSE and state.pending_kong_kind == 'concealed_kong'
            can_rob = not rob_concealed or any(s.family == 'thirteen_orphans' for s in shapes(counts_of(state.players[player].concealed + [state.pending_tile]), len(state.players[player].melds)))
            if player not in chosen and can_rob and state.pending_tile // 4 in self._waits_for(player):
                hand = state.players[player]
                hand.temporary_furiten = True
                if hand.riichi:
                    hand.riichi_furiten = True
        if chosen:
            if len(chosen) == 3 and self.config.abortive_draws and self.config.triple_ron_abort:
                self._finish('triple_ron')
                return
            winners = chosen if self.config.multi_ron else chosen[:1]
            self._win(winners)
            return
        if state.phase == Phase.ROB_KONG_RESPONSE:
            self._confirm_kong(source, Action(K(state.pending_kong_kind), state.pending_kong_tiles, meld_index=state.pending_meld))
            return
        self._accept_riichi()
        if state.four_kans_pending:
            self._finish('four_kans')
            return
        calls = [(p, state.replies[p]) for p in state.responders if state.replies[p].kind in (K.PENG, K.OPEN_KONG)]
        if not calls:
            calls = [(p, state.replies[p]) for p in state.responders if state.replies[p].kind == K.CHI]
        if calls:
            player, action = calls[0]
            self._interrupt()
            if action.kind == K.OPEN_KONG:
                self._confirm_kong(player, action, source=source, called_tile=state.pending_tile)
                return
            hand = state.players[player]
            for tile in action.tiles:
                hand.concealed.remove(tile)
            meld = Meld(action.kind.value, tuple(sorted(action.tiles + (state.pending_tile,))), source, state.pending_tile)
            hand.melds.append(meld)
            hand.forbidden_discards = self._call_forbidden(action, state.pending_tile)
            state.players[source].river[-1].called_by = player
            hand.last_drawn, hand.draw_source = None, None
            self._mark_pao(player, source)
            self._emit(action.kind.value, {'player': player, 'source': source, 'meld': asdict(meld)})
            self._window(Phase.SELF_TURN, player, (player,))
            return
        if self.config.abortive_draws:
            if all(p.riichi for p in state.players):
                self._finish('four_riichi')
                return
            if not state.any_call and all(p.discard_count == 1 for p in state.players):
                first = [p.river[0].tile // 4 for p in state.players]
                if len(set(first)) == 1 and first[0] in range(27, 31):
                    self._finish('four_winds')
                    return
        self._draw((source + 1) % 4)

    def _mark_pao(self, player, source):
        if source is None or not self.config.pao:
            return
        melds = self.state.players[player].melds
        sets = {m.tiles[0] // 4 for m in melds if m.kind != 'chi'}
        for name, needed in (('大三元', {31, 32, 33}), ('大四喜', {27, 28, 29, 30})):
            if needed <= sets and name not in self.state.players[player].pao:
                self.state.players[player].pao[name] = source
        if sum(m.kind.endswith('kong') for m in melds) == 4:
            self.state.players[player].pao.setdefault('四杠子', source)

    def _confirm_kong(self, player, action, *, source=None, called_tile=None):
        state, hand = self.state, self.state.players[player]
        self._reveal_pending()
        for tile in action.tiles:
            hand.concealed.remove(tile)
        if action.kind == K.ADDED_KONG:
            meld = hand.melds[action.meld_index]
            meld.kind, meld.added_tile = 'added_kong', action.tiles[0]
            meld.tiles = tuple(sorted(meld.tiles + action.tiles))
        else:
            meld = Meld(action.kind.value, tuple(sorted(action.tiles + (() if called_tile is None else (called_tile,)))), source, called_tile)
            hand.melds.append(meld)
            if source is not None:
                state.players[source].river[-1].called_by = player
                self._mark_pao(player, source)
        state.kong_count += 1
        state.pending_kong_tiles, state.pending_kong_kind = (), None
        makers = sum(any(m.kind.endswith('kong') for m in p.melds) for p in state.players)
        state.four_kans_pending = self.config.abortive_draws and state.kong_count == 4 and makers > 1
        self._emit('kong_confirmed', {'player': player, 'kong_id': state.kong_count, 'meld': asdict(meld)})
        if action.kind == K.CONCEALED_KONG or not self.config.delayed_open_kan_dora:
            state.indicator_count += 1
            self._emit('dora_revealed', {'indicators': self.indicators()})
        else:
            state.delayed_indicators += 1
        self._draw(player, 'kong')

    def _draw(self, player, source='wall', **unused):
        state = self.state
        if not state.remaining_tiles:
            self._finish('wall_exhausted')
            return
        if source == 'kong':
            tile = state.dead_wall[state.rinshan_count]
            state.tail -= 1
            state.dead_wall[state.rinshan_count] = state.wall[state.tail]
            state.rinshan_count += 1
        else:
            tile = state.wall[state.head]
            state.head += 1
        hand = state.players[player]
        hand.concealed.append(tile)
        hand.concealed.sort()
        hand.last_drawn, hand.draw_source = tile, source
        hand.temporary_furiten = False
        hand.forbidden_discards = ()
        state.pending_kong_tiles, state.pending_kong_kind = (), None
        state.discard_kong_id = None
        self._emit('draw', {'player': player, 'source': source, 'remaining': state.remaining_tiles}, private={player: {'tile': tile}})
        self._window(Phase.SELF_TURN, player, (player,))

    def _transfer(self, payer, receiver, amount, reason, **kwargs):
        if not amount:
            return
        transfer = Transfer(len(self.state.transfers), payer, receiver, amount, reason)
        self.state.transfers.append(transfer)
        if payer == 4:
            self.state.riichi_sticks -= amount // 1000
        else:
            self.state.scores[payer] -= amount
        if receiver == 4:
            self.state.riichi_sticks += amount // 1000
        else:
            self.state.scores[receiver] += amount
        self._emit('payment', asdict(transfer))

    def _pay_value(self, player, value, source=None):
        dealer = player == self.config.dealer
        liabilities = []
        if value.yakuman and self.config.pao:
            for item in value.items:
                name = '大四喜' if item.name == '大四喜' else item.name
                if name in self.state.players[player].pao:
                    liabilities.append((self.state.players[player].pao[name], item.fan // 13 * 8000))
        remaining = value.base_points - sum(base for _, base in liabilities)
        for liable, base in liabilities:
            full = base * (6 if dealer else 4)
            if source is None or source == liable:
                self._transfer(liable, player, full, 'pao')
            else:
                self._transfer(liable, player, full // 2, 'pao')
                self._transfer(source, player, full // 2, 'ron')
        if source is not None:
            self._transfer(source, player, round100(remaining * (6 if dealer else 4)), 'ron')
            self._transfer(source, player, self.config.honba * 300, 'honba')
        else:
            for payer in range(4):
                if payer != player:
                    self._transfer(payer, player, round100(remaining * (2 if dealer or payer == self.config.dealer else 1)), 'tsumo')
            if remaining == 0 and liabilities and len({p for p, _ in liabilities}) == 1:
                self._transfer(liabilities[0][0], player, self.config.honba * 300, 'honba')
            else:
                for payer in range(4):
                    if payer != player:
                        self._transfer(payer, player, self.config.honba * 100, 'honba')

    def _win(self, winners, self_draw=False):
        state, source = self.state, self.state.actor
        values = {p: self._self_value(p) if self_draw else self._ron_value(p) for p in winners}
        tile = state.players[source].last_drawn if self_draw else state.pending_tile
        if not self_draw and state.phase == Phase.ROB_KONG_RESPONSE:
            state.players[source].concealed.remove(tile)
            state.players[source].last_drawn, state.players[source].draw_source = None, None
            state.robbed_tiles.append(tile)
            self._emit('kong_robbed', {'player': source, 'tile': tile, 'winners': winners})
        elif not self_draw:
            state.players[source].river[-1].winner_refs.extend(winners)
        for player in winners:
            value = values[player]
            self._pay_value(player, value, None if self_draw else source)
            self._record_win(Win(player, None if self_draw else source, tile, self_draw, value))
        if any(state.players[p].riichi for p in winners):
            self._emit('ura_revealed', {'indicators': self.indicators(True)})
        if state.riichi_sticks:
            self._transfer(4, winners[0], state.riichi_sticks * 1000, 'riichi_award')
        self._finish('tsumo' if self_draw else 'ron')

    def _finish(self, reason):
        state = self.state
        detail = {'ready': {}, 'noten': [], 'nagashi': [], 'dealer_continues': False}
        if reason == 'wall_exhausted':
            ready = [p for p in range(4) if self._waits_for(p)]
            detail['ready'] = {str(p): {'waits': list(self._waits_for(p))} for p in ready}
            detail['noten'] = [p for p in range(4) if p not in ready]
            nagashi = [p for p, hand in enumerate(state.players) if hand.river and all(outside(d.tile // 4) and d.called_by is None for d in hand.river) and all(m.kind == 'concealed_kong' for m in hand.melds)] if self.config.nagashi_mangan else []
            detail['nagashi'] = nagashi
            if nagashi:
                for winner in nagashi:
                    for payer in range(4):
                        if payer != winner:
                            self._transfer(payer, winner, 4000 if winner == self.config.dealer or payer == self.config.dealer else 2000, 'nagashi_mangan')
                            self._transfer(payer, winner, self.config.honba * 100, 'honba')
            elif 0 < len(ready) < 4:
                owed = {p: 3000 // len(detail['noten']) for p in detail['noten']}
                due = {p: 3000 // len(ready) for p in ready}
                for payer in detail['noten']:
                    for receiver in ready:
                        amount = min(owed[payer], due[receiver])
                        self._transfer(payer, receiver, amount, 'noten')
                        owed[payer] -= amount
                        due[receiver] -= amount
            # Nagashi is treated as an exhaustive draw for dealer continuation and deposits.
            detail['dealer_continues'] = self.config.dealer in ready
        elif reason in ('ron', 'tsumo'):
            detail['dealer_continues'] = any(w.player == self.config.dealer for w in state.wins)
        else:
            detail['dealer_continues'] = True
        detail['riichi_sticks'] = state.riichi_sticks
        state.end_reason, state.settlement = reason, detail
        state.pending_riichi = None
        state.pending_kong_tiles, state.pending_kong_kind = (), None
        self._window(Phase.TERMINAL, state.actor, ())
        self._emit('round_ended', {'reason': reason, 'scores': list(state.scores), 'settlement': detail})

    def observe(self, player):
        observation = super().observe(player)
        state = self.state
        for seat, item in enumerate(observation['players']):
            hand = state.players[seat]
            item.update(riichi=hand.riichi, double_riichi=hand.double_riichi, riichi_discard=hand.riichi_discard,
                        seat_wind=27 + (seat - self.config.dealer) % 4)
            if seat == player:
                item.update(furiten=self._furiten(player), temporary_furiten=hand.temporary_furiten,
                            riichi_furiten=hand.riichi_furiten, ippatsu=hand.ippatsu)
        observation.update(variant='riichi', dealer=self.config.dealer, round_wind=self.config.round_wind,
                           pending_tile=state.pending_tile,
                           hand_number=self.config.hand_number, honba=self.config.honba, riichi_sticks=state.riichi_sticks,
                           dora_indicators=self.indicators(), red_fives=list(self.config.red_fives),
                           ura_indicators=self.indicators(True) if state.phase == Phase.TERMINAL and any(p.riichi and not p.can_act for p in state.players) else [])
        from .public_history import observation_context
        observation.update(observation_context(self))
        return deepcopy(observation)

    def assert_invariants(self):
        state = self.state
        if not 0 <= state.head <= state.tail <= 122 or len(state.dead_wall) != 14:
            raise AssertionError('invalid live / dead wall boundaries')
        if not 0 <= state.rinshan_count <= 4 or state.kong_count != state.rinshan_count:
            raise AssertionError('invalid replacement draw count')
        if state.indicator_count + state.delayed_indicators != state.kong_count + 1 or not 1 <= state.indicator_count <= 5:
            raise AssertionError('invalid dora indicator count')
        owned = list(state.wall[state.head:state.tail]) + state.dead_wall + state.robbed_tiles
        winners = {w.player: w for w in state.wins}
        if len(winners) != len(state.wins) or len(winners) > 3:
            raise AssertionError('invalid winners')
        for player, hand in enumerate(state.players):
            owned.extend(hand.concealed)
            for meld in hand.melds:
                if meld.kind not in ('chi', 'peng', 'open_kong', 'concealed_kong', 'added_kong'):
                    raise AssertionError('invalid meld kind')
                expected = 4 if meld.kind.endswith('kong') else 3
                kinds = sorted(t // 4 for t in meld.tiles)
                if len(meld.tiles) != expected:
                    raise AssertionError('invalid meld size')
                if meld.kind == 'chi':
                    if kinds[0] >= 27 or kinds != list(range(kinds[0], kinds[0] + 3)) or kinds[0] // 9 != kinds[-1] // 9:
                        raise AssertionError('invalid chi')
                elif len(set(kinds)) != 1:
                    raise AssertionError('invalid triplet/quad')
                if meld.kind == 'concealed_kong' and meld.source_player is not None:
                    raise AssertionError('concealed quad has a source')
                owned.extend(meld.tiles)
            for discard in hand.river:
                if discard.called_by is None:
                    owned.append(discard.tile)
                elif not any(m.called_tile == discard.tile and m.source_player == player for m in state.players[discard.called_by].melds):
                    raise AssertionError('called discard lacks owning meld')
            effective = len(hand.concealed) + 3 * len(hand.melds)
            if state.phase == Phase.TERMINAL:
                expected = 14 if player in winners and winners[player].self_draw else 13
                if not state.wins and effective == 14 and player == state.actor:
                    expected = 14  # nine-terminals abort before discard
            else:
                expected = 13 + int(player == state.actor and state.phase in (Phase.SELF_TURN, Phase.ROB_KONG_RESPONSE))
            if effective != expected:
                raise AssertionError(f'wrong effective count for player {player}: {effective}, expected {expected}')
            if hand.last_drawn is not None and hand.last_drawn not in hand.concealed:
                raise AssertionError('last drawn is not owned')
            if hand.riichi and not hand.closed:
                raise AssertionError('riichi hand is open')
            if hand.can_act == (player in winners):
                raise AssertionError('winner activity mismatch')
        if Counter(owned) != Counter(range(136)):
            raise AssertionError('physical inventory mismatch')
        scores = list(self.config.initial_scores)
        bank = state.initial_bank
        for index, transfer in enumerate(state.transfers):
            if transfer.id != index or transfer.amount <= 0 or transfer.amount % 100 or transfer.payer == transfer.receiver or not 0 <= transfer.payer <= 4 or not 0 <= transfer.receiver <= 4:
                raise AssertionError('invalid point transfer')
            if transfer.payer == 4:
                bank -= transfer.amount
            else:
                scores[transfer.payer] -= transfer.amount
            if transfer.receiver == 4:
                bank += transfer.amount
            else:
                scores[transfer.receiver] += transfer.amount
        if scores != state.scores or bank != state.riichi_sticks * 1000 or bank < 0:
            raise AssertionError('scores and deposits disagree with ledger')
        if sum(scores) + bank != sum(self.config.initial_scores) + state.initial_bank:
            raise AssertionError('points are not conserved')
        if set(state.replies) - set(state.responders):
            raise AssertionError('response outside window')
        if state.phase == Phase.TERMINAL and (state.responders or state.end_reason is None):
            raise AssertionError('incomplete terminal state')
