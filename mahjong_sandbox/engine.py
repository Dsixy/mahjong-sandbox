"""Authoritative, synchronous blood-battle reference engine.

Only submit() changes state. Parallel decisions remain sealed until resolution.
No timers, I/O, model calls, or global RNG live in the rules loop.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import hashlib
import json
import random
from typing import Any

from .model import (
    Action, ActionKind, Discard, Event, Meld, Phase, PlayerState, RuleConfig,
    State, Transfer, Transition, Win,
)
from .scoring import WinContext, evaluate_hand, ready_values
from .tiles import SICHUAN_TILES, kind_of


class IllegalAction(ValueError):
    """Rejected commands leave state, event history, and replay unchanged."""


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class Engine:
    def __init__(
        self, config: RuleConfig | None = None, seed: int = 0, *,
        wall: list[int] | tuple[int, ...] | None = None, exchange_offset: int | None = None,
    ) -> None:
        self.config = config or RuleConfig()
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        rng = random.Random(seed)
        deck = list(SICHUAN_TILES) if wall is None else list(wall)
        if any(type(t) is not int for t in deck) or sorted(deck) != list(SICHUAN_TILES):
            raise ValueError("wall must be a permutation of the 108 physical tiles")
        if wall is None:
            rng.shuffle(deck)
        if exchange_offset is None:
            exchange_offset = rng.choice((1, 2, 3))
        if type(exchange_offset) is not int or exchange_offset not in (1, 2, 3):
            raise ValueError("exchange offset must be 1, 2, or 3")
        self.exchange_offset = exchange_offset
        self.initial_wall = tuple(deck)
        self.events: list[Event] = []
        self.commands: list[dict[str, Any]] = []
        players = [PlayerState() for _ in range(4)]
        self.state = State(players, deck, 0, len(deck), Phase.DECLARE_SUIT, self.config.dealer)
        self._emit("round_started", {"dealer": self.config.dealer, "rules": asdict(self.config)})
        # 13 each, followed by the dealer's 14th tile, before exchange/定缺.
        for _ in range(13):
            for offset in range(4):
                player = (self.config.dealer + offset) % 4
                players[player].concealed.append(deck[self.state.head])
                self.state.head += 1
        dealer = players[self.config.dealer]
        dealer.last_drawn = deck[self.state.head]
        dealer.draw_source = "initial"
        dealer.concealed.append(dealer.last_drawn)
        self.state.head += 1
        for player, hand in enumerate(players):
            hand.concealed.sort()
            self._emit("dealt", {"player": player, "count": len(hand.concealed)},
                       private={player: {"tiles": list(hand.concealed)}})
        phase = Phase.EXCHANGE if self.config.exchange_three else Phase.DECLARE_SUIT
        self._window(phase, self.config.dealer, tuple(range(4)))
        self.assert_invariants()

    def _emit(self, kind: str, data: dict[str, Any], *, private=None, visible_to=None) -> None:
        self.events.append(Event(len(self.events), kind, deepcopy(data), deepcopy(private or {}), visible_to))

    def _window(self, phase: Phase, actor: int, responders: tuple[int, ...], *, tile=None, meld=None) -> None:
        state = self.state
        state.phase, state.actor = phase, actor
        state.responders, state.replies = responders, {}
        state.pending_tile, state.pending_meld = tile, meld
        state.decision_id += 1

    def next_player(self) -> int | None:
        return next((p for p in self.state.responders if p not in self.state.replies), None)

    def _other_active(self, player: int) -> tuple[int, ...]:
        return tuple((player + offset) % 4 for offset in range(1, 4)
                     if self.state.players[(player + offset) % 4].can_act)

    def _self_value(self, player: int):
        hand = self.state.players[player]
        if hand.last_drawn is None and hand.draw_source != "initial":
            return None
        return evaluate_hand(hand, self.config, context=WinContext(
            self_draw=True, kong_draw=hand.draw_source == "kong",
            last_tile=self.state.remaining_tiles == 0,
        ))

    def _ron_value(self, player: int):
        state = self.state
        if state.pending_tile is None or not state.players[player].can_act:
            return None
        rob = state.phase == Phase.ROB_KONG_RESPONSE
        value = evaluate_hand(state.players[player], self.config, tile_kind=kind_of(state.pending_tile),
                              context=WinContext(rob_kong=rob,
                              kong_discard=not rob and state.discard_kong_id is not None,
                              last_tile=not rob and state.remaining_tiles == 0))
        if value and self.config.passed_ron_lock and value.fan <= state.players[player].passed_ron_fan:
            return None
        return value

    def legal_actions(self, player: int) -> tuple[Action, ...]:
        state = self.state
        if type(player) is not int or not 0 <= player < 4:
            raise ValueError("invalid player")
        if player not in state.responders or player in state.replies or state.phase == Phase.TERMINAL:
            return ()
        hand = state.players[player]
        tiles = sorted(hand.concealed)
        if state.phase == Phase.EXCHANGE:
            return tuple(Action(ActionKind.EXCHANGE, selected)
                         for suit in range(3)
                         for selected in combinations([t for t in tiles if kind_of(t) // 9 == suit], 3))
        if state.phase == Phase.DECLARE_SUIT:
            return tuple(Action(ActionKind.DECLARE_SUIT, suit=suit) for suit in range(3))
        if state.phase == Phase.SELF_TURN:
            value = self._self_value(player)
            if value and self.config.force_win_last_four and state.remaining_tiles <= 3:
                return (Action(ActionKind.TSUMO),)
            discardable = [t for t in tiles if kind_of(t) // 9 == hand.missing_suit] if hand.has_missing_tiles() else tiles
            result = [Action(ActionKind.DISCARD, (tile,)) for tile in discardable]
            if value:
                result.append(Action(ActionKind.TSUMO))
            if state.remaining_tiles and not hand.has_missing_tiles():
                by_kind: dict[int, list[int]] = {}
                for tile in tiles:
                    by_kind.setdefault(kind_of(tile), []).append(tile)
                if len(hand.melds) < 4:
                    for same in by_kind.values():
                        if len(same) == 4:
                            result.append(Action(ActionKind.CONCEALED_KONG, tuple(same)))
                for index, meld in enumerate(hand.melds):
                    if meld.kind == "peng":
                        for tile in by_kind.get(kind_of(meld.tiles[0]), ()):
                            result.append(Action(ActionKind.ADDED_KONG, (tile,), meld_index=index))
            return tuple(result)
        value = self._ron_value(player)
        if value and self.config.force_win_last_four and state.remaining_tiles <= 3:
            return (Action(ActionKind.RON),)
        result = [Action(ActionKind.PASS)]
        if value:
            result.append(Action(ActionKind.RON))
        if state.phase == Phase.DISCARD_RESPONSE and not hand.has_missing_tiles():
            kind = kind_of(state.pending_tile)
            same = [tile for tile in tiles if kind_of(tile) == kind]
            if kind // 9 != hand.missing_suit and len(hand.melds) < 4:
                result.extend(Action(ActionKind.PENG, consumed) for consumed in combinations(same, 2))
                if len(same) == 3 and state.remaining_tiles:
                    result.append(Action(ActionKind.OPEN_KONG, tuple(same)))
        return tuple(result)

    def submit(self, player: int, action: Action, decision_id: int) -> Transition:
        state = self.state
        if type(player) is not int or not 0 <= player < 4:
            raise IllegalAction("invalid player")
        if type(decision_id) is not int or decision_id != state.decision_id:
            raise IllegalAction("stale decision_id")
        if not isinstance(action, Action) or action not in self.legal_actions(player):
            raise IllegalAction("action is not legal for this player and decision")
        before_events = len(self.events)
        before_scores = state.scores.copy()
        self.commands.append({"player": player, "decision_id": decision_id, "action": action.to_dict()})
        state.revision += 1
        state.replies[player] = action
        if state.phase == Phase.SELF_TURN:
            self._resolve_self(action)
        elif len(state.replies) == len(state.responders):
            if state.phase == Phase.EXCHANGE:
                self._resolve_exchange()
            elif state.phase == Phase.DECLARE_SUIT:
                for seat, reply in state.replies.items():
                    state.players[seat].missing_suit = reply.suit
                self._emit("missing_suits_declared", {"suits": [p.missing_suit for p in state.players]})
                self._window(Phase.SELF_TURN, self.config.dealer, (self.config.dealer,))
            else:
                self._resolve_responses()
        self.assert_invariants()
        return Transition(tuple(deepcopy(self.events[before_events:])),
                          tuple(now - old for now, old in zip(state.scores, before_scores)),
                          state.decision_id, state.phase == Phase.TERMINAL)

    def _resolve_exchange(self) -> None:
        state = self.state
        selections = {p: reply.tiles for p, reply in state.replies.items()}
        for player, tiles in selections.items():
            hand = state.players[player]
            for tile in tiles:
                hand.concealed.remove(tile)
            if hand.last_drawn in tiles:
                hand.last_drawn = None
            self._emit("exchange_selected", {"player": player, "tiles": list(tiles)}, visible_to=(player,))
        received = {}
        for player, tiles in selections.items():
            target = (player + self.exchange_offset) % 4
            state.players[target].concealed.extend(tiles)
            state.players[target].concealed.sort()
            received[target] = {"received": list(tiles)}
        # Exchange is not a draw: no incoming tile is marked as 摸切.
        state.players[self.config.dealer].last_drawn = None
        self._emit("exchange_completed", {"offset": self.exchange_offset}, private=received)
        self._window(Phase.DECLARE_SUIT, self.config.dealer, tuple(range(4)))

    def _resolve_self(self, action: Action) -> None:
        state = self.state
        actor = state.actor
        hand = state.players[actor]
        if action.kind == ActionKind.DISCARD:
            tile = action.tiles[0]
            hand.concealed.remove(tile)
            tsumogiri = tile == hand.last_drawn
            hand.river.append(Discard(tile, tsumogiri))
            state.discard_kong_id = state.pending_kong_id if hand.draw_source == "kong" else None
            hand.last_drawn, hand.draw_source = None, None
            self._emit("discard", {"player": actor, "tile": tile, "tsumogiri": tsumogiri})
            self._window(Phase.DISCARD_RESPONSE, actor, self._other_active(actor), tile=tile)
        elif action.kind == ActionKind.TSUMO:
            value = self._self_value(actor)
            for payer in self._other_active(actor):
                self._transfer(payer, actor, value.payment, "tsumo")
            self._record_win(Win(actor, None, hand.last_drawn, True, value))
            self._continue_after(actor)
        elif action.kind == ActionKind.CONCEALED_KONG:
            self._confirm_kong(actor, action)
        elif action.kind == ActionKind.ADDED_KONG:
            state.pending_kong_id = None
            self._emit("kong_proposed", {"player": actor, "tile": action.tiles[0], "meld_index": action.meld_index})
            self._window(Phase.ROB_KONG_RESPONSE, actor, self._other_active(actor),
                         tile=action.tiles[0], meld=action.meld_index)

    def _resolve_responses(self) -> None:
        state = self.state
        source = state.actor
        rob = state.phase == Phase.ROB_KONG_RESPONSE
        offers = {p: self._ron_value(p) for p in state.responders}
        winners = [p for p in state.responders if state.replies[p].kind == ActionKind.RON]
        # Voluntary refusal is tracked even if a different player's call wins priority.
        if self.config.passed_ron_lock:
            for player, value in offers.items():
                if value and player not in winners:
                    state.players[player].passed_ron_fan = max(state.players[player].passed_ron_fan, value.fan)
        if winners:
            tile = state.pending_tile
            if rob:
                state.players[source].concealed.remove(tile)
                state.players[source].last_drawn = None
                state.players[source].draw_source = None
                state.robbed_tiles.append(tile)
                self._emit("kong_robbed", {"player": source, "tile": tile, "winners": winners})
            else:
                state.players[source].river[-1].winner_refs.extend(winners)
            for winner in winners:
                value = offers[winner]
                self._transfer(source, winner, value.payment, "ron")
                self._record_win(Win(winner, source, tile, False, value))
            if not rob and self.config.kong_discard_transfer and state.discard_kong_id is not None:
                income = sum(t.amount for t in state.transfers
                             if t.kong_id == state.discard_kong_id and t.reason in self.KONG_REASONS and t.receiver == source)
                amount, remainder = divmod(income, len(winners))
                for index, winner in enumerate(winners):
                    self._transfer(source, winner, amount + int(index < remainder), "kong_discard_transfer",
                                   kong_id=state.discard_kong_id)
            self._continue_after(source)
        elif rob:
            self._confirm_kong(source, Action(ActionKind.ADDED_KONG, (state.pending_tile,), meld_index=state.pending_meld))
        else:
            calls = [(p, state.replies[p]) for p in state.responders
                     if state.replies[p].kind in (ActionKind.PENG, ActionKind.OPEN_KONG)]
            if calls:
                player, action = calls[0]  # closest active seat; peng/open-kong share priority
                if action.kind == ActionKind.OPEN_KONG:
                    self._confirm_kong(player, action, source=source, called_tile=state.pending_tile)
                else:
                    hand = state.players[player]
                    for tile in action.tiles:
                        hand.concealed.remove(tile)
                    hand.melds.append(Meld("peng", tuple(sorted(action.tiles + (state.pending_tile,))), source, state.pending_tile))
                    state.players[source].river[-1].called_by = player
                    hand.last_drawn, hand.draw_source = None, None
                    state.pending_kong_id = None
                    self._emit("peng", {"player": player, "source": source, "meld": asdict(hand.melds[-1])})
                    self._window(Phase.SELF_TURN, player, (player,))
            else:
                self._continue_after(source)

    KONG_REASONS = ("concealed_kong", "added_kong", "open_kong")

    def _confirm_kong(self, player: int, action: Action, *, source=None, called_tile=None) -> None:
        state = self.state
        hand = state.players[player]
        for tile in action.tiles:
            hand.concealed.remove(tile)
        if action.kind == ActionKind.ADDED_KONG:
            meld = hand.melds[action.meld_index]
            meld.kind = "added_kong"
            meld.added_tile = action.tiles[0]
            meld.tiles = tuple(sorted(meld.tiles + action.tiles))
            reason, payers, amount = "added_kong", self._other_active(player), self.config.base_points
        elif action.kind == ActionKind.CONCEALED_KONG:
            meld = Meld("concealed_kong", action.tiles)
            hand.melds.append(meld)
            reason, payers, amount = "concealed_kong", self._other_active(player), 2 * self.config.base_points
        else:
            meld = Meld("open_kong", tuple(sorted(action.tiles + (called_tile,))), source, called_tile)
            hand.melds.append(meld)
            state.players[source].river[-1].called_by = player
            reason, payers, amount = "open_kong", (source,), 2 * self.config.base_points
        state.kong_count += 1
        kong_id = state.kong_count
        self._emit("kong_confirmed", {"player": player, "kong_id": kong_id, "meld": asdict(meld)})
        for payer in payers:
            self._transfer(payer, player, amount, reason, kong_id=kong_id)
        self._draw(player, "kong", kong_id=kong_id)

    def _draw(self, player: int, source: str = "wall", *, kong_id=None) -> None:
        state = self.state
        if state.remaining_tiles == 0:
            self._finish("wall_exhausted")
            return
        if source == "kong":
            state.tail -= 1
            tile = state.wall[state.tail]
        else:
            tile = state.wall[state.head]
            state.head += 1
        hand = state.players[player]
        hand.concealed.append(tile)
        hand.concealed.sort()
        hand.last_drawn, hand.draw_source = tile, source
        hand.passed_ron_fan = -1
        state.pending_kong_id = kong_id
        state.discard_kong_id = None
        self._emit("draw", {"player": player, "source": source, "remaining": state.remaining_tiles},
                   private={player: {"tile": tile}})
        self._window(Phase.SELF_TURN, player, (player,))

    def _transfer(self, payer: int, receiver: int, amount: int, reason: str, *, kong_id=None, refund_of=None) -> None:
        if amount == 0:
            return
        transfer = Transfer(len(self.state.transfers), payer, receiver, amount, reason, kong_id, refund_of)
        self.state.transfers.append(transfer)
        self.state.scores[payer] -= amount
        self.state.scores[receiver] += amount
        self._emit("payment", asdict(transfer))

    def _record_win(self, win: Win) -> None:
        self.state.wins.append(win)
        self.state.players[win.player].can_act = False
        self._emit("win", {"win": asdict(win), "concealed": list(self.state.players[win.player].concealed)})

    def _continue_after(self, source: int) -> None:
        if len(self.state.wins) >= 3:
            self._finish("three_winners")
        elif self.state.remaining_tiles == 0:
            self._finish("wall_exhausted")
        else:
            self._draw(self._other_active(source)[0])

    def _finish(self, reason: str) -> None:
        state = self.state
        detail: dict[str, Any] = {"flowers": [], "ready": {}, "noten": [], "refunds": []}
        if reason == "wall_exhausted":
            active = [p for p in range(4) if state.players[p].can_act]
            flowers = [p for p in active if state.players[p].has_missing_tiles()]
            ready = {p: ready_values(state.players[p], self.config) for p in active if p not in flowers}
            ready = {p: values for p, values in ready.items() if values}
            noten = [p for p in active if p not in flowers and p not in ready]
            detail["flowers"], detail["noten"] = flowers, noten
            detail["ready"] = {str(p): {"waits": sorted(values), "max_payment": max(v.payment for v in values.values())}
                               for p, values in ready.items()}
            if self.config.check_flower_pigs:
                for flower in flowers:
                    for receiver in range(4):
                        if receiver not in flowers:
                            self._transfer(flower, receiver, self.config.base_points * (1 << self.config.fan_cap), "flower_pig")
            if self.config.check_noten:
                for payer in noten:
                    for receiver, values in ready.items():
                        self._transfer(payer, receiver, max(v.payment for v in values.values()), "noten")
            if self.config.refund_noten_kongs:
                refundable = [t for t in state.transfers if t.reason in self.KONG_REASONS and t.receiver in noten + flowers]
                for transfer in refundable:
                    self._transfer(transfer.receiver, transfer.payer, transfer.amount, "kong_refund",
                                   kong_id=transfer.kong_id, refund_of=transfer.id)
                    detail["refunds"].append(transfer.id)
        state.end_reason, state.settlement = reason, detail
        self._window(Phase.TERMINAL, state.actor, ())
        state.pending_kong_id, state.discard_kong_id = None, None
        self._emit("round_ended", {"reason": reason, "scores": list(state.scores), "settlement": detail})

    def observe(self, player: int) -> dict[str, Any]:
        if type(player) is not int or not 0 <= player < 4:
            raise ValueError("invalid player")
        state = self.state
        players = []
        for seat, hand in enumerate(state.players):
            item = {"player": seat, "concealed_count": len(hand.concealed), "melds": [asdict(m) for m in hand.melds],
                    "river": [asdict(d) for d in hand.river], "missing_suit": hand.missing_suit, "can_act": hand.can_act}
            if seat == player or not hand.can_act:
                item["concealed"] = list(hand.concealed)
            if seat == player:
                item.update(last_drawn=hand.last_drawn, passed_ron_fan=hand.passed_ron_fan)
            players.append(item)
        visible = [event.for_player(player) for event in self.events]
        return deepcopy({"rule_version": self.config.rule_version, "player": player,
                         "phase": state.phase.value, "actor": state.actor, "decision_id": state.decision_id,
                         "remaining_tiles": state.remaining_tiles, "scores": list(state.scores),
                         "players": players, "wins": [asdict(w) for w in state.wins],
                         "legal_actions": [action.to_dict() for action in self.legal_actions(player)],
                         "events": [event for event in visible if event is not None], "end_reason": state.end_reason})

    def snapshot(self) -> dict[str, Any]:
        """Full trusted/debug state. Never pass this to an ordinary policy."""
        return deepcopy(asdict(self.state))

    def state_hash(self) -> str:
        return digest(self.snapshot())

    def event_hash(self) -> str:
        return digest([asdict(event) for event in self.events])

    def assert_invariants(self) -> None:
        state = self.state
        if not 0 <= state.head <= state.tail <= len(state.wall):
            raise AssertionError("invalid wall boundaries")
        owned = list(state.wall[state.head:state.tail]) + list(state.robbed_tiles)
        win_by_player = {win.player: win for win in state.wins}
        if len(win_by_player) != len(state.wins) or len(state.wins) > 3:
            raise AssertionError("duplicate or excess winners")
        for seat, hand in enumerate(state.players):
            owned.extend(hand.concealed)
            for meld in hand.melds:
                expected = 3 if meld.kind == "peng" else 4
                if meld.kind not in ("peng", *self.KONG_REASONS) or len(meld.tiles) != expected:
                    raise AssertionError("invalid meld structure")
                if len({kind_of(t) for t in meld.tiles}) != 1:
                    raise AssertionError("mixed-kind triplet or quad")
                if hand.missing_suit is not None and kind_of(meld.tiles[0]) // 9 == hand.missing_suit:
                    raise AssertionError("meld contains the declared missing suit")
                owned.extend(meld.tiles)
            for discard in hand.river:
                if discard.called_by is None:
                    owned.append(discard.tile)
                elif not any(m.called_tile == discard.tile and m.source_player == seat
                             for m in state.players[discard.called_by].melds):
                    raise AssertionError("called discard has no owning meld")
            effective = len(hand.concealed) + 3 * len(hand.melds)
            if not hand.can_act:
                if seat not in win_by_player or effective != (14 if win_by_player[seat].self_draw else 13):
                    raise AssertionError("invalid winning hand ownership")
            else:
                if seat in win_by_player:
                    raise AssertionError("winner remains active")
                extra = ((state.phase in (Phase.EXCHANGE, Phase.DECLARE_SUIT) and seat == self.config.dealer)
                         or (state.phase in (Phase.SELF_TURN, Phase.ROB_KONG_RESPONSE) and seat == state.actor))
                if effective != 13 + int(extra):
                    raise AssertionError(f"wrong effective tile count for player {seat}: {effective}")
            if hand.last_drawn is not None and hand.last_drawn not in hand.concealed:
                raise AssertionError("last-drawn tile is not owned")
            if state.phase not in (Phase.EXCHANGE, Phase.DECLARE_SUIT) and hand.missing_suit not in (0, 1, 2):
                raise AssertionError("missing suit not declared")
        if Counter(owned) != Counter(SICHUAN_TILES):
            raise AssertionError("physical tiles are missing or multiply owned")
        scores = [0] * 4
        for index, transfer in enumerate(state.transfers):
            if transfer.id != index or transfer.amount <= 0 or transfer.payer == transfer.receiver:
                raise AssertionError("invalid transfer")
            scores[transfer.payer] -= transfer.amount
            scores[transfer.receiver] += transfer.amount
        if scores != state.scores or sum(state.scores) != 0:
            raise AssertionError("scores disagree with zero-sum ledger")
        if set(state.replies) - set(state.responders):
            raise AssertionError("reply outside current decision window")
        if any(not state.players[p].can_act for p in state.responders):
            raise AssertionError("inactive player asked to act")
        if state.phase == Phase.TERMINAL and (state.responders or state.end_reason is None):
            raise AssertionError("incomplete terminal state")
