from copy import deepcopy
import unittest

from mahjong_sandbox import Action, ActionKind as K, Engine, IllegalAction, Phase, RuleConfig
from mahjong_sandbox.scoring import ready_values
from tests.helpers import exhaust_wall, pass_rest, scenario, select, submit


class OpeningTests(unittest.TestCase):
    def test_initial_deal_and_exchange_are_atomic(self):
        engine = Engine(seed=12, exchange_offset=2)
        self.assertEqual([len(p.concealed) for p in engine.state.players], [14, 13, 13, 13])
        self.assertEqual(engine.state.remaining_tiles, 55)
        before = [p.concealed.copy() for p in engine.state.players]
        chosen = {}
        for seat in range(4):
            chosen[seat] = engine.legal_actions(seat)[0].tiles
            engine.submit(seat, engine.legal_actions(seat)[0], engine.state.decision_id)
            if seat < 3:
                self.assertEqual([p.concealed for p in engine.state.players], before)
        for seat in range(4):
            self.assertEqual(set(engine.state.players[seat].concealed),
                             set(before[seat]) - set(chosen[seat]) | set(chosen[(seat - 2) % 4]))
        self.assertEqual(engine.state.phase, Phase.DECLARE_SUIT)
        self.assertIsNone(engine.state.players[0].last_drawn)

    def test_suit_declarations_are_sealed(self):
        engine = Engine(RuleConfig(exchange_three=False, dealer=3))
        before = engine.observe(1)
        engine.submit(0, Action(K.DECLARE_SUIT, suit=1), engine.state.decision_id)
        self.assertEqual(engine.observe(1), before)
        for player in (1, 2, 3):
            engine.submit(player, Action(K.DECLARE_SUIT, suit=2), engine.state.decision_id)
        self.assertEqual([p.missing_suit for p in engine.state.players], [1, 2, 2, 2])
        self.assertEqual(engine.next_player(), 3)

    def test_hidden_deal_and_defensive_observation_copy(self):
        engine = Engine(seed=21)
        obs = engine.observe(0)
        self.assertNotIn("concealed", obs["players"][1])
        self.assertNotIn("wall", obs)
        dealt = [e for e in obs["events"] if e["kind"] == "dealt"]
        self.assertIn("tiles", dealt[0]["data"])
        self.assertNotIn("tiles", dealt[1]["data"])
        obs["players"][0]["concealed"].clear()
        obs["events"][0]["data"]["rules"]["fan_cap"] = 99
        self.assertEqual(len(engine.state.players[0].concealed), 14)
        self.assertEqual(engine.observe(0)["events"][0]["data"]["rules"]["fan_cap"], 5)

    def test_observation_is_independent_of_opponent_tiles_and_future_wall(self):
        engine = Engine(seed=34)
        before = engine.observe(0)
        one, two = engine.state.players[1].concealed, engine.state.players[2].concealed
        one[0], two[0] = two[0], one[0]
        wall, head = engine.state.wall, engine.state.head
        wall[head], wall[head + 1] = wall[head + 1], wall[head]
        engine.assert_invariants()
        self.assertEqual(engine.observe(0), before)

    def test_invalid_and_stale_commands_are_side_effect_free(self):
        engine = Engine(seed=9)
        before = (engine.state_hash(), engine.event_hash(), deepcopy(engine.commands))
        for action, decision in ((Action(K.TSUMO), engine.state.decision_id),
                                 (engine.legal_actions(0)[0], engine.state.decision_id + 1),
                                 (Action(K.EXCHANGE, (0, 1, 2)), engine.state.decision_id)):
            with self.assertRaises(IllegalAction):
                engine.submit(0, action, decision)
            self.assertEqual((engine.state_hash(), engine.event_hash(), engine.commands), before)
        action = engine.legal_actions(0)[0]
        engine.submit(0, action, engine.state.decision_id)
        with self.assertRaises(IllegalAction):
            engine.submit(0, action, engine.state.decision_id)

    def test_action_schema_has_strict_parameters(self):
        self.assertEqual(Action("pass"), Action(K.PASS))
        for args in ((K.PASS, (0,)), (K.PENG, (0, 0)), (K.DISCARD, (True,)),
                     (K.EXCHANGE, (2, 1, 0)), (K.ADDED_KONG, (0,))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Action(*args)
        with self.assertRaises(ValueError):
            Action(K.DECLARE_SUIT, suit=True)

    def test_transition_projection_keeps_exchange_selections_private(self):
        engine = Engine(seed=2)
        for player in (0, 1, 2):
            engine.submit(player, engine.legal_actions(player)[0], engine.state.decision_id)
        transition = engine.submit(3, engine.legal_actions(3)[0], engine.state.decision_id)
        projected = transition.for_player(0)
        selections = [event for event in projected["events"] if event["kind"] == "exchange_selected"]
        self.assertEqual([event["data"]["player"] for event in selections], [0])
        completed = next(event for event in projected["events"] if event["kind"] == "exchange_completed")
        self.assertIn("received", completed["data"])
        self.assertNotIn("private", completed)


class TurnTests(unittest.TestCase):
    def test_missing_suit_must_be_discarded_first(self):
        engine = scenario({0: "1s"})
        actions = engine.legal_actions(0)
        self.assertTrue(actions)
        self.assertTrue(all(a.kind == K.DISCARD and a.tiles[0] // 4 // 9 == 2 for a in actions))

    def test_peng_owns_called_tile_and_requires_discard_without_draw(self):
        engine = scenario({0: "4p", 1: "44p"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        before = engine.state.remaining_tiles
        submit(engine, 1, K.PENG, tile_kind=12)
        pass_rest(engine)
        self.assertEqual(engine.state.actor, 1)
        self.assertEqual(engine.state.remaining_tiles, before)
        self.assertEqual(engine.state.players[0].river[-1].called_by, 1)
        self.assertEqual(len(engine.state.players[1].melds[0].tiles), 3)
        self.assertNotIn(K.TSUMO, {a.kind for a in engine.legal_actions(1)})
        engine.assert_invariants()

    def test_ron_beats_peng_and_winner_is_skipped(self):
        engine = scenario({0: "4p", 1: "123456789m1234p", 2: "44p"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        before = engine.observe(2)
        submit(engine, 1, K.RON)
        self.assertEqual(engine.observe(2), before)
        submit(engine, 2, K.PENG, tile_kind=12)
        pass_rest(engine)
        self.assertEqual([w.player for w in engine.state.wins], [1])
        self.assertFalse(engine.state.players[1].can_act)
        self.assertEqual(engine.state.players[2].melds, [])
        self.assertEqual(engine.state.actor, 2)
        self.assertEqual(engine.state.scores, [-1, 1, 0, 0])

    def test_three_ron_winners_share_a_reference_not_tile_ownership(self):
        engine = scenario({0: "4p", **{p: "123456789m1234p" for p in (1, 2, 3)}})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        for player in (1, 2, 3):
            submit(engine, player, K.RON)
        self.assertEqual(engine.state.end_reason, "three_winners")
        self.assertEqual(engine.state.scores, [-3, 1, 1, 1])
        self.assertEqual(engine.state.players[0].river[-1].winner_refs, [1, 2, 3])
        self.assertEqual(len({w.tile for w in engine.state.wins}), 1)
        self.assertEqual(engine.legal_actions(0), ())
        engine.assert_invariants()

    def test_self_draw_only_collects_from_active_players(self):
        engine = scenario({0: "123456789m12344p", 1: "123456789m1234p"}, head_tiles="4p")
        submit(engine, 0, K.TSUMO)
        self.assertEqual(engine.state.scores, [6, -2, -2, -2])
        self.assertEqual(engine.state.actor, 1)
        self.assertFalse(engine.state.players[0].can_act)
        self.assertNotIn(0, engine.state.responders)
        submit(engine, 1, K.TSUMO)
        self.assertEqual(engine.state.scores, [6, 2, -4, -4])
        self.assertEqual(engine.state.actor, 2)

    def test_opening_self_draw_after_exchange_has_no_independent_drawn_tile(self):
        engine = scenario({0: "123456789m12344p"})
        engine.state.players[0].last_drawn = None
        engine.state.players[0].draw_source = "initial"
        submit(engine, 0, K.TSUMO)
        self.assertIsNone(engine.state.wins[0].tile)
        self.assertEqual(engine.state.scores, [6, -2, -2, -2])

    def test_passed_ron_lock_persists_when_peng_skips_the_draw(self):
        engine = scenario({0: "4p", 1: "123456789m1234p", 2: "44p1p"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        submit(engine, 1, K.PASS)
        submit(engine, 2, K.PENG, tile_kind=12)
        pass_rest(engine)
        self.assertEqual(engine.state.players[1].passed_ron_fan, 0)
        submit(engine, 2, K.DISCARD, tile_kind=9)
        self.assertNotIn(K.RON, {a.kind for a in engine.legal_actions(1)})

    def test_response_arrival_order_does_not_change_outcome(self):
        engine = scenario({0: "4p", 1: "123456789m1234p", 2: "44p"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        actions = {1: select(engine, 1, K.RON), 2: select(engine, 2, K.PENG), 3: Action(K.PASS)}
        other = deepcopy(engine)
        for player in (1, 2, 3):
            engine.submit(player, actions[player], engine.state.decision_id)
        for player in (3, 2, 1):
            other.submit(player, actions[player], other.state.decision_id)
        self.assertEqual(engine.state_hash(), other.state_hash())

    def test_passed_ron_lock_allows_higher_fan_and_resets_only_on_draw(self):
        engine = scenario({0: "4p", 1: "123456789m1234p"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        submit(engine, 1, K.PASS)
        submit(engine, 2, K.PASS)
        # Last reply triggers resolution and next draw; player 1 draws and clears lock.
        submit(engine, 3, K.PASS)
        self.assertEqual(engine.state.players[1].passed_ron_fan, -1)
        # Independently inspect the same waiting hand at a valid discard window.
        engine = scenario({0: "4p", 1: "123456789m1234p"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        engine.state.players[1].passed_ron_fan = 0
        self.assertNotIn(K.RON, {a.kind for a in engine.legal_actions(1)})
        engine.state.discard_kong_id = 1  # a higher-fan 杠上炮 offer
        self.assertIn(K.RON, {a.kind for a in engine.legal_actions(1)})

    def test_no_kong_after_wall_is_empty_and_last_tile_response_resolves(self):
        engine = scenario({0: "1111m"})
        exhaust_wall(engine)
        self.assertNotIn(K.CONCEALED_KONG, {a.kind for a in engine.legal_actions(0)})
        submit(engine, 0, K.DISCARD)
        self.assertEqual(engine.state.phase, Phase.DISCARD_RESPONSE)
        pass_rest(engine)
        self.assertEqual(engine.state.end_reason, "wall_exhausted")

    def test_optional_last_four_forces_legal_win(self):
        engine = scenario({0: "123456789m12344p"}, config=RuleConfig(force_win_last_four=True))
        exhaust_wall(engine)
        self.assertEqual(engine.legal_actions(0), (Action(K.TSUMO),))

    def test_last_discard_can_still_be_won_before_exhaustive_settlement(self):
        engine = scenario({0: "4p", 1: "123456789m1234p"})
        exhaust_wall(engine)
        submit(engine, 0, K.DISCARD, tile_kind=12)
        submit(engine, 1, K.RON)
        pass_rest(engine)
        self.assertEqual(engine.state.wins[0].value.payment, 2)
        self.assertEqual(engine.state.end_reason, "wall_exhausted")
        self.assertFalse(engine.state.players[1].can_act)

    def test_missing_suit_blocks_calls_as_well_as_winning(self):
        engine = scenario({0: "4p", 1: "44p1s"})
        submit(engine, 0, K.DISCARD, tile_kind=12)
        self.assertEqual(engine.legal_actions(1), (Action(K.PASS),))


class KongTests(unittest.TestCase):
    def test_open_kong_payment_and_tail_replacement(self):
        engine = scenario({0: "1p", 1: "111p"})
        submit(engine, 0, K.DISCARD, tile_kind=9)
        replacement = engine.state.wall[engine.state.tail - 1]
        before = engine.state.remaining_tiles
        submit(engine, 1, K.OPEN_KONG)
        pass_rest(engine)
        self.assertEqual(engine.state.players[1].last_drawn, replacement)
        self.assertEqual(engine.state.remaining_tiles, before - 1)
        self.assertEqual(engine.state.scores, [-2, 2, 0, 0])
        self.assertEqual(engine.state.players[1].melds[-1].kind, "open_kong")

    def test_concealed_kong_payment(self):
        engine = scenario({0: "1111m"})
        submit(engine, 0, K.CONCEALED_KONG, tile_kind=0)
        self.assertEqual(engine.state.scores, [6, -2, -2, -2])
        self.assertEqual(engine.state.players[0].melds[-1].kind, "concealed_kong")
        engine.assert_invariants()

    def added_fixture(self):
        return scenario({0: "5p", 1: "123456789m46p77p"}, melds={0: [("peng", "555p", 2)]})

    def test_added_kong_is_tentative_until_responses_complete(self):
        engine = self.added_fixture()
        before = engine.state.remaining_tiles
        submit(engine, 0, K.ADDED_KONG)
        self.assertEqual(engine.state.phase, Phase.ROB_KONG_RESPONSE)
        self.assertEqual(engine.state.scores, [0, 0, 0, 0])
        self.assertEqual(engine.state.players[0].melds[0].kind, "peng")
        pass_rest(engine)
        self.assertEqual(engine.state.players[0].melds[0].kind, "added_kong")
        self.assertEqual(engine.state.scores, [3, -1, -1, -1])
        self.assertEqual(engine.state.remaining_tiles, before - 1)

    def test_robbed_kong_has_no_kong_payment_or_replacement(self):
        engine = self.added_fixture()
        before = engine.state.remaining_tiles
        submit(engine, 0, K.ADDED_KONG)
        robbed = engine.state.pending_tile
        submit(engine, 1, K.RON)
        pass_rest(engine)
        self.assertEqual(engine.state.players[0].melds[0].kind, "peng")
        self.assertEqual(engine.state.robbed_tiles, [robbed])
        self.assertEqual(engine.state.scores, [-2, 2, 0, 0])
        self.assertEqual(engine.state.remaining_tiles, before - 1)  # next active player draws normally
        self.assertFalse(any(t.reason in engine.KONG_REASONS for t in engine.state.transfers))
        self.assertEqual(engine.state.actor, 2)
        engine.assert_invariants()


class SettlementTests(unittest.TestCase):
    def test_optional_kong_discard_income_is_split_among_multiple_winners(self):
        engine = scenario({0: "4p", **{p: "123456789m1234p" for p in (1, 2, 3)}},
                          melds={0: [("open_kong", "8888p", 1)]},
                          config=RuleConfig(kong_discard_transfer=True))
        engine.state.players[0].draw_source = "kong"
        engine.state.pending_kong_id = engine.state.kong_count = 1
        engine._transfer(1, 0, 2, "open_kong", kong_id=1)
        submit(engine, 0, K.DISCARD, tile_kind=12)
        for player in (1, 2, 3):
            submit(engine, player, K.RON)
        self.assertEqual(engine.state.scores, [-6, 1, 3, 2])
        engine.assert_invariants()

    def test_flower_pig_noten_and_refund_exact_ledger(self):
        engine = scenario({0: "147m147p258m9m", 1: "123456789m1234p",
                           2: "111222333s1234p", 3: "147m147p258m258p9m"},
                          phase=Phase.DISCARD_RESPONSE, melds={0: [("concealed_kong", "9999p", None)]})
        self.assertEqual(ready_values(engine.state.players[0], engine.config), {})
        self.assertEqual(ready_values(engine.state.players[3], engine.config), {})
        engine._transfer(1, 0, 2, "concealed_kong", kong_id=1)
        exhaust_wall(engine)
        engine._finish("wall_exhausted")
        self.assertEqual(engine.state.scores, [31, 34, -96, 31])
        self.assertEqual(engine.state.settlement["flowers"], [2])
        self.assertEqual(engine.state.settlement["noten"], [0, 3])
        self.assertEqual(engine.state.settlement["refunds"], [0])
        self.assertEqual(engine.state.settlement["ready"]["1"]["max_payment"], 1)
        engine.assert_invariants()

    def test_invariant_catches_duplicate_tile_and_score_drift(self):
        engine = Engine()
        engine.state.players[0].concealed[0] = engine.state.players[1].concealed[0]
        with self.assertRaises(AssertionError):
            engine.assert_invariants()
        engine = Engine()
        engine.state.scores[0] += 1
        with self.assertRaises(AssertionError):
            engine.assert_invariants()
