import unittest
from mahjong_sandbox.model import Meld, PlayerState, RuleConfig
from mahjong_sandbox.scoring import WinContext, evaluate_hand, ready_values
from mahjong_sandbox.tiles import parse_hand


def hand(text, missing=2):
    return PlayerState(concealed=[4 * kind + copy for kind, count in enumerate(parse_hand(text)) for copy in range(count)],
                       missing_suit=missing)


class ScoringTests(unittest.TestCase):
    def test_known_values(self):
        examples = [
            ("123456789m12344p", 0, 1),
            ("111222333m44455p", 1, 2),
            ("11122233344455m", 3, 8),
            ("11223344556677m", 4, 16),
            ("11112233445566m", 5, 32),
            ("11112222334455m", 5, 32),
            ("111123m456p789p55p", 1, 2),
            ("123789m123789p11p", 2, 4),
            ("12312378978911m", 5, 32),
            ("12378978978911m", 4, 16),
            ("222555m888p222p55p", 3, 8),
        ]
        for text, fan, payment in examples:
            with self.subTest(text=text):
                value = evaluate_hand(hand(text), RuleConfig())
                self.assertEqual((value.fan, value.payment), (fan, payment))
        self.assertEqual(evaluate_hand(hand("11112222334455m"), RuleConfig()).raw_fan, 6)

    def test_best_shape_not_first_shape(self):
        value = evaluate_hand(hand("11223344556677m"), RuleConfig())
        self.assertEqual(value.family, "seven_pairs")
        self.assertEqual(value.payment, 16)

    def test_self_draw_modes_and_context(self):
        player = hand("123456789m12344p")
        for mode, fan, payment in (("base", 0, 2), ("fan", 1, 2), ("none", 0, 1)):
            value = evaluate_hand(player, RuleConfig(self_draw_bonus=mode), context=WinContext(self_draw=True))
            self.assertEqual((value.fan, value.payment), (fan, payment))
        value = evaluate_hand(player, RuleConfig(base_points=3), context=WinContext(rob_kong=True))
        self.assertEqual(value.payment, 6)
        value = evaluate_hand(player, RuleConfig(), context=WinContext(self_draw=True, kong_draw=True, last_tile=True))
        self.assertEqual(value.payment, 5)

    def test_missing_suit_and_unavailable_tiles(self):
        self.assertIsNone(evaluate_hand(hand("123456789m12344p", missing=0), RuleConfig()))
        self.assertIsNone(evaluate_hand(hand("11122233344455z"), RuleConfig()))
        self.assertIsNone(evaluate_hand(hand("123456789m12344p", missing=None), RuleConfig()))

    def test_gold_hook_uses_melds_and_quad_roots(self):
        player = hand("55p")
        for kind in (0, 1, 2, 12):
            player.melds.append(Meld("peng", tuple(4 * kind + i for i in range(3))))
        value = evaluate_hand(player, RuleConfig())
        self.assertEqual((value.fan, value.payment), (2, 4))
        player.melds[0] = Meld("concealed_kong", (0, 1, 2, 3))
        self.assertEqual(evaluate_hand(player, RuleConfig()).fan, 3)

    def test_ready_value_is_maximum_ordinary_ron(self):
        player = hand("123456789m1234p")
        values = ready_values(player, RuleConfig())
        self.assertEqual(set(values), {9, 12})
        self.assertTrue(all(v.payment == 1 for v in values.values()))
        player.missing_suit = 0
        self.assertEqual(ready_values(player, RuleConfig()), {})

    def test_config_rejects_ambiguous_values(self):
        for kwargs in ({"fan_cap": -1}, {"rule_version": "unknown"}, {"base_points": 0},
                       {"exchange_three": 1}, {"dealer": True}, {"self_draw_bonus": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RuleConfig(**kwargs)
