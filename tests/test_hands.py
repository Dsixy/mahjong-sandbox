import unittest
from mahjong_sandbox.hands import waits, winning_shapes
from mahjong_sandbox.tiles import counts_of, kind_name, parse_hand


class HandTests(unittest.TestCase):
    def test_standard_and_open_hands(self):
        self.assertTrue(winning_shapes(parse_hand("123456789m12344p")))
        self.assertTrue(winning_shapes(parse_hand("12344p"), meld_count=3))
        self.assertTrue(winning_shapes(parse_hand("44p"), meld_count=4))
        self.assertFalse(winning_shapes(parse_hand("12345p"), meld_count=3))

    def test_seven_pairs_and_quad_variant(self):
        counts = parse_hand("11113355779922m")
        self.assertEqual([s.family for s in winning_shapes(counts)], ["seven_pairs"])
        self.assertFalse(winning_shapes(counts, seven_pairs_quads=False))
        self.assertFalse(winning_shapes(parse_hand("1122334455667m")))

    def test_ambiguous_decompositions_are_all_preserved(self):
        shapes = winning_shapes(parse_hand("11223344556677m"))
        self.assertIn("seven_pairs", {s.family for s in shapes})
        self.assertGreater(sum(s.family == "standard" for s in shapes), 1)

    def test_suits_do_not_wrap_and_honors_are_not_sequences(self):
        counts = [0] * 34
        counts[7] = counts[8] = counts[9] = 1
        counts[10] = 2
        self.assertFalse(winning_shapes(counts, meld_count=3))
        self.assertFalse(winning_shapes(parse_hand("12344z"), meld_count=3))
        self.assertTrue(winning_shapes(parse_hand("11144z"), meld_count=3))

    def test_waits_are_structural_and_no_fifth_copy(self):
        self.assertEqual(waits(parse_hand("123456789m1234p")), (9, 12))
        self.assertEqual(waits(parse_hand("4p"), meld_count=4), (12,))
        self.assertNotIn(0, waits(parse_hand("1111m222333p456p")))
        self.assertEqual(waits(parse_hand("44p"), meld_count=4), ())

    def test_tile_helpers_reject_corrupt_inventory(self):
        self.assertEqual(kind_name(33), "C")
        self.assertEqual(counts_of([0, 1, 4])[:2], (2, 1))
        for value in ("11111m", "0m", "8z", "123", "1x", "１２３m"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_hand(value)
        with self.assertRaises(ValueError):
            counts_of([0, 0])
        with self.assertRaises(ValueError):
            winning_shapes([0] * 33)
