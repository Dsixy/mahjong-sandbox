from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from mahjong_sandbox import Engine, Phase, RuleConfig, export_replay, replay, run_round, save_replay


class ReplayTests(unittest.TestCase):
    def test_three_winners_from_real_deal_exchange_and_declarations(self):
        from examples.three_winners import build_example
        engine = build_example()
        self.assertEqual(engine.state.end_reason, "three_winners")
        self.assertEqual(engine.state.scores, [-3, 1, 1, 1])
        self.assertEqual(len(engine.commands), 12)
        restored = replay(export_replay(engine))
        self.assertEqual(restored.state_hash(), engine.state_hash())
        self.assertEqual(restored.event_hash(), engine.event_hash())

    def test_complete_and_partial_replays(self):
        for config in (RuleConfig(), RuleConfig(exchange_three=False, dealer=2)):
            engine = Engine(config, seed=42)
            self.assertEqual(replay(export_replay(engine)).state_hash(), engine.state_hash())
            run_round(engine, policy_seed=15, prefer_win=True)
            restored = replay(export_replay(engine))
            self.assertEqual(restored.state_hash(), engine.state_hash())
            self.assertEqual(restored.event_hash(), engine.event_hash())
            self.assertEqual(restored.state.phase, Phase.TERMINAL)

    def test_file_round_trip(self):
        engine = run_round(Engine(seed=77), policy_seed=77)
        with tempfile.TemporaryDirectory() as temp:
            path = save_replay(engine, Path(temp) / "nested" / "round.json")
            self.assertEqual(replay(path).state_hash(), engine.state_hash())

    def test_replay_rejects_tampering(self):
        data = export_replay(run_round(Engine(seed=91), policy_seed=91))
        variants = []
        for key, value in (("engine_version", "unknown"), ("format_version", 99)):
            changed = deepcopy(data)
            changed[key] = value
            variants.append(changed)
        changed = deepcopy(data)
        changed["commands"][0]["decision_id"] = 999
        variants.append(changed)
        changed = deepcopy(data)
        changed["expected"]["scores"][0] += 1
        variants.append(changed)
        changed = deepcopy(data)
        changed["initial_wall"][0] = changed["initial_wall"][1]
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                replay(changed)

    def test_seeded_random_rounds_complete_and_are_reproducible(self):
        for seed in range(30):
            with self.subTest(seed=seed):
                config = RuleConfig(exchange_three=seed % 2 == 0, dealer=seed % 4,
                                    self_draw_bonus=("base", "fan", "none")[seed % 3],
                                    force_win_last_four=seed % 5 == 0)
                engine = run_round(Engine(config, seed), policy_seed=seed, prefer_win=seed % 2 == 0)
                self.assertEqual(engine.state.phase, Phase.TERMINAL)
                self.assertEqual(sum(engine.state.scores), 0)
                self.assertLess(len(engine.commands), 600)
                engine.assert_invariants()
                if seed < 3:
                    second = run_round(Engine(config, seed), policy_seed=seed, prefer_win=seed % 2 == 0)
                    self.assertEqual(engine.state_hash(), second.state_hash())
                    self.assertEqual(replay(export_replay(engine)).state_hash(), engine.state_hash())
