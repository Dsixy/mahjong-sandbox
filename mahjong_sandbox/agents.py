"""Reproducible baselines that only consume player observations."""
import random
from typing import Any

from .engine import Engine
from .model import Action, Phase


class RandomAgent:
    def __init__(self, seed: int = 0, *, prefer_win: bool = False) -> None:
        self.rng = random.Random(seed)
        self.prefer_win = prefer_win

    def act(self, observation: dict[str, Any]) -> Action:
        candidates = observation["legal_actions"]
        if self.prefer_win:
            winning = [a for a in candidates if a["kind"] in ("ron", "tsumo")]
            if winning:
                candidates = winning
        if not candidates:
            raise ValueError("agent called without a decision")
        return Action.from_dict(self.rng.choice(candidates))


def run_round(engine: Engine, *, policy_seed: int = 0, prefer_win: bool = False, max_commands: int = 2000) -> Engine:
    agents = [RandomAgent(policy_seed * 4 + p, prefer_win=prefer_win) for p in range(4)]
    for _ in range(max_commands):
        if engine.state.phase == Phase.TERMINAL:
            return engine
        player = engine.next_player()
        if player is None:
            raise AssertionError("nonterminal environment has no decision")
        observation = engine.observe(player)
        engine.submit(player, agents[player].act(observation), observation["decision_id"])
    raise RuntimeError("round exceeded max_commands; this is an error, not a valid termination")
