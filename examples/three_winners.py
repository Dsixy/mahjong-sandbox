"""A constructed physical wall demonstrating a complete default-rule 一炮三响.

Only the public Engine command API is used after constructing the wall; no state
is edited to force the outcome. This is a rules example, not a random-policy run.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mahjong_sandbox import Action, ActionKind, Engine, save_replay
from mahjong_sandbox.tiles import SICHUAN_TILES, parse_hand


def build_example() -> Engine:
    pool = list(SICHUAN_TILES)

    def allocate(text):
        tiles = []
        for kind, count in enumerate(parse_hand(text)):
            for _ in range(count):
                tile = next(t for t in pool if t // 4 == kind)
                pool.remove(tile)
                tiles.append(tile)
        return tiles

    hands = {p: allocate("123456789m1234p") for p in (1, 2, 3)}
    hands[0] = allocate("123456789m1235p")
    dealer_extra = allocate("4p")[0]
    wall = [hands[p][index] for index in range(13) for p in range(4)] + [dealer_extra] + pool
    engine = Engine(seed=0, wall=wall, exchange_offset=1)
    # Swap 123m: everyone keeps the same kinds, while physical copies rotate.
    for player in range(4):
        action = next(a for a in engine.legal_actions(player) if {t // 4 for t in a.tiles} == {0, 1, 2})
        engine.submit(player, action, engine.state.decision_id)
    for player in range(4):
        engine.submit(player, Action(ActionKind.DECLARE_SUIT, suit=2), engine.state.decision_id)
    engine.submit(0, Action(ActionKind.DISCARD, (dealer_extra,)), engine.state.decision_id)
    for player in (1, 2, 3):
        engine.submit(player, Action(ActionKind.RON), engine.state.decision_id)
    return engine


if __name__ == "__main__":
    engine = build_example()
    path = save_replay(engine, Path(__file__).with_name("three_winners.json"))
    print(json.dumps({"replay": str(path), "scores": engine.state.scores,
                      "end_reason": engine.state.end_reason, "commands": len(engine.commands)}, indent=2))
