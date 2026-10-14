"""Inventory-valid scenario builder for targeted rules tests, not production setup."""
from mahjong_sandbox.engine import Engine
from mahjong_sandbox.model import Discard, Meld, Phase, PlayerState, RuleConfig, State
from mahjong_sandbox.tiles import SICHUAN_TILES, parse_hand


def scenario(hands=None, *, actor=0, phase=Phase.SELF_TURN, suits=(2, 2, 2, 2), melds=None, config=None,
             head_tiles="", tail_tiles=""):
    hands, melds = hands or {}, melds or {}
    pool = list(SICHUAN_TILES)
    players = [PlayerState(missing_suit=suit) for suit in suits]

    def allocate(text):
        result = []
        for kind, count in enumerate(parse_hand(text)):
            for _ in range(count):
                tile = next((t for t in pool if t // 4 == kind), None)
                if tile is None:
                    raise ValueError(f"fixture overuses kind {kind}")
                pool.remove(tile)
                result.append(tile)
        return result

    for player, specs in melds.items():
        for kind, text, source in specs:
            tiles = tuple(allocate(text))
            called = None if source is None else tiles[0]
            players[player].melds.append(Meld(kind, tiles, source, called))
            if source is not None:
                players[source].river.append(Discard(called, False, called_by=player))
    for player, text in hands.items():
        players[player].concealed = allocate(text)
    reserved_head, reserved_tail = allocate(head_tiles), allocate(tail_tiles)
    for player, hand in enumerate(players):
        extra = phase == Phase.SELF_TURN and player == actor
        count = 13 + int(extra) - 3 * len(hand.melds)
        if len(hand.concealed) > count:
            raise ValueError("fixture has too many concealed tiles")
        while len(hand.concealed) < count:
            hand.concealed.append(pool.pop(0))
        hand.concealed.sort()
    if phase == Phase.SELF_TURN:
        players[actor].last_drawn = players[actor].concealed[-1]
        players[actor].draw_source = "wall"
    engine = Engine(config or RuleConfig(exchange_three=False), 0)
    pool = reserved_head + pool + reserved_tail
    engine.state = State(players, pool, 0, len(pool), phase, actor)
    engine.events.clear()
    engine.commands.clear()
    engine._window(phase, actor, (actor,) if phase == Phase.SELF_TURN else engine._other_active(actor))
    engine.assert_invariants()
    return engine


def select(engine, player, kind, *, tile_kind=None):
    actions = [a for a in engine.legal_actions(player) if a.kind == kind
               and (tile_kind is None or a.tiles and a.tiles[0] // 4 == tile_kind)]
    if not actions:
        raise AssertionError(f"no legal {kind} for player {player}")
    return actions[0]


def submit(engine, player, kind, *, tile_kind=None):
    engine.submit(player, select(engine, player, kind, tile_kind=tile_kind), engine.state.decision_id)


def pass_rest(engine):
    from mahjong_sandbox.model import ActionKind
    decision = engine.state.decision_id
    while engine.state.decision_id == decision:
        player = engine.next_player()
        if player is None:
            break
        submit(engine, player, ActionKind.PASS)


def exhaust_wall(engine):
    state = engine.state
    state.players[0].river.extend(Discard(t, False) for t in state.wall[state.head:state.tail])
    state.head = state.tail
    engine.assert_invariants()
