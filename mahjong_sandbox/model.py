"""Serializable reference state. Histories reference tiles; they do not own them."""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .tiles import kind_of

RULE_VERSION = "sichuan_xuezhan_v1"
ENGINE_VERSION = "0.1.0"


@dataclass(frozen=True)
class RuleConfig:
    rule_version: str = RULE_VERSION
    exchange_three: bool = True
    base_points: int = 1
    fan_cap: int = 5
    self_draw_bonus: str = "base"  # base / fan / none
    seven_pairs_quads: bool = True
    passed_ron_lock: bool = True
    last_tile_bonus: bool = True
    force_win_last_four: bool = False
    kong_discard_transfer: bool = False
    check_flower_pigs: bool = True
    check_noten: bool = True
    refund_noten_kongs: bool = True
    dealer: int = 0

    def __post_init__(self) -> None:
        if self.rule_version != RULE_VERSION:
            raise ValueError("unsupported rule version")
        if type(self.base_points) is not int or self.base_points <= 0:
            raise ValueError("base_points must be positive")
        if type(self.fan_cap) is not int or not 0 <= self.fan_cap <= 10:
            raise ValueError("fan_cap must be in [0, 10]")
        if self.self_draw_bonus not in ("base", "fan", "none"):
            raise ValueError("invalid self-draw bonus")
        if type(self.dealer) is not int or not 0 <= self.dealer < 4:
            raise ValueError("invalid dealer")
        for name in (
            "exchange_three", "seven_pairs_quads", "passed_ron_lock", "last_tile_bonus",
            "force_win_last_four", "kong_discard_transfer", "check_flower_pigs",
            "check_noten", "refund_noten_kongs",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


class Phase(StrEnum):
    EXCHANGE = "exchange"
    DECLARE_SUIT = "declare_suit"
    SELF_TURN = "self_turn"
    DISCARD_RESPONSE = "discard_response"
    ROB_KONG_RESPONSE = "rob_kong_response"
    TERMINAL = "terminal"


class ActionKind(StrEnum):
    EXCHANGE = "exchange"
    DECLARE_SUIT = "declare_suit"
    DISCARD = "discard"
    PENG = "peng"
    OPEN_KONG = "open_kong"
    CONCEALED_KONG = "concealed_kong"
    ADDED_KONG = "added_kong"
    TSUMO = "tsumo"
    RON = "ron"
    PASS = "pass"
    CHI = "chi"
    RIICHI = "riichi"
    ABORT_NINE = "abort_nine"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    tiles: tuple[int, ...] = ()
    suit: int | None = None
    meld_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ActionKind(self.kind))
        if type(self.tiles) is not tuple or any(type(t) is not int or not 0 <= t < 136 for t in self.tiles):
            raise ValueError("action tiles must be a tuple of physical tile IDs")
        if tuple(sorted(set(self.tiles))) != self.tiles:
            raise ValueError("action tiles must be sorted and unique")
        sizes = {ActionKind.EXCHANGE: 3, ActionKind.DISCARD: 1, ActionKind.PENG: 2,
                 ActionKind.OPEN_KONG: 3, ActionKind.CONCEALED_KONG: 4, ActionKind.ADDED_KONG: 1,
                 ActionKind.CHI: 2, ActionKind.RIICHI: 1}
        if len(self.tiles) != sizes.get(self.kind, 0):
            raise ValueError("wrong number of tiles for action")
        if self.kind == ActionKind.DECLARE_SUIT:
            if type(self.suit) is not int or self.suit not in (0, 1, 2):
                raise ValueError("declare_suit needs suit 0, 1, or 2")
        elif self.suit is not None:
            raise ValueError("unexpected suit parameter")
        if self.kind == ActionKind.ADDED_KONG:
            if type(self.meld_index) is not int or not 0 <= self.meld_index < 4:
                raise ValueError("added_kong needs a meld index")
        elif self.meld_index is not None:
            raise ValueError("unexpected meld index")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "tiles": list(self.tiles),
                "suit": self.suit, "meld_index": self.meld_index}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Action":
        if not isinstance(value, dict) or set(value) != {"kind", "tiles", "suit", "meld_index"}:
            raise ValueError("invalid action schema")
        tiles = value["tiles"]
        if not isinstance(tiles, list) or any(type(t) is not int for t in tiles):
            raise ValueError("invalid action tiles")
        if any(value[name] is not None and type(value[name]) is not int for name in ("suit", "meld_index")):
            raise ValueError("invalid action parameter")
        return cls(ActionKind(value["kind"]), tuple(tiles), value["suit"], value["meld_index"])


@dataclass
class Meld:
    kind: str
    tiles: tuple[int, ...]
    source_player: int | None = None
    called_tile: int | None = None
    added_tile: int | None = None


@dataclass
class Discard:
    tile: int
    tsumogiri: bool
    called_by: int | None = None
    winner_refs: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class FanItem:
    name: str
    fan: int


@dataclass(frozen=True)
class HandValue:
    family: str
    items: tuple[FanItem, ...]
    raw_fan: int
    fan: int
    payment: int


@dataclass(frozen=True)
class Win:
    player: int
    source_player: int | None
    tile: int | None
    self_draw: bool
    value: HandValue


@dataclass
class PlayerState:
    concealed: list[int] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    river: list[Discard] = field(default_factory=list)
    missing_suit: int | None = None
    can_act: bool = True
    last_drawn: int | None = None
    draw_source: str | None = None
    passed_ron_fan: int = -1

    def has_missing_tiles(self) -> bool:
        return self.missing_suit is not None and any(kind_of(t) // 9 == self.missing_suit for t in self.concealed)


@dataclass(frozen=True)
class Transfer:
    id: int
    payer: int
    receiver: int
    amount: int
    reason: str
    kong_id: int | None = None
    refund_of: int | None = None


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    data: dict[str, Any]
    private: dict[int, dict[str, Any]] = field(default_factory=dict)
    visible_to: tuple[int, ...] | None = None

    def for_player(self, player: int) -> dict[str, Any] | None:
        if self.visible_to is not None and player not in self.visible_to:
            return None
        from copy import deepcopy
        result = deepcopy(self.data)
        result.update(deepcopy(self.private.get(player, {})))
        return {"seq": self.seq, "kind": self.kind, "data": result}


@dataclass
class State:
    players: list[PlayerState]
    wall: list[int]
    head: int
    tail: int
    phase: Phase
    actor: int
    decision_id: int = 0
    revision: int = 0
    responders: tuple[int, ...] = ()
    replies: dict[int, Action] = field(default_factory=dict)
    pending_tile: int | None = None
    pending_meld: int | None = None
    pending_kong_id: int | None = None
    discard_kong_id: int | None = None
    scores: list[int] = field(default_factory=lambda: [0] * 4)
    transfers: list[Transfer] = field(default_factory=list)
    wins: list[Win] = field(default_factory=list)
    robbed_tiles: list[int] = field(default_factory=list)
    kong_count: int = 0
    end_reason: str | None = None
    settlement: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_tiles(self) -> int:
        return self.tail - self.head


@dataclass(frozen=True)
class Transition:
    events: tuple[Event, ...]
    score_delta: tuple[int, ...]
    decision_id: int
    terminal: bool

    def for_player(self, player: int) -> dict[str, Any]:
        if type(player) is not int or not 0 <= player < 4:
            raise ValueError("invalid player")
        events = [event.for_player(player) for event in self.events]
        return {"events": [event for event in events if event is not None],
                "score_delta": list(self.score_delta), "decision_id": self.decision_id,
                "terminal": self.terminal}
