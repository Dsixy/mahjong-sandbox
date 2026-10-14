"""Versioned four-player riichi profile; tile IDs and action schema are shared."""
from dataclasses import dataclass, field
from .model import PlayerState, State

RIICHI_RULE_VERSION = "riichi_4p_v1"


@dataclass(frozen=True)
class RiichiConfig:
    rule_version: str = RIICHI_RULE_VERSION
    dealer: int = 0
    round_wind: int = 27
    hand_number: int = 1
    honba: int = 0
    riichi_sticks: int = 0
    initial_scores: tuple[int, ...] = (25000, 25000, 25000, 25000)
    red_fives: tuple[int, ...] = (16, 52, 88)
    open_tanyao: bool = True
    kuikae_forbidden: bool = True
    riichi_min_live: int = 4
    multi_ron: bool = True
    triple_ron_abort: bool = True
    abortive_draws: bool = True
    nagashi_mangan: bool = True
    kiriage_mangan: bool = False
    counted_yakuman: bool = True
    double_yakuman: bool = True
    pao: bool = True
    delayed_open_kan_dora: bool = True

    def __post_init__(self):
        object.__setattr__(self, "initial_scores", tuple(self.initial_scores))
        object.__setattr__(self, "red_fives", tuple(self.red_fives))
        if self.rule_version != RIICHI_RULE_VERSION:
            raise ValueError("unsupported riichi rule version")
        if type(self.dealer) is not int or not 0 <= self.dealer < 4:
            raise ValueError("invalid dealer")
        if type(self.round_wind) is not int or self.round_wind not in range(27, 31):
            raise ValueError("invalid round wind")
        if type(self.hand_number) is not int or not 1 <= self.hand_number <= 4:
            raise ValueError("invalid hand number")
        for name in ("honba", "riichi_sticks", "riichi_min_live"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"invalid {name}")
        if len(self.initial_scores) != 4 or any(type(n) is not int or n % 100 for n in self.initial_scores):
            raise ValueError("initial_scores must contain four integer hundred-point scores")
        if len(set(self.red_fives)) != len(self.red_fives) or any(type(t) is not int or t // 4 not in (4, 13, 22) for t in self.red_fives):
            raise ValueError("red_fives must be unique physical five IDs")
        for name in ("open_tanyao", "kuikae_forbidden", "multi_ron", "triple_ron_abort", "abortive_draws",
                     "nagashi_mangan", "kiriage_mangan", "counted_yakuman", "double_yakuman", "pao", "delayed_open_kan_dora"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass
class RiichiPlayer(PlayerState):
    riichi: bool = False
    double_riichi: bool = False
    ippatsu: bool = False
    riichi_discard: int | None = None
    riichi_waits: tuple[int, ...] = ()
    temporary_furiten: bool = False
    riichi_furiten: bool = False
    discard_count: int = 0
    forbidden_discards: tuple[int, ...] = ()
    pao: dict[str, int] = field(default_factory=dict)

    @property
    def closed(self):
        return all(m.kind == "concealed_kong" for m in self.melds)


@dataclass
class RiichiState(State):
    dead_wall: list[int] = field(default_factory=list)
    rinshan_count: int = 0
    indicator_count: int = 1
    delayed_indicators: int = 0
    any_call: bool = False
    pending_riichi: int | None = None
    pending_kong_tiles: tuple[int, ...] = ()
    pending_kong_kind: str | None = None
    four_kans_pending: bool = False
    riichi_sticks: int = 0
    initial_bank: int = 0
