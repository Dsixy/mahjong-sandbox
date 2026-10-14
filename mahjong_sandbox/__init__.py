"""Headless Mahjong sandbox: authoritative engine, public views, and replay."""
from .agents import RandomAgent, run_round
from .engine import Engine, IllegalAction
from .hands import HandShape, waits, winning_shapes
from .model import Action, ActionKind, Phase, RuleConfig, ENGINE_VERSION
from .replay import export_replay, replay, save_replay
from .riichi_engine import RiichiEngine
from .riichi_model import RiichiConfig
from .riichi_scoring import RiichiContext, RiichiValue, riichi_waits, score_hand
from .riichi_agent import RiichiAgent, riichi_shanten
from .riichi_match import RiichiMatch, MatchConfig
from .fast import FastEngine, FastRiichiEngine, RuntimeOptions
from .batch import BatchConfig, SyncBatchEnv
from .parallel import ProcessBatchEnv
from .bots import make_bot

__version__ = ENGINE_VERSION
__all__ = ["Action", "ActionKind", "Engine", "HandShape", "IllegalAction", "Phase", "RandomAgent",
           "RuleConfig", "export_replay", "replay", "run_round", "save_replay", "waits", "winning_shapes",
           "RiichiEngine", "RiichiConfig", "RiichiContext", "RiichiValue", "riichi_waits", "score_hand",
           "RiichiAgent", "riichi_shanten", "RiichiMatch", "MatchConfig",
           "FastEngine", "FastRiichiEngine", "RuntimeOptions", "BatchConfig", "SyncBatchEnv",
           "ProcessBatchEnv", "make_bot"]
