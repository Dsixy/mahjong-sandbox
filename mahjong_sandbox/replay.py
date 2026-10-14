"""Portable deterministic command replays with full-state and event checksums."""
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .engine import Engine
from .model import Action, ENGINE_VERSION, RuleConfig
from .riichi_model import RiichiConfig
from .riichi_engine import RiichiEngine

REPLAY_FORMAT_VERSION = 1


def export_replay(engine: Engine) -> dict[str, Any]:
    if hasattr(engine, 'runtime') and not engine.runtime.record:
        raise ValueError('replay export requires RuntimeOptions(record=True)')
    if hasattr(engine, 'rounds'):
        return engine.export()
    return {
        "format_version": REPLAY_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "config": asdict(engine.config),
        "seed": engine.seed,
        "initial_wall": list(engine.initial_wall),
        "exchange_offset": getattr(engine, 'exchange_offset', None),
        "commands": json.loads(json.dumps(engine.commands)),
        "expected": {
            "state_hash": engine.state_hash(), "event_hash": engine.event_hash(),
            "end_reason": engine.state.end_reason, "scores": list(engine.state.scores),
            "command_count": len(engine.commands),
        },
    }


def save_replay(engine: Engine, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export_replay(engine), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def replay(source: dict[str, Any] | str | Path) -> Engine:
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get('format_version') == 2:
        from .riichi_match import replay_match
        return replay_match(data)
    if not isinstance(data, dict) or data.get("format_version") != REPLAY_FORMAT_VERSION:
        raise ValueError("unsupported replay format")
    if data.get("engine_version") != ENGINE_VERSION:
        raise ValueError("replay requires a different engine version")
    engine = replay_engine(data)
    for index, command in enumerate(data["commands"]):
        try:
            if set(command) != {"player", "decision_id", "action"}:
                raise ValueError("invalid command schema")
            engine.submit(command["player"], Action.from_dict(command["action"]), command["decision_id"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"invalid replay command {index}: {error}") from error
    actual = export_replay(engine)["expected"]
    if actual != data["expected"]:
        raise ValueError("replay verification failed: state, events, or result was altered")
    return engine


def replay_engine(data):
    """Instantiate an unplayed round from its complete, versioned replay config."""
    if set(data) != {'format_version', 'engine_version', 'config', 'seed', 'initial_wall',
                    'exchange_offset', 'commands', 'expected'}:
        raise ValueError('replay requires the complete exported schema')
    if not isinstance(data['config'], dict) or not isinstance(data['commands'], list):
        raise ValueError('replay config must be an object and commands must be a list')
    config_type = RiichiConfig if data['config'].get('rule_version') == RiichiConfig().rule_version else RuleConfig
    if set(data['config']) != set(asdict(config_type())):
        raise ValueError('replay must contain the complete rule configuration')
    if config_type is RiichiConfig:
        if data['exchange_offset'] is not None:
            raise ValueError('riichi has no exchange offset')
        return RiichiEngine(RiichiConfig(**data['config']), data['seed'], wall=data['initial_wall'])
    return Engine(RuleConfig(**data['config']), data['seed'], wall=data['initial_wall'], exchange_offset=data['exchange_offset'])
