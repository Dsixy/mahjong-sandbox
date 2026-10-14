"""End-to-end multi-profile soak validation; intentionally not a GPU benchmark."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mahjong_sandbox import Engine, Phase, RuleConfig, export_replay, replay, run_round


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    start = time.perf_counter()
    reasons, event_counts = Counter(), Counter()
    replays, commands, wins = 0, 0, 0
    for index in range(args.games):
        seed = args.seed + index
        config = RuleConfig() if index % 3 == 0 else RuleConfig(
            exchange_three=index % 2 == 0, dealer=index % 4,
            self_draw_bonus=("base", "fan", "none")[index % 3],
            fan_cap=index % 6, seven_pairs_quads=index % 7 != 0,
            passed_ron_lock=index % 5 != 0, last_tile_bonus=index % 4 != 0,
            force_win_last_four=index % 11 == 0, kong_discard_transfer=index % 7 == 0,
            check_flower_pigs=index % 13 != 0, check_noten=index % 17 != 0,
            refund_noten_kongs=index % 19 != 0,
        )
        engine = run_round(Engine(config, seed), policy_seed=seed, prefer_win=index % 2 == 0)
        if engine.state.phase != Phase.TERMINAL or engine.state.end_reason not in ("wall_exhausted", "three_winners"):
            raise AssertionError(f"seed {seed} did not terminate normally")
        engine.assert_invariants()
        if index % 25 == 0 or any(e.kind == "kong_robbed" for e in engine.events):
            data = export_replay(engine)
            # JSON round-trip, so replay never relies on Python-only object types.
            restored = replay(json.loads(json.dumps(data)))
            if restored.state_hash() != engine.state_hash():
                raise AssertionError(f"seed {seed} diverged on replay")
            replays += 1
        reasons[engine.state.end_reason] += 1
        event_counts.update(e.kind for e in engine.events)
        commands += len(engine.commands)
        wins += len(engine.state.wins)
        if (index + 1) % 100 == 0:
            print(f"validated {index + 1}/{args.games} complete rounds", flush=True)
    report = {"status": "passed", "games": args.games, "seed_start": args.seed,
              "seed_end": args.seed + args.games - 1, "commands": commands, "wins": wins,
              "end_reasons": dict(reasons), "event_counts": dict(event_counts),
              "verified_replays": replays, "default_profile": asdict(RuleConfig()),
              "elapsed_seconds": round(time.perf_counter() - start, 3)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
