import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .agents import run_round, RandomAgent
from .greedy import GreedyAgent
from .riichi_agent import RiichiAgent
from .riichi_engine import RiichiEngine
from .riichi_match import RiichiMatch, MatchConfig
from .engine import Engine
from .model import RuleConfig, Phase
from .replay import replay, save_replay


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="川麻 / 日麻游戏与 AI 沙盒")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="run complete seeded rounds")
    simulate.add_argument("--games", type=int, default=1)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--policy", choices=("random", "win-first", "greedy"), default="random")
    simulate.add_argument("--variant", choices=("sichuan", "riichi"), default="sichuan")
    simulate.add_argument("--match", choices=("hand", "east", "south"), default="hand")
    simulate.add_argument("--no-exchange", action="store_true")
    simulate.add_argument("--replay-dir", type=Path)
    check = subparsers.add_parser("replay", help="re-execute and verify a replay")
    check.add_argument("path", type=Path)
    web = subparsers.add_parser("serve", help="start the playable local web table")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    evaluation = subparsers.add_parser('evaluate', help='seat-balanced policy evaluation against three greedy bots')
    evaluation.add_argument('--variant', choices=('sichuan', 'riichi'), default='riichi')
    evaluation.add_argument('--blocks', type=int, default=50)
    evaluation.add_argument('--workers', type=int, default=4)
    evaluation.add_argument('--seed', type=int, default=0)
    evaluation.add_argument('--policies', nargs='+', choices=('random', 'win-first', 'greedy', 'defensive', 'value'), default=('random', 'win-first', 'greedy', 'defensive', 'value'))
    evaluation.add_argument('--backend', choices=('auto', 'python', 'numba'), default='auto')
    performance = subparsers.add_parser('benchmark', help='reference/optimized profiles and 1000-board process scaling')
    performance.add_argument('--games', type=int, default=20)
    performance.add_argument('--boards', type=int, default=1000)
    performance.add_argument('--workers', type=int, nargs='+', default=(1, 2, 4))
    performance.add_argument('--seed', type=int, default=10000)
    training = subparsers.add_parser('training-smoke', help='run candidate REINFORCE and check terminal rewards')
    training.add_argument('--variant', choices=('sichuan', 'riichi'), default='sichuan')
    training.add_argument('--boards', type=int, default=32)
    training.add_argument('--steps', type=int, default=160)
    training.add_argument('--workers', type=int, default=0)
    training.add_argument('--seed', type=int, default=32)
    training.add_argument('--backend', choices=('auto', 'python', 'numba'), default='auto')
    batches = subparsers.add_parser('batch-smoke', help='advance many live boards to natural termination and check rewards')
    batches.add_argument('--variant', choices=('sichuan', 'riichi'), default='riichi')
    batches.add_argument('--boards', type=int, default=1000)
    batches.add_argument('--workers', type=int, default=4)
    batches.add_argument('--seed', type=int, default=700)
    batches.add_argument('--backend', choices=('auto', 'python', 'numba'), default='auto')
    batches.add_argument('--self-play', action='store_true')
    for item in (evaluation, performance, training, batches):
        item.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command in ('evaluate', 'benchmark', 'training-smoke', 'batch-smoke'):
        if args.command == 'evaluate':
            from .evaluation import evaluate
            result = evaluate(variant=args.variant, blocks=args.blocks, workers=args.workers,
                              seed=args.seed, policies=args.policies, backend=args.backend)
        elif args.command == 'benchmark':
            from .benchmark import benchmark
            result = benchmark(games=args.games, boards=args.boards, workers=args.workers, seed=args.seed)
        elif args.command == 'training-smoke':
            from .training import training_smoke
            result = training_smoke(n=args.boards, steps=args.steps, seed=args.seed, variant=args.variant,
                                    backend=args.backend, workers=args.workers)
        else:
            from .benchmark import batch_trial
            result = batch_trial(args.boards, args.workers, variant=args.variant, backend=args.backend,
                                 seed=args.seed, controlled_player=None if args.self_play else 0)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
        print(text)
        return 0
    if args.command == "serve":
        from .webserver import serve
        serve(args.host, args.port)
        return 0
    if args.command == "replay":
        result = replay(args.path)
        if isinstance(result, RiichiMatch):
            print(json.dumps({"verified": True, **result.summary()}, ensure_ascii=False, indent=2))
            return 0
        engine = result
        print(json.dumps({"verified": True, "end_reason": engine.state.end_reason,
                          "scores": engine.state.scores, "commands": len(engine.commands),
                          "state_hash": engine.state_hash()}, ensure_ascii=False, indent=2))
        return 0
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.variant != 'riichi' and args.match != 'hand': parser.error('--match requires --variant riichi')
    start = time.perf_counter()
    reasons = Counter()
    event_counts = Counter()
    commands = 0
    winners = 0
    total_scores = [0] * 4
    for index in range(args.games):
        seed = args.seed + index
        match = RiichiMatch(MatchConfig(length=args.match), seed=seed) if args.match != 'hand' else None
        engine = match.engine if match else RiichiEngine(seed=seed) if args.variant == 'riichi' else Engine(RuleConfig(exchange_three=not args.no_exchange), seed)
        agents = [(RiichiAgent if args.variant == 'riichi' else GreedyAgent)(seed * 4+p) if args.policy == 'greedy' else RandomAgent(seed*4+p, prefer_win=args.policy == 'win-first') for p in range(4)]
        while True:
            while engine.state.phase != Phase.TERMINAL:
                player = engine.next_player()
                obs = engine.observe(player)
                engine.submit(player, agents[player].act(obs), obs['decision_id'])
            event_counts.update(event.kind for event in engine.events)
            commands += len(engine.commands)
            winners += len(engine.state.wins)
            if not match: break
            match.complete_hand()
            if match.ended: break
            engine = match.next_hand()
        reasons[match.end_reason if match else engine.state.end_reason] += 1
        total_scores = [a + b for a, b in zip(total_scores, match.final_scores if match else engine.state.scores)]
        if args.replay_dir:
            save_replay(match or engine, args.replay_dir / f"round_{seed}.json")
    elapsed = time.perf_counter() - start
    print(json.dumps({"games": args.games, "policy": args.policy, "rules": engine.config.rule_version, "match": args.match,
                      "end_reasons": dict(reasons), "winners": winners, "commands": commands,
                      "events": dict(event_counts), "total_scores": total_scores,
                      "elapsed_seconds": round(elapsed, 3), "games_per_second": round(args.games / elapsed, 2)},
                     ensure_ascii=False, indent=2))
    return 0
