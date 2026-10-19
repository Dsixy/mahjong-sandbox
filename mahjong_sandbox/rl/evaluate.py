"""Paired-wall, four-seat held-out evaluation of saved models and existing bots."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from ..bots import POLICIES, make_bot
from ..evaluation import _interval
from ..fast import FastEngine, RuntimeOptions
from ..model import RuleConfig, Phase
from .agent import RLAgent
from .checkpoint import checksum, model_path, read_metadata, resolve_device

_WORKER_AGENTS = {}


def _initialize(models, backend, deterministic):
    global _WORKER_AGENTS
    torch.set_num_threads(1)
    _WORKER_AGENTS = {label: RLAgent(path, backend=backend, deterministic=deterministic)
                      for label, path in models.items()}


def play_game(seed, seat, candidate, opponent, *, agents=None, rules=None, backend='auto', deterministic=True):
    agents = _WORKER_AGENTS if agents is None else agents
    rules = rules or RuleConfig()
    engine = FastEngine(rules, seed=seed, runtime=RuntimeOptions(operator_backend=backend))
    lineup = [make_bot(opponent, variant='sichuan', seed=seed * 4 + p, backend=backend) for p in range(4)]
    lineup[seat] = agents[candidate] if candidate in agents else make_bot(candidate, variant='sichuan', seed=seed * 4 + seat, backend=backend)
    if candidate in agents and not deterministic:
        torch.manual_seed((seed * 4 + seat) % (2**63))
    while engine.state.phase != Phase.TERMINAL:
        if engine.command_count >= 2000:
            raise RuntimeError('nonterminating evaluation hand')
        player = engine.next_player()
        action = lineup[player].act(engine.observe(player))
        engine.submit(player, action, engine.state.decision_id)
    engine.assert_invariants(force=True)
    return {'seed': seed, 'seat': seat, 'policy': candidate, 'opponent': opponent,
            'win': int(any(w.player == seat for w in engine.state.wins)),
            'deal_in': int(any(w.source_player == seat for w in engine.state.wins)),
            'net_score': engine.state.scores[seat], 'scores': list(engine.state.scores),
            'commands': engine.command_count, 'end_reason': engine.state.end_reason}


def _block(args):
    seed, candidate, opponent, rule_values, backend, deterministic = args
    return [play_game(seed, seat, candidate, opponent, rules=RuleConfig(**rule_values),
                      backend=backend, deterministic=deterministic) for seat in range(4)]


def summarize(rows, *, seed=0):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['seed'], []).append(row)
    means = {key: [float(np.mean([r[key] for r in group])) for group in grouped.values()]
             for key in ('win', 'deal_in', 'net_score')}
    rng = np.random.default_rng(seed)
    return {'games': len(rows), 'wall_blocks': len(grouped),
            'win_rate': float(np.mean(means['win'])), 'deal_in_rate': float(np.mean(means['deal_in'])),
            'mean_net_score': float(np.mean(means['net_score'])),
            'ci95': {key: _interval(values, rng, bounded=key != 'net_score') for key, values in means.items()}}


def validate_model(model, metadata, *, blocks=8, seed=100000, backend='auto'):
    agent = RLAgent(model=model, metadata=metadata, backend=backend)
    rows = [play_game(seed + b, seat, 'rl', 'greedy', agents={'rl': agent}, rules=agent.rules, backend=backend)
            for b in range(blocks) for seat in range(4)]
    return summarize(rows, seed=seed)


def evaluate_models(model, *, baseline_model=None, opponents=('greedy',), baselines=POLICIES,
                    blocks=50, seed=200000, workers=4, backend='auto', deterministic=True,
                    include_games=True, device='cpu', baseline_label='warmstart'):
    if type(blocks) is not int or blocks < 1 or type(workers) is not int or workers < 1:
        raise ValueError('positive blocks and worker counts required')
    if not opponents or len(set(opponents)) != len(opponents) or any(x not in POLICIES for x in opponents):
        raise ValueError('unique known opponent policies required')
    if len(set(baselines)) != len(baselines) or any(x not in POLICIES for x in baselines):
        raise ValueError('unique known baseline policies required')
    device = resolve_device(device)
    if device != 'cpu' and workers != 1:
        raise ValueError('GPU evaluation uses --workers 1 to keep one GPU model owner')
    models = {'rl': str(model_path(model).resolve())}
    if baseline_model:
        if not isinstance(baseline_label, str) or not baseline_label or baseline_label == 'rl' or baseline_label in POLICIES:
            raise ValueError('baseline model label must be unique')
        models[baseline_label] = str(model_path(baseline_model).resolve())
    metadata = {k: read_metadata(p) for k, p in models.items()}
    rules = RuleConfig(**metadata['rl']['rules'])
    for p in models.values():
        read_metadata(p, expected_rules=rules)
    validation_seeds = {int(m['training'].get('validation_seed', -999999)) for m in metadata.values()}
    validation_sizes = {int(m['training'].get('validation_blocks', 0)) for m in metadata.values()}
    # Prevent accidental reuse of checkpoint-selection walls as the test set.
    for start in validation_seeds:
        if max(seed, start) < min(seed + blocks, start + max(validation_sizes)):
            raise ValueError('test walls overlap the checkpoint-selection validation walls')
    candidates = tuple(models) + tuple(baselines)
    work = [(seed + b, candidate, opponent, asdict(rules), backend, deterministic)
            for opponent in opponents for candidate in candidates for b in range(blocks)]
    start = time.perf_counter()
    if workers == 1:
        torch.set_num_threads(1)
        agents = {label: RLAgent(path, device=device, backend=backend, deterministic=deterministic)
                  for label, path in models.items()}
        rows = [play_game(wall, seat, candidate, opponent, agents=agents, rules=rules,
                         backend=backend, deterministic=deterministic)
                for wall, candidate, opponent, *_ in work for seat in range(4)]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                initializer=_initialize, initargs=(models, backend, deterministic)) as pool:
            rows = [row for block in pool.map(_block, work, chunksize=2) for row in block]
    elapsed = time.perf_counter() - start
    matchups, comparisons = {}, {}
    for opponent in opponents:
        matchups[opponent] = {}
        by_policy = {}
        for policy in candidates:
            subset = [r for r in rows if r['opponent'] == opponent and r['policy'] == policy]
            matchups[opponent][policy] = summarize(subset, seed=seed)
            by_policy[policy] = {b: [r for r in subset if r['seed'] == b] for b in range(seed, seed + blocks)}
        comparisons[opponent] = {}
        for policy in candidates[1:]:
            rng, stats = np.random.default_rng(seed), {}
            for key in ('win', 'deal_in', 'net_score'):
                differences = [np.mean([r[key] for r in by_policy['rl'][b]]) - np.mean([r[key] for r in by_policy[policy][b]]) for b in range(seed, seed + blocks)]
                stats[key] = {'rl_minus_baseline': float(np.mean(differences)), 'ci95': _interval(differences, rng)}
            comparisons[opponent][policy] = stats
    return {'format': 'sichuan_rl_evaluation_v1', 'variant': 'sichuan', 'rules': asdict(rules),
            'models': {k: {'path': p, 'sha256': checksum(p), 'timesteps': metadata[k]['timesteps'],
                            'source_sha256': metadata[k]['source_sha256']} for k, p in models.items()},
            'wall_seeds': [seed, seed + blocks - 1], 'blocks': blocks, 'seat_rotations': 4,
            'interval_method': '95% paired-wall block bootstrap, 2000 resamples; conservative exact boundary intervals',
            'deterministic_policy': deterministic, 'workers': workers, 'device': device,
            'games': len(rows), 'elapsed_seconds': elapsed, 'games_per_second': len(rows) / elapsed,
            'commands': sum(r['commands'] for r in rows),
            'end_reasons': dict(Counter(r['end_reason'] for r in rows)),
            'matchups': matchups, 'paired_comparisons': comparisons,
            **({'game_records': rows} if include_games else {})}
