"""Seat-balanced paired-wall policy evaluation and complete-game workloads."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from .bots import make_bot, POLICIES
from .engine import Engine
from .fast import FastEngine, FastRiichiEngine, RuntimeOptions
from .model import Phase, RuleConfig
from .riichi_engine import RiichiEngine
from .riichi_model import RiichiConfig
from .operators import numpy


def run_game(seed=0, *, variant='riichi', policies=('greedy',) * 4,
             backend='auto', reference=False, config=None):
    if variant not in ('sichuan', 'riichi') or len(policies) != 4:
        raise ValueError('a known variant and four policies required')
    config = config or (RiichiConfig() if variant == 'riichi' else RuleConfig())
    if reference:
        cls = RiichiEngine if variant == 'riichi' else Engine
        engine = cls(config, seed=seed)
    else:
        cls = FastRiichiEngine if variant == 'riichi' else FastEngine
        engine = cls(config, seed=seed, runtime=RuntimeOptions(operator_backend=backend))
    initial = tuple(engine.state.scores)
    agents = [make_bot(policy, variant=variant, seed=seed * 4 + p, backend=backend,
                       reference=reference) for p, policy in enumerate(policies)]
    commands = 0
    while engine.state.phase != Phase.TERMINAL:
        if commands >= 2000:
            raise RuntimeError('nonterminating game')
        p = engine.next_player()
        engine.submit(p, agents[p].act(engine.observe(p)), engine.state.decision_id)
        commands += 1
    engine.assert_invariants()
    return {'seed': seed, 'scores': engine.state.scores,
            'gains': [a - b for a, b in zip(engine.state.scores, initial)],
            'winners': [w.player for w in engine.state.wins],
            'deal_ins': sorted({w.source_player for w in engine.state.wins if w.source_player is not None}),
            'reason': engine.state.end_reason, 'commands': commands}


def _paired_block(args):
    variant, seed, policy, backend, rules = args
    games = []
    for seat in range(4):
        lineup = ['greedy'] * 4
        lineup[seat] = policy
        game = run_game(seed, variant=variant, policies=lineup, backend=backend, config=rules)
        top = max(game['scores'])
        tied = sum(s == top for s in game['scores'])
        games.append({'win': int(seat in game['winners']), 'deal_in': int(seat in game['deal_ins']),
                      'gain': game['gains'][seat],
                      'top_score': 1 / tied if game['scores'][seat] == top else 0,
                      'draw': int(not game['winners']), 'commands': game['commands'],
                      'reason': game['reason']})
    return policy, seed, games


def _interval(block_means, rng, *, bounded=False):
    np = numpy()
    values = np.asarray(block_means, dtype=float)
    if len(values) < 2:
        return None
    resamples = rng.integers(0, len(values), (2000, len(values)))
    # A zero-width bootstrap at a boundary would falsely imply certainty.
    # Each block mean lies in [0,1]. Its mean is bounded by the probability of
    # any positive (or any non-one) block, giving an exact conservative bound.
    tail = .025 ** (1 / len(values))
    if bounded and np.all(values == 0):
        return [0., 1 - tail]
    if bounded and np.all(values == 1):
        return [tail, 1.]
    return np.quantile(values[resamples].mean(axis=1), (.025, .975)).tolist()


def evaluate(*, variant='riichi', blocks=50, seed=0, policies=POLICIES,
             workers=4, backend='auto', rules=None):
    if type(blocks) is not int or blocks <= 0 or type(workers) is not int or workers <= 0:
        raise ValueError('positive blocks and workers required')
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError('unique known policies required')
    if variant not in ('sichuan', 'riichi'):
        raise ValueError('unknown variant')
    work = [(variant, seed + b, p, backend, rules) for b in range(blocks) for p in policies]
    start = time.perf_counter()
    if workers == 1:
        rows = list(map(_paired_block, work))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
            rows = list(pool.map(_paired_block, work, chunksize=2))
    elapsed = time.perf_counter() - start
    np, summaries, total_commands, reasons = numpy(), {}, 0, Counter()
    for policy in policies:
        grouped = [games for p, _, games in rows if p == policy]
        flat = [g for group in grouped for g in group]
        rng = np.random.default_rng(seed)
        metrics = {}
        for name in ('win', 'deal_in', 'gain', 'top_score', 'draw'):
            means = [sum(g[name] for g in group) / 4 for group in grouped]
            metrics[name] = {'mean': float(np.mean(means)), 'ci95': _interval(means, rng, bounded=name != 'gain')}
        summaries[policy] = {'games': len(flat), 'metrics': metrics}
        total_commands += sum(g['commands'] for g in flat)
        reasons.update(g['reason'] for g in flat)
    games = len(work) * 4
    return {'variant': variant, 'opponents': 'three greedy bots', 'blocks': blocks,
            'wall_seeds': [seed, seed + blocks - 1], 'seat_rotations': 4,
            'interval_method': '95% paired-block percentile bootstrap, 2000 resamples; conservative exact bounds for all-zero/all-one bounded metrics',
            'workers': workers, 'backend': backend, 'games': games, 'commands': total_commands,
            'elapsed_seconds': elapsed, 'games_per_second': games / elapsed,
            'commands_per_second': total_commands / elapsed, 'end_reasons': dict(reasons),
            'policies': summaries}
