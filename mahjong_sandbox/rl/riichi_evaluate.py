"""Paired riichi matches with point, placement and observable style metrics."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from ..bots import POLICIES, make_bot
from ..fast import FastRiichiEngine, RuntimeOptions
from ..model import Phase
from ..operators import shanten_batch
from ..riichi_model import RiichiConfig
from ..riichi_match import RiichiMatch, MatchConfig
from .agent import RLAgent
from .checkpoint import read_metadata, model_path, checksum, resolve_device

_AGENTS = {}


def _initialize(models, backend, deterministic):
    torch.set_num_threads(1)
    _AGENTS.update({k: RLAgent(v, backend=backend, deterministic=deterministic) for k, v in models.items()})


def _threat_choice(obs, action, counters, backend):
    if action.kind.value not in ('discard', 'riichi') or obs['players'][obs['player']]['riichi']:
        return
    threats = [p for p in obs['players'] if p['player'] != obs['player'] and p['riichi']]
    if not threats:
        return
    safe_sets = [{d['tile'] // 4 for d in p['river']} for p in threats]
    kinds = sorted({a['tiles'][0] // 4 for a in obs['legal_actions'] if a['kind'] in ('discard', 'riichi')})
    if len(kinds) < 2:
        return
    counters['threat_decisions'] += 1
    if not any(all(k in river for river in safe_sets) for k in kinds):
        return
    counters['safe_opportunities'] += 1
    chosen = action.tiles[0] // 4
    chosen_safe = all(chosen in river for river in safe_sets)
    if chosen_safe:
        counters['safe_choices'] += 1
    own = obs['players'][obs['player']]
    counts = np.bincount([t // 4 for t in own['concealed']], minlength=34)
    after = np.tile(counts, (len(kinds), 1))
    after[np.arange(len(kinds)), kinds] -= 1
    distances = shanten_batch(after, len(own['melds']), variant='riichi', backend=backend)
    if chosen_safe and distances[kinds.index(chosen)] > min(distances):
        counters['shanten_backoff'] += 1


def play_match(seed, seat, candidate, opponent, *, rules=None, agents=None, backend='auto',
               deterministic=True, match_length='east', collect_style=True):
    rules, agents = rules or RiichiConfig(), _AGENTS if agents is None else agents
    # Adjacent logical evaluation blocks must not reuse each other's match walls.
    wall_seed = seed * 256
    factory = partial(FastRiichiEngine, runtime=RuntimeOptions(operator_backend=backend))
    match = RiichiMatch(MatchConfig(length=match_length), rules, wall_seed, engine_factory=factory, record=False) if match_length != 'hand' else None
    engine = match.engine if match else factory(rules, seed=wall_seed)
    lineup = [make_bot(opponent, variant='riichi', seed=seed * 4 + p, backend=backend) for p in range(4)]
    lineup[seat] = agents[candidate] if candidate in agents else make_bot(candidate, variant='riichi', seed=seed * 4 + seat, backend=backend)
    if candidate in agents and not deterministic:
        torch.manual_seed((seed * 4 + seat) % (2**63))
    counters, actions, hands = Counter(), Counter(), []
    commands = 0
    while True:
        while engine.state.phase != Phase.TERMINAL:
            if engine.command_count >= 2000:
                raise RuntimeError('nonterminating riichi evaluation hand')
            p = engine.next_player()
            obs = engine.observe(p)
            action = lineup[p].act(obs)
            if p == seat:
                actions[action.kind.value] += 1
                if collect_style:
                    _threat_choice(obs, action, counters, backend)
            engine.submit(p, action, engine.state.decision_id)
        engine.assert_invariants(force=True)
        commands += engine.command_count
        own = engine.state.players[seat]
        wins = [w for w in engine.state.wins if w.player == seat]
        win_income = sum(t.amount for t in engine.state.transfers if t.receiver == seat and t.reason in ('ron', 'tsumo', 'pao', 'honba', 'riichi_award')) if wins else 0
        hands.append({'seed': engine.seed, 'win': int(bool(wins)),
            'deal_in': int(any(w.source_player == seat for w in engine.state.wins)),
            'riichi': int(own.riichi), 'open': int(any(m.kind != 'concealed_kong' for m in own.melds)),
            'kongs': sum(m.kind in ('open_kong', 'concealed_kong', 'added_kong') for m in own.melds),
            'self_draw': sum(w.self_draw for w in wins), 'win_han': sum(w.value.fan for w in wins),
            'win_income': win_income, 'end_reason': engine.state.end_reason})
        if not match:
            scores = list(engine.state.scores)
            ranking = sorted(range(4), key=lambda p: (-scores[p], (p - rules.dealer) % 4))
            end_reason = engine.state.end_reason
            break
        match.complete_hand()
        if match.ended:
            scores, ranking, end_reason = match.final_scores, match.summary()['ranking'], match.end_reason
            assert sum(scores) == sum(rules.initial_scores) + rules.riichi_sticks * 1000
            break
        if len(hands) >= 128:
            raise RuntimeError('riichi evaluation match exceeded 128 hands')
        engine = match.next_hand()
    counts = {key: sum(h[key] for h in hands) for key in ('win', 'deal_in', 'riichi', 'open', 'kongs', 'self_draw', 'win_han', 'win_income')}
    return {'seed': seed, 'wall_seed': wall_seed, 'seat': seat, 'policy': candidate, 'opponent': opponent,
            'hands': len(hands), **counts, 'net_score': scores[seat] - rules.initial_scores[seat],
            'scores': list(scores), 'rank': ranking.index(seat) + 1, 'commands': commands,
            'end_reason': end_reason, 'style_counters': dict(counters), 'actions': dict(actions), 'hand_records': hands}


def _block(args):
    seed, candidate, opponent, values, backend, deterministic, length, style = args
    return [play_match(seed, seat, candidate, opponent, rules=RiichiConfig(**values),
                      backend=backend, deterministic=deterministic, match_length=length, collect_style=style) for seat in range(4)]


def _totals(rows):
    keys = ('hands', 'win', 'deal_in', 'riichi', 'open', 'kongs', 'self_draw', 'win_han', 'win_income', 'net_score')
    total = {key: sum(r[key] for r in rows) for key in keys}
    total.update(matches=len(rows), rank=sum(r['rank'] for r in rows),
                 first_place=sum(r['rank'] == 1 for r in rows), last_place=sum(r['rank'] == 4 for r in rows))
    for key in ('threat_decisions', 'safe_opportunities', 'safe_choices', 'shanten_backoff'):
        total[key] = sum(r['style_counters'].get(key, 0) for r in rows)
    return total


METRICS = {'win': ('win', 'hands'), 'deal_in': ('deal_in', 'hands'), 'net_score': ('net_score', 'matches'),
           'rank': ('rank', 'matches'), 'first_place': ('first_place', 'matches'), 'last_place': ('last_place', 'matches'),
           'riichi': ('riichi', 'hands'), 'open': ('open', 'hands'), 'kongs': ('kongs', 'hands'),
           'self_draw_share': ('self_draw', 'win'), 'win_han': ('win_han', 'win'), 'win_income': ('win_income', 'win'),
           'safe_choice': ('safe_choices', 'safe_opportunities'), 'shanten_backoff': ('shanten_backoff', 'safe_opportunities')}


def _estimate(totals, numerator, denominator):
    return totals[numerator] / totals[denominator] if totals[denominator] else None


def summarize(rows, *, seed=0):
    total = _totals(rows)
    blocks = [_totals([r for r in rows if r['seed'] == s]) for s in sorted({r['seed'] for r in rows})]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(blocks), (2000, len(blocks)))
    estimates, intervals = {}, {}
    for name, (num, den) in METRICS.items():
        estimates[name] = _estimate(total, num, den)
        a = np.array([b[num] for b in blocks])[samples].sum(axis=1)
        b = np.array([x[den] for x in blocks])[samples].sum(axis=1)
        valid = b > 0
        intervals[name] = np.quantile(a[valid] / b[valid], [.025, .975]).tolist() if valid.any() else None
    return {'games': len(rows), 'wall_blocks': len(blocks), 'hands': total['hands'],
            'win_rate': estimates['win'], 'deal_in_rate': estimates['deal_in'],
            'mean_net_score': estimates['net_score'], 'mean_rank': estimates['rank'],
            'first_place_rate': estimates['first_place'], 'last_place_rate': estimates['last_place'],
            'style': {k: v for k, v in estimates.items() if k not in ('win', 'deal_in', 'net_score', 'rank', 'first_place', 'last_place')},
            'counts': total, 'ci95': intervals}


def validate_model(model, metadata, *, blocks=4, seed=100000, backend='auto'):
    agent = RLAgent(model=model, metadata=metadata, backend=backend)
    length = metadata.get('training', {}).get('config', {}).get('match_length', 'east')
    rows = [play_match(seed + b, seat, 'rl', 'greedy', agents={'rl': agent}, rules=agent.rules,
                      backend=backend, match_length=length, collect_style=False) for b in range(blocks) for seat in range(4)]
    return summarize(rows, seed=seed)


def evaluate_models(model, *, baseline_model=None, opponents=('greedy',), baselines=POLICIES,
                    blocks=16, seed=200000, workers=4, backend='auto', deterministic=True,
                    include_games=True, device='cpu', baseline_label='warmstart'):
    device = resolve_device(device)
    if device != 'cpu' and workers != 1:
        raise ValueError('GPU evaluation requires workers=1')
    models = {'rl': str(model_path(model).resolve())}
    if baseline_model:
        if not baseline_label or baseline_label == 'rl' or baseline_label in POLICIES:
            raise ValueError('baseline model label must be unique')
        models[baseline_label] = str(model_path(baseline_model).resolve())
    metadata = {k: read_metadata(p) for k, p in models.items()}
    rules = RiichiConfig(**metadata['rl']['rules'])
    length = metadata['rl']['training'].get('config', {}).get('match_length', 'east')
    for label, path in models.items():
        read_metadata(path, expected_rules=rules)
        config = metadata[label]['training']
        start, count = config.get('validation_seed', -99999), config.get('validation_blocks', 0)
        if max(seed, start) < min(seed + blocks, start + count):
            raise ValueError('test walls overlap checkpoint-selection walls')
    work = [(seed + b, c, opponent, asdict(rules), backend, deterministic, length, True)
            for opponent in opponents for c in (*models, *baselines) for b in range(blocks)]
    begin = time.perf_counter()
    if workers == 1:
        torch.set_num_threads(1)
        agents = {k: RLAgent(p, backend=backend, device=device, deterministic=deterministic) for k, p in models.items()}
        rows = [play_match(s, seat, c, o, agents=agents, rules=rules, backend=backend,
                          deterministic=deterministic, match_length=length) for s, c, o, *_ in work for seat in range(4)]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_initialize,
                                 initargs=(models, backend, deterministic)) as pool:
            rows = [r for block in pool.map(_block, work, chunksize=1) for r in block]
    elapsed = time.perf_counter() - begin
    matchups, comparisons = {}, {}
    for opponent in opponents:
        subsets = {p: [r for r in rows if r['opponent'] == opponent and r['policy'] == p] for p in (*models, *baselines)}
        matchups[opponent] = {p: summarize(r, seed=seed) for p, r in subsets.items()}
        comparisons[opponent] = {}
        for p in list(subsets)[1:]:
            rng = np.random.default_rng(seed)
            paired = {}
            block_rows = [[r for r in subsets['rl'] if r['seed'] == s] for s in range(seed, seed + blocks)]
            base_rows = [[r for r in subsets[p] if r['seed'] == s] for s in range(seed, seed + blocks)]
            indices = rng.integers(0, blocks, (2000, blocks))
            for name, (num, den) in list(METRICS.items())[:6]:
                a, b = [_totals(x) for x in block_rows], [_totals(x) for x in base_rows]
                a_num, a_den = np.array([x[num] for x in a]), np.array([x[den] for x in a])
                b_num, b_den = np.array([x[num] for x in b]), np.array([x[den] for x in b])
                delta = a_num.sum() / a_den.sum() - b_num.sum() / b_den.sum()
                samples = a_num[indices].sum(axis=1) / a_den[indices].sum(axis=1) - b_num[indices].sum(axis=1) / b_den[indices].sum(axis=1)
                paired[name] = {'rl_minus_baseline': float(delta), 'ci95': np.quantile(samples, [.025, .975]).tolist()}
            comparisons[opponent][p] = paired
    return {'format': 'riichi_rl_evaluation_v1', 'variant': 'riichi', 'match_length': length,
            'rules': asdict(rules), 'models': {k: {'path': p, 'timesteps': metadata[k]['timesteps'], 'sha256': checksum(p),
                'source_sha256': metadata[k]['source_sha256']} for k, p in models.items()},
            'wall_seeds': [seed, seed + blocks - 1], 'wall_stride': 256, 'blocks': blocks, 'seat_rotations': 4,
            'interval_method': '95% paired-match-seed block bootstrap, 2000 resamples; hand ratios resample counts',
            'style_definitions': {'riichi': 'accepted riichi hands / hands', 'open': 'hands with an open meld / hands',
                'safe_choice': 'genbutsu against every riichi opponent, conditional on having a safe legal discard and not already being in riichi',
                'shanten_backoff': 'chosen genbutsu worsens shanten versus best legal discard under the same condition, with at least two discard kinds; behavioral proxy, not inferred intent'},
            'device': device, 'workers': workers, 'deterministic_policy': deterministic,
            'games': len(rows), 'hands': sum(r['hands'] for r in rows), 'elapsed_seconds': elapsed,
            'matchups': matchups, 'paired_comparisons': comparisons, **({'game_records': rows} if include_games else {})}
