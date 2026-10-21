"""Resume deterministic held-out matches without changing the frozen experiment.

Only inference buffer sizes and scheduling differ from the standard evaluator.
Each cache key includes the frozen model hashes, source, plan and full match task.
Separate CPU instances may precompute the same cache in reverse task order.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def initialize(models, backend, deterministic):
    import torch
    from mahjong_sandbox.rl import riichi_evaluate as evaluator
    from mahjong_sandbox.rl.agent import RLAgent
    from mahjong_sandbox.rl.checkpoint import load_checkpoint
    torch.set_num_threads(1)
    evaluator._AGENTS.clear()
    for label, path in models.items():
        model, metadata = load_checkpoint(path, device='cpu', n_steps=1, n_envs=1)
        evaluator._AGENTS[label] = RLAgent(model=model, metadata=metadata,
            backend=backend, deterministic=deterministic)


def cached_block(args, *, cache_root, identity):
    from dataclasses import asdict
    from mahjong_sandbox.rl.riichi_evaluate import play_match
    from mahjong_sandbox.riichi_model import RiichiConfig
    from mahjong_sandbox.riichi_match import MatchConfig
    seed, candidate, opponent, rules, backend, deterministic, length, style, match_config = args
    if length != 'hand':
        match_config=asdict(MatchConfig(**match_config) if isinstance(match_config,dict) else match_config or MatchConfig(length=length))
    args=(seed,candidate,opponent,rules,backend,deterministic,length,style,match_config)
    key = digest({'identity': identity, 'task': args})
    destination = Path(cache_root) / (key + '.json')
    if not deterministic:
        raise ValueError('persistent match cache requires deterministic policies')
    if destination.exists():
        receipt = json.loads(destination.read_text())
        rows = receipt['rows']
        if receipt['key'] != key or receipt['rows_sha256'] != digest(rows):
            raise ValueError('held-out match cache checksum changed')
        if len(rows) != 4 or any((row['seed'], row['seat'], row['policy'], row['opponent']) !=
                (seed, seat, candidate, opponent) for seat, row in enumerate(rows)):
            raise ValueError('held-out match cache task changed')
        return rows
    rows = [play_match(seed, seat, candidate, opponent, rules=RiichiConfig(**rules),
        backend=backend, deterministic=deterministic, match_length=length,
        collect_style=style, match_config=match_config) for seat in range(4)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'key': key, 'rows_sha256': digest(rows), 'rows': rows},
        separators=(',', ':')) + '\n')
    os.replace(temporary, destination)
    return rows


def setup(root):
    from mahjong_sandbox.rl.checkpoint import source_fingerprint, read_metadata
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if source_fingerprint() != plan['source_sha256']:
        raise ValueError('evaluation source differs from frozen plan')
    frozen = json.loads((root / 'test-models.json').read_text())
    for entry in frozen.values():
        if read_metadata(entry['path'])['model_sha256'] != entry['sha256']:
            raise ValueError('frozen test model changed')
    identity = {'source_sha256': plan['source_sha256'], 'plan_sha256': digest(plan),
        'models': {label: entry['sha256'] for label, entry in frozen.items()},
        'cache_format': 'riichi_test_block_v1'}
    return plan, frozen, partial(cached_block, cache_root=str(root / 'test-blocks'), identity=identity)


def install(root):
    """Install before evaluate_models; preserve its match and bootstrap semantics."""
    from mahjong_sandbox.rl import riichi_evaluate as evaluator
    _, _, block = setup(root)
    evaluator._initialize = initialize
    evaluator._block = block


def evaluate_curriculum(root, workers):
    """The curriculum freezes its model list before this wrapper opens the cache."""
    from scripts import run_riichi_curriculum as curriculum
    from mahjong_sandbox.rl import riichi_evaluate as evaluator
    original = evaluator.evaluate_models
    def evaluate(model, **kwargs):
        install(root)
        return original(model, **kwargs)
    evaluator.evaluate_models = evaluate
    try:
        return curriculum.evaluate(Path(root), workers)
    finally:
        evaluator.evaluate_models = original


def selected_v2(plan,frozen):
    from mahjong_sandbox.rl.checkpoint import read_metadata
    labels=['v2_'+str(seed) for seed in plan['train_seeds'] if 'v2_'+str(seed) in frozen]
    if not labels:raise ValueError('no planned V2 RL seed in the frozen test models')
    return max(labels,key=lambda label:read_metadata(frozen[label]['path'])['training']['validation_score'])


def precompute(root, workers=8):
    from dataclasses import asdict
    from mahjong_sandbox.rl.checkpoint import read_metadata
    from mahjong_sandbox.riichi_model import RiichiConfig
    plan, frozen, block = setup(root)
    selected = selected_v2(plan,frozen)
    models = {'rl': frozen[selected]['path'], **{label: entry['path'] for label, entry in frozen.items() if label != selected}}
    rules = asdict(RiichiConfig(**read_metadata(models['rl'])['rules']))
    tasks = [(plan['test_seed'] + b, candidate, 'greedy', rules, 'numba', True, 'south', True, plan['match_config'])
        for candidate in (*models, 'greedy', 'defensive') for b in range(plan['eval_blocks'])]
    # Meet the main evaluator in the middle; duplicate tasks remain deterministic.
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=initialize,
            initargs=(models, 'numba', True)) as pool:
        for index, _ in enumerate(pool.map(block, reversed(tasks), chunksize=1), 1):
            if index % 32 == 0:
                print(json.dumps({'event': 'test_cache_progress', 'visited_blocks': index, 'total_blocks': len(tasks)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    precompute(args.root, args.workers)


if __name__ == '__main__':
    main()
