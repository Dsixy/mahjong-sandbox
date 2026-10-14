"""Reproducible correctness-checked timing, including cold process/JIT costs."""
import cProfile
import os
import platform
import pstats
import sys
from pathlib import Path
import time
from .batch import BatchConfig, SyncBatchEnv, ACTION_KINDS
from .evaluation import run_game
from .operators import numpy, native_available, shanten_batch, waits_batch, discard_ukeire_batch


def metadata():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'logical_cpus': os.cpu_count(), 'affinity_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    for name in ('numpy', 'numba', 'llvmlite'):
        try:
            from importlib.metadata import version
            result[name] = version(name)
        except ImportError:
            result[name] = None
    for path in ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/memory.max'):
        if Path(path).exists():
            result[Path(path).name] = Path(path).read_text().strip()
    return result


def _clear():
    from .greedy import _regular_distance
    from .operators import shanten, structural_waits
    _regular_distance.cache_clear()
    shanten.cache_clear()
    structural_waits.cache_clear()


def _rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def batch_trial(n=1000, workers=4, *, variant='riichi', backend='auto', seed=700, controlled_player=0):
    np = numpy()
    from .parallel import ProcessBatchEnv
    cfg = BatchConfig(variant=variant, seed=seed, backend=backend, opponent_policy='greedy', controlled_player=controlled_player)
    clock = time.perf_counter()
    env = ProcessBatchEnv(n, workers, cfg) if workers else SyncBatchEnv(n, cfg)
    startup = time.perf_counter() - clock
    elapsed_steps, decisions, latencies, peak_rss = 0, 0, [], 0
    rollout = time.perf_counter()
    warm = 0
    with env:
        b = env.observation()
        initial = b['scores'].copy()
        rewards = np.zeros((n, 4))
        for step in range(400):
            active = ~b['terminated']
            if not active.any():
                break
            a = b['action_mask'].argmax(axis=1).astype(np.int64)
            win = b['action_mask'] & np.isin(b['actions'][:, :, 0], [ACTION_KINDS.index('ron'), ACTION_KINDS.index('tsumo')])
            winning = win.any(axis=1)
            a[winning] = win[winning].argmax(axis=1)
            a[~active] = -1
            clock = time.perf_counter()
            b = env.step(a, b['tokens'])
            seconds = time.perf_counter() - clock
            rewards += b['rewards']
            if step < 3:
                warm += seconds
            else:
                elapsed_steps += seconds
                decisions += int(active.sum())
                latencies.append(seconds)
            pids = [os.getpid()] + [p.pid for p in getattr(env, 'processes', ())]
            resident = [_rss(pid) for pid in pids]
            peak_rss = max(peak_rss, sum(x for x in resident if x is not None))
        total = time.perf_counter() - rollout
        if not b['terminated'].all():
            raise AssertionError('batch did not finish all hands')
        np.testing.assert_array_equal(rewards, b['scores'] - initial)
        shared = getattr(env, 'shared_bytes', sum(v.nbytes for v in env.buffers.values()))
    return {'variant': variant, 'seed': seed, 'backend': backend, 'live_boards': n, 'workers': workers,
            'controlled_player': controlled_player, 'opponent_policy': 'greedy' if controlled_player is not None else None,
            'startup_seconds': startup, 'first_three_steps_seconds': warm,
            'complete_rollout_seconds': total, 'steady_step_seconds': elapsed_steps,
            'steady_learner_decisions': decisions, 'decisions_per_second': decisions / elapsed_steps,
            'batch_latency_p50_ms': float(np.quantile(latencies, .5) * 1000),
            'batch_latency_p95_ms': float(np.quantile(latencies, .95) * 1000),
            'shared_buffer_bytes': shared, 'sampled_rss_sum_bytes': peak_rss,
            'rss_note': 'sum of RSS includes shared-page duplicates; sampled at step boundaries',
            'completed_hands': int(b['terminated'].sum()), 'rewards_verified': True}


def benchmark(*, games=20, boards=1000, workers=(1, 2, 4), seed=10000):
    np = numpy()
    if games < 1 or boards < 1:
        raise ValueError('positive games/boards required')
    backend = 'numba' if native_available() else 'python'
    cold = time.perf_counter()
    run_game(-123, variant='riichi', backend=backend)
    run_game(-123, variant='sichuan', backend=backend)
    cold = time.perf_counter() - cold
    serial, profiles = [], {}
    for variant in ('sichuan', 'riichi'):
        expected = None
        for mode in ('reference', 'lean-python', 'lean-native') if native_available() else ('reference', 'lean-python'):
            _clear()
            chosen = 'numba' if mode == 'lean-native' else 'python'
            run_game(-123, variant=variant, backend=chosen, reference=mode == 'reference')
            clock = time.perf_counter()
            rows = [run_game(seed + i, variant=variant, backend=chosen, reference=mode == 'reference') for i in range(games)]
            elapsed = time.perf_counter() - clock
            results = [(g['scores'], g['commands'], g['winners'], g['reason']) for g in rows]
            if expected is None:
                expected = results
            if results != expected:
                raise AssertionError('benchmark policies or rules differ across backends')
            serial.append({'variant': variant, 'mode': mode, 'games': games, 'elapsed_seconds': elapsed,
                           'games_per_second': games / elapsed,
                           'commands_per_second': sum(g['commands'] for g in rows) / elapsed,
                           'identical_results': True})
            print(f'{variant} {mode}: {games / elapsed:.2f} games/s', file=sys.stderr, flush=True)
        for mode in ('reference', 'lean-native'):
            _clear()
            profiler = cProfile.Profile()
            with profiler:
                for i in range(3):
                    run_game(seed + i, variant=variant, backend=backend, reference=mode == 'reference')
            stats = pstats.Stats(profiler)
            top = sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:12]
            profiles[f'{variant}-{mode}'] = [{'function': f'{Path(file).name}:{line}:{name}',
                                               'calls': v[1], 'cumulative_seconds': v[3]}
                                              for (file, line, name), v in top]
    rng = np.random.default_rng(seed)
    counts = np.zeros((1000, 34), dtype=np.int16)
    for row in counts:
        for tile in rng.choice(136, 14, replace=False):
            row[tile // 4] += 1
    operators = []
    for fn in (shanten_batch, waits_batch, discard_ukeire_batch):
        inputs = counts.copy()
        if fn is waits_batch:
            for row in inputs:
                row[np.flatnonzero(row)[0]] -= 1
        args = (inputs, inputs) if fn is discard_ukeire_batch else (inputs,)
        outputs = []
        for chosen in ('python', backend):
            _clear()
            fn(*(a[:1] for a in args), backend=chosen)
            clock = time.perf_counter()
            out = fn(*args, backend=chosen)
            elapsed = time.perf_counter() - clock
            outputs.append(out)
            operators.append({'operator': fn.__name__, 'backend': chosen, 'rows': 1000, 'seconds': elapsed})
            print(f'{fn.__name__} {chosen}: {elapsed:.3f}s / 1000 rows', file=sys.stderr, flush=True)
        for a, b in zip(outputs[0] if isinstance(outputs[0], tuple) else (outputs[0],), outputs[1] if isinstance(outputs[1], tuple) else (outputs[1],)):
            np.testing.assert_array_equal(a, b)
    batches = []
    for w in workers:
        trial = batch_trial(boards, w, backend=backend, seed=seed)
        batches.append(trial)
        print(f'{boards} boards, {w} workers: {trial["decisions_per_second"]:.1f} decisions/s', file=sys.stderr, flush=True)
    return {'hardware': metadata(), 'seed': seed, 'cold_operator_warmup_seconds': cold,
            'serial': serial, 'operators': operators, 'batches': batches, 'profiles': profiles}
