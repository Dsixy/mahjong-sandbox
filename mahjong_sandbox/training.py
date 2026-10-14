"""A small real REINFORCE smoke test for the vector environment contract.

This is a readiness check, not a claim to train a competitive Mahjong model.
"""
import time
from .batch import ACTION_KINDS, BatchConfig, SyncBatchEnv
from .operators import numpy

FEATURE_COUNT = 36


def candidate_features(batch):
    """Compact visible-only features; each menu entry has its own identity."""
    np = numpy()
    actions = batch['actions']
    n, m = actions.shape[:2]
    x = np.zeros((n, m, FEATURE_COUNT), dtype=np.float32)
    x[:, :, 0] = 1
    codes = np.maximum(actions[:, :, 0], 0)
    x[:, :, 1:14] = np.eye(len(ACTION_KINDS), dtype=np.float32)[codes]
    tiles = actions[:, :, 2:6]
    kinds = np.maximum(tiles, 0) // 4
    rows = np.arange(n)[:, None, None]
    valid = tiles >= 0
    x[:, :, 14:18] = batch['own_counts'][rows, kinds] / 4 * valid
    x[:, :, 18:22] = batch['known_counts'][rows, kinds] / 4 * valid
    x[:, :, 22] = (batch['red_mask'][rows, np.maximum(tiles, 0)] * valid).sum(axis=2) / 4
    x[:, :, 23] = ((kinds % 9) * valid).sum(axis=2) / 32
    x[:, :, 24] = ((kinds // 9) * valid).sum(axis=2) / 12
    x[:, :, 25:31] = np.eye(6, dtype=np.float32)[batch['meta'][:, 0]][:, None, :]
    players = batch['meta'][:, 1]
    scores = batch['scores']
    x[:, :, 31] = ((scores[np.arange(n), players] - scores.mean(axis=1)) / 25000)[:, None]
    x[:, :, 32] = (batch['meta'][:, 3] / 84)[:, None]
    suits = actions[:, :, 6]
    suit_sizes = batch['own_counts'][:, :27].reshape(n, 3, 9).sum(axis=2)
    declaring = suits >= 0
    x[:, :, 33] = suit_sizes[np.arange(n)[:, None], np.maximum(suits, 0)] / 14 * declaring
    x[:, :, 34] = np.maximum(suits, 0) / 2
    x[:, :, 35] = np.maximum(actions[:, :, 7], 0) / 3
    x *= batch['action_mask'][:, :, None]
    return x


def masked_policy(batch, weights, rng):
    np = numpy()
    x = candidate_features(batch)
    logits = x @ weights
    mask = batch['action_mask']
    logits = np.where(mask, logits, -1e30)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits) * mask
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-20)
    cumulative = probabilities.cumsum(axis=1)
    samples = (cumulative < rng.random((len(mask), 1)) * cumulative[:, -1:]).sum(axis=1)
    samples = np.minimum(samples, mask.shape[1] - 1)
    closed = batch['terminated'] | batch['truncated']
    samples[closed] = -1
    if not np.all(mask[np.flatnonzero(~closed), samples[~closed]]):
        raise AssertionError('masked sampler selected an illegal action')
    expected = np.einsum('nm,nmf->nf', probabilities, x)
    gradient = x[np.arange(len(mask)), np.maximum(samples, 0)] - expected
    gradient[closed] = 0
    return samples, gradient


def training_smoke(*, n=32, steps=160, seed=32, variant='sichuan', backend='auto', workers=0):
    np = numpy()
    rng = np.random.default_rng(seed)
    weights = np.zeros(FEATURE_COUNT, dtype=np.float32)
    eligibility = np.zeros((n, FEATURE_COUNT), dtype=np.float32)
    returns = np.zeros(n)
    reward_sum = np.zeros((n, 4), dtype=np.float64)
    finished = np.zeros(n, dtype=bool)
    config = BatchConfig(variant=variant, seed=seed, controlled_player=0,
                         opponent_policy='greedy', backend=backend)
    start = time.perf_counter()
    if workers:
        from .parallel import ProcessBatchEnv
        env = ProcessBatchEnv(n, workers, config)
    else:
        env = SyncBatchEnv(n, config)
    startup = time.perf_counter() - start
    episodes, updates, learner_decisions, baseline, inference_seconds = 0, 0, 0, 0., 0.
    started = time.perf_counter()
    with env:
        batch = env.observation()
        initial = batch['scores'].copy()
        for _ in range(steps):
            clock = time.perf_counter()
            actions, gradient = masked_policy(batch, weights, rng)
            inference_seconds += time.perf_counter() - clock
            learner_decisions += int((actions >= 0).sum())
            eligibility += gradient
            batch = env.step(actions, batch['tokens'])
            reward_sum += batch['rewards']
            scale = 1000 if variant == 'riichi' else 32
            returns += batch['rewards'][:, 0] / scale
            done = np.flatnonzero(batch['terminated'] & ~finished)
            if len(done):
                np.testing.assert_array_equal(reward_sum[done], batch['scores'][done] - initial[done])
                episodes += len(done)
                finished[done] = True
            # Hold weights fixed while every trajectory in this batch is
            # collected. Closed lanes wait with action -1 and zero reward.
            if finished.all():
                advantage = returns - baseline
                update = (advantage[:, None] * eligibility).mean(axis=0)
                norm = np.linalg.norm(update)
                weights += .01 * update / max(1, norm)
                if not np.isfinite(weights).all():
                    raise AssertionError('nonfinite model update')
                baseline = .9 * baseline + .1 * float(returns.mean())
                updates += 1
                eligibility.fill(0)
                returns.fill(0)
                reward_sum.fill(0)
                finished.fill(False)
                batch = env.reset_done()
                initial = batch['scores'].copy()
        elapsed = time.perf_counter() - started
    if not episodes or not np.any(weights):
        raise AssertionError('smoke test did not complete episodes and update weights; increase steps')
    return {'algorithm': 'linear candidate REINFORCE', 'variant': variant, 'environments': n,
            'feature_count': FEATURE_COUNT,
            'workers': workers, 'seed': seed, 'steps': steps, 'episodes': episodes,
            'updates': updates, 'learner_decisions': learner_decisions,
            'startup_seconds': startup, 'rollout_seconds': elapsed, 'inference_seconds': inference_seconds,
            'decisions_per_second': learner_decisions / elapsed,
            'weight_norm': float(np.linalg.norm(weights)), 'weights': weights.tolist(),
            'reward_conservation_verified': True, 'illegal_actions': 0, 'finite_gradients': True}
