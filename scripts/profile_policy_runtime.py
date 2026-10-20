"""Bounded rollout/update timing; never writes training checkpoints or opponent pools.

Use the same immutable checkpoint, manifest, image and configuration for comparisons.
The first update includes warmup; steady-state summaries exclude it. CUDA timings
synchronize at measured boundaries. Run separately from a production learner.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from types import MethodType

import torch
from stable_baselines3.common.callbacks import BaseCallback

from mahjong_sandbox.rl.checkpoint import load_checkpoint, read_metadata, source_fingerprint
from mahjong_sandbox.rl.env import SichuanVecEnv
from mahjong_sandbox.rl.train import TrainConfig


class Callback(BaseCallback):
    def _on_step(self):
        return True


def profile(config, checkpoint, manifest, updates):
    torch.set_num_threads(config.torch_threads)
    metadata = read_metadata(checkpoint, expected_rules=config.rules)
    seed = (config.seed + metadata['timesteps'] + 1000003) % 2**32
    counters = {}

    class TimedEnv(SichuanVecEnv):
        def _encode(self, batch):
            started = time.perf_counter()
            result = super()._encode(batch)
            counters['feature_seconds'] = counters.get('feature_seconds', 0.) + time.perf_counter() - started
            return result

        def step_wait(self):
            started = time.perf_counter()
            result = super().step_wait()
            counters['environment_step_seconds'] = counters.get('environment_step_seconds', 0.) + time.perf_counter() - started
            return result

    env = TimedEnv(config.n_envs, config.workers, seed=seed, opponents=config.opponents,
        backend=config.backend, reward_scale=config.reward_scale, rules=config.rules,
        hints=config.hints, opponent_manifest=manifest,
        self_play_probability=config.self_play_probability)
    rows = []
    started = time.perf_counter()
    try:
        model, _ = load_checkpoint(checkpoint, env=env, device=config.device,
            expected_rules=config.rules, learning_rate=config.learning_rate,
            n_steps=config.rollout_steps, batch_size=config.batch_size, n_epochs=config.epochs,
            gamma=config.gamma, gae_lambda=config.gae_lambda, clip_range=config.clip_range,
            ent_coef=config.entropy_coef, vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm, target_kl=config.target_kl)
        model.set_random_seed(seed)

        def synchronize():
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)

        original_forward = model.policy.forward
        original_collect, original_train = model.collect_rollouts, model.train

        def forward(_policy, *args, **kwargs):
            synchronize()
            begin = time.perf_counter()
            result = original_forward(*args, **kwargs)
            synchronize()
            counters['learner_forward_seconds'] = counters.get('learner_forward_seconds', 0.) + time.perf_counter() - begin
            return result

        def collect(_model, *args, **kwargs):
            counters.clear()
            synchronize()
            begin = time.perf_counter()
            result = original_collect(*args, **kwargs)
            synchronize()
            counters['rollout_seconds'] = time.perf_counter() - begin
            return result

        def update(_model):
            synchronize()
            begin = time.perf_counter()
            epochs_before = model._n_updates
            original_train()
            synchronize()
            row = dict(counters, update=len(rows) + 1,
                ppo_update_seconds=time.perf_counter() - begin,
                optimizer_epochs=model._n_updates - epochs_before,
                decisions=config.n_envs * config.rollout_steps)
            row['total_seconds'] = row['rollout_seconds'] + row['ppo_update_seconds']
            row['decisions_per_second'] = row['decisions'] / row['total_seconds']
            row['environment_nonencode_seconds'] = row.get('environment_step_seconds', 0.) - row.get('feature_seconds', 0.)
            rows.append(row)
            print(json.dumps({'event': 'profile_update', 'device': config.device,
                'workers': config.workers, 'n_envs': config.n_envs, **row}), flush=True)

        model.policy.forward = MethodType(forward, model.policy)
        model.collect_rollouts = MethodType(collect, model)
        model.train = MethodType(update, model)
        model.learn(total_timesteps=config.n_envs * config.rollout_steps * updates,
            callback=Callback(), reset_num_timesteps=False, progress_bar=False)
        steady = rows[1:] or rows
        seconds = sum(r['total_seconds'] for r in steady)
        return {'config': asdict(config), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': metadata['model_sha256'], 'parameters': sum(p.numel() for p in model.policy.parameters()),
            'source_sha256': source_fingerprint(), 'env_seed': seed,
            'elapsed_seconds_including_startup': time.perf_counter() - started,
            'completed_episodes': env.completed_episodes, 'updates': rows,
            'steady_decisions_per_second': sum(r['decisions'] for r in steady) / seconds,
            'steady_rollout_fraction': sum(r['rollout_seconds'] for r in steady) / seconds,
            'steady_update_fraction': sum(r['ppo_update_seconds'] for r in steady) / seconds,
            'actual_device': str(model.device)}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True, help='A frozen copy, referencing immutable snapshots')
    parser.add_argument('--output', required=True, help='New diagnostic report path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, nargs='+', default=[8])
    parser.add_argument('--updates', type=int, default=3)
    args = parser.parse_args()
    if args.updates < 2:
        parser.error('at least two updates required to distinguish warmup')
    output = Path(args.output)
    if output.exists():
        parser.error('diagnostic report already exists')
    config = TrainConfig(**json.loads(Path(args.config).read_text()))
    if not config.self_play_probability:
        parser.error('this profile expects the self-play training configuration')
    manifest = Path(args.manifest)
    manifest_before = manifest.read_bytes()
    report = {'observed_utc': datetime.now(timezone.utc).isoformat(),
        'torch': torch.__version__, 'python': platform.python_version(),
        'hostname': platform.node(),
        'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), None),
        'cpu_affinity_count': len(os.sched_getaffinity(0)),
        'opponent_manifest': json.loads(manifest_before),
        'notes': ['No checkpoint, validation, or snapshot publication during timings.',
                  'First update excluded from steady-state summaries.',
                  'Environment step includes feature encoding; do not add these overlapping counters.',
                  'CUDA boundaries synchronized; instrumentation can add overhead.'],
        'profiles': []}
    for workers in args.workers:
        result = profile(replace(config, device=args.device, workers=workers),
            args.checkpoint, str(manifest.resolve()), args.updates)
        report['profiles'].append(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if manifest.read_bytes() != manifest_before:
        raise RuntimeError('opponent manifest changed during profiling')
    print(json.dumps({'event': 'profile_complete', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
