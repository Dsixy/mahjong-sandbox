"""Bounded, restartable experiments for preemptible workers and persistent output."""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
from .checkpoint import read_metadata, source_fingerprint
from .train import TrainConfig, train


def newest_checkpoint(output, initial, *, rules):
    """A torn latest pair cannot hide an earlier immutable completed update."""
    output = Path(output)
    candidates = [Path(initial), output / 'latest.zip', *output.glob('checkpoints/step_*.zip')]
    ranked = []
    for path in candidates:
        try:
            steps = int(json.loads(path.with_suffix('.json').read_text())['timesteps'])
            ranked.append((steps, str(path)))
        except (OSError, ValueError, KeyError):
            continue
    for _, path in sorted(ranked, reverse=True):
        try:
            return Path(path), read_metadata(path, expected_rules=rules)
        except (OSError, ValueError, KeyError):
            print(json.dumps({'event': 'invalid_checkpoint_skipped', 'path': path}), flush=True)
    raise ValueError('no intact compatible checkpoint to resume')


def run(config, output, initial, *, additional_steps, evaluate=True, eval_blocks=100):
    if type(additional_steps) is not int or additional_steps < 1:
        raise ValueError('additional_steps must be positive')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_metadata = read_metadata(initial, expected_rules=config.rules)
        quantum = config.n_envs * config.rollout_steps
        target = initial_metadata['timesteps'] + ((additional_steps + quantum - 1) // quantum) * quantum
        experiment = {'version': 1, 'initial_sha256': initial_metadata['model_sha256'],
            'initial_timesteps': initial_metadata['timesteps'], 'target_timesteps': target,
            'additional_steps_requested': additional_steps, 'config': asdict(config),
            'evaluation_blocks': eval_blocks, 'evaluate': evaluate, 'source_sha256': source_fingerprint()}
        experiment = json.loads(json.dumps(experiment))
        manifest = output / 'experiment.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved != experiment:
                raise ValueError('experiment settings changed; use a new output directory')
        else:
            manifest.write_text(json.dumps(experiment, indent=2) + '\n')
        resume, metadata = newest_checkpoint(output, initial, rules=config.rules)
        print(json.dumps({'event': 'experiment_resume', 'checkpoint': str(resume),
                          'timesteps': metadata['timesteps'], 'target': target}), flush=True)
        if metadata['timesteps'] < target:
            report = train(output, replace(config, total_timesteps=target - metadata['timesteps']), resume=resume)
            if report['interrupted']:
                return 75  # RJob retries against the same bounded target.
        if evaluate and not (output / 'evaluation.json').exists():
            from .evaluate import evaluate_models
            # Fixed held-out walls; best is chosen on an entirely different seed set.
            stats = evaluate_models(output / 'best.zip', baseline_model=initial,
                opponents=('greedy',), blocks=eval_blocks, seed=200000, workers=1,
                backend=config.backend, device=config.device, baseline_label='initial')
            temporary = output / 'evaluation.tmp'
            temporary.write_text(json.dumps(stats, indent=2) + '\n')
            temporary.replace(output / 'evaluation.json')
        (output / 'complete.json').write_text(json.dumps({'target_timesteps': target,
            'evaluation': str(output / 'evaluation.json') if evaluate else None}, indent=2) + '\n')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--additional-steps', type=int, default=10000000)
    parser.add_argument('--eval-blocks', type=int, default=100)
    parser.add_argument('--no-evaluation', action='store_true')
    args = parser.parse_args()
    return run(TrainConfig(**json.loads(args.config.read_text())), args.output, args.initial,
        additional_steps=args.additional_steps, evaluate=not args.no_evaluation, eval_blocks=args.eval_blocks)


if __name__ == '__main__':
    raise SystemExit(main())
