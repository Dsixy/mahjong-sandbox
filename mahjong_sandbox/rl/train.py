"""Train, validate, checkpoint and resume Mahjong MaskablePPO on CPU or CUDA."""
from collections import deque
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from ..bots import POLICIES
from ..model import RuleConfig
from ..riichi_model import RiichiConfig
from .checkpoint import load_checkpoint, read_metadata, save_checkpoint, resolve_device
from .env import MahjongVecEnv
from .evaluate import validate_model
from .imitation import pretrain
from .policy import MahjongPolicy
from .placement import DEFAULT_RANK_REWARDS, rank_rewards as validate_rank_rewards, validation_objective
from .schema import rules_from_values, feature_module


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 65536
    n_envs: int = 64
    workers: int = 4
    rollout_steps: int = 64
    batch_size: int = 256
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 1.
    gae_lambda: float = .95
    clip_range: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float = .03
    reward_scale: float | None = None
    reward_mode: str = 'score'
    rank_rewards: tuple[float, ...] = DEFAULT_RANK_REWARDS
    validation_workers: int = 1
    hidden: int = 128
    seed: int = 42
    backend: str = 'auto'
    device: str = 'cpu'
    torch_threads: int = 1
    opponents: tuple[str, ...] = ('greedy', 'defensive', 'value')
    hints: bool = True
    bc_samples: int = 8000
    bc_epochs: int = 8
    checkpoint_every: int = 8
    validation_every: int = 8
    validation_blocks: int = 8
    validation_seed: int = 100000
    self_play_probability: float = 0.
    snapshot_every: int = 8
    snapshot_pool_size: int = 4
    variant: str = 'sichuan'
    match_length: str = 'hand'
    match_config: object = None
    feature_version: str | None = None
    anchor_directories: tuple[str, ...] = ()
    anchor_coefficient: float = 0.
    anchor_steps: int = 2
    anchor_batch_size: int = 256
    skip_forced_actions: bool | None = None
    rules: RuleConfig | RiichiConfig | None = None

    def __post_init__(self):
        if self.variant not in ('sichuan', 'riichi'):
            raise ValueError('unknown training variant')
        if self.reward_mode not in ('score', 'rank'):
            raise ValueError('unknown reward mode')
        object.__setattr__(self, 'rank_rewards', validate_rank_rewards(self.rank_rewards))
        if self.reward_scale is None:
            object.__setattr__(self, 'reward_scale', 1. if self.reward_mode == 'rank' else 10000. if self.variant == 'riichi' else 32.)
        if self.reward_mode == 'rank' and (self.variant != 'riichi' or self.match_length == 'hand' or self.reward_scale != 1 or self.gamma != 1):
            raise ValueError('pure rank reward requires a riichi match, scale=1 and gamma=1')
        if self.skip_forced_actions is None:
            object.__setattr__(self, 'skip_forced_actions', self.variant == 'riichi')
        if type(self.skip_forced_actions) is not bool:
            raise ValueError('skip_forced_actions must be boolean')
        if self.match_length not in ('hand', 'east', 'south') or (self.variant != 'riichi' and self.match_length != 'hand'):
            raise ValueError('match training requires riichi')
        if self.rules is None:
            object.__setattr__(self, 'rules', RiichiConfig() if self.variant == 'riichi' else RuleConfig())
        if isinstance(self.rules, dict):
            object.__setattr__(self, 'rules', rules_from_values(self.rules))
        feature_module(self.rules,self.feature_version)
        if self.match_config is not None:
            from ..riichi_match import MatchConfig
            if isinstance(self.match_config,dict):
                object.__setattr__(self,'match_config',MatchConfig(**self.match_config))
            if self.variant!='riichi' or not isinstance(self.match_config,MatchConfig) or self.match_config.length!=self.match_length:
                raise ValueError('match config must agree with riichi match_length')
        if isinstance(self.opponents, list):
            object.__setattr__(self, 'opponents', tuple(self.opponents))
        if isinstance(self.anchor_directories,list):
            object.__setattr__(self,'anchor_directories',tuple(self.anchor_directories))
        if (not np.isfinite(self.anchor_coefficient) or self.anchor_coefficient < 0 or
                bool(self.anchor_directories) != bool(self.anchor_coefficient) or
                type(self.anchor_steps) is not int or self.anchor_steps < 1 or
                type(self.anchor_batch_size) is not int or self.anchor_batch_size < 1 or
                (self.anchor_coefficient and self.variant != 'riichi')):
            raise ValueError('teacher anchor requires riichi datasets, positive coefficient and batch settings')
        for key in ('total_timesteps', 'n_envs', 'rollout_steps', 'batch_size', 'epochs', 'hidden', 'torch_threads', 'checkpoint_every', 'validation_every', 'snapshot_every', 'snapshot_pool_size', 'validation_workers'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('workers', 'bc_samples', 'bc_epochs', 'validation_blocks', 'validation_seed', 'seed'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.workers > self.n_envs or not 0 <= self.seed < 2**32:
            raise ValueError('invalid workers or RNG seed')
        size = self.n_envs * self.rollout_steps
        if size < 2 or self.batch_size < 2 or size % self.batch_size:
            raise ValueError('rollout size must be divisible by minibatch size, both >= 2')
        if self.bc_samples and not self.bc_epochs:
            raise ValueError('positive bc_epochs required when pretraining')
        for key in ('learning_rate', 'reward_scale', 'clip_range', 'max_grad_norm', 'target_kl'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('gamma/lambda outside valid range')
        if any(not np.isfinite(v) or v < 0 for v in (self.entropy_coef, self.value_coef)):
            raise ValueError('invalid loss coefficients')
        if not self.opponents or any(p not in POLICIES for p in self.opponents):
            raise ValueError('known opponent policies required')
        if self.backend not in ('auto', 'python', 'numba') or type(self.hints) is not bool:
            raise ValueError('invalid operator or feature settings')
        if not isinstance(self.rules, RiichiConfig if self.variant == 'riichi' else RuleConfig):
            raise ValueError('training variant and rules do not match')
        if not np.isfinite(self.self_play_probability) or not 0 <= self.self_play_probability <= 1:
            raise ValueError('self_play_probability must be in [0,1]')



class TrainingInterrupted(Exception):
    """A termination request reached a completed optimizer update boundary."""


class TrainingCallback(BaseCallback):
    def __init__(self, output, config, context):
        super().__init__()
        self.output, self.config, self.context = Path(output), config, context
        self.episodes = deque(maxlen=512)
        self.validations, self.history = [], []
        self.last_step = -1
        self.rollout_updates = 0
        self.last_validation = -1
        self.best_score = -float('inf')
        self.stop_requested = False
        self.anchor = None
        if config.anchor_coefficient:
            from .distillation import TeacherAnchor
            self.anchor=TeacherAnchor(config.anchor_directories,feature_version=config.feature_version,
                coefficient=config.anchor_coefficient,steps=config.anchor_steps,
                batch_size=config.anchor_batch_size,seed=config.seed)
        if (self.output / 'best.zip').exists():
            saved = read_metadata(self.output / 'best.zip', expected_rules=config.rules)
            settings = saved['training']
            if (settings.get('validation_seed') == config.validation_seed and
                    settings.get('validation_blocks') == config.validation_blocks and saved['hints'] == config.hints and
                    settings.get('config', {}).get('reward_mode', 'score') == config.reward_mode and
                    tuple(settings.get('config', {}).get('rank_rewards', DEFAULT_RANK_REWARDS)) == config.rank_rewards and
                    saved['feature_version']==(config.feature_version or saved['feature_version'])):
                self.best_score = float(settings.get('validation_score', -float('inf')))
        self.started = time.perf_counter()

    def emit(self, event):
        event = {'elapsed_seconds': time.perf_counter() - self.started, **event}
        with (self.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(event, ensure_ascii=False), flush=True)
        return event

    def _on_training_start(self):
        self.last_step = self.model.num_timesteps
        self.emit({'event': 'start', 'timesteps': self.model.num_timesteps,
                   'device': str(self.model.device), 'config': asdict(self.config)})

    def _on_step(self):
        for info in self.locals.get('infos', ()):
            if 'metrics' in info:
                self.episodes.append(info['metrics'])
        return True

    def finish_iteration(self, final=False):
        steps = self.model.num_timesteps
        if steps == self.last_step:
            return
        self.last_step = steps
        self.rollout_updates += 1
        iteration = self.rollout_updates
        stats = {'event': 'ppo_update', 'timesteps': steps, 'optimizer_epochs': self.model._n_updates,
                 'rollout_updates_this_run': self.rollout_updates}
        if self.anchor is not None:
            stats.update(self.anchor.update(self.model))
        for key, value in self.model.logger.name_to_value.items():
            if key.startswith('train/') and np.isscalar(value):
                numeric = float(value)
                if not np.isfinite(numeric) and key != 'train/explained_variance':
                    raise FloatingPointError(f'nonfinite training metric: {key}')
                stats[key] = numeric if np.isfinite(numeric) else None
        if self.episodes:
            stats['recent_completed_episodes'] = len(self.episodes)
            stats.update({f'train/{key}': float(np.mean([r[key] for r in self.episodes]))
                          for key in ('win', 'deal_in', 'net_score', 'rank', 'rank_utility', 'first_place', 'last_place')
                          if all(key in r for r in self.episodes)})
        self.history.append(self.emit(stats))
        if iteration % self.config.checkpoint_every == 0 or final:
            save_checkpoint(self.model, self.output / 'checkpoints' / f'step_{steps}.zip',
                            rules=self.config.rules, hints=self.config.hints, training=self.context)
        if self.config.self_play_probability and iteration % self.config.snapshot_every == 0:
            from .selfplay import publish_snapshot
            pool = publish_snapshot(self.model, self.output / 'selfplay', rules=self.config.rules,
                hints=self.config.hints, training=self.context, pool_size=self.config.snapshot_pool_size)
            self.emit({'event': 'self_play_pool', 'timesteps': steps, **pool})
        if self.config.validation_blocks and (iteration % self.config.validation_every == 0 or final):
            self.validate()

    def validate(self):
        metadata = {'rules': asdict(self.config.rules), 'hints': self.config.hints, 'training': self.context}
        stats = validate_model(self.model, metadata, blocks=self.config.validation_blocks,
                               seed=self.config.validation_seed, backend=self.config.backend, workers=self.config.validation_workers)
        event = self.emit({'event': 'validation', 'timesteps': self.model.num_timesteps, **stats})
        self.validations.append(event)
        self.last_validation = self.model.num_timesteps
        # Only PPO-trained checkpoints compete here; the BC-only model is kept
        # separately for a held-out ablation instead of masquerading as RL.
        if validation_objective(stats, self.config) > self.best_score:
            self.best_score = validation_objective(stats, self.config)
            save_checkpoint(self.model, self.output / 'best.zip', rules=self.config.rules,
                hints=self.config.hints, training={**self.context, 'validation_score': self.best_score,
                    'validation_metric': 'rank_utility' if self.config.reward_mode == 'rank' else 'mean_net_score'})

    def _on_rollout_start(self):
        if self.model._n_updates:
            self.finish_iteration()
        if self.stop_requested:
            raise TrainingInterrupted()

    def _on_training_end(self):
        self.finish_iteration(final=True)


def train(output, config=None, *, resume=None):
    config = config or TrainConfig()
    device = resolve_device(config.device)
    torch.set_num_threads(config.torch_threads)
    if resume:
        previous = read_metadata(resume, expected_rules=config.rules)
        old = previous['training'].get('config', {})
        if previous['timesteps'] and (old.get('reward_mode', 'score') != config.reward_mode or
                (config.reward_mode == 'rank' and tuple(old.get('rank_rewards', DEFAULT_RANK_REWARDS)) != config.rank_rewards)):
            raise ValueError('changing trained reward units requires a zero-step warmstart, not optimizer resume')
        if previous['hints'] != config.hints:
            raise ValueError('cannot change hint features while resuming')
    else:
        previous = None
    version=config.feature_version or (previous['feature_version'] if previous else
        'riichi_visible_v2' if config.variant=='riichi' else 'sichuan_visible_v1')
    if previous and version!=previous['feature_version']:
        raise ValueError('feature versions cannot change on resume; create a new cold start')
    config=replace(config,feature_version=version)
    output = Path(output)
    if any((output / name).exists() for name in ('latest.zip', 'best.zip', 'warmstart.zip')) and not resume:
        raise ValueError('output already contains a model; use --resume or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    if resume and (output / 'training.json').exists():
        (output / f'training_before_{previous["timesteps"]}.json').write_bytes((output / 'training.json').read_bytes())
    (output / 'config.json').write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2) + '\n')
    # Resume restores policy, value network and Adam state. New games use a new
    # seed stream; in-flight walls/trajectories are intentionally not restored.
    env_seed = (config.seed + (previous['timesteps'] + 1000003 if previous else 0)) % 2**32
    env = MahjongVecEnv(config.n_envs, config.workers, seed=env_seed, opponents=config.opponents,
        backend=config.backend, reward_scale=config.reward_scale, rules=config.rules, hints=config.hints,
        opponent_manifest=output.resolve() / 'selfplay' / 'opponents.json' if config.self_play_probability else None,
        self_play_probability=config.self_play_probability, variant=config.variant, match_length=config.match_length,
        skip_forced_actions=config.skip_forced_actions, reward_mode=config.reward_mode, rank_rewards=config.rank_rewards,
        feature_version=config.feature_version,match_config=config.match_config)
    started = time.perf_counter()
    bc_report = previous['training'].get('imitation') if previous else None
    initial_validation = None
    try:
        hyperparameters = dict(learning_rate=config.learning_rate, n_steps=config.rollout_steps,
            batch_size=config.batch_size, n_epochs=config.epochs, gamma=config.gamma,
            gae_lambda=config.gae_lambda, clip_range=config.clip_range, ent_coef=config.entropy_coef,
            vf_coef=config.value_coef, max_grad_norm=config.max_grad_norm, target_kl=config.target_kl)
        if resume:
            model, _ = load_checkpoint(resume, env=env, device=device, expected_rules=config.rules, **hyperparameters)
            # Loading can preserve an older seed; use this run's new wall stream.
            model.set_random_seed(env_seed)
            start_steps = model.num_timesteps
        else:
            model = MaskablePPO(MahjongPolicy, env, seed=config.seed, device=device,
                policy_kwargs={'hidden': config.hidden, 'variant': config.variant}, verbose=0, **hyperparameters)
            start_steps = 0
        context = {'config': asdict(config), 'validation_seed': config.validation_seed,
                   'validation_blocks': config.validation_blocks, 'env_seed': env_seed,
                   'resumed_from': str(resume) if resume else None,
                   'resumed_model_sha256': previous['model_sha256'] if previous else None,
                   'resume_restores': 'model, value function, Adam; new hands, not exact in-flight state'}
        if config.bc_samples and not resume:
            bc_report = pretrain(model, config.bc_samples, epochs=config.bc_epochs,
                seed=(config.seed + 1234) % 2**32, rules=config.rules, backend=config.backend, hints=config.hints,
                feature_version=config.feature_version,match_length=config.match_length,match_config=config.match_config)
            (output / 'imitation.json').write_text(json.dumps(bc_report, indent=2) + '\n')
            save_checkpoint(model, output / 'warmstart.zip', rules=config.rules, hints=config.hints,
                            training={**context, 'imitation': bc_report})
            print(json.dumps({'event': 'imitation_complete', **bc_report}), flush=True)
        context['imitation'] = bc_report
        if config.self_play_probability:
            from .selfplay import publish_snapshot
            publish_snapshot(model, output / 'selfplay', rules=config.rules, hints=config.hints,
                training=context, pool_size=config.snapshot_pool_size)
        if config.validation_blocks:
            initial_validation = validate_model(model, {'rules': asdict(config.rules), 'hints': config.hints, 'training': context},
                blocks=config.validation_blocks, seed=config.validation_seed, backend=config.backend, workers=config.validation_workers)
            print(json.dumps({'event': 'initial_validation', **initial_validation}), flush=True)
        callback = TrainingCallback(output, config, context)
        def request_stop(signum, frame):
            callback.stop_requested = True
        old_handler = signal.signal(signal.SIGTERM, request_stop)
        interrupted = False
        try:
            model.learn(total_timesteps=config.total_timesteps, callback=callback,
                        reset_num_timesteps=not bool(resume), progress_bar=False)
        except TrainingInterrupted:
            interrupted = True
        finally:
            signal.signal(signal.SIGTERM, old_handler)
        latest = save_checkpoint(model, output / 'latest.zip', rules=config.rules, hints=config.hints, training=context)
        if not (output / 'best.zip').exists():
            save_checkpoint(model, output / 'best.zip', rules=config.rules, hints=config.hints, training=context)
        report = {'algorithm': 'sb3-contrib MaskablePPO', 'config': asdict(config),
            'actual_device': str(model.device), 'start_timesteps': start_steps,
            'final_timesteps': model.num_timesteps, 'additional_timesteps': model.num_timesteps - start_steps,
            'optimizer_epochs': model._n_updates, 'rollout_updates_this_run': callback.rollout_updates,
            'completed_episodes_this_run': env.completed_episodes,
            'noninteractive_episodes_skipped': env.noninteractive_episodes_skipped,
            'elapsed_seconds': time.perf_counter() - started, 'imitation': bc_report,
            'initial_validation': initial_validation, 'validations': callback.validations,
            'latest_model_sha256': latest['model_sha256'], 'source_sha256': latest['source_sha256'],
            'latest': str(output / 'latest.zip'), 'best': str(output / 'best.zip'),
            'resume': context['resume_restores'], 'illegal_actions': 0}
        report['interrupted'] = interrupted
        (output / 'training.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        return report
    finally:
        env.close()
