"""Explicit adapter from the stage-5 batch contract to SB3 VecEnv autoreset."""
from dataclasses import replace
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from ..batch import BatchConfig, SyncBatchEnv, MAX_ACTIONS
from ..parallel import ProcessBatchEnv
from ..model import RuleConfig
from .features import STATE_DIM, HINT_DIM, encode_batch


def observation_space():
    return spaces.Dict({
        'state': spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
        'candidates': spaces.Box(-1, 135, (MAX_ACTIONS, 8), np.int16),
        'hints': spaces.Box(-np.inf, np.inf, (MAX_ACTIONS, HINT_DIM), np.float32),
        'mask': spaces.Box(0, 1, (MAX_ACTIONS,), np.bool_),
    })


class SichuanVecEnv(VecEnv):
    def __init__(self, n=32, workers=0, *, seed=0, opponents=('greedy',),
                 backend='auto', reward_scale=32., rules=None, hints=True, rotate_seats=True):
        if type(n) is not int or n < 1 or type(workers) is not int or not 0 <= workers <= n:
            raise ValueError('positive environment count and workers in 0..N required')
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('reward_scale must be finite and positive')
        self.config = BatchConfig(variant='sichuan', seed=seed, backend=backend,
            rules=rules or RuleConfig(), rotate_seats=rotate_seats, opponent_pool=tuple(opponents))
        if not opponents:
            raise ValueError('at least one opponent policy required')
        self.workers, self.reward_scale, self.hints = workers, reward_scale, hints
        self.core = self.raw = self.features = None
        self.pending = None
        self.episode_returns = np.zeros(n)
        self.episode_lengths = np.zeros(n, np.int64)
        self.initial_scores = np.zeros((n, 4), np.int64)
        self.completed_episodes = 0
        self.render_mode = None
        super().__init__(n, observation_space(), spaces.Discrete(MAX_ACTIONS))

    def _encode(self, batch):
        return encode_batch(batch, rules=self.config.rules, backend=self.config.backend, hints=self.hints)

    def reset(self):
        requested_seed = self._seeds[0]
        if self.core is None or requested_seed is not None:
            if self.core is not None:
                self.core.close()
            if requested_seed is not None:
                self.config = replace(self.config, seed=int(requested_seed))
            self.core = (ProcessBatchEnv(self.num_envs, self.workers, self.config) if self.workers
                         else SyncBatchEnv(self.num_envs, self.config))
            self.raw = self.core.observation()
        else:
            self.raw = self.core.reset()
        self.pending = None
        self.initial_scores = self.raw['scores'].copy()
        self.episode_returns.fill(0)
        self.episode_lengths.fill(0)
        self.features = self._encode(self.raw)
        self._reset_seeds()
        self._reset_options()
        return self.features

    def step_async(self, actions):
        if self.pending is not None:
            raise RuntimeError('step already pending')
        if self.raw is None:
            raise RuntimeError('reset the VecEnv before stepping')
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or actions.dtype.kind not in 'iu':
            raise ValueError('integer actions [N] required')
        if np.any(actions < 0) or np.any(actions >= MAX_ACTIONS) or not np.all(self.features['mask'][np.arange(self.num_envs), actions]):
            raise ValueError('action not in the semantic legal menu')
        self.pending = actions.copy()
        if self.workers:
            self.core.step_async(actions, self.raw['tokens'])

    def step_wait(self):
        if self.pending is None:
            raise RuntimeError('no pending step')
        players = self.raw['meta'][:, 1].copy()
        terminal = self.core.step_wait() if self.workers else self.core.step(self.pending, self.raw['tokens'])
        self.pending = None
        if terminal['truncated'].any():
            raise RuntimeError('unexpected rule-engine truncation')
        rewards = terminal['rewards'][np.arange(self.num_envs), players] / self.reward_scale
        self.episode_returns += rewards
        self.episode_lengths += 1
        done = terminal['terminated'].copy()
        infos = [{} for _ in range(self.num_envs)]
        indices = np.flatnonzero(done)
        if len(indices):
            final_features = self._encode({k: v[indices] for k, v in terminal.items()})
            for local, i in enumerate(indices):
                wins = terminal['wins'][i]
                net = int(terminal['scores'][i, players[i]] - self.initial_scores[i, players[i]])
                if not np.isclose(self.episode_returns[i] * self.reward_scale, net, atol=1e-4):
                    raise AssertionError('training reward lost part of the blood-battle settlement')
                infos[i] = {'terminal_observation': {k: v[local].copy() for k, v in final_features.items()},
                            'TimeLimit.truncated': False,
                            'episode': {'r': float(self.episode_returns[i]), 'l': int(self.episode_lengths[i])},
                            'metrics': {'win': int(np.any(wins[:, 0] == players[i])),
                                        'deal_in': int(np.any(wins[:, 1] == players[i])), 'net_score': net,
                                        'seat': int(players[i])}}
            self.completed_episodes += len(indices)
            self.raw = self.core.reset_done()
            self.initial_scores[indices] = self.raw['scores'][indices]
            self.episode_returns[indices] = 0
            self.episode_lengths[indices] = 0
        else:
            self.raw = terminal
        self.features = self._encode(self.raw)
        return self.features, rewards.astype(np.float32), done, infos

    def get_attr(self, attr_name, indices=None):
        if attr_name == 'render_mode':
            return [None for _ in self._get_indices(indices)]
        if attr_name == 'action_masks':
            return [lambda i=i: self.features['mask'][i].copy() for i in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def has_attr(self, attr_name):
        return attr_name in ('action_masks', 'render_mode')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == 'action_masks':
            return [self.features['mask'][i].copy() for i in self._get_indices(indices)]
        raise AttributeError(method_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError('change training configuration by constructing a new environment')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        if self.core is not None:
            self.core.close()
            self.core = None
