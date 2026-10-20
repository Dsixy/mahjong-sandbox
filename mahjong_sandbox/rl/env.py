"""Explicit adapter from the stage-5 batch contract to SB3 VecEnv autoreset."""
from dataclasses import replace
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from ..batch import BatchConfig, SyncBatchEnv, MAX_ACTIONS
from ..parallel import ProcessBatchEnv
from ..model import RuleConfig
from ..riichi_model import RiichiConfig
from .features import STATE_DIM, HINT_DIM, encode_batch
from .schema import feature_module
from .placement import DEFAULT_RANK_REWARDS, rank_rewards as validate_rank_rewards, ranking


def observation_space(rules=None):
    schema = feature_module(rules or RuleConfig())
    return spaces.Dict({
        'state': spaces.Box(-np.inf, np.inf, (schema.STATE_DIM,), np.float32),
        'candidates': spaces.Box(-1, 135, (MAX_ACTIONS, 8), np.int16),
        'hints': spaces.Box(-np.inf, np.inf, (MAX_ACTIONS, schema.HINT_DIM), np.float32),
        'mask': spaces.Box(0, 1, (MAX_ACTIONS,), np.bool_),
    })


class MahjongVecEnv(VecEnv):
    def __init__(self, n=32, workers=0, *, seed=0, opponents=('greedy',),
                 backend='auto', reward_scale=32., rules=None, hints=True, rotate_seats=True,
                 opponent_manifest=None, self_play_probability=0., variant='sichuan', match_length='hand',
                 skip_forced_actions=None, reward_mode='score', rank_rewards=DEFAULT_RANK_REWARDS):
        if type(n) is not int or n < 1 or type(workers) is not int or not 0 <= workers <= n:
            raise ValueError('positive environment count and workers in 0..N required')
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('reward_scale must be finite and positive')
        if reward_mode not in ('score', 'rank') or (reward_mode == 'rank' and (variant != 'riichi' or match_length == 'hand' or reward_scale != 1)):
            raise ValueError('rank reward requires a riichi match and reward_scale=1')
        self.reward_mode, self.rank_rewards = reward_mode, validate_rank_rewards(rank_rewards)
        self.config = BatchConfig(variant=variant, seed=seed, backend=backend,
            rules=rules or (RiichiConfig() if variant == 'riichi' else RuleConfig()), rotate_seats=rotate_seats, opponent_pool=tuple(opponents),
            opponent_manifest=str(opponent_manifest) if opponent_manifest else None,
            self_play_probability=self_play_probability, match_length=match_length,
            skip_forced_actions=(variant == 'riichi') if skip_forced_actions is None else skip_forced_actions)
        if not opponents:
            raise ValueError('at least one opponent policy required')
        self.workers, self.reward_scale, self.hints = workers, reward_scale, hints
        self.core = self.raw = self.features = None
        self.pending = None
        self.episode_returns = np.zeros(n)
        self.episode_lengths = np.zeros(n, np.int64)
        self.initial_scores = np.zeros((n, 4), np.int64)
        self.prefix_rewards = np.zeros(n)
        self.completed_episodes = 0
        self.noninteractive_episodes_skipped = 0
        self.render_mode = None
        super().__init__(n, observation_space(self.config.rules), spaces.Discrete(MAX_ACTIONS))

    def _encode(self, batch):
        return encode_batch(batch, rules=self.config.rules, backend=self.config.backend, hints=self.hints)

    def _skip_closed_starts(self):
        # An opponent can end a hand before this seat gets its first choice.
        # Gym reset must still return a live decision, never an all-false mask.
        for _ in range(128):
            closed = self.raw['terminated'] | self.raw['truncated']
            if not closed.any():
                return
            self.noninteractive_episodes_skipped += int(closed.sum())
            self.raw = self.core.reset_done()
        raise RuntimeError('could not obtain an interactive episode start')

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
        self._skip_closed_starts()
        self.pending = None
        self.initial_scores = self.raw['scores'].copy()
        if self.config.match_length != 'hand':
            self.initial_scores[:] = self.config.rules.initial_scores
            players = self.raw['meta'][:, 1].astype(int)
            if self.reward_mode == 'score':
                self.prefix_rewards = (self.raw['scores'] - self.initial_scores)[np.arange(self.num_envs), players].astype(float)
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
        rewards = ((terminal['rewards'][np.arange(self.num_envs), players] + self.prefix_rewards) / self.reward_scale
                   if self.reward_mode == 'score' else np.zeros(self.num_envs))
        self.prefix_rewards.fill(0)
        done = terminal['terminated'].copy()
        if self.reward_mode == 'rank':
            for i in np.flatnonzero(done):
                rank = ranking(terminal['scores'][i], self.config.rules.dealer).index(int(players[i])) + 1
                rewards[i] = self.rank_rewards[rank - 1]
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        indices = np.flatnonzero(done)
        if len(indices):
            final_features = self._encode({k: v[indices] for k, v in terminal.items()})
            for local, i in enumerate(indices):
                wins = terminal['wins'][i]
                net = int(terminal['scores'][i, players[i]] - self.initial_scores[i, players[i]])
                if self.reward_mode == 'score' and not np.isclose(self.episode_returns[i] * self.reward_scale, net, atol=1e-4):
                    raise AssertionError('training reward lost part of the match settlement')
                metrics = {'net_score': net, 'seat': int(players[i])}
                if self.config.match_length != 'hand':
                    rank = ranking(terminal['scores'][i], self.config.rules.dealer).index(int(players[i])) + 1
                    utility = self.rank_rewards[rank - 1]
                    if self.reward_mode == 'rank' and not np.isclose(self.episode_returns[i], utility, atol=1e-7):
                        raise AssertionError('rank reward must equal final placement utility')
                    metrics.update(rank=rank, rank_utility=utility, first_place=int(rank == 1), last_place=int(rank == 4))
                else:
                    metrics.update(win=int(np.any(wins[:, 0] == players[i])), deal_in=int(np.any(wins[:, 1] == players[i])))
                infos[i] = {'terminal_observation': {k: v[local].copy() for k, v in final_features.items()},
                            'TimeLimit.truncated': False,
                            'episode': {'r': float(self.episode_returns[i]), 'l': int(self.episode_lengths[i])},
                            'metrics': metrics}
            self.completed_episodes += len(indices)
            self.raw = self.core.reset_done()
            self._skip_closed_starts()
            self.initial_scores[indices] = self.raw['scores'][indices]
            if self.config.match_length != 'hand':
                self.initial_scores[indices] = self.config.rules.initial_scores
                next_players = self.raw['meta'][indices, 1].astype(int)
                if self.reward_mode == 'score':
                    self.prefix_rewards[indices] = (self.raw['scores'][indices] - self.initial_scores[indices])[np.arange(len(indices)), next_players]
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


SichuanVecEnv = MahjongVecEnv


class RiichiVecEnv(MahjongVecEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variant', 'riichi')
        kwargs.setdefault('reward_scale', 1. if kwargs.get('reward_mode') == 'rank' else 10000.)
        super().__init__(*args, **kwargs)
