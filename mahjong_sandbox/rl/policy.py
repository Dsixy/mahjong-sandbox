"""SB3 MaskablePPO policy with shared candidate scoring, independent of menu order."""
import math
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from ..batch import ACTION_KINDS, MAX_ACTIONS
from .features import STATE_DIM, HINT_DIM


class MenuExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space['state'].shape[0] + MAX_ACTIONS * (8 + observation_space['hints'].shape[-1] + 1))

    def forward(self, observations):
        return torch.cat([observations[k].flatten(1) for k in ('state', 'candidates', 'hints', 'mask')], dim=1)


class CandidateHeads(nn.Module):
    def __init__(self, hidden=128, state_dim=STATE_DIM, hint_dim=HINT_DIM, tile_kinds=27):
        super().__init__()
        self.state_dim, self.hint_dim, self.tile_kinds = state_dim, hint_dim, tile_kinds
        self.latent_dim_pi, self.latent_dim_vf = MAX_ACTIONS, hidden
        self.context = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.query = nn.Linear(hidden, 64)
        self.action = nn.Sequential(nn.Linear(tile_kinds + len(ACTION_KINDS) + 4 + 5 + 1 + hint_dim, 64),
                                    nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.bias = nn.Linear(64, 1)

    def _actor(self, features, context):
        n = len(features)
        begin = self.state_dim
        candidates = features[:, begin:begin + MAX_ACTIONS * 8].reshape(n, MAX_ACTIONS, 8)
        begin += MAX_ACTIONS * 8
        hints = features[:, begin:begin + MAX_ACTIONS * self.hint_dim].reshape(n, MAX_ACTIONS, self.hint_dim)
        mask = features[:, -MAX_ACTIONS:].bool()
        # Rollout storage remains fixed shape; neural work skips padded columns.
        columns = torch.arange(MAX_ACTIONS, device=features.device)
        width = max(1, int(torch.where(mask, columns + 1, 0).max().item()))
        c = candidates[:, :width].long()
        tiles = c[:, :, 2:6]
        present = tiles >= 0
        kinds = (tiles.clamp_min(0) // 4).clamp_max(self.tile_kinds - 1)
        hist = (torch.nn.functional.one_hot(kinds, self.tile_kinds) * present.unsqueeze(-1)).sum(dim=2) / 4
        code = torch.nn.functional.one_hot(c[:, :, 0].clamp_min(0), len(ACTION_KINDS))
        suit = torch.nn.functional.one_hot((c[:, :, 6] + 1).clamp_min(0), 4)
        meld = torch.nn.functional.one_hot((c[:, :, 7] + 1).clamp_min(0), 5)
        action = self.action(torch.cat((hist, code, suit, meld, c[:, :, 1:2].clamp_min(0) / 4,
                                       hints[:, :width]), dim=-1).float())
        logits = (action * self.query(context)[:, None, :]).sum(dim=-1) / math.sqrt(64) + self.bias(action).squeeze(-1)
        return torch.nn.functional.pad(logits, (0, MAX_ACTIONS - width))

    def forward(self, features):
        context = self.context(features[:, :self.state_dim])
        return self._actor(features, context), context

    def forward_actor(self, features):
        return self._actor(features, self.context(features[:, :self.state_dim]))

    def forward_critic(self, features):
        return self.context(features[:, :self.state_dim])


class MahjongPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, hidden=128, variant='sichuan', **kwargs):
        self.hidden, self.variant = hidden, variant
        if variant not in ('sichuan', 'riichi'):
            raise ValueError('unknown policy variant')
        kwargs.setdefault('features_extractor_class', MenuExtractor)
        kwargs.setdefault('normalize_images', False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.mlp_extractor = CandidateHeads(self.hidden, self.observation_space['state'].shape[0],
            self.observation_space['hints'].shape[-1], 34 if self.variant == 'riichi' else 27)
        # The head already emits candidate logits; a learned 512x512 projection
        # here would reintroduce arbitrary menu-position dependence.
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.hidden, 1)
        if self.ortho_init:
            self.mlp_extractor.apply(lambda m: self.init_weights(m, gain=math.sqrt(2)))
            self.init_weights(self.mlp_extractor.bias, gain=.01)
            self.init_weights(self.value_net, gain=1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'hidden': self.hidden, 'variant': self.variant}
