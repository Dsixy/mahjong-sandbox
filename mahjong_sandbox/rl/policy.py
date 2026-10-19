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
        super().__init__(observation_space, STATE_DIM + MAX_ACTIONS * (8 + HINT_DIM + 1))

    def forward(self, observations):
        return torch.cat([observations[k].flatten(1) for k in ('state', 'candidates', 'hints', 'mask')], dim=1)


class CandidateHeads(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.latent_dim_pi, self.latent_dim_vf = MAX_ACTIONS, hidden
        self.context = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.query = nn.Linear(hidden, 64)
        self.action = nn.Sequential(nn.Linear(27 + len(ACTION_KINDS) + 4 + 5 + 1 + HINT_DIM, 64),
                                    nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.bias = nn.Linear(64, 1)

    def _actor(self, features, context):
        n = len(features)
        begin = STATE_DIM
        candidates = features[:, begin:begin + MAX_ACTIONS * 8].reshape(n, MAX_ACTIONS, 8)
        begin += MAX_ACTIONS * 8
        hints = features[:, begin:begin + MAX_ACTIONS * HINT_DIM].reshape(n, MAX_ACTIONS, HINT_DIM)
        mask = features[:, -MAX_ACTIONS:].bool()
        # Rollout storage remains fixed shape; neural work skips padded columns.
        columns = torch.arange(MAX_ACTIONS, device=features.device)
        width = max(1, int(torch.where(mask, columns + 1, 0).max().item()))
        c = candidates[:, :width].long()
        tiles = c[:, :, 2:6]
        present = tiles >= 0
        kinds = (tiles.clamp_min(0) // 4).clamp_max(26)
        hist = (torch.nn.functional.one_hot(kinds, 27) * present.unsqueeze(-1)).sum(dim=2) / 4
        code = torch.nn.functional.one_hot(c[:, :, 0].clamp_min(0), len(ACTION_KINDS))
        suit = torch.nn.functional.one_hot((c[:, :, 6] + 1).clamp_min(0), 4)
        meld = torch.nn.functional.one_hot((c[:, :, 7] + 1).clamp_min(0), 5)
        action = self.action(torch.cat((hist, code, suit, meld, c[:, :, 1:2].clamp_min(0) / 4,
                                       hints[:, :width]), dim=-1).float())
        logits = (action * self.query(context)[:, None, :]).sum(dim=-1) / math.sqrt(64) + self.bias(action).squeeze(-1)
        return torch.nn.functional.pad(logits, (0, MAX_ACTIONS - width))

    def forward(self, features):
        context = self.context(features[:, :STATE_DIM])
        return self._actor(features, context), context

    def forward_actor(self, features):
        return self._actor(features, self.context(features[:, :STATE_DIM]))

    def forward_critic(self, features):
        return self.context(features[:, :STATE_DIM])


class MahjongPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, hidden=128, **kwargs):
        self.hidden = hidden
        kwargs.setdefault('features_extractor_class', MenuExtractor)
        kwargs.setdefault('normalize_images', False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.mlp_extractor = CandidateHeads(self.hidden)
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
        return {**super()._get_constructor_parameters(), 'hidden': self.hidden}
