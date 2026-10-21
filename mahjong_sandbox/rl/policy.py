"""SB3 MaskablePPO policy with shared candidate scoring, independent of menu order."""
import math
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from ..batch import ACTION_KINDS, MAX_ACTIONS
from .features import STATE_DIM, HINT_DIM
from ..public_history import EVENT_KINDS, EVENT_DETAILS, MAX_PUBLIC_EVENTS


class PublicInformationEncoder(nn.Module):
    """Encode all ordered public tokens without quadratic attention storage."""
    OUTPUT_DIM = 128

    def __init__(self, state_dim):
        super().__init__()
        self.tile=nn.Embedding(35,4)
        self.rivers=nn.Sequential(nn.Linear(4*136*18,32),nn.Tanh())
        self.melds=nn.Sequential(nn.Linear(4*4*41,32),nn.Tanh())
        self.event=nn.Embedding(len(EVENT_KINDS),8)
        self.actor=nn.Embedding(6,4)
        self.history_tile=nn.Embedding(35,8)
        self.detail=nn.Embedding(len(EVENT_DETAILS),4)
        self.meld_index=nn.Embedding(5,2)
        self.position=nn.Embedding(MAX_PUBLIC_EVENTS,8)
        self.history=nn.Sequential(nn.Linear(45,32),nn.Tanh())
        self.temporal=nn.Conv1d(32,32,3,padding=1)
        self.query=nn.Linear(state_dim,32)
        self.pool=nn.Sequential(nn.Linear(96,64),nn.Tanh())

    @staticmethod
    def bits(value, width):
        return ((value[...,None] >> torch.arange(width,device=value.device))&1).float()

    def forward(self, obs):
        river=obs['rivers'].long();present=(river[...,0]!=34)[...,None]
        encoded=torch.cat((self.tile(river[...,0]),river[...,1:2].float(),
            self.bits(river[...,2],4),torch.nn.functional.one_hot(river[...,3],5).float(),
            self.bits(river[...,4],4)),dim=-1)*present
        rivers=self.rivers(encoded.flatten(1))
        meld=obs['melds'].long();present=(meld[...,0]!=5)[...,None]
        encoded=torch.cat((torch.nn.functional.one_hot(meld[...,0],6).float(),
            torch.nn.functional.one_hot(meld[...,1],5).float(),self.tile(meld[...,2:6]).flatten(-2),
            self.bits(meld[...,6],4),self.tile(meld[...,7]),meld[...,8:9].float(),
            self.tile(meld[...,9]),meld[...,10:11].float()),dim=-1)*present
        melds=self.melds(encoded.flatten(1))
        mask=obs['history_mask'].bool();width=max(1,int(mask.sum(1).max().item()))
        mask=mask[:,:width];history=obs['public_history'][:,:width].long()
        numeric=torch.stack((history[...,5]/136,history[...,6],history[...,7]/32000,
            history[...,9]/70,history[...,10]/136,history[...,11],
            torch.arange(width,device=history.device).expand(len(history),-1)/MAX_PUBLIC_EVENTS),dim=-1)
        encoded=torch.cat((self.event(history[...,0]),self.actor(history[...,1]),self.actor(history[...,2]),
            self.history_tile(history[...,3]),self.detail(history[...,8]),
            self.meld_index((history[...,4]+1).clamp_min(0)),numeric,
            self.position(torch.arange(width,device=history.device)).expand(len(history),-1,-1)),dim=-1)
        encoded=self.history(encoded)*mask[...,None]
        encoded=torch.tanh(self.temporal(encoded.transpose(1,2)).transpose(1,2))*mask[...,None]
        logits=(encoded*self.query(obs['state'])[:,None,:]).sum(-1)/math.sqrt(32)
        weights=logits.masked_fill(~mask,-1e9).softmax(-1)*mask
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-9)
        attended=(encoded*weights[...,None]).sum(1)
        mean=encoded.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        last=encoded[torch.arange(len(history),device=history.device),(mask.sum(1)-1).clamp_min(0)]
        return torch.cat((rivers,melds,self.pool(torch.cat((attended,mean,last),dim=-1))),dim=-1)


class MenuExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        state_dim=observation_space['state'].shape[0]
        extra=PublicInformationEncoder.OUTPUT_DIM if 'public_history' in observation_space.spaces else 0
        super().__init__(observation_space, state_dim + extra + MAX_ACTIONS * (8 + observation_space['hints'].shape[-1] + 1))
        self.context_dim=state_dim+extra
        self.public_encoder=PublicInformationEncoder(state_dim) if extra else None

    def forward(self, observations):
        parts=[observations['state']]
        if self.public_encoder is not None:
            parts.append(self.public_encoder(observations))
        parts.extend(observations[k].flatten(1) for k in ('candidates','hints','mask'))
        return torch.cat(parts,dim=1)


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
        self.mlp_extractor = CandidateHeads(self.hidden, self.features_extractor.context_dim,
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
