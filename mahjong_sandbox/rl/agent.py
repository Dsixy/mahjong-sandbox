"""Load once, then act from exactly the same public features used in training."""
from dataclasses import asdict
from ..model import Action, RuleConfig
from .features import encode_batch, observation_batch
from .checkpoint import load_checkpoint
from .schema import rules_from_values, compatible_observation_rules, schema_from_space


class RLAgent:
    def __init__(self, path=None, *, model=None, metadata=None, backend='auto',
                 device='cpu', deterministic=True):
        if model is None:
            model, metadata = load_checkpoint(path, device=device)
        if metadata is None:
            raise ValueError('model metadata is required')
        self.model, self.metadata = model, metadata
        self.rules = rules_from_values(metadata['rules'])
        self.feature_version=schema_from_space(self.rules,model.observation_space).FEATURE_VERSION
        if metadata.get('feature_version',self.feature_version)!=self.feature_version:
            raise ValueError('agent metadata and model feature versions disagree')
        self.backend, self.deterministic = backend, deterministic

    def act(self, observation):
        if self.feature_version=='riichi_visible_v2':
            from ..public_history import HISTORY_VERSION
            if observation.get('public_history_version')!=HISTORY_VERSION or 'match_context' not in observation:
                raise ValueError('v2 agent requires complete public history and explicit match context')
        announced = next((e['data']['rules'] for e in observation['events'] if e['kind'] == 'round_started'), None)
        if announced is None or not compatible_observation_rules(self.rules, announced):
            raise ValueError('observed rules differ from checkpoint rules')
        if self.deterministic and len(observation['legal_actions']) == 1:
            return Action.from_dict(observation['legal_actions'][0])
        batch = observation_batch(observation)
        features = encode_batch(batch, rules=self.rules, backend=self.backend, hints=self.metadata['hints'],
                                feature_version=self.feature_version)
        action, _ = self.model.predict(features, action_masks=features['mask'], deterministic=self.deterministic)
        return Action.from_dict(observation['legal_actions'][int(action[0])])
