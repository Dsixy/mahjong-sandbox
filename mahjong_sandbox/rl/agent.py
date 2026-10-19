"""Load once, then act from exactly the same public features used in training."""
from dataclasses import asdict
from ..model import Action, RuleConfig
from .features import encode_batch, observation_batch
from .checkpoint import load_checkpoint


class RLAgent:
    def __init__(self, path=None, *, model=None, metadata=None, backend='auto',
                 device='cpu', deterministic=True):
        if model is None:
            model, metadata = load_checkpoint(path, device=device)
        if metadata is None:
            raise ValueError('model metadata is required')
        self.model, self.metadata = model, metadata
        self.rules = RuleConfig(**metadata['rules'])
        self.backend, self.deterministic = backend, deterministic

    def act(self, observation):
        announced = next((e['data']['rules'] for e in observation['events'] if e['kind'] == 'round_started'), None)
        if announced != asdict(self.rules):
            raise ValueError('observed rules differ from checkpoint rules')
        batch = observation_batch(observation)
        features = encode_batch(batch, rules=self.rules, backend=self.backend, hints=self.metadata['hints'])
        action, _ = self.model.predict(features, action_masks=features['mask'], deterministic=self.deterministic)
        return Action.from_dict(observation['legal_actions'][int(action[0])])
