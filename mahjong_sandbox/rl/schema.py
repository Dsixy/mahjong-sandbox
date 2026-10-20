"""Rule and feature contracts, including legacy Sichuan checkpoints."""
from dataclasses import asdict
from ..model import RuleConfig
from ..riichi_model import RiichiConfig

ROUND_FIELDS = {'dealer', 'round_wind', 'hand_number', 'honba', 'riichi_sticks', 'initial_scores'}


def rules_from_values(values):
    cls = RiichiConfig if values.get('rule_version') == 'riichi_4p_v1' else RuleConfig
    return cls(**values)


def variant_of(rules):
    return 'riichi' if isinstance(rules, RiichiConfig) else 'sichuan'


def feature_module(rules):
    if isinstance(rules, RiichiConfig):
        from . import riichi_features
        return riichi_features
    from . import features
    return features


def compatible_observation_rules(expected, announced):
    actual = rules_from_values(announced)
    if type(actual) is not type(expected):
        return False
    left, right = asdict(expected), asdict(actual)
    if isinstance(expected, RiichiConfig):
        left = {k: v for k, v in left.items() if k not in ROUND_FIELDS}
        right = {k: v for k, v in right.items() if k not in ROUND_FIELDS}
    return left == right
