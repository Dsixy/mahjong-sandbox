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


def feature_module(rules, feature_version=None):
    if isinstance(rules, RiichiConfig):
        if feature_version == 'riichi_visible_v2':
            from . import riichi_features_v2
            return riichi_features_v2
        if feature_version not in (None,'riichi_visible_v1'):
            raise ValueError('unknown riichi feature version')
        from . import riichi_features
        return riichi_features
    if feature_version not in (None,'sichuan_visible_v1'):
        raise ValueError('unknown Sichuan feature version')
    from . import features
    return features


def schema_from_space(rules, space):
    versions=('riichi_visible_v1','riichi_visible_v2') if isinstance(rules,RiichiConfig) else ('sichuan_visible_v1',)
    for version in versions:
        schema=feature_module(rules,version)
        if space['state'].shape==(schema.STATE_DIM,) and space['hints'].shape==(512,schema.HINT_DIM):
            expected={'state','hints','mask','candidates',*getattr(schema,'EXTRA_SHAPES',{})}
            if (set(space.spaces)==expected and space['candidates'].shape==(512,8) and space['mask'].shape==(512,) and
                    all(space[k].shape==v for k,v in getattr(schema,'EXTRA_SHAPES',{}).items())):
                return schema
    raise ValueError('model observation space does not match a supported feature contract')


def compatible_observation_rules(expected, announced):
    actual = rules_from_values(announced)
    if type(actual) is not type(expected):
        return False
    left, right = asdict(expected), asdict(actual)
    if isinstance(expected, RiichiConfig):
        left = {k: v for k, v in left.items() if k not in ROUND_FIELDS}
        right = {k: v for k, v in right.items() if k not in ROUND_FIELDS}
    return left == right
