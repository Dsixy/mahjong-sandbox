"""Atomic SB3 model/optimizer checkpoints with an explicit feature/rule contract."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import uuid
from ..batch import ACTION_KINDS
from ..model import RuleConfig
from .features import FEATURE_VERSION, STATE_DIM, HINT_DIM
from .schema import feature_module, rules_from_values, variant_of, schema_from_space

CHECKPOINT_VERSION = 'mahjong_maskable_ppo_v1'


def model_path(path):
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / ('best.zip' if (path / 'best.zip').exists() else 'latest.zip')
    if path.suffix != '.zip':
        path = path.with_suffix('.zip')
    return path


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for p in sorted(root.rglob('*.py')):
        digest.update(str(p.relative_to(root)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def save_checkpoint(model, path, *, rules=None, hints=True, training=None):
    import torch
    if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
        raise FloatingPointError('refusing to save nonfinite model parameters')
    rules = rules or RuleConfig()
    schema = schema_from_space(rules,model.observation_space)
    if (model.observation_space['state'].shape != (schema.STATE_DIM,) or
            model.observation_space['hints'].shape[-1] != schema.HINT_DIM or
            getattr(model.policy, 'variant', 'sichuan') != variant_of(rules)):
        raise ValueError('model architecture does not match checkpoint rules/features')
    path = model_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.zip')
    metadata_path = path.with_suffix('.json')
    metadata_tmp = metadata_path.with_name(metadata_path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        model.save(temporary)
        metadata = {'format': CHECKPOINT_VERSION, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'sb3-contrib MaskablePPO', 'feature_version': schema.FEATURE_VERSION,
            'state_dim': schema.STATE_DIM, 'hint_dim': schema.HINT_DIM, 'action_kinds': list(ACTION_KINDS),
            'rules': asdict(rules), 'hints': bool(hints),
            'observation_shapes': {k:list(v.shape) for k,v in model.observation_space.spaces.items()},
            'model_sha256': checksum(temporary), 'source_sha256': source_fingerprint(),
            'timesteps': int(model.num_timesteps), 'optimizer_updates': int(model._n_updates),
            'versions': {name: version(name) for name in ('torch', 'numpy', 'stable_baselines3', 'sb3-contrib')},
            'training': training or {}}
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return metadata


def read_metadata(path, *, expected_rules=None):
    path = model_path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    rules = rules_from_values(metadata['rules'])
    schema = feature_module(rules,metadata.get('feature_version'))
    expected = (CHECKPOINT_VERSION, schema.FEATURE_VERSION, schema.STATE_DIM, schema.HINT_DIM, list(ACTION_KINDS))
    actual = tuple(metadata.get(k) for k in ('format', 'feature_version', 'state_dim', 'hint_dim', 'action_kinds'))
    if actual != expected:
        raise ValueError('incompatible checkpoint/feature/action schema')
    shapes={'state':[schema.STATE_DIM],'candidates':[512,8],'hints':[512,schema.HINT_DIM],'mask':[512],
            **{k:list(v) for k,v in getattr(schema,'EXTRA_SHAPES',{}).items()}}
    if metadata.get('observation_shapes',shapes)!=shapes or (getattr(schema,'EXTRA_SHAPES',{}) and 'observation_shapes' not in metadata):
        raise ValueError('checkpoint structured observation shapes differ from feature contract')
    if expected_rules is not None and asdict(expected_rules) != asdict(rules):
        raise ValueError('checkpoint rules differ from evaluation/training rules')
    if metadata['model_sha256'] != checksum(path):
        raise ValueError('checkpoint checksum mismatch')
    if type(metadata.get('hints')) is not bool:
        raise ValueError('invalid checkpoint feature settings')
    return metadata


def load_checkpoint(path, *, env=None, device='cpu', expected_rules=None, **kwargs):
    from sb3_contrib import MaskablePPO
    metadata = read_metadata(path, expected_rules=expected_rules)
    model = MaskablePPO.load(model_path(path), env=env, device=resolve_device(device), **kwargs)
    if schema_from_space(rules_from_values(metadata['rules']),model.observation_space).FEATURE_VERSION!=metadata['feature_version']:
        raise ValueError('saved model observation space differs from checkpoint metadata')
    return model, metadata

def resolve_device(device):
    import torch
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    parsed = torch.device(device)
    if parsed.type not in ('cpu', 'cuda'):
        raise ValueError('this trainer supports cpu or cuda devices')
    if parsed.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu for CPU testing')
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise ValueError('CUDA device index is not visible')
    return str(parsed)

