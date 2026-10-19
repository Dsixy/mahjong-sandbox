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
    path = model_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.zip')
    metadata_path = path.with_suffix('.json')
    metadata_tmp = metadata_path.with_name(metadata_path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        model.save(temporary)
        metadata = {'format': CHECKPOINT_VERSION, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'algorithm': 'sb3-contrib MaskablePPO', 'feature_version': FEATURE_VERSION,
            'state_dim': STATE_DIM, 'hint_dim': HINT_DIM, 'action_kinds': list(ACTION_KINDS),
            'rules': asdict(rules or RuleConfig()), 'hints': bool(hints),
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
    expected = (CHECKPOINT_VERSION, FEATURE_VERSION, STATE_DIM, HINT_DIM, list(ACTION_KINDS))
    actual = tuple(metadata.get(k) for k in ('format', 'feature_version', 'state_dim', 'hint_dim', 'action_kinds'))
    if actual != expected:
        raise ValueError('incompatible checkpoint/feature/action schema')
    rules = RuleConfig(**metadata['rules'])
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

