"""Historical self play: frozen opponents selected once per seat and whole hand.

Workers infer on CPU; only the learner owns CUDA. An atomically published manifest
changes future hands, never opponents in flight. Immutable snapshots also provide
an audit trail and survive preemption along with ordinary training checkpoints.
"""
from collections import OrderedDict
import json
import os
from pathlib import Path
import uuid
from .agent import RLAgent
from .checkpoint import read_metadata, save_checkpoint


def publish_snapshot(model, directory, *, rules, hints, training, pool_size=4):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'opponents.json'
    # Unique immutable names also cover resumes from an older checkpoint.
    path = directory / f'step_{model.num_timesteps}_{uuid.uuid4().hex[:8]}.zip'
    metadata = save_checkpoint(model, path, rules=rules, hints=hints, training=training)
    entries = json.loads(manifest.read_text())['snapshots'] if manifest.exists() else []
    entries.append({'path': str(path), 'timesteps': metadata['timesteps'], 'sha256': metadata['model_sha256']})
    payload = {'version': 1, 'snapshots': entries[-pool_size:]}
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, manifest)
    return payload


class SnapshotPool:
    def __init__(self, manifest, *, rules, backend, probability):
        import torch
        torch.set_num_threads(1)
        self.manifest, self.rules, self.backend = Path(manifest), rules, backend
        self.probability = probability
        self.entries, self.cache = [], OrderedDict()

    def refresh(self):
        data = json.loads(self.manifest.read_text())
        if data.get('version') != 1 or not data.get('snapshots'):
            raise ValueError('invalid or empty self-play manifest')
        self.entries = data['snapshots']

    def opponent(self, seed, fallback):
        if (seed & 0xffffffff) / 2**32 >= self.probability:
            return fallback
        entry = self.entries[(seed >> 32) % len(self.entries)]
        path = entry['path']
        if path not in self.cache:
            metadata = read_metadata(path, expected_rules=self.rules)
            if metadata['model_sha256'] != entry['sha256']:
                raise ValueError('self-play snapshot differs from manifest')
            self.cache[path] = RLAgent(path, backend=self.backend, device='cpu', deterministic=True)
        agent = self.cache[path]
        self.cache.move_to_end(path)
        # Existing hands retain references to their frozen agents after eviction.
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return agent
