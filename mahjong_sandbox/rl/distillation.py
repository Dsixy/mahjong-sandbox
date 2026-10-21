"""Sharded, match-separated greedy distillation and learner-state aggregation.

The teacher remains observation-only. Its equally preferred moves form a target
set, so a random tie-break is not treated as the only correct action.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np

from ..batch import allocate, encode_observation
from ..bots import make_bot
from ..fast import FastRiichiEngine, RuntimeOptions
from ..model import Action, Phase
from ..riichi_match import MatchConfig, RiichiMatch
from ..riichi_model import RiichiConfig
from .features import encode_batch
from .schema import feature_module

GROUPS = ('discard', 'call', 'riichi', 'win', 'kong')
FIDELITY_THRESHOLDS = {'all':.94,'discard':.94,'call':.90,'riichi':.98,'win':.995,'threat':.90}


def fidelity_key(audit):
    """Protect critical decisions, then choose population teacher fidelity.

    Macro loss can be dominated by one overconfident error among a handful of
    kan examples, selecting a substantially worse common-discard policy.
    """
    shortfalls=[max(0.,threshold-audit.get(name,{}).get('teacher_set_accuracy',0.))
                for name,threshold in FIDELITY_THRESHOLDS.items()]
    return (sum(value>0 for value in shortfalls),sum(shortfalls),
            -audit['all']['teacher_set_accuracy'],audit['all']['set_cross_entropy'])


class ChoiceRecorder(random.Random):
    """Record the teacher's final random tie, without changing its RNG stream."""
    choices = None

    def choice(self, sequence):
        self.choices = list(sequence)
        return super().choice(sequence)


def teacher_targets(teacher, observation, raw):
    teacher.rng.choices = None
    chosen = teacher.act(observation)
    choices = teacher.rng.choices or [chosen.to_dict()]
    schema = feature_module(RiichiConfig(), 'riichi_visible_v2')
    mask = np.zeros(512, bool)
    for action in choices:
        mask[schema.canonical_index(raw, Action.from_dict(action))] = True
    label = schema.canonical_index(raw, chosen)
    if not mask[label]:
        raise AssertionError('teacher target omitted its actual move')
    return chosen, label, mask


def decision_group(observation):
    kinds = {a['kind'] for a in observation['legal_actions']}
    if kinds.intersection(('ron', 'tsumo')): return GROUPS.index('win')
    if 'riichi' in kinds: return GROUPS.index('riichi')
    if kinds.intersection(('concealed_kong', 'added_kong')): return GROUPS.index('kong')
    if kinds.intersection(('chi', 'peng', 'open_kong')): return GROUPS.index('call')
    return GROUPS.index('discard')


def _collect_shard(job):
    directory, index, count, seed, rules_values, match_values, backend, learner_path, mix = job
    rules = RiichiConfig(**rules_values)
    config = MatchConfig(**match_values)
    learner = None
    if learner_path:
        import torch
        from .agent import RLAgent
        torch.set_num_threads(1)
        learner = RLAgent(learner_path, backend=backend, device='cpu', deterministic=True)
    factory = partial(FastRiichiEngine, runtime=RuntimeOptions(operator_backend=backend))
    rng = random.Random(seed)
    blocks, labels, targets, groups, threats, contexts, matches = [], [], [], [], [], [], []
    pending, hands, commands, match_count = [], 0, 0, 0
    schema = feature_module(rules, 'riichi_visible_v2')
    started = time.perf_counter()
    while len(labels) < count:
        if match_count >= 256:
            raise RuntimeError('demonstration shard exhausted its reserved match seeds')
        logical_seed = seed + match_count
        match = RiichiMatch(config, rules, logical_seed * 256, record=False, engine_factory=factory)
        teachers = [make_bot('greedy', variant='riichi', seed=logical_seed * 4 + p, backend=backend) for p in range(4)]
        for p, teacher in enumerate(teachers): teacher.rng = ChoiceRecorder(logical_seed * 4 + p)
        learner_seat = match_count % 4
        while not match.ended and len(labels) < count:
            engine = match.engine
            while engine.state.phase != Phase.TERMINAL and len(labels) < count:
                p = engine.next_player(); observation = engine.observe(p)
                if engine.command_count >= 2000: raise RuntimeError('nonterminating demonstration hand')
                if len(observation['legal_actions']) == 1:
                    engine.submit(p, Action.from_dict(observation['legal_actions'][0]), engine.state.decision_id)
                    commands += 1
                    continue
                raw = allocate(1, public_history=True)
                encode_observation(observation, engine.legal_actions(p), raw)
                chosen, label, acceptable = teacher_targets(teachers[p], observation, raw)
                meaningful = schema.canonical_mask(raw).sum() > 1
                # DAgger queries every decision of the learner, including places
                # it reaches after its own earlier mistakes, rather than only
                # trajectories generated by the expert.
                if meaningful and (learner is None or p == learner_seat):
                    pending.append(raw); labels.append(label); targets.append(acceptable)
                    groups.append(decision_group(observation))
                    threats.append(any(x['player'] != p and x['riichi'] for x in observation['players']))
                    contexts.append(int(observation['round_wind'] >= 28 and observation['hand_number'] >= 3))
                    matches.append(logical_seed)
                    if len(pending) >= 64 or len(labels) == count:
                        combined = {k: np.concatenate([x[k] for x in pending]) for k in raw}
                        values = encode_batch(combined, rules=rules, backend=backend, feature_version='riichi_visible_v2')
                        values['v1_state'] = encode_batch(combined, rules=rules, backend=backend, feature_version='riichi_visible_v1')['state']
                        blocks.append(values); pending = []
                action = learner.act(observation) if learner is not None and p == learner_seat and rng.random() >= mix else chosen
                engine.submit(p, action, engine.state.decision_id); commands += 1
            if engine.state.phase == Phase.TERMINAL:
                hands += 1; engine.assert_invariants(force=True); match.complete_hand()
                if not match.ended: match.next_hand()
        match_count += 1
    features = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    payload = {**features, 'labels': np.asarray(labels, np.int16), 'acceptable': np.asarray(targets, bool),
        'groups': np.asarray(groups, np.int8), 'threat': np.asarray(threats, bool),
        'late': np.asarray(contexts, bool), 'match_ids': np.asarray(matches, np.int64)}
    path = Path(directory) / f'shard_{index:05d}.npz'; temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **payload)
    temporary.replace(path)
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'samples': len(labels), 'match_ids': sorted(set(matches)), 'hands': hands, 'commands': commands,
        'groups': {name: int((payload['groups'] == i).sum()) for i, name in enumerate(GROUPS)},
        'threat': int(payload['threat'].sum()), 'late': int(payload['late'].sum()),
        'elapsed_seconds': time.perf_counter() - started}


def collect_dataset(directory, samples, *, seed, workers=4, shard_size=1024, rules=None,
                    match_config=None, backend='numba', learner=None, teacher_mix=0.):
    from .checkpoint import source_fingerprint, read_metadata
    if samples < 1 or workers < 1 or shard_size < 1 or not 0 <= teacher_mix <= 1:
        raise ValueError('positive collection settings and teacher_mix in [0,1] required')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    rules, match_config = rules or RiichiConfig(), match_config or MatchConfig(length='south')
    learner_path = str(Path(learner).resolve()) if learner else None
    plan = {'format': 'riichi_distillation_dataset_v1', 'samples': samples, 'seed': seed,
        'shard_size': shard_size, 'reserved_matches_per_shard': 256, 'rules': asdict(rules),
        'match_config': asdict(match_config), 'source_sha256': source_fingerprint(),
        'learner_sha256': read_metadata(learner)['model_sha256'] if learner else None,
        'teacher_mix': teacher_mix, 'backend': backend}
    plan=json.loads(json.dumps(plan))
    receipt = directory / 'manifest.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['plan'] != plan: raise ValueError('dataset settings changed')
        for item in result['shards']:
            if hashlib.sha256((directory / item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('dataset shard checksum mismatch')
        return result
    jobs = [(str(directory), i, min(shard_size, samples-i*shard_size), seed+i*256,
             asdict(rules), asdict(match_config), backend, learner_path, teacher_mix)
            for i in range((samples+shard_size-1)//shard_size)]
    started = time.perf_counter(); records = []
    if workers == 1:
        iterator = map(_collect_shard, jobs)
        for item in iterator:
            records.append(item); print(json.dumps({'event':'demonstration_shard', **item}), flush=True)
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as pool:
            for item in pool.map(_collect_shard, jobs):
                records.append(item); print(json.dumps({'event':'demonstration_shard', **item}), flush=True)
    result = {'plan': plan, 'shards': records, 'elapsed_seconds': time.perf_counter()-started}
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    return result


def read_shard(path, feature_version):
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    fields = ('state', 'candidates', 'hints', 'mask')
    if feature_version == 'riichi_visible_v2': fields += ('rivers', 'melds', 'public_history', 'history_mask')
    elif feature_version != 'riichi_visible_v1': raise ValueError('unknown riichi distillation version')
    features = {k: values[k] for k in fields}
    if feature_version == 'riichi_visible_v1': features['state'] = values['v1_state']
    return features, {k: values[k] for k in ('labels','acceptable','groups','threat','late','match_ids')}


def dataset_paths(directories):
    paths, match_ids = [], set()
    for directory in directories:
        root = Path(directory); manifest = json.loads((root/'manifest.json').read_text())
        for record in manifest['shards']:
            paths.append(root / record['path']); match_ids.update(record['match_ids'])
    return paths, match_ids


def set_loss(distribution, acceptable, weights=None):
    """Negative log mass on the teacher's tie set; legal softmax is upstream."""
    import torch
    valid = torch.as_tensor(acceptable, device=distribution.probs.device, dtype=torch.bool)
    if not bool(valid.any(-1).all()): raise ValueError('empty teacher target set')
    losses = -torch.logsumexp(distribution.logits.masked_fill(~valid, -torch.inf), dim=-1)
    if weights is None: return losses.mean()
    weights = torch.as_tensor(weights, device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum()


def audit_policy(model, directories, *, batch_size=256):
    import torch
    from .schema import schema_from_space
    version = schema_from_space(RiichiConfig(), model.observation_space).FEATURE_VERSION
    paths, _ = dataset_paths(directories); totals = Counter(); sums = Counter()
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for path in paths:
            features, metadata = read_shard(path, version)
            for start in range(0, len(metadata['labels']), batch_size):
                ids = np.arange(start, min(start+batch_size, len(metadata['labels'])))
                observation, _ = model.policy.obs_to_tensor({k:v[ids] for k,v in features.items()})
                dist = model.policy.get_distribution(observation, action_masks=features['mask'][ids]).distribution
                choices = dist.probs.argmax(-1).cpu().numpy()
                acceptable = metadata['acceptable'][ids]
                agreement = acceptable[np.arange(len(ids)), choices]
                exact = choices == metadata['labels'][ids]
                mass = (dist.probs * torch.as_tensor(acceptable, device=model.device)).sum(-1).cpu().numpy()
                loss = -np.log(np.maximum(mass, 1e-30))
                strata = {'all':np.ones(len(ids),bool), 'threat':metadata['threat'][ids], 'late':metadata['late'][ids]}
                strata.update({name:metadata['groups'][ids] == i for i,name in enumerate(GROUPS)})
                for name, mask in strata.items():
                    totals[name] += int(mask.sum()); sums[name+'_agreement'] += int(agreement[mask].sum())
                    sums[name+'_exact'] += int(exact[mask].sum()); sums[name+'_mass'] += float(mass[mask].sum())
                    sums[name+'_loss'] += float(loss[mask].sum())
    return {name:{'samples':n, 'teacher_set_accuracy':sums[name+'_agreement']/n,
        'sampled_action_accuracy':sums[name+'_exact']/n, 'teacher_set_probability':sums[name+'_mass']/n,
        'set_cross_entropy':sums[name+'_loss']/n} for name,n in totals.items() if n}


def fit_policy(model, train_directories, validation_directory, output, *, epochs=12,
               batch_size=256, learning_rate=3e-4, seed=42, training=None):
    import torch
    from .checkpoint import save_checkpoint
    from .schema import schema_from_space
    train_paths, train_matches = dataset_paths(train_directories)
    _, validation_matches = dataset_paths([validation_directory])
    if train_matches.intersection(validation_matches):
        raise ValueError('training and validation share a whole match')
    if epochs < 1 or batch_size < 1 or learning_rate <= 0: raise ValueError('invalid distillation settings')
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    version = schema_from_space(RiichiConfig(), model.observation_space).FEATURE_VERSION
    counts = Counter()
    for directory in train_directories:
        for shard in json.loads((Path(directory)/'manifest.json').read_text())['shards']:
            counts.update(shard['groups'])
    total = sum(counts.values())
    # Rare wins, riichi and calls cannot be drowned out by routine discards.
    group_weights = np.array([min(8., np.sqrt(total/max(1,len(GROUPS)*counts[g]))) for g in GROUPS])
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate, eps=1e-5)
    rng = np.random.default_rng(seed); history = []; best = None; started = time.perf_counter()
    for epoch in range(epochs):
        total_loss, samples = 0., 0
        for number in rng.permutation(len(train_paths)):
            features, metadata = read_shard(train_paths[number], version)
            order = rng.permutation(len(metadata['labels']))
            model.policy.set_training_mode(True)
            for start in range(0,len(order),batch_size):
                ids = order[start:start+batch_size]
                obs,_ = model.policy.obs_to_tensor({k:v[ids] for k,v in features.items()})
                dist = model.policy.get_distribution(obs,action_masks=features['mask'][ids]).distribution
                loss = set_loss(dist,metadata['acceptable'][ids],group_weights[metadata['groups'][ids]])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite distillation loss')
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.);optimizer.step()
                total_loss += float(loss.detach())*len(ids);samples += len(ids)
        audit = audit_policy(model,[validation_directory],batch_size=batch_size)
        row = {'event':'distillation_epoch','epoch':epoch+1,'samples':samples,
            'training_weighted_set_cross_entropy':total_loss/samples,'validation':audit,
            'elapsed_seconds':time.perf_counter()-started}
        history.append(row)
        with (output/'imitation-metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        # Selection uses held-out teacher loss, never match test results.
        score = fidelity_key(audit)
        if best is None or score < best:
            best = score
            save_checkpoint(model,output/'warmstart.zip',rules=RiichiConfig(),training={**(training or {}),
                'imitation':{'dataset_directories':[str(Path(p).resolve()) for p in train_directories],
                    'validation_directory':str(Path(validation_directory).resolve()),
                    'selection':'fewer fidelity-gate failures, smaller shortfall, higher overall teacher-set accuracy, lower overall cross entropy',
                    'selected_epoch':epoch+1,'validation':audit,'seed':seed}})
    result={'feature_version':version,'epochs':epochs,'training_samples_per_epoch':total,
        'group_weights':dict(zip(GROUPS,group_weights.tolist())),'history':history,
        'best_validation_selection_key':list(best),'elapsed_seconds':time.perf_counter()-started}
    (output/'imitation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


class TeacherAnchor:
    """Small supervised updates after PPO, using its own saved Adam optimizer."""
    def __init__(self, directories, *, feature_version, coefficient, steps, batch_size=256, seed=42):
        self.paths,_ = dataset_paths(directories)
        if not self.paths or coefficient <= 0 or steps < 1: raise ValueError('invalid teacher anchor')
        self.version,self.coefficient,self.steps,self.batch_size = feature_version,coefficient,steps,batch_size
        self.rng = np.random.default_rng(seed);self.cached=None;self.number=-1

    def update(self,model):
        import torch
        losses=[];model.policy.set_training_mode(True)
        for _ in range(self.steps):
            number=int(self.rng.integers(len(self.paths)))
            if number!=self.number:
                self.cached=read_shard(self.paths[number],self.version);self.number=number
            features,metadata=self.cached
            ids=self.rng.integers(len(metadata['labels']),size=self.batch_size)
            obs,_=model.policy.obs_to_tensor({k:v[ids] for k,v in features.items()})
            dist=model.policy.get_distribution(obs,action_masks=features['mask'][ids]).distribution
            loss=set_loss(dist,metadata['acceptable'][ids])
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite teacher anchor loss')
            model.policy.optimizer.zero_grad();(loss*self.coefficient).backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),model.max_grad_norm)
            model.policy.optimizer.step();losses.append(float(loss.detach()))
        model.policy.set_training_mode(False)
        return {'anchor/set_cross_entropy':float(np.mean(losses)), 'anchor/coefficient':self.coefficient,
                'anchor/optimizer_steps':self.steps}
