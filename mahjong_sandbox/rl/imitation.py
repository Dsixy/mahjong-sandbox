"""Optional, explicitly reported behavior-cloning initialization from a public bot."""
from collections import Counter
import random
from functools import partial
import numpy as np
import torch
from ..batch import allocate, encode_observation, derived_seed
from ..bots import make_bot
from ..fast import FastEngine, FastRiichiEngine, RuntimeOptions
from ..model import Phase
from .features import canonical_mask, canonical_index, encode_batch
from .schema import variant_of, feature_module, schema_from_space


def collect_demonstrations(samples, *, seed=1234, rules=None, backend='auto', hints=True,
                           feature_version=None,match_length='hand',match_config=None):
    if samples < 1:
        raise ValueError('positive demonstration sample count required')
    blocks, labels, phases = [], [], Counter()
    pending, pending_labels = [], []
    hands, commands = 0, 0
    variant = variant_of(rules)
    schema = feature_module(rules,feature_version)
    match=None;matches=0
    while len(labels) + len(pending_labels) < samples:
        if match_length!='hand' and (match is None or match.ended):
            from ..riichi_match import RiichiMatch,MatchConfig
            if variant!='riichi':raise ValueError('match demonstrations require riichi')
            match=RiichiMatch(match_config or MatchConfig(length=match_length),rules,
                derived_seed(seed,matches,0),record=False,
                engine_factory=partial(FastRiichiEngine,runtime=RuntimeOptions(operator_backend=backend)))
            matches+=1
        engine = match.engine if match else (FastRiichiEngine if variant == 'riichi' else FastEngine)(rules, seed=derived_seed(seed, hands, 0),
                            runtime=RuntimeOptions(operator_backend=backend))
        bots = [make_bot('greedy', variant=variant, seed=derived_seed(seed, hands, 0, p + 1),
                         backend=backend) for p in range(4)]
        while engine.state.phase != Phase.TERMINAL:
            p = engine.next_player()
            obs = engine.observe(p)
            action = bots[p].act(obs)
            raw = allocate(1,public_history=feature_version=='riichi_visible_v2')
            encode_observation(obs, engine.legal_actions(p), raw)
            # Forced passes convey no policy-learning signal; keep branch decisions.
            if schema.canonical_mask(raw).sum() > 1:
                pending.append(raw)
                pending_labels.append(schema.canonical_index(raw, action))
                phases[obs['phase']] += 1
                if len(pending) >= 128 or len(labels) + len(pending) == samples:
                    combined = {k: np.concatenate([r[k] for r in pending]) for k in raw}
                    blocks.append(encode_batch(combined, rules=engine.config, backend=backend, hints=hints,feature_version=feature_version))
                    labels.extend(pending_labels)
                    pending, pending_labels = [], []
            engine.submit(p, action, engine.state.decision_id)
            commands += 1
            if len(labels) >= samples:
                break
            if engine.command_count >= 2000:
                raise RuntimeError('nonterminating demonstration hand')
        hands += 1
        if match and engine.state.phase==Phase.TERMINAL:
            match.complete_hand()
            if not match.ended:match.next_hand()
    features = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    return features, np.asarray(labels, np.int64), {'teacher': 'greedy', 'samples': samples,
        'hands_started': hands, 'commands': commands, 'phase_counts': dict(phases), 'seed': seed}


def pretrain(model, samples=8000, *, epochs=8, seed=1234, rules=None, backend='auto', hints=True, batch_size=256,
             feature_version=None,match_length='hand',match_config=None):
    feature_version=feature_version or schema_from_space(rules,model.observation_space).FEATURE_VERSION
    features, labels, report = collect_demonstrations(samples, seed=seed, rules=rules, backend=backend, hints=hints,
        feature_version=feature_version,match_length=match_length,match_config=match_config)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3, eps=1e-5)
    losses, accuracies = [], []
    model.policy.set_training_mode(True)
    for _ in range(epochs):
        order = rng.permutation(samples)
        total_loss, correct = 0., 0
        for begin in range(0, samples, batch_size):
            ids = order[begin:begin + batch_size]
            obs, _ = model.policy.obs_to_tensor({k: v[ids] for k, v in features.items()})
            targets = torch.as_tensor(labels[ids], device=model.device)
            dist = model.policy.get_distribution(obs, action_masks=features['mask'][ids])
            loss = -dist.log_prob(targets).mean()
            if not torch.isfinite(loss):
                raise AssertionError('nonfinite imitation loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            correct += int((dist.mode() == targets).sum())
        losses.append(total_loss / samples)
        accuracies.append(correct / samples)
    model.policy.set_training_mode(False)
    return {**report, 'epochs': epochs, 'training_cross_entropy': losses,
            'training_accuracy': accuracies, 'ppo_timesteps': 0}
