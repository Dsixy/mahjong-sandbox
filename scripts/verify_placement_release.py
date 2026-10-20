"""Verify CPU loading and Adam continuation using a temporary model copy."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mahjong_sandbox.fast import FastRiichiEngine, RuntimeOptions
from mahjong_sandbox.riichi_model import RiichiConfig
from mahjong_sandbox.rl.agent import RLAgent
from mahjong_sandbox.rl.checkpoint import checksum, load_checkpoint
from mahjong_sandbox.rl.train import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='mahjong_sandbox/models/riichi-rank/best.zip')
    parser.add_argument('--output', default='runs/placement-20261020/release-runtime-proof.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = Path(args.model)
    before_sha = checksum(path)
    model, metadata = load_checkpoint(path, device='cpu')
    config = metadata['training']['config']
    assert config['reward_mode'] == 'rank' and config['match_length'] == 'south'
    assert config['reward_scale'] == config['gamma'] == 1
    state = model.policy.optimizer.state_dict()['state']
    assert state
    steps = [float(item['step']) for item in state.values()]
    assert min(steps) > 0
    assert all(p.device.type == 'cpu' and bool(torch.isfinite(p).all())
               for p in model.policy.parameters())
    assert all(v.device.type == 'cpu' and bool(torch.isfinite(v).all())
               for item in state.values() for v in item.values() if isinstance(v, torch.Tensor))
    agent = RLAgent(model=model, metadata=metadata, backend='numba')
    rules = replace(RiichiConfig(), dealer=2, round_wind=28, hand_number=3,
                    honba=2, riichi_sticks=2, initial_scores=(21000, 29000, 18000, 30000))
    engine = FastRiichiEngine(rules, seed=99117, runtime=RuntimeOptions(operator_backend='numba'))
    player = engine.next_player()
    obs = engine.observe(player)
    action = agent.act(obs)
    assert action.to_dict() in obs['legal_actions']
    engine.submit(player, action, engine.state.decision_id)
    parameters = {key: value.clone() for key, value in model.policy.state_dict().items()}
    with tempfile.TemporaryDirectory(prefix='placement-release-resume-') as directory:
        result = train(directory, TrainConfig(
            variant='riichi', match_length='south', reward_mode='rank',
            rank_rewards=tuple(config['rank_rewards']), reward_scale=1, gamma=1, gae_lambda=1,
            total_timesteps=32, n_envs=4, workers=0, rollout_steps=8, batch_size=16,
            epochs=1, learning_rate=1e-4, hidden=128, bc_samples=0, bc_epochs=0,
            validation_blocks=0, backend='numba', self_play_probability=0,
            checkpoint_every=1, seed=777777), resume=path)
        restored, after = load_checkpoint(Path(directory) / 'latest.zip', device='cpu')
        assert after['timesteps'] == metadata['timesteps'] + 32
        assert after['optimizer_updates'] == metadata['optimizer_updates'] + 1
        assert any(not torch.equal(value, restored.policy.state_dict()[key])
                   for key, value in parameters.items())
        new_steps = [float(item['step'])
                     for item in restored.policy.optimizer.state_dict()['state'].values()]
        assert min(new_steps) > min(steps)
        proof = {
            'format': 'placement_release_runtime_proof_v1', 'model_sha256': before_sha,
            'cpu_load': True, 'finite_parameters_and_optimizer': True,
            'reward_mode': config['reward_mode'], 'match_length': config['match_length'],
            'dynamic_round_legal_action': action.to_dict(),
            'saved_optimizer_steps': [min(steps), max(steps)],
            'resume_optimizer_steps': [min(new_steps), max(new_steps)],
            'resumed_from_timesteps': metadata['timesteps'],
            'resumed_to_timesteps': after['timesteps'],
            'optimizer_epochs_before': metadata['optimizer_updates'],
            'optimizer_epochs_after': after['optimizer_updates'],
            'parameters_changed': True, 'illegal_actions': result['illegal_actions'],
            'promoted_checkpoint_unchanged': checksum(path) == before_sha,
        }
        assert proof['promoted_checkpoint_unchanged'] and proof['illegal_actions'] == 0
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(proof, indent=2) + '\n')
        print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
