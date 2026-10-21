"""Lazy, CPU-only access to the bundled high-difficulty opponent."""
from functools import lru_cache
from pathlib import Path
import threading

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/sichuan-high/best.zip'
MODEL_PATHS = {'sichuan': MODEL_PATH, 'riichi': MODEL_PATH.parents[1] / 'riichi-high/best.zip'}
_LOAD_LOCK = threading.Lock()


class TrainedOpponent:
    def __init__(self, agent):
        self.agent, self.rules = agent, agent.rules
        self.lock = threading.Lock()

    def act(self, observation):
        # SB3 predict changes the policy's training mode. Serialize access when
        # HTTP sessions share this model; every decision still uses its own view.
        with self.lock:
            return self.agent.act(observation)


@lru_cache(maxsize=2)
def _load_opponent(variant):
    if variant not in MODEL_PATHS:
        raise ValueError('unknown trained opponent variant')
    path = MODEL_PATHS[variant]
    try:
        import torch
        from .agent import RLAgent
    except ImportError as error:
        raise ValueError('高难度 AI 需要 RL 依赖，请在服务端安装 .[fast,rl] 后重试。') from error
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ValueError('高难度 AI 模型文件缺失，请重新安装完整的游戏包。')
    torch.set_num_threads(1)
    return TrainedOpponent(RLAgent(path, device='cpu', deterministic=True))


def high_difficulty_opponent(variant='sichuan'):
    with _LOAD_LOCK:
        return _load_opponent(variant)
