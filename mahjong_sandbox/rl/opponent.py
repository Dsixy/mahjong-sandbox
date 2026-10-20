"""Lazy, CPU-only access to the bundled high-difficulty opponent."""
from functools import lru_cache
from pathlib import Path
import threading

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/sichuan-high/best.zip'
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


@lru_cache(maxsize=1)
def _load_opponent():
    try:
        import torch
        from .agent import RLAgent
    except ImportError as error:
        raise ValueError('高难度 AI 需要 RL 依赖，请在服务端安装 .[fast,rl] 后重试。') from error
    if not MODEL_PATH.is_file() or not MODEL_PATH.with_suffix('.json').is_file():
        raise ValueError('高难度 AI 模型文件缺失，请重新安装完整的游戏包。')
    torch.set_num_threads(1)
    return TrainedOpponent(RLAgent(MODEL_PATH, device='cpu', deterministic=True))


def high_difficulty_opponent():
    with _LOAD_LOCK:
        return _load_opponent()
