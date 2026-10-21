# 川麻 CPU 示例模型

`best.zip` 是 69,632 次 PPO 环境决策后选择的模型，包含策略、价值网络和 Adam 状态；`warmstart.zip` 仅完成 8000 条 bot 示范的行为克隆，用于独立对照。各自 `.json` 文件记录规则、特征版本、依赖、来源和 SHA-256。

```bash
.venv/bin/python -m mahjong_sandbox rl-evaluate --model models/sichuan-ppo-demo/best.zip --blocks 50 --workers 4 --output runs/demo-evaluation.json
# GPU/RJob 中继续训练，需相应 PyTorch/CUDA 环境。
python -m mahjong_sandbox rl-train --resume models/sichuan-ppo-demo/best.zip --config configs/rl/sichuan-gpu.json --output /outputs/mahjong/sichuan-ppo
```

训练与评测说明见 [RL 文档](../../docs/RL.md)。该短程模型用于验证完整流程，不代表已训练出强于现有牌效 bot 的策略。
