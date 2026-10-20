# 川麻高难度 AI

网页“高难度 · 训练 AI”使用本目录的 `best.zip`。模型只消费本家可见观察，使用合法动作掩码，在 CPU 上确定性推理；多个牌桌共享一个模型并串行执行推理。普通牌效、随机策略及日麻不加载 PyTorch。

## 检查点与选择依据

- 训练步数：6,361,088；Adam epoch：3,137。
- 模型 SHA-256：`3daa6cf1262373af425d379d6710b289e54bc70e6b562aac4fbc353d997b0241`。
- 规则：`sichuan_xuezhan_v1`，开启换三张，底分 1，5 番封顶，自摸加底。
- 包含策略、价值网络和 Adam 状态，保留原始 `.json` 元数据及训练源码哈希；可在 CPU 加载或继续训练。
- 沿用长程实验的固定验证集最佳检查点：牌墙 100000–100015、四座轮换，共 64 手，验证平均净分 +1.078125。后续 CPU 续训的最高验证均分为 +0.46875，没有覆盖此模型。

固定监测集使用牌墙 300000–300099、四座轮换，每个模型 400 手，对阵三家 greedy。该检查点胡牌率 **59.00%**、放铳率 **44.50%**、平均净分 **+0.4625**；后续 10,293,248、10,555,392、11,079,680 步模型的平均净分依次为 +0.3925、+0.1075、+0.0200。此处以原验证集选择模型，监测集用于报告对照，没有把最后一步视为最优模型。

“高难度”是游戏档位名称。该检查点净分的 95% 区间为 **[-0.1051, 1.0825]**，尚未证明稳定强于 greedy。监测结果与来源见 `evaluation.json`；完整前后对照见仓库 `docs/sandbox-progress-evaluation.json`。此前短程流程样例仍保留在 `models/sichuan-ppo-demo/`。

## 使用

在项目根目录安装 `.[fast,rl]` 后运行 `python -m mahjong_sandbox serve`，在“新开一局”选择川麻和“高难度 · 训练 AI”。界面固定开启换三张；不兼容的 API 请求会拒绝。模型、元数据与本说明也包含在 Python wheel 中。

```bash
python -m mahjong_sandbox rl-evaluate \
  --model mahjong_sandbox/models/sichuan-high/best.zip \
  --blocks 100 --seed 400000 --workers 4 \
  --output runs/high-ai-evaluation.json

python -m mahjong_sandbox rl-train \
  --resume mahjong_sandbox/models/sichuan-high/best.zip \
  --config configs/rl/policy-cpu-sandbox.json \
  --output runs/high-ai-continuation
```
