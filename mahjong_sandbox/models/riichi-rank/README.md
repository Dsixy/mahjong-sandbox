# 四人立直：顺位奖励实验模型

本目录保存以完整半庄最终排名训练的检查点，奖励一至四位分别为 +1.5、+0.5、−0.5、−1.5；中途奖励为 0，gamma=1，不混合净点数或和牌奖励。训练和评测使用 CPU，模型/价值函数/Adam 可在 CPU 加载和续训。

- 训练来源：Git `8147bb2e83fdcb6957bad5ca308db0355b29c7c4`，包源码 SHA-256 `32e3cca7a1c39312f5c6d3a2db8c619453362177a03f5600754419832c2e5e9b`。
- 两个训练种子各跑 score/rank 一组，各 524,288 步；固定预算最终模型为主比较。
- 本目录检查点由两个顺位模型的固定验证集表现选择，未使用测试集选模。训练种子 **20261**，检查点 **393,216 步**，PPO 优化 epoch **96**。
- 模型 SHA-256：`01aa8063c16b872a08a5ca5ef2783ca3f917d516417960d048cfcf261e675427`。
- 验证种子 120000–120015、四座轮换，平均排名 utility +0.125000。
- 同目录 `warmstart.zip/.json` 是四组共用的 0 PPO 步模仿初始化，SHA-256 `82562978b74ec4adb6426ce0d2b4d68baaad3b81433f00d617f1ea5d23c50b57`，保留它便于复现实验。

**这是负结果实验检查点：独立测试平均顺位差于 greedy 基线，保留用于复现和继续训练，未替换默认高难度档位。**

独立测试种子 600000–600255、四座轮换，共 1,024 场半庄、10126 手，对阵三家 greedy。所发布检查点平均顺位 **2.6680**，一位率 **20.51%**，四位率 **30.76%**，净点数 **-2755.3/场**。这个验证最佳模型属于次分析；两种奖励的公平主比较及配对区间见仓库 `docs/PLACEMENT-RL.md`，完整汇总见本目录 `evaluation.json`。

配置为有赤、食断的四人立直半庄，特征仅使用本家可见信息、34 种牌、2,350 维状态、12 维候选提示，合法动作掩码与旧立直模型兼容。该模型是明确目标下的训练实验，不代表赛事级实力，也不把纯平均顺位等同于所有赛事的奖金规则。

```bash
python -m mahjong_sandbox rl-evaluate \
  --model mahjong_sandbox/models/riichi-rank/best.zip \
  --baseline-model mahjong_sandbox/models/riichi-high/best.zip \
  --blocks 256 --seed 800000 --workers 16 \
  --output runs/rank-new-evaluation.json

python -m mahjong_sandbox rl-train \
  --resume mahjong_sandbox/models/riichi-rank/best.zip \
  --config configs/rl/riichi-south-rank.json --workers 8 \
  --output runs/rank-continuation
```

模型与原始元数据应一起保存；加载会核对特征、规则和文件校验和。默认游戏中的高难度档位仍使用既有净点数模型，便于保留对照。
