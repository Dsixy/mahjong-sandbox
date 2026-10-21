# 四人立直训练 AI

网页选择“四人立直”与“高难度 · 训练 AI”使用本目录的 `best.zip`，只消费本家可见信息，以合法动作掩码在 CPU 上确定性推理。训练配置为有赤、食断的四人立直，推理允许庄家、场风、局数、本场、供托和分数随比赛变化。

## 检查点

- CPU 沙盒训练 262,144 个可决策步，128 个并发环境、16 个进程，完成 3,370 场完整东风战，非法动作 0。
- 先以 greedy 的 12,000 条决策做 12 轮行为克隆，再以 MaskablePPO 训练；每个对手席位有 25% 概率从历史模型池取样，否则使用 greedy / defensive / value。对手在整场内固定。
- 发布检查点为 **196,608 步**，Adam epoch **96**；它由固定验证集 100000–100003、四座轮换的 16 场东风战选择，验证平均净点数 **−262.5**。最后一步的验证分数没有超过它。行为克隆初始模型不参与最佳 PPO 检查点选择。
- 模型 SHA-256：`ba1cca88138fa89091b334e63bdf0fb00f622f2c80e0ebce796a616e5151866e`。
- 训练源码 Git commit：`acdcb48845f1bcc6ab4ba39bca2713e9c897af1a`，包源码指纹：`5a628fa6bf63b5cdc7c3d47ad1a529dbdf37bb9e712bfb9df82126b8481249f4`。发布时增加了网页接入和环境 reset 保护；没有修改正在训练的源码。
- 原始 `.json` 元数据保留策略、价值网络、Adam、规则、特征版本及校验和的来源。CPU 恢复后续训 32 步已验证，参数和 Adam 步数确实更新，原检查点未变。

训练奖励为完整东风战净点数 / 10,000，包含跨局立直棒与终场供托，没有额外的立直、鸣牌、胡牌或避四奖励。不能将此原型称为已经优化顺位的比赛级策略。“高难度”是游戏档位名称，模型强度以独立评测为准。

独立测试、相对基线的配对区间及牌风指标见 `evaluation.json`。可复现命令与结果分析见仓库 `docs/RIICHI-RL.md`，运行记录与加载/续训验证见 `docs/riichi-training.json`。

## 使用

```bash
python -m pip install '.[fast,rl]'
python -m mahjong_sandbox serve

# 从本检查点继续训练，输出到新目录。
python -m mahjong_sandbox rl-train \
  --resume mahjong_sandbox/models/riichi-high/best.zip \
  --config configs/rl/riichi-cpu.json --timesteps 262144 \
  --output runs/riichi-continuation

# 使用另一组测试种子，避免把选模墙作为测试墙。
python -m mahjong_sandbox rl-evaluate \
  --model mahjong_sandbox/models/riichi-high/best.zip \
  --blocks 100 --seed 500000 --workers 4 \
  --output runs/riichi-new-evaluation.json
```

网页高难度固定开启赤宝牌和食断；不兼容的 API 规则会拒绝。模型与本目录说明包含在 Python wheel 中。训练器也接受 `--device cuda`，本次训练、评测及发布验证使用 CPU。

<!-- measured-results -->

## 固定权重确认测试

种子 400000–400063、四座轮换，每个策略 256 场东风战；PPO 实际完成 1385 手。对手为三家 greedy，权重在测试前已固定。模型胡牌率 **23.18%**、放铳率 **17.26%**、平均净点数 **+578.9/场**、平均顺位 **2.457**、四位率 **23.05%**。

对 greedy 净分差的 95% 配对区间为 **[-684, +1781]**；对行为克隆初始模型的净分差为 +1166.8，区间 **[-563, +2823]**。逐策略结果、定义与区间见 `evaluation.json`。

副露率 10.40%，立直率 35.52%，每百手杠 2.60 次，偏门清进攻；在 2155 次符合条件的防守机会中，有 0 次牺牲向听选现物。少副露不等于已学会弃和。
