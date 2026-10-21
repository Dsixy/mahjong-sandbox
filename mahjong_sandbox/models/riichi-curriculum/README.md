# 四人立直：蒸馏与混合自博弈实验

**本轮未通过独立测试发布门槛，保留为实验模型；默认高难度 AI 不变。**

本目录保留 V2 主模型及 V1 同流程对照，均由预先规定的验证牌墙选出。各训练种子实际完成 524,288 个决策；所选检查点可以早于训练终点。独立测试与发布判定见 [实验报告](../../../docs/RIICHI-CURRICULUM.md)，完整数字见 [analysis.json](../../../docs/riichi-curriculum/analysis.json)。

| 文件 | 输入 | 用途 |
|---|---|---|
| `best.zip/.json` | V2 完整公开信息 | 两个 V2 种子间验证选出的主模型 |
| `warmstart.zip/.json` | V2 | 通过教师验收的蒸馏模型，0 PPO 步 |
| `replica.zip/.json` | V2 | 第二个训练种子的验证检查点，供复现 |
| `control.zip/.json` | V1 | 相同新训练流程的输入对照 |
| `control-warmstart.zip/.json` | V1 | 对照组蒸馏模型，0 PPO 步 |

RL 文件保留完整策略、价值函数、Adam 与原始序列化内容，字节哈希对应测试前冻结的检查点。包源码 SHA-256 为 `309f385b9b5bbdeb52c304b533a626c6e585c1a06aba8a874e21010a49d275ec`。原发布第一版及顺位实验权重另保留在既有目录。

以下命令在新的牌墙上评测；默认继承模型的半庄规则：

```bash
python -m mahjong_sandbox rl-evaluate \
  --model mahjong_sandbox/models/riichi-curriculum/best.zip \
  --baseline-model mahjong_sandbox/models/riichi-curriculum/warmstart.zip \
  --baselines greedy defensive --blocks 256 --seed 2300000 --workers 4 \
  --output runs/riichi-curriculum-new-test.json

python -m mahjong_sandbox rl-train \
  --resume mahjong_sandbox/models/riichi-curriculum/best.zip \
  --config configs/rl/riichi-curriculum-continue.json \
  --output runs/riichi-curriculum-continuation
```

续训例子追加 131,072 个决策、64 场并发，以较低的进程数运行；加载原策略、价值函数及 Adam，重新开始牌局。该例显式关闭教师保留损失，因为安装包未附带训练数据集。要重现带教师保留损失的完整流程，请使用仓库的 `scripts/run_riichi_curriculum_pipeline.py` 重新生成数据，或使用校验过的原始持久数据目录。这属于新的训练实验，不改变本轮已冻结的对比。

CPU 加载及临时副本的真实 Adam 续训证据见 [V2 核验](../../../docs/riichi-curriculum/v2-cpu-resume-proof.json) 和 [V1 核验](../../../docs/riichi-curriculum/v1-cpu-resume-proof.json)。

独立测试中，V2 主模型平均位次 2.5313、胡牌率 22.29%、放铳率 16.82%、净点数 +116.8/半庄。相对 greedy 的位次差 95% 区间为 [−0.0303, +0.0938]；相对第一版为 [−0.1016, +0.0645]。验证总体最佳是 `control.zip`，其独立测试平均位次 2.5273；也没有证实超过 greedy。`replica.zip` 的测试点估计为 2.4697，作为预先规划的第二个种子保留，主模型选择保持测试前的结果。
