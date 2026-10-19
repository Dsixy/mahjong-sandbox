# 阶段 6：川麻血战到底 RL

本阶段实现的是可训练、可保存、可恢复、可独立评测的神经网络策略。规则使用现有 `sichuan_xuezhan_v1`：108 张牌、换三张、定缺、多人胡牌后继续、查花猪/查叫及退杠。episode 始终是整手血战，不在学习者先胡时提前结束。

## 算法与框架选择

使用 **PyTorch + SB3-contrib 2.9 MaskablePPO**，复用框架的 PPO clipping、GAE、价值函数、熵正则、Adam、梯度裁剪、rollout buffer、模型保存加载。项目只实现麻将观察/动作编码、候选评分策略、批量环境适配、实验管理与评测。阶段 5 的 NumPy 线性 REINFORCE 留作接口验收，不作为本阶段主训练器。

选择依据：

| 方案 | 本项目的适配工作 | 选择 |
| --- | --- | --- |
| 自建训练循环 | 需要自己维护 PPO、GAE、优化器恢复和掩码一致性 | 当前不增加这部分维护范围 |
| CleanRL 风格单文件 PPO | 容易修改，但仍需维护改写后的训练/恢复流程 | 作为算法实现对照 |
| SB3-contrib MaskablePPO | 已支持离散动作掩码和 Dict 观察；需定制候选策略与 VecEnv | 采用 |

依据：[PPO 原论文](https://arxiv.org/abs/1707.06347)、[动作掩码研究](https://arxiv.org/abs/2006.14171)、[MaskablePPO 官方文档](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)、[CleanRL PPO 文档](https://docs.cleanrl.dev/rl-algorithms/ppo/)。这里使用前馈网络，未实现 LSTM/对手隐状态推断；牌河通过公开统计和最近 8 张的顺序编码进入模型。

## 输入、输出与奖励

| 输入 | 形状 | 内容 |
| --- | --- | --- |
| `state` | `[N,1549] float32` | 本家暗牌及可见牌的四级计数、各家河牌/副露/亮牌统计、各家最近 8 张河牌、缺门/行牌状态、相对分数、阶段、相对行动方/庄家、余牌、公开换牌方向、本家过胡限制、待应牌与本家摸牌 |
| `candidates` | `[N,512,8] int16` | 当前合法动作的类型、消耗牌、定缺花色及副露索引 |
| `hints` | `[N,512,7] float32` | 动作后结构距离、最佳弃牌有效进张、相关花色数量、缺门消耗、可见牌数量、距离是否最优及摸切标识 |
| `mask` | `[N,512] bool` | 合法语义候选，其他位置概率严格为 0 |

所有座位按“本家、下家、对家、上家”排列。牌墙、其他家的暗牌和密封选择不进入模型；训练与 `RLAgent.act(observation)` 共用 `encode_observation()` 和同一套特征函数。结构距离和有效进张也只使用可见牌。没有把现有 bot 的动作/评分作为 PPO 的隐藏输入。

动作输出为当前候选上的 categorical 分布，另有一个标量 `V(observation)`。动作包括换三张、定缺、弃牌、碰、明杠/暗杠/加杠、自摸、荣和与过。候选共享评分网络，不使用把 512 个位置各自当成固定动作类别的输出层；重排菜单会相应重排概率，价值估计保持不变。仅合并牌种、参数和摸切标识相同的实体牌副本，保留摸切/手切的公开信息差异。

每个 worker 管理多手牌。学习座位按 `(全局 lane + episode) % 4` 轮换，三个对手从配置的 bot 池按独立种子选取。一次 step 从学习者动作推进至其下次决策或整手终局，其间全部净分变化都进入奖励。学习者胡牌退出后，剩余玩家继续结算，延后收益不会丢失。

默认奖励为 **原始净分增量 / 32**，`gamma=1`，没有额外胡牌奖励、向听奖励或非法动作罚分。优化目标与整手平均净得分一致；胡牌率、放铳率作为独立评测指标。接口逐手核对累计奖励与最终分差。SB3 的自动重置只发生在适配层：先保存 `terminal_observation`、奖励和 done，再重置；真正终局不 bootstrap，rollout 长度边界继续使用价值估计。

`--no-hints` 可关闭结构距离等领域特征，摸切标识仍保留为动作身份。它是独立实验配置，不允许在恢复同一模型时悄悄切换。默认模型尺寸约 128 维上下文、64 维动作表示，适合作为可继续扩展的基线。

## 安装与 CPU 训练

已实测 Python 3.12/3.13、PyTorch 2.14.1 CPU、SB3/SB3-contrib 2.9.0、NumPy 2.3.5、Numba 0.68.0。参考游戏仍不需要 RL 依赖。

```bash
# 已有可用的 .venv 时不必重建。
python3 -m venv .venv
.venv/bin/python -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install '.[fast,rl]'

.venv/bin/python -m mahjong_sandbox rl-train \
  --config configs/rl/sichuan-cpu.json \
  --output runs/sichuan-cpu
```

默认先收集 8000 条 **存在多个语义候选的决策** 做 8 轮行为克隆，再执行 65,536 次环境决策的 PPO。只有一种动作的强制决策不算模仿样本；模仿报告记录教师、牌局数量、阶段覆盖、训练集交叉熵和准确率。该准确率不等于独立测试胜率。`--bc-samples 0` 可从随机初始化进行纯 PPO。

可覆盖参数，例如 `--boards 32 --workers 4 --timesteps 16384 --rollout-steps 64 --batch-size 256`。timesteps 是学习者环境决策数，包含响应决策，不是麻将完整局数；框架按完整 rollout 更新，实际值可能向上取整。

训练目录包含：

| 文件 | 作用 |
| --- | --- |
| `config.json`, `metrics.jsonl` | 实验配置、训练损失/KL/熵、近期训练指标和验证记录 |
| `imitation.json`, `warmstart.zip/.json` | 可选的模仿初始化及其报告，PPO timestep 为 0 |
| `checkpoints/step_*.zip/.json` | 定期模型与优化器检查点 |
| `latest.zip/.json` | 最新完成训练的模型 |
| `best.zip/.json` | 固定验证牌墙上平均净得分最高的 PPO 模型；BC-only 不参与此选择 |
| `training.json` | 实际决策数、更新数、完成牌局、耗时、验证和模型校验和 |

默认每 8 个 rollout 更新做检查点及 8 个牌墙 × 4 个座位的验证，验证种子从 100000 开始。验证集用于模型选择，最终对战测试另用种子；评测入口会拒绝与已记录验证区间重叠的测试种子。

## 加载与恢复

```bash
.venv/bin/python -m mahjong_sandbox rl-train \
  --resume runs/sichuan-cpu/latest.zip \
  --output runs/sichuan-cpu --timesteps 65536
```

这里的 timesteps 是**追加训练量**。恢复策略、价值网络和 Adam 状态，保持累计 timestep；使用新的牌墙种子流和新的完整牌局，不恢复中途牌墙/轨迹或保证逐 bit 延续。`training_before_<timestep>.json` 保留之前报告。CLI 默认继承检查点中的配置，再应用配置文件和显式参数覆盖。已有模型的网络架构保持不变，`--hidden` 用于新模型初始化。规则、特征版本或 hint 设置不兼容会拒绝加载；保存文件附有 SHA-256 校验和及依赖版本。

在现有引擎里使用保存的策略：

```python
from mahjong_sandbox.rl.agent import RLAgent

agent = RLAgent('runs/sichuan-cpu/best.zip', device='cpu')
# 与原有 bot 一样，只传该玩家的可见 observation。
action = agent.act(engine.observe(player))
engine.submit(player, action, engine.state.decision_id)
```

## 独立对战评测

```bash
.venv/bin/python -m mahjong_sandbox rl-evaluate \
  --model runs/sichuan-cpu/best.zip \
  --baseline-model runs/sichuan-cpu/warmstart.zip \
  --opponents greedy --blocks 50 --seed 200000 --workers 4 \
  --output runs/sichuan-cpu/evaluation.json

# 只让 RL 分别挑战五种三家同策略对手。
.venv/bin/python -m mahjong_sandbox rl-evaluate \
  --model runs/sichuan-cpu/best.zip --baselines \
  --opponents random win-first greedy defensive value \
  --blocks 25 --seed 300000 --workers 4 \
  --output runs/sichuan-cpu/matchups.json
```

第一条默认比较 RL、warmstart、随机、胡牌优先、牌效、防守和价值牌策略，对手均为三家 greedy；每种候选用相同牌墙轮坐四个座位，以减少发牌和庄家差异。第二条提供直接对战矩阵。默认取模型最大概率动作；`--stochastic` 使用按牌墙/座位播种的采样。

报告输出 `win_rate`、`deal_in_rate`、`mean_net_score`、95% 区间和逐局原始记录，并计算 RL 相对各基线的成对差值区间。区间按独立牌墙块 bootstrap，不能把同一牌墙的四个座位当成四个独立样本。胡牌/放铳分别表示本家至少胡过一次/至少成为一次荣和来源；川麻两者可以同时发生。平均净得分包括杠分、查花猪、查叫和退杠。测试结果只说明该规则、样本和对手池下的表现。

## GPU / RJob 入口

RJob 只需准备代码和依赖、分配 GPU/CPU、挂载持久输出目录，然后调用：

```bash
# 在 GPU 镜像中安装与 CUDA 环境匹配的 PyTorch，再安装本项目 extras。
python -m pip install '.[fast,rl]'
bash scripts/train_sichuan_rl.sh \
  --config configs/rl/sichuan-gpu.json --output /outputs/mahjong/sichuan-ppo

# 从 CPU 检查点在 GPU 上接着训练。
python -m mahjong_sandbox rl-train \
  --resume /inputs/sichuan/latest.zip --config configs/rl/sichuan-gpu.json \
  --output /outputs/mahjong/sichuan-ppo
```

GPU 配置预留 256 局、8 个 CPU worker、128 步 rollout、1024 minibatch 和 200 万追加决策。资源数与输出挂载路径由实际 RJob 配置决定；这里不猜测平台任务字段，也没有提交 GPU 作业。`MAHJONG_PYTHON` 可指定入口脚本使用的解释器。

`--device cpu/cuda/cuda:0/auto` 控制 PyTorch 推理与反向传播。请求 CUDA 而不可用时明确失败，不静默改成 CPU。牌局与 NumPy/Numba 特征仍运行在 CPU，worker 使用 spawn；GPU 由训练主进程持有。保存的模型可跨设备加载，当前每个训练进程使用一张 GPU，不含 DDP。评测若使用 GPU，需 `--device cuda --workers 1`，避免每个评测进程各占一份 GPU 上下文。

CPU 试训的 10 秒活动栈采样中，约四分之一落在有效进张算子，神经网络反向传播/前向也占较大比例；GPU 不会消除环境和牌效算子的 CPU 成本。该采样不是整次训练墙钟占比，详见 [采样报告](stage6-cpu-profile.json)。后续 GPU 实测应分别看环境、特征、传输、网络更新的耗时；本次不声称 GPU 加速比。

## 本次实测与验收

本地带可选依赖的完整 130 项测试通过；无 NumPy/Torch 的参考环境也通过回归，24 项可选测试跳过。新增 10 项 RL 测试覆盖共用编码、信息边界、摸切去重、菜单重排、掩码、终局奖励和自动重置、轮换座位、进程一致性、模型及优化器恢复、续训、验证/测试隔离和模仿初始化。

代码已通过 Git bundle 迁移至开发机 `/workspace/mahjong-sandbox-stage6`，使用独立 `.rl-venv` 完成 10 项 RL 测试、1024 决策训练、两次追加 256 决策续训和 24 局最终评测，记录见 [开发机验收](stage6-dev-validation.json)。这部分是兼容性验证，不是策略强度证据。最终 69,632 步检查点已从本地 Python 3.12 迁移至开发机 Python 3.13，成功加载并提交合法动作，见 [模型迁移验收](stage6-model-portability.json)。

<!-- RESULTS:START -->
CPU 试训先用 8000 条多候选 bot 示范初始化，随后完成 65,536 + 4,096 = **69,632 次环境决策**、**1,730 手完整血战**。恢复训练确实接续累计步数、改变模型参数并保留 Adam 状态；模型在当前规则下没有提交非法动作。初次运行含示范采集约 754.6 秒，追加运行约 122.9 秒。这是功能试训耗时，包含验证及并发回归负载，不是吞吐量极限。

独立测试种子 200000–200049，50 个牌墙 × 4 个座位，每个候选 200 局，共 1400 局；全部对阵三家 greedy：

| 策略 | 胡牌率 | 放铳率 | 平均净得分 | 净得分 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| PPO 模型 | 71.5% | 33.0% | -0.03 | [-0.55, +0.46] |
| 仅模仿初始化 | 65.5% | 38.5% | -0.34 | [-0.76, +0.07] |
| random | 1.5% | 65.5% | -5.43 | [-6.62, -4.60] |
| win-first | 2.0% | 65.0% | -5.38 | [-6.58, -4.57] |
| greedy | 70.5% | 38.0% | +0.00 | [+0.00, +0.00] |
| defensive | 66.0% | 39.0% | -0.12 | [-0.33, +0.04] |
| value | 53.5% | 43.5% | -1.18 | [-1.73, -0.68] |

相对 greedy，PPO 的胡牌率差 +1.0 个百分点、放铳率差 −5.0 个百分点、平均净分差 −0.03；成对差值区间均包含 0，尚不能证明它更强。相对仅模仿初始化，胡牌率提高 6.0 个百分点，平均净分提高 0.31，但净分差的 95% 区间约为 [−0.22, +0.87]，也不能确认稳定收益提升。没有用这些测试结果反过来调参或选择检查点。

另外用测试种子 300000–300024 直接挑战五种对手池，每个 100 局，共 500 局：

| 三家对手 | RL 胡牌率 | RL 放铳率 | RL 平均净得分 |
| --- | ---: | ---: | ---: |
| random | 75.0% | 0.0% | +3.46 |
| win-first | 76.0% | 0.0% | +3.50 |
| greedy | 75.0% | 32.0% | +0.32 |
| defensive | 74.0% | 31.0% | +0.85 |
| value | 79.0% | 27.0% | +2.04 |

两组试验合计 1900 局，对手池和牌墙不同，不能合并成一个总体胜率。完整逐局记录及差值区间见 [公平比较](stage6-evaluation.json)、[直接对战](stage6-matchups.json)。

[训练报告](stage6-training.json)、[训练日志](stage6-metrics.jsonl)、[完整验收记录](stage6-validation.json)。可直接加载的 [PPO 检查点](../models/sichuan-ppo-demo/best.zip) 与 [模仿初始化检查点](../models/sichuan-ppo-demo/warmstart.zip) 已随代码保留；各自的 JSON 清单记录版本、规则、训练来源和校验和。

训练报告的 `optimizer_epochs`（早期检查点中的 `optimizer_updates`）沿用 SB3 的 epoch 计数，不是 Adam minibatch 步数；本次共有 17 个 rollout 更新。RJob 后续可在该检查点上继续训练；CUDA 路径已预留，但没有 GPU 硬件实跑结果。
<!-- RESULTS:END -->
