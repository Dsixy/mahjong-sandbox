# 长程 GPU PPO 与历史自博弈

2026-10-20：该轮 GPU 训练及最终评测已完成。后续先用 CPU RJob，现已迁至 CPU 沙盒；当前资源、恢复点、固定目标及独立评测见 [CPU 沙盒说明](CPU-SANDBOX.md)，此前记录见 [CPU RJob 说明](CPU-RJOB.md)。下文保留原 GPU 部署记录。

`configs/rl/policy-longrun.json` 从阶段 6 的 69,632 步 CPU 模型继续训练。单 GPU 负责学习者批量推理和 PPO 更新，8 个 CPU worker 管理 128 手牌，历史对手在 worker 上做 CPU 推理。对手模型在整手牌内冻结；每个对手座位有 50% 概率从最近 4 个历史快照中均匀抽取，否则使用 greedy/defensive/value。初始模型也进入池，每 8 次完整更新发布新快照。这是历史模型自博弈，不是四座位同时更新。

每个 rollout 含 8,192 次学习者决策。每次更新保存完整策略、价值网络、Adam 及 SHA-256；每 32 次更新在独立固定的 64 手验证牌局上选择 best。学习率为 1e-4。训练目标是追加至少 1,000 万次决策，向完整 rollout 取整：从 69,632 到 10,072,064 步，追加 10,002,432 步。决策数不是完成牌局数。

## 可抢占执行

输出目录必须放在持久共享存储，并在重启时挂到相同绝对路径。脚本和训练环境均放在本实验目录，GPU worker 不下载依赖；准备环境与正式训练使用同一个固定 digest 的 CUDA 镜像。

```bash
# EXPERIMENT_ROOT 包含 source/（当前 Git 提交）与 runtime/（该镜像内准备的环境）。
bash "$EXPERIMENT_ROOT/source/scripts/run_policy_gpu.sh" "$EXPERIMENT_ROOT"
```

使用平台支持的 `rjob submit --task-type=idle`、单副本、1 GPU、16 CPU、64,000 MiB 内存；闲时任务不指定 priority/配额组。CLI 设置 `--restart-policy=restartjobonfailure --backoff_limit=20 --termination-grace-period-seconds=30`，模板中的策略值为 `RestartJobOnFailure`；先 dry-run 检查当前平台接受的实际模板。GPFS mount、镜像和作业名由部署环境传入。任务名使用 `policy-optimization-*`。

`rl.longrun` 把目标总步数、初始模型哈希、配置、代码哈希绑定到 `experiment.json`。重启扫描 `latest` 和不可变 `checkpoints/step_*`，校验后选择步数最多的完整文件对；损坏或保存中断的文件会跳过。目标不会因每次重启重复增加。文件锁阻止同目录两个学习者并发写入。

如果 `best` 的模型/元数据替换被打断，控制器按验证日志选择已有的完整同一步检查点并重建文件对；只用验证集记录恢复选择，不查询最终测试成绩。最后一条未写完的日志会忽略。

收到 SIGTERM 时尽可能完成当前 rollout 和优化，再在更新边界保存退出，返回 75 以触发重试。平台仅给 30 秒时可能强杀，此时从上一完整更新恢复；中途牌局、未完成 rollout 和随机数流不精确恢复。检查点包含初始策略/优化器；对手池使用持久不可变快照。框架不提供跨平台逐位复现。

`gpu-runtime.json` 是真实 CUDA 矩阵乘与反向传播证据；`metrics.jsonl` 的 start/ppo_update 证明实际模型设备、优化与有限损失。启动脚本在 CUDA 不可用时明确失败。GPU 利用率可能受规则推进及单动作历史对手 CPU 推理限制，不能从资源申请推断 GPU 加速比。

后续 CPU/H200 同机分段实测及资源选择见 [运行时间对照](RL-PERFORMANCE.md)：当前模型 GPU 更新明显更快，但整轮吞吐约提升 13%，采样流水线仍是主要优化方向。

最终独立评测用 100 个未用于选模的牌墙、四座轮换，在三家 greedy 对手下比较 RL、初始 CPU 模型和五种现有 bot，共 2,800 手，保存 `evaluation.json`。标签 `initial` 指继续训练前的模型；胡牌率、放铳率、平均净得分及配对区间沿用阶段 6 的定义。`complete.json` 仅在达到目标且评测完成后生成。长程训练进行中不能据此声称优于基线。

手动运行控制器也支持 CPU 回归：

```bash
python -m mahjong_sandbox.rl.longrun --config CPU_CONFIG.json \
  --output PERSISTENT_OUTPUT --initial models/sichuan-ppo-demo/best.zip \
  --additional-steps 512 --no-evaluation
```

## 本次部署记录

代码提交为 `c1977b7`，135 项本地回归测试通过；固定 CUDA 镜像内的 15 项 RL/自博弈测试通过。镜像内 Python 3.12.12、PyTorch 2.8.0+cu128、CUDA build 12.8；依赖全部安装在实验私有 runtime，GPU 任务离线运行。

先尝试闲时队列；等待驱逐冷却窗口后仍未分配，已确认停止且没有 GPU 绑定。正式任务改用同项目 `llmagent_gpu` 普通队列：单卡、16 CPU、64,000 MiB、priority 9、preemptible=yes、平台 auto-restart=true、backoff_limit=20；普通任务使用平台要求的 Never 重启策略，由平台自动恢复机制重新调度，控制器据持久检查点继续。资源预测发现 `gpu-lg-cmc-h-h200-1680.host.h.pjlab.org.cn` 是已有任务占用的部分空闲节点，于生成的原生 RJob 模板中加入该节点的软亲和性；资源变化时允许调度到其他可用节点。正式作业名为 `policy-optimization-20261019-main`，已调度至该部分空闲节点。完整资源清单和实际运行状态由 `longrun-rjob.json` 记录。

本次发现平台 admission webhook 对服务端 dry-run 会产生空 PodGroup 副作用，因此只用 `rjob submit --dry-run=true` 生成客户端模板。预检产生的空组已核对 UID、创建时间、无 RJob/replica 后删除；正式作业独立创建，保留实际 UID。

```bash
# 在开发机当前会话中设置非敏感路由参数，不改动全局配置。
export KUBEBRAIN_CLUSTER_ENTRY=https://h.pjlab.org.cn
export KUBEBRAIN_NAMESPACE=ailab-llmagent
rjob get policy-optimization-20261019-main
rjob logs job policy-optimization-20261019-main --tail-lines=50
```

持久输出：`/mnt/shared-storage-gpfs2/gpfs2-agent-multimodel/user/dongyi/projects/policy-optimization-20261019/output/`。指标在 `metrics.jsonl`，模型在 `checkpoints/`，自博弈池在 `selfplay/`，最终成绩在 `evaluation.json`。
