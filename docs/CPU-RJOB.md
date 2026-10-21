# CPU RJob 续训

2026-10-20：该作业已停止，最新模型和固定目标已迁至更充足的 CPU 沙盒，见 [当前沙盒状态与 AI 指标](CPU-SANDBOX.md)。下文保留 RJob 部署记录。

本轮使用纯 CPU 作业 `policy-cpu-continuation-small-20261020-80380532`，namespace 为 `ailab-llmagent`，配额组为 `llmagent_cpu_task`，资源为 4 CPU、16,000 MiB、0 GPU。8 CPU 的初次请求未分配节点，已停止；较小请求已分配到 `lg-cmc-h-cpu-0078.host.h.pjlab.org.cn`。

上一轮 GPU 作业已成功完成，从 69,632 到 10,072,064 次学习者决策，追加 10,002,432 步，并完成 2,800 手独立对战评测。该作业状态为 Succeeded，不再占用 GPU。完整最佳模型、最终模型及原始评测已取回本地 `runs/gpu-longrun-20261019/`。

## 恢复与本轮上限

从上一轮 **最终** `latest.zip` 继续，而不是重新初始化：累计步数 10,072,064、优化器 epoch 4,949，SHA-256 为 `47ee2906095f2eedc125d908a8b9dfae744f9eceefb8c66a6e08501658f76d90`。策略、价值网络和 Adam 一起恢复；新牌局使用新的随机流。上一轮验证集选出的 `best.zip` 独立保留，不能把 best 的 6,361,088 步误认为上一轮没有达到总目标。

本轮先追加至少 1,000,000 步，按 8,192 步的完整 rollout 取整，共 123 次更新、1,007,616 步，目标累计 **11,079,680**。设备与进程数改为 CPU / 4 worker；128 局、64 步 rollout、1,024 minibatch、4 epoch、学习率 1e-4、自博弈概率 0.5 和其余学习/规则参数沿用上一轮。配置见 `configs/rl/policy-cpu-continuation.json`。

复制上一轮最近 4 个不可变历史对手快照的清单，在新目录继续发布快照；模型、优化器和旧池文件的哈希已核对。每个对手在整手牌中冻结，每 8 次更新刷新池；每次完整更新保存检查点。每 32 次更新做独立于最终测试牌墙的验证并选择 best。CPU 续训的 best 由本轮验证产生，旧 GPU best 保留在旧目录。

平台自动重启已开启，`backoffLimit=20`。控制器将初始哈希、配置、源码和累计目标绑定到新 `experiment.json`；重启从最新完整文件对恢复，目标不重复增加。训练包仍是已验证的 `c1977b7`，包哈希 `189975bd9843123bb59149c16c560432cea2924e019c02dcaa739cbee6abf513`。

## 持久目录与操作

运行根目录：

```text
/mnt/shared-storage-gpfs2/gpfs2-agent-multimodel/user/dongyi/projects/policy-optimization-20261019/cpu-continuation-20261020/
```

目录中包含独立的 `initial.zip/.json`、`config.json`、`handoff.json`、原生 RJob spec 和 `output/`。`source/` 与 `runtime/` 指向同一持久实验内已准备的冻结源码/环境；作业使用准备该环境时的同一固定镜像 digest。该 runtime 不能直接在不同 Python 基础环境的开发机上运行。

`scripts/run_policy_cpu.sh RUN_ROOT` 会隐藏 CUDA 设备，验证 CPU 矩阵反向传播、模型和 Adam 的 CPU 恢复，然后调用有累计步数上限的控制器。输出 `cpu-runtime.json`、`metrics.jsonl`、不可变 `checkpoints/` 和 `selfplay/`；`latest.zip` 在训练结束或可处理的中断后产生。

```bash
export KUBEBRAIN_CLUSTER_ENTRY=https://h.pjlab.org.cn
export KUBEBRAIN_NAMESPACE=ailab-llmagent
rjob get policy-cpu-continuation-small-20261020-80380532
rjob logs job policy-cpu-continuation-small-20261020-80380532 --tail-lines=30
```

达到目标后自动评测：100 个测试牌墙 × 4 座轮换 × 7 个候选，共 2,800 手，三家对手为 greedy；候选包括本轮 best、CPU 续训前的最终 GPU 模型，以及 random/win-first/greedy/defensive/value。报告输出胡牌率、放铳率、平均净得分和配对区间。新报告的 `initial` 指 10,072,064 步 GPU 最终模型，含义与上一轮的 69,632 步初始模型不同。

## 上一轮结果与验证

上一轮最佳模型在三家 greedy 对手下的 400 手评测：胡牌率 58.5%、放铳率 47.75%、平均净得分 +0.45；净得分 95% 区间约 [-0.09, 1.06]。相对原 69,632 步模型的净得分差区间也包含 0，不能声称已经显著优于该模型或 greedy。结果摘要见 [独立评测](longrun-final-evaluation.json)，旧最佳和最终模型均已在本地 CPU 成功加载并核对有限参数与 Adam 设备。

CPU 启动脚本通过 Bash 语法检查，配置通过 `TrainConfig`、固定目标与训练包哈希校验。还用该脚本完成了本地端到端验收：恢复模型与 Adam、追加 16 步、发布自博弈快照、保存检查点、自动评测 28 手并产生 `complete.json`，全程设备为 CPU，CUDA 可见设备为 0。正式作业的 CPU 矩阵反向传播与初始模型/Adam 恢复也已通过，64 手初始验证结果与 GPU 结束时的同牌墙验证相同。实际节点、恢复证明和完成更新数记录在 [CPU 作业凭据](cpu-continuation-rjob.json)。

正式 CPU 作业已完成至少 1 次完整优化并保存哈希有效的检查点：累计 10,080,256 步，优化器 epoch 从 4,949 增至 4,953。此次核对时作业为 Running。
