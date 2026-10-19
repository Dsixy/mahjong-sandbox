# 长程 GPU PPO 与历史自博弈

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

最终独立评测用 100 个未用于选模的牌墙、四座轮换，在三家 greedy 对手下比较 RL、初始 CPU 模型和五种现有 bot，共 2,800 手，保存 `evaluation.json`。标签 `initial` 指继续训练前的模型；胡牌率、放铳率、平均净得分及配对区间沿用阶段 6 的定义。`complete.json` 仅在达到目标且评测完成后生成。长程训练进行中不能据此声称优于基线。

手动运行控制器也支持 CPU 回归：

```bash
python -m mahjong_sandbox.rl.longrun --config CPU_CONFIG.json \
  --output PERSISTENT_OUTPUT --initial models/sichuan-ppo-demo/best.zip \
  --additional-steps 512 --no-evaluation
```
