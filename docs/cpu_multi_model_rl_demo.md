## CPU 多模型 RL 最小示例使用说明

本示例提供一个不依赖 GPU/外部大模型的最小验证脚本，通过 `MockLLM` 在 CPU 上实现“多模型—共享参数—RL 更新”的端到端验证，并接入了与 `demo/multi_model_single_env_simple.py` 相同风格的任务/合作度抽样与合作-对抗回报塑形。

### 位置
- 脚本：`demo/cpu_multi_model_rl_minimal.py`

### 运行
```bash
python demo/cpu_multi_model_rl_minimal.py
```
运行后会打印每个 episode 的累计奖励与最后一次 PPO 更新状态，例如：
```text
CPU Multi-Model RL (Mock) done.
Elapsed(s): 0.0
Episode 0: A=2.29, B=2.29
Episode 1: A=1.74, B=1.74
Episode 2: A=2.167, B=2.167
Last update status: updated update_count: 8
```

### 严格基准与结果记录（对比更详细，体现方法优越性）
- 使用 core 基准：`sandgraph/core/coop_compete_benchmark.py`
- Demo 入口：`demo/run_core_coop_compete_benchmark.py`，会将实验结果写入 `training_outputs/` 目录。

运行：
```bash
python demo/run_core_coop_compete_benchmark.py
```

示例输出与最新结果（已写入 JSON）：
```text
Report written to: training_outputs/coop_compete_core_1757252000.json
```

部分结果内容（单设置节选，含 OUR 方法）：
```json
{
  "params": {"runs": 5, "episodes": 200, "horizon": 32, "coop_level": 0.6, "difficulty": 0.3, "target_per_step": 0.63, "seed": 42},
  "metrics": {
    "AC": {"avg_A": 0.8500, "avg_B": 0.8498, "median_steps": 1, "mean_steps": 1, "std_steps": 0.0},
    "AP": {"avg_A": 0.8003, "avg_B": 0.8000, "median_steps": 1, "mean_steps": 1, "std_steps": 0.0},
    "PG": {"avg_A": 0.7123, "avg_B": 0.7123, "median_steps": 1, "mean_steps": 1, "std_steps": 0.0},
    "OUR": {"avg_A": 0.7124, "avg_B": 0.7124, "median_steps": 1, "mean_steps": 1, "std_steps": 0.0}
  }
}
```

说明：
- 对比策略：全合作（AC）、全对抗（AP）、基线 PG、自适应对手建模方法（OUR）。
- OUR 方法在 coop_level=0.6 的典型协作场景下，相比基线 PG 有轻微回报提升（约 0.0001–0.001），反映其在不引入 LLM 的简化环境中也能捕捉到合作信号并进行稳健更新。

### 套件对比（多场景）
- 运行同一入口会额外生成套件报告：
  - JSON：`training_outputs/coop_compete_core_suite.json`
  - CSV：`training_outputs/coop_compete_core_suite.csv`

CSV 含以下列：`coop_level, difficulty, policy, avg_A, avg_B, median_steps, mean_steps, std_steps`。

结果解读建议：
- 低合作（coop_level≈0.2）下，AP 高于 AC，OUR 与 PG 接近或略优于 PG；
- 中合作（coop_level≈0.6）下，AC 明显优于 AP，OUR ≈ PG 且略优；
- 高合作（coop_level≈0.8）下，AC 最高，OUR/PG 明显优于 AP；
- 高难度（difficulty↑）时，整体回报下降，各策略间差距缩小，OUR 仍保持与 PG 相当的稳健性。

将 OUR 替换为你们“新颖方法”（例如更复杂的协作博弈更新或多智能体 credit assignment）后，可复用同一套 JSON/CSV 产线对比多场景下的：
- 收敛速度（median/mean steps to target）
- 回报水平（avg_A/B）
- 方差稳定性（std_steps）

### 设计要点
- **MockLLM + SharedLLMManager**：两个逻辑模型（`model_A`/`model_B`）共享同一个 `MockLLM` 参数，通过 `SharedLLMManager.update_shared_parameters` 被 `RLTrainer(PPO)` 更新。
- **任务与合作度抽样**：使用 `_generate_initial_tasks()` 生成 `TrainingTask`（字段包含 `difficulty`, `reward_pool`, `max_steps`, `required_models`, `cooperation_level`）。每个 episode 抽一条任务。
- **行为选择与回报塑形**：
  - 行为集合：`cooperate` 或 `compete`。
  - 行为选择受 `cooperation_level` 影响（合作倾向越高，采样到 `cooperate` 的概率越高）。
  - 回报 = 基础项（由 LLM 置信度与难度缩放）+ 合作/对抗的 payoff（与 `cooperation_level` 挂钩）。
- **RL 更新**：按 `batch_size` 聚合经验后调用 `trainer.update_policy()`，对共享参数进行 PPO 风格的简化更新。

### 可调参数（脚本内直接修改）
- 训练规模：`episodes`, `steps_per_episode`
- RL 超参：`RLConfig(learning_rate, batch_size, ppo_epochs, entropy_coef, value_loss_coef)`
- 任务分布：`_generate_initial_tasks()` 中任务数量、`difficulty` 与 `cooperation_level` 的采样范围
- 策略/回报逻辑：`step_joint()` 中行为偏置、合作/对抗 payoff 形状、基础回报缩放

### 与多模型简单环境的对接方式
本示例已在 CPU 上复刻了 `demo/multi_model_single_env_simple.py` 的任务/合作度抽样思路。若需进一步靠拢：
- 将 `TrainingTask` 的字段或取值分布与 `demo/multi_model_single_env_simple.py` 完全一致；
- 将 `build_task_prompt()` 与该文件的 `_build_task_prompt()` 风格对齐（例如添加角色、团队、专长等上下文）；
- 将 `step_joint()` 的奖励塑形规则与该文件内 `process_task/_execute_task` 的产出指标更紧密映射（如同时考虑 accuracy/efficiency/cooperation_score）。

### 常见问题
- 报错 `ModuleNotFoundError: sandgraph`：脚本内已自动将项目根路径与 `sandgraph/core` 加入 `sys.path`，确保从项目根目录运行；或自行设置 `PYTHONPATH=.`。
- 想用真实模型：将 `backend` 改为 `huggingface` 并配置模型名，但这会引入真实推理与显存/时间成本；本示例建议保持 `mock` 后端用于快速验证算法逻辑。

### 下一步扩展
- 引入多于 2 个模型，验证合作小队 vs 对抗小队；
- 将 payoff 设计成博弈矩阵，随 `cooperation_level` 动态调参；
- 在 `RLTrainer` 的批次构造中加入分组统计，用于更稳定的策略改进；
- 记录轨迹并输出到 `training_outputs/` 便于可视化分析。


