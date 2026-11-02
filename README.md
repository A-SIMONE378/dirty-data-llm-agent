# Verifiable Dirty-Data LLM Agent: SFT and GRPO

本仓库整理了两个相互衔接、独立呈现的个人项目：项目 A 构建可验证的脏数据分析 Agent 并完成 LoRA 监督微调；项目 B 在同一工具环境和评测口径上继续开展 GRPO 训练，通过负结果分析确定方案边界。

仓库只保留代码、配置、测试、实验摘要和项目报告。原始数据、模型权重、LoRA Adapter、完整轨迹和 GPU 缓存因体积、许可与测试集泄漏风险未上传。

## 项目概览

### Project A — 面向脏数据分析的可验证 LLM Agent 构建与监督微调

- 将 45 张真实 UCI 表格转化为 4,360 道结构化任务，覆盖 6 类可复现数据污染。
- 实现 Canonical Data、Task Spec、Programmatic Executor、Reference Cleaner 与严格 Verifier，保证任务答案可执行、可回放、可审计。
- 构建表格检查、受限 Python 执行与答案提交三类工具；从 6,400 次教师采样中筛选出 3,133 条多轮 SFT 轨迹。
- 在同一组 120 道内部冻结测试与领域泛化任务上，Qwen3-1.7B 严格通过率由 17.5% 提升至 71.7%，Qwen3-4B 由 53.3% 提升至 85.0%。

[查看 Project A 代码与复现说明](project-a-sft/README.md) · [查看完整项目报告](project-a-sft/reports/项目A报告.md)

### Project B — 多轮工具调用 Agent 的 GRPO 训练与失败诊断

- 将 Project A 的三工具 Agent 封装为可重放的多轮强化学习环境，奖励由程序执行结果和严格 Verifier 共同决定。
- 从 800 道训练任务生成 6,400 条候选轨迹，筛选出 336 道具有组内差异、可提供有效学习信号的任务。
- 完成短程与完整 GRPO 对照训练；最佳严格通过率为 67.5%，未超过同一测试集上的 SFT 基线 71.7%。
- 结合轨迹与运行日志定位工具格式、额外动作、内存异常和环境版本差异等候选因素，据此停止继续扩展，保留完整负结果和限制说明。

[查看 Project B 代码与复现说明](project-b-grpo/README.md) · [查看完整项目报告](project-b-grpo/reports/项目B报告.md)

## 代码结构

```text
.
├── project-a-sft/          # 数据治理、任务合成、Agent、教师采样、LoRA SFT 与评测
│   ├── src/dirty_da/
│   ├── tests/
│   ├── scripts/
│   ├── configs/
│   ├── results/
│   ├── docs/
│   └── reports/
├── project-b-grpo/         # GRPO 环境、奖励、Rollout 筛选、训练与失败诊断
│   ├── src/dirty_da/
│   ├── tests/
│   ├── scripts/
│   ├── configs/
│   ├── results/
│   ├── docs/
│   └── reports/
└── docs/
    └── RESUME_PROJECTS.md  # 与简历分类一致的项目描述
```

## 技术路线

```text
Project A
Canonical Tables
  -> Controlled Corruptions
  -> Structured Task Specs
  -> Programmatic Answers
  -> Three-Tool Agent
  -> Teacher Trajectories
  -> Qwen3 LoRA SFT
  -> Strict Verifier Evaluation

Project B
SFT Agent + Same Tools
  -> Group Rollouts
  -> Verifier-Based Rewards
  -> Comparable-Task Filtering
  -> GRPO Training
  -> Same Frozen Evaluation
  -> Failure Diagnosis and Stop Decision
```

## 快速验收 Project A

公开仓库中的核心 CPU 测试不需要模型权重、原始数据包或 GPU：

```bash
cd project-a-sft
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -v \
  tests/test_canonical.py \
  tests/test_corruption.py \
  tests/test_corruption_fixtures.py \
  tests/test_cleaner_fixtures.py \
  tests/test_executor_verifier.py \
  tests/test_a5_data.py \
  tests/test_a5_format.py \
  tests/test_a5_train.py \
  tests/test_a5_eval.py
```

完整测试还会验证 54 张数据表、4,360 道任务和教师轨迹，因此需要先恢复未公开的数据包。缺少这些资产时，不应把 `python -m pytest` 的数据依赖失败解释为代码逻辑失败。

Project B 是建立在 Project A 源码、A2 任务和 A5 SFT Adapter 之上的增量实验。其环境准备、文件覆盖方式和 GPU 复现命令见 [Project B README](project-b-grpo/README.md)。

## 结果口径与限制

- 所有通过率均为项目内部严格 Verifier 接受率，不是公开排行榜成绩。
- A6 使用 80 道测试任务和 40 道领域泛化任务；数据按表隔离，避免同表记录跨训练与测试集合。
- Project B 只覆盖 Qwen3-1.7B、单随机种子和有限任务规模；负结果不能推广为 GRPO 在其他环境中无效。
- 仓库不提供一键下载数据或模型的脚本；复现 GPU 实验需要用户自行准备合法的数据与本地 Qwen3 权重。

