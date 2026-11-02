# 项目B报告

## 在 SFT 之上做多轮 Agent RL：环境、奖励、数据过滤与 GRPO 负结果

> 文档状态：2026-10-29。B1–B4 已完成；B4 为负结果；B5/B6 未启动，B 分支关闭。
> 交叉引用：项目A报告（2026-10-29）已同步 B4 负结果；两份报告使用同一批 A6 120 题 test/domain，SFT 基线为 86/120（71.7%）。
> 适用读者：懂 SFT、Agent rollout、多轮工具调用、GRPO、沙盒评测和 GPU 训练工程的技术人员。
> 结论先行：B3 用 6,400 次 SFT rollout 从 800 个训练题中筛出 336 个“有梯度”的题；B4 用 TRL GRPO 训练 Qwen3-1.7B，但 13-step、42-step、336-step 三种规模都没有超过 SFT。SFT 在 A6 120 题 test/domain 上是 86/120（71.7%），B4 最好的是 42-step adapter 的 81/120（67.5%）。因此 B4 不是“RL 还没调好”的正面结果，而是一个有对照、有失败分析的负结果；继续做 B5 没有依据。

## 当前最终状态（2026-10-29）

### B1

- 1,000/1,000 rollout 完成，parallel=4
- 全部 reward=1、terminal=accepted
- replay 一致；0 failure；`b1_acceptance_pass=true`
- 审计文件：`artifacts/b1_environment_audit.json`

### B2

- reward = outcome + 可选 format bonus + 可选 efficiency penalty；anti-cheat 命中直接归零
- 13 个 reward 测试通过
- 未读 `df`、硬编码 `pd.DataFrame`、答案无工具证据、private field 泄漏均返回 0

### B3

- 800 个 train 任务、28 张表、每题 8 次 SFT rollout，共 6,400 次
- accepted 4,177（**65.3%**）
- 130 题 0/8、334 题 8/8；保留 **336 题**
- 保留题平均 pass rate **56.0%**
- `b3_filter_pass=true`
- 产物：`artifacts/b3_rollouts/{tasks.json,filtered_tasks.json,summary.json,shard0..5}`

### B4

| 实验 | 训练规模 | train loss | 训练时间 | A6 120题 | 结果 |
|---|---:|---:|---:|---:|---|
| train100 | 13 step, G=8 | 0.0397 | 423s | 73/120 = 60.8% | 低于 SFT |
| full1 | 42 step, G=8，epoch=0.125 | 0.05142 | 1,876.76s | 81/120 = 67.5% | 低于 SFT |
| full2 | 336 step, G=8，1 epoch，format reward | 0.05454 | 17,265.78s | 80/120 = 66.7% | 低于 SFT |
| full3 | 336 step, G=8，1 epoch，outcome-only | 0.05022 | 15,154.33s | 79/120 = 65.8% | 低于 SFT |

- SFT 基线：86/120 = **71.7%**
- B4 最好的 adapter 仍比 SFT 低 **4.2 个百分点**
- 336-step 没有比 42-step 更好；format reward 和 outcome-only 都没有救回来
- B4 full2/full3 adapter、评测结果已备份到本地归档

### 未完成/已关闭

- B5：reward/课程/规模正式对比，未启动
- B6：reward hacking 统计、最终 RL 报告，未启动
- 没有 B4 的 reward 曲线、熵曲线、anti-cheat 拦截统计和人工 review
- 没有第二 seed；没有完整 benchmark；没有把 B4 放到 4B 上重复

---

## B 项目报告（简历版）

### 一句话

在项目 A 的 3,133 条 SFT 数据和 1.7B SFT adapter 之上，构建了可重放的多轮工具 Agent 环境、带 anti-cheat 的可验证 reward 和 6,400 次 rollout 的 GRPO 数据过滤流程；用 TRL 完成 13/42/336-step 的 GRPO 对照实验，发现 RL 在 120 题 held-out test/domain 上没有超过 SFT（最高 67.5% vs 71.7%），定位到工具格式、terminal 语义、长轨迹行为漂移和训练/评测环境差异后，主动停止 B5。

### 最终结果：从多少到多少

“accuracy”均指严格 verifier 的 accepted accuracy；B4 与 SFT 使用同一批 120 条 A6 test/domain 题。

| 设置 | accepted | accuracy | 相对 SFT |
|---|---:|---:|---:|
| A5 SFT 1.7B | 86/120 | **71.7%** | 基线 |
| B4 train100，13 step | 73/120 | 60.8% | -10.9pp |
| B4 full1，42 step | 81/120 | **67.5%** | -4.2pp |
| B4 full2，336 step，format reward | 80/120 | 66.7% | -5.0pp |
| B4 full3，336 step，outcome-only | 79/120 | 65.8% | -5.9pp |

分 split：

| 设置 | test（80题） | domain（40题） |
|---|---:|---:|
| SFT | 54/80 = 67.5% | 32/40 = 80.0% |
| train100 | 44/80 = 55.0% | 29/40 = 72.5% |
| full1 | 51/80 = 63.7% | 30/40 = 75.0% |
| full2 | 51/80 = 63.7% | 29/40 = 72.5% |
| full3 | 49/80 = 61.3% | 30/40 = 75.0% |

### 这些数字说明什么

1. **B3 的数据过滤是有效的工程结果，但不是“数据越多越好”。** 800 题中只有 336 题同时满足“有成功样本”和“有失败样本”，适合 GRPO 组内比较；130 题全失败、334 题全成功，没有组内梯度。保留题的平均 pass rate 从 65.3% 降到 56.0%，说明过滤确实把训练集中到了“刚好不会”的题上。
2. **B4 的 42-step 比 13-step 好，但 336-step 没有继续变好。** 13-step 60.8%，42-step 67.5%，336-step 66.7%/65.8%。这不是“再多训一点就超过 SFT”的曲线；在 1 epoch 处已经出现平台/退化。
3. **RL 没有超过 SFT。** 最好的 B4 仍低 4.2pp；336-step 反而略低于 42-step。仅凭训练 loss 下降不能说明策略更好；B4 的 train loss 在 0.050–0.055 附近，和评测结果没有正相关。
4. **format reward 不是关键。** full2 默认 format bonus=0.05，full3 设为 0；两者结果 66.7% vs 65.8%，差异只有 1 题。问题不在“有没有给格式奖励”。
5. **RL 改变了行为，但没有把行为变成更高的正确率。** B4 平均步骤 4.38–4.46，SFT 只有 2.86；B4 的 `run_python` 调用 307–314 次/120题，SFT 只有 121 次；B4 产生 203–225 次 MemoryError，SFT 为 0。B4 更愿意多写代码、多跑工具，但更多动作没有转化为更高的 accepted。
6. **B4 的格式错误反而更少。** SFT 有 18 次 invalid_action，B4 只有 1–2 次；但 B4 的 rejected 19–22、max_steps 16，说明“会调用工具”不等于“会做对题”。这是一个典型的行为漂移：格式变好，正确率没变好。

### 项目展现了什么能力

| 能力 | 证据 |
|---|---|
| Agent RL 环境工程 | 把 A3 三工具环境适配成可 reset/step/get_reward 的 TRL environment；支持多轮工具调用、独立 episode、reward 计算 |
| 可验证 reward | outcome + format + efficiency + anti-cheat；13 个测试覆盖未读 df、硬编码、无证据、private field、超步 |
| RL 数据工程 | 800 题 ×8 = 6,400 次 rollout；pass-rate 过滤；336 题；确定性与可重放 |
| 训练工程 | 独立 venv（transformers 5.2 / TRL 1.16）；LoRA 继续训练；G=8；bf16；gradient checkpointing；单卡/多卡调度 |
| 评测设计 | 冻结 A6 test/domain；SFT 与 B4 同题对照；分 split 报告；多次训练规模对照 |
| 故障诊断 | Qwen 工具参数格式、terminal 后重复 submit、`max_completion_length`、dirty cache fingerprint、训练/评测环境不一致 |
| 研究判断 | 在 336-step 仍低于 SFT 时停止 B5，而不是继续烧卡；把负结果写成可复现实验 |

### 简历 bullets（可直接改写）

- Built a replayable multi-turn tool-use RL environment on top of a 3,133-example SFT pipeline, with 800 training tasks, 6,400 SFT rollouts, pass-rate filtering to **336** group-comparable tasks, and a deterministic reward with anti-hardcoding/evidence checks.
- Implemented TRL GRPO with environment-owned reward, G=8 group sampling, 4,096-token completion budget, LoRA continuation from a 1.7B SFT adapter, and 13/42/336-step controlled runs.
- Evaluated all RL adapters on the same frozen 120-task test/domain set as SFT: best RL **67.5%** vs SFT **71.7%**; 336-step did not improve over 42-step, and format/outcome reward variants both failed to beat SFT.
- Diagnosed the negative result: 19.4% malformed Qwen tool calls in B3 rollouts, terminal-after-submit loop, 2.5× more tool calls and 200+ MemoryErrors in RL trajectories, plus a transformers 5.2 training / 4.57 evaluation mismatch; stopped B5 instead of claiming an unverified RL gain.
- Produced a fully backed-up experiment trail: configs, adapters, eval manifests, 6-shard rollout results, dirty-table cache, and reproducible commands; all critical artifacts downloaded to the local archive.

### 不能夸大的地方

- B4 没有超过 SFT；不能写“GRPO 提升 agent 能力”。
- B4 full1 只有 42 step、manifest 记录 epoch=0.125，不是完整 1 epoch；full2/full3 才是 1 epoch。
- 评测只有 120 条 A6 test/domain、单 seed；没有完整 benchmark，也没有 4B RL 复现。
- B4 训练在 `project-b` 的 transformers 5.2 下完成，评测在 base 的 4.57.6 下完成；虽然 SFT 与 B4 使用同一评测环境，仍不能排除 tokenizer/chat template 差异。
- B3 的保留题是按一次 SFT rollout 的 pass rate 筛的；B4 训练过程中没有重新做动态组过滤。
- 没有 B4 的 reward 曲线、策略熵、anti-cheat 命中统计和人工 review；不能把 train loss 当作策略质量。
- B4 的 336-step 结果不是“RL 一定无效”的普遍结论；它只说明当前这套环境、reward、数据和 1.7B 设置下无效。

### 面试可讲的 3 个真实案例

1. **B3 中 19.4% 的 rollout 是 Qwen 工具格式错误。** 6,400 条里有 1,244 条 `QwenOutputError`：438 条“arguments 不是 JSON object”，143 条“没有返回工具调用”，以及大量 malformed JSON。不是模型不会做题，而是输出协议不稳定；这也是为什么 B3 的保留题只有 336，且保留题里仍有 35.5% invalid_action。后续如果重开，应先做 constrained decoding 或更严格的工具调用重试，而不是直接加 GRPO 步数。
2. **B4 环境允许 submit 后继续调用工具。** 代码里 `_submitted` 被设置但没有在 `_call` 中强制终止；prompt 只说“submit 后停止”，实际靠 `max_tool_calling_iterations=6` 兜底。smoke 阶段能看到重复 submit/多余工具调用。评测 harness 在 terminal 后停止，所以评测轨迹没有暴露这个问题；训练侧会浪费工具预算，并给组内 reward 增加噪声。正确修法是环境层硬性关闭 episode，而不是继续加 prompt。
3. **`max_completion_length=256` 把工具结果也算进去。** 第一版 smoke 生成被截断，TRL 回滚，reward 全 0；改为 4,096 后才正常工作。另一个环境坑是 `project-b` 的 pyarrow/pandas 与 A2 的 dirty-table fingerprint 不一致，直接在 B4 环境重算会报 `Dirty table fingerprint differs from the A2 record`；最后在 base 环境预生成 336 个 dirty CSV 缓存，B4 只读缓存，避免污染 A 环境。

---

## 0. 执行摘要

项目 B 的目标不是“再做一个 SFT”，而是验证：在项目 A 的 SFT agent 之上，GRPO 能否利用组内成功/失败差异，把 held-out test/domain 的 accepted accuracy 再往上推。

链路：

```text
A5 SFT adapter
  → B3：800 个 train 题 × 8 次 rollout，过滤出 336 个有梯度的题
  → B4：TRL GRPO，G=8，多轮工具环境，13/42/336 step
  → A6 120 题 test/domain 对照评测
  → 结果：B4 最高 67.5%，SFT 71.7%，B4 关闭
```

| 阶段 | 状态 | 关键结果 |
|---|---|---|
| B1 | 完成 | 1,000/1,000 rollout，reward 全 1，replay 一致 |
| B2 | 完成 | reward + anti-cheat；13 个测试通过 |
| B3 | 完成 | 6,400 rollout；336 题保留；pass@1 65.3% → kept 56.0% |
| B4 | 完成，负结果 | 13/42/336 step；最好 67.5% vs SFT 71.7% |
| B5 | 未启动 | 没有继续做 reward/课程/规模正式对比 |
| B6 | 未启动 | 没有 reward hacking 统计和最终 RL 报告 |

真正花时间的不是“把 GRPO 跑起来”，而是：

1. 把 A3 的 reset/step 语义适配成 TRL environment；
2. 把 800 个训练题做成可重放、可过滤的 rollout 集；
3. 处理 Qwen 工具调用格式、terminal 后重复 submit、completion 长度和 dirty-cache fingerprint；
4. 用同一套 120 题评测证明 RL 没有收益，并及时停止。

---

## 1. 目标、边界与复用关系

### 1.1 目标

- 从 A5 的 1.7B SFT adapter 继续训练，不重新做 SFT。
- 只在 A2 的 **train split** 上训练；test/domain 继续封存。
- 用多轮工具调用环境，让模型自己决定 inspect → run_python → submit。
- 用可验证 reward 和 GRPO 组内比较，验证是否能超过 SFT。
- 如果无效，要能给出“为什么无效”的证据，而不是只报一个坏数字。

### 1.2 边界

B 复用 A 的：

- A2 结构化任务、dirty profile、strict verifier；
- A3 的 `inspect_table` / `run_python` / `submit_answer` 三工具和沙盒；
- A5 的 1.7B SFT adapter；
- A6 的 120 题 test/domain 评测协议。

B 不做：

- 不重跑 A4 教师采样；
- 不把 test/domain 当训练数据；
- 不把 expected_answer、task_spec、canonical_fingerprint 暴露给策略；
- 不为 B 升级 A 的 base 环境；
- 不在 A 还在跑时安装 TRL/verl/vLLM，避免污染 A。

---

## 2. B1：环境封装

### 2.1 API

`src/dirty_da/b_env.py` 提供：

- `reset()`：创建新的 `AgentEpisode`，返回问题、公开工具 schema、最大步数、当前步数；不返回 expected answer。
- `step(tool, arguments)`：调用 A3 工具，直到 terminal 或总超时；terminal 时计算 B2 reward。
- `replay(actions)`：同一 task/actions 重放，返回 reward、terminal status、完整 public trajectory。
- 每个环境对象独立；可在进度池/线程池里并行。
- 支持 `from_task_record(...)` 和 `from_frame(...)` 两种构造方式。

### 2.2 审计

`scripts/run_b1_environment_audit.py` 使用固定 fixture：

```text
inspect_table → run_python → submit_answer
```

检查：

1. reward 全为 1；
2. terminal 全 accepted；
3. 无异常；
4. 重放 reward 一致；
5. observation 不含 `expected_answer`。

结果：

```json
{
  "rollouts_completed": 1000,
  "rollouts_requested": 1000,
  "parallel": 4,
  "all_rewards_one": true,
  "all_terminal_accepted": true,
  "replay_consistent": true,
  "failure_count": 0,
  "b1_acceptance_pass": true
}
```

B1 是纯 CPU/沙盒审计，不占 GPU；它证明环境契约在进入 RL 之前是稳定的。

---

## 3. B2：奖励设计

### 3.1 Reward 组成

`src/dirty_da/b_reward.py`：

```text
total = outcome + format_bonus + efficiency
```

- `outcome`：terminal accepted = 1，否则 0。
- `format`：accepted 且 `submit_answer` 结构合法时给很小 bonus，默认 0.05。
- `efficiency`：可选 `-step_penalty * steps`，默认 0。
- `anti_cheat`：命中任一规则，total 直接归零。

anti-cheat 规则：

- 没有成功读取 `df`；
- 提交值不被工具输出支持；
- private field / 非法 trajectory；
- 代码里出现 `pd.DataFrame(` / `pd.read_csv(` 等硬编码表构造。

`RewardBreakdown` 保存 outcome/format/efficiency/anti_cheat/reasons，便于后续统计 reward hacking。

### 3.2 测试

`tests/test_b_reward.py` 覆盖：

- 正确数据框轨迹 = 1.0；
- 未读 df = 0；
- 硬编码 DataFrame = 0；
- 答案无工具证据 = 0；
- rejected = 0；
- private field = 0；
- efficiency 惩罚；
- reward config 边界。

13 个测试通过。B4 的训练 reward 直接使用同一个 `compute_b_reward`，不是另写一套训练 reward。

---

## 4. B3：800 题 rollout 与 336 题过滤

### 4.1 题目池

`configs/b3_tasks.json`：

- 从 A2 train tasks 选 800 个；
- 28 张表，按表均衡，确定性哈希排序；
- 每题 8 次采样；
- pass-rate 过滤区间 `0.01–0.99`；
- 课程顺序：`single_step → grouped → cleaning_required → judgment`。

产物：

- `artifacts/b3_tasks/index.json`
- `artifacts/b3_tasks/manifest.json`

### 4.2 Rollout 配置

`configs/b3_rollout.json`：

| 项 | 值 |
|---|---:|
| 模型 | Qwen3-1.7B + A5 SFT adapter |
| samples/task | 8 |
| temperature / top_p | 0.8 / 0.95 |
| max_new_tokens | 256 |
| max_context | 28,672 |
| max_steps | 10 |
| shard | 6 |

运行方式：

```bash
# 每个 shard 一个进程；机器上按可用 GPU 分配
CUDA_VISIBLE_DEVICES=0 python scripts/run_b3_rollouts.py \
  --shard-index 0 --shard-count 6 --resume \
  --model /root/autodl-tmp/Qwen3-1.7B

python scripts/merge_b3_rollouts.py --shards 6
```

### 4.3 结果

6,400 次 rollout：

| 状态 | 次数 | 占比 |
|---|---:|---:|
| accepted | 4,177 | 65.3% |
| invalid_action | 1,244 | 19.4% |
| rejected | 937 | 14.6% |
| max_steps | 42 | 0.7% |
| **合计** | **6,400** | **100%** |

- 平均步数 2.78；accepted 平均 3.00，invalid_action 平均 1.56，max_steps 固定 10。
- `invalid_action` 全部是 `QwenOutputError`，不是 verifier 拒绝。
- top 错误：
  - 438 次 `Qwen tool arguments must be a JSON object.`
  - 143 次 `Qwen must return exactly one tool call; found 0.`
  - 15 次 `Qwen tool call must contain exactly name and arguments.`
  - 其余是 malformed JSON 前缀（长代码/长答案被截断）。

### 4.4 过滤

| 类别 | 题数 |
|---|---:|
| 0/8，全失败 | 130 |
| 8/8，全成功 | 334 |
| 0 < pass < 1，保留 | **336** |
| 合计 | 800 |

保留题：

- mean pass rate **0.5599**
- pass@8 = 336/336 = 100%
- 全 800 题的 pass@8 = 670/800 = 83.75%
- 难度：cleaning_required 129 / judgment 81 / single_step 65 / grouped 61
- 28 张表都有题；单表 4–18 题

保留题的 terminal 分布：

| 状态 | 次数 | 占比 |
|---|---:|---:|
| accepted | 1,505 | 56.0% |
| invalid_action | 953 | 35.5% |
| rejected | 195 | 7.3% |
| max_steps | 35 | 1.3% |
| **合计** | **2,688** | **100%** |

关键观察：保留题里仍有 **35.5% invalid_action**，远高于全量的 19.4%。这说明“有成功也有失败”的题，失败很多不是算错，而是工具调用格式失败。B4 的 reward 只能看到最终 terminal，无法区分“算错”和“格式坏”；这是后续 GRPO 组内信号噪声的重要来源。

### 4.5 B3 产物

```text
artifacts/b3_rollouts/
├── tasks.json                 # 800 题及每题 terminal 统计
├── filtered_tasks.json        # 336 个保留题
├── summary.json               # b3_filter_pass
├── shard0..5/
│   ├── results.jsonl
│   └── trajectories/
```

---

## 5. B4：框架选择与环境

### 5.1 框架决策

候选：

1. TRL `GRPOTrainer` + `environment_factory`；
2. verl 的 Agentic RL / multi-turn；
3. 自写 rollout + GRPO loss；
4. vLLM 只做采样加速。

最终选 TRL，原因：

- 已有 GRPO、group sampling、多轮 environment 接口；
- `environment_factory` 支持 environment-owned `get_reward`；
- 比 verl 安装/配置轻；
- 比自写 GRPO 少实现 logprob/mask/组内 advantage 的风险。

代价：

- `environment_factory` 要求 `transformers>=5.2.0`；
- A5 的 base 环境是 `transformers 4.57.6`；
- 因此为 B 建独立 venv，不升级 A 的环境。

### 5.2 独立环境

`/root/.venvs/project-b`：

| 包 | 版本 |
|---|---:|
| transformers | 5.2.0 |
| trl | 1.16.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| torch | 2.8.0+cu128 |
| jmespath | 安装用于 tool schema |

base `/root/miniconda3` 保持 transformers 4.57.6，用于 A5/A6 和 B3；B4 训练只用 project-b。

### 5.3 B4 环境

`src/dirty_da/b4_env.py`：

- `B4EnvironmentFactory` 接收 task_ids、runtime_root、reward_config、dirty_cache；
- 只允许 train split；
- `reset(task_id)`：
  - 从 `artifacts/b4_dirty_cache/{task_id}.csv` 读取 dirty table；
  - 构造 `AgentEpisode`，注入 expected_answer，但不向策略暴露；
- `inspect_table` / `run_python` / `submit_answer`：调用 A3 工具；
- `get_reward()`：调用 B2 `compute_b_reward`；
- 训练环境 `max_steps=10`、run timeout 15s、memory 8GB、CPU 120s。

### 5.4 Dirty cache 问题

如果直接在 project-b 里用 A2 记录重算 dirty table，会出现：

```text
Dirty table fingerprint differs from the A2 record
```

原因：

- project-b 安装了 pyarrow，pandas 字符串/类型推断与 A2 生成环境不同；
- fingerprint 对 dtype/字符串处理敏感；
- A2 的 fingerprint 是在 base 环境生成的。

解决：

- 在 base `/root/miniconda3` 运行 `scripts/prepare_b4_dirty_cache.py`；
- 为 336 个保留题各生成一个 dirty CSV + manifest；
- 目录约 46MB；
- B4 只读缓存，不重算 fingerprint。

这不是“绕过校验”，而是把“数据生成环境”和“训练环境”解耦；缓存本身仍由 A2 的 task record 和 corruption profile 生成。

### 5.5 Terminal 语义问题

`B4TaskEnvironment`：

- `submit_answer` 设置 `_submitted=True`；
- 但 `_call` 没有检查 `_submitted` 后直接拒绝；
- prompt 里写“After submit_answer, stop calling tools”；
- 实际靠 `max_tool_calling_iterations=6` 兜底。

结果：

- smoke 阶段能看到重复 submit / terminal 后的多余工具调用；
- 评测 harness 在 terminal 后停止，所以评测轨迹不会体现；
- 训练侧会浪费工具预算，并给组内 reward 增加噪声。

### 5.6 Completion 长度问题

第一版 smoke 使用 `max_completion_length=256`：

- 这个预算包含 tool result token；
- 生成很快被截断，TRL 回滚；
- 表现为只调用 1 次工具、reward 0。

改为 `max_completion_length=4096` 后才正常。

### 5.7 训练配置

共同配置：

| 项 | 值 |
|---|---:|
| 初始策略 | A5 SFT adapter |
| G | 8 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 8 |
| max_completion_length | 4096 |
| max_tool_calling_iterations | 6 |
| temperature / top_p | 0.8 / 0.95 |
| learning_rate | 1e-6 |
| beta（KL） | 0 |
| precision | bf16 |
| gradient checkpointing | true |
| seed | 20261027 |
| vLLM | off |

训练任务池：B3 的 336 个保留题；不碰 test/domain。

---

## 6. B4 实验与结果

### 6.1 训练规模

| 实验 | task_limit | max_steps | epoch（manifest） | train_loss | runtime |
|---|---:|---:|---:|---:|---:|
| train100 | 100 | 13 | 未记录 | 0.0397 | 423s |
| full1 | 336 | 42 | 0.125 | 0.05142 | 1,876.76s |
| full2 | 336 | 336 | 1.0 | 0.05454 | 17,265.78s |
| full3 | 336 | 336 | 1.0 | 0.05022 | 15,154.33s |

说明：

- full1 只有 42 step，manifest 记录 epoch=0.125；不能当作完整 1 epoch。
- full2/full3 是 1 epoch；full2 使用默认 format bonus=0.05，full3 使用 format bonus=0。
- 训练 loss 不是评测指标；full1/full2/full3 的 loss 差异不能直接解释 accuracy。

### 6.2 评测结果

同一批 120 条 A6 test/domain：

| 模型 | accepted | accuracy | test | domain |
|---|---:|---:|---:|---:|
| SFT | 86/120 | **71.7%** | 54/80 = 67.5% | 32/40 = 80.0% |
| B4 train100 | 73/120 | 60.8% | 44/80 = 55.0% | 29/40 = 72.5% |
| B4 full1 | 81/120 | **67.5%** | 51/80 = 63.7% | 30/40 = 75.0% |
| B4 full2 | 80/120 | 66.7% | 51/80 = 63.7% | 29/40 = 72.5% |
| B4 full3 | 79/120 | 65.8% | 49/80 = 61.3% | 30/40 = 75.0% |

### 6.3 终止状态

| 模型 | accepted | rejected | max_steps | sandbox_error | invalid_action |
|---|---:|---:|---:|---:|---:|
| SFT | 86 | 16 | 0 | 0 | 18 |
| train100 | 73 | 19 | 17 | 2 | 9 |
| full1 | 81 | 19 | 16 | 3 | 1 |
| full2 | 80 | 19 | 16 | 3 | 2 |
| full3 | 79 | 22 | 16 | 2 | 1 |

解释：

- B4 的 `invalid_action` 明显少于 SFT：格式更稳定；
- 但 `max_steps` 从 0 升到 16，`rejected` 也更多；
- B4 把“格式问题”换成了“跑太多步/算错”的问题，净正确率反而下降。

### 6.4 行为统计

从 120 条评测轨迹统计：

| 模型 | 平均步数 | run_python 调用 | MemoryError 次数 |
|---|---:|---:|---:|
| SFT | 2.86 | 121 | 0 |
| train100 | 4.54 | 333 | 225 |
| full1 | 4.42 | 310 | 209 |
| full2 | 4.38 | 307 | 203 |
| full3 | 4.46 | 314 | 214 |

B4 的策略明显更“爱写代码”：

- 平均多 1.5 步；
- `run_python` 调用约 2.5 倍；
- 出现 200+ 次 MemoryError；
- 但没有换来更高 accepted。

这说明 RL 优化了“继续尝试/继续调用工具”的行为，却没有优化到“最终答案正确”。在有限的 120 题上，这是负迁移。

---

## 7. 为什么没有超过 SFT：失败分析

### 7.1 训练数据太少且太难

336 题是“SFT 有成功也有失败”的题，适合组内比较，但绝对数量小；每题 8 个样本，组内奖励差异有限。full2/full3 各 1 epoch 后没有提升，说明这 336 题不足以支撑稳定的策略改进。

### 7.2 工具格式噪声

B3 全量 19.4% invalid_action；保留题 35.5%。这些失败大多不是“算错”，而是：

- arguments 不是 JSON object；
- 没有返回工具调用；
- 返回多个工具调用；
- JSON 被长代码/长答案截断。

在 GRPO 里，这些轨迹会被压到低 reward，但同组里可能没有“格式正确且算对”的对照，梯度信号不稳定。

### 7.3 Terminal 后重复 submit

训练环境没有硬性关闭 terminal；只靠 prompt 和 6 次工具上限。模型可能：

- 已经 submit accepted，还继续 submit；
- 浪费工具预算；
- 让组内轨迹长度差异变大。

这是环境语义问题，不是模型“不听话”；应在环境层强制终止。

### 7.4 长轨迹行为漂移

B4 平均 4.4 步、300+ 次 run_python、200+ MemoryError；SFT 只有 2.86 步、121 次 run_python、0 MemoryError。RL 让模型更愿意尝试，但在当前 sandbox/工具预算下，更多尝试没有带来更多正确。

可能原因：

- reward 只看 terminal，不看过程质量；
- 没有 step penalty（full2/full3 的 step_penalty=0）；
- 组内 reward 稀疏；
- 336 题覆盖不足，模型学到“多跑工具”而不是“更准”。

### 7.5 train/eval 环境不一致

- B4 训练：project-b，transformers 5.2.0；
- B4 评测：base，transformers 4.57.6；
- tokenizer/chat template/生成行为可能有细微差异；
- SFT 和 B4 使用同一评测环境，所以相对比较仍有效；但不能排除“训练环境不匹配”进一步压低了 B4。

这也是为什么不能把 B4 的负结果直接推广成“GRPO 对 agent 无效”；它只说明当前设置无效。

### 7.6 SFT 已经很强

SFT 在 120 题上是 71.7%，不是 17.5% 的 base；RL 要提升的是一个已经能完成大部分任务的策略。336 题、单 seed、1 epoch、outcome-only/format reward，很难在这个基础上提供稳定增益。

---

## 8. 决策：为什么停止 B5

B4 的事实：

- 13-step：60.8%；
- 42-step：67.5%；
- 336-step：66.7% / 65.8%；
- 全部低于 SFT 71.7%。

如果继续 B5，需要先解决：

1. terminal 语义；
2. 工具格式/constrained decoding；
3. 动态组过滤；
4. reward shaping/step penalty；
5. 训练/评测环境一致；
6. 更大训练池和多 seed。

这些都不是“再加一张卡跑 1 epoch”能解决的。继续投入的期望收益低，因此选择：

```text
B4 作为负结果关闭；B5/B6 不启动；不再为 B 租卡。
```

这不是“项目失败”，而是“用一个有对照的实验否决了一条路线”。

---

## 9. 如果未来重开 B：最小实验顺序

如果以后还有 GPU 和时间，建议按下面顺序，不要直接重跑 336-step：

1. **修 terminal 语义**：submit 后环境立即关闭；禁止继续工具调用；评测/训练一致。
2. **修工具格式**：对 Qwen 工具调用做 constrained decoding 或严格重试；先让 invalid_action 降到 <5%。
3. **做 50 题 pilot**：G=8，检查组内 reward 方差、平均步数、MemoryError；不通过就不放大。
4. **动态组过滤**：每轮重新剔除全 0/全 1 组，避免无效梯度。
5. **reward 消融**：outcome-only、+format、+step penalty；先小规模，不要一次上 336 题。
6. **环境一致**：训练和评测用同一 transformers/tokenizer 版本，或至少在两个版本都评测。
7. **多 seed**：至少 2–3 个 seed；用 120 题 test/domain 做最终对照，不用训练题选模型。
8. **再考虑 4B**：1.7B 容量可能不足；但 4B 之前先把 1.7B 的负结果修成可复现的基线。

---

## 10. 可复现实验与文件路径

### 10.1 B1/B2

```bash
python scripts/run_b1_environment_audit.py --rollouts 1000 --parallel 4
python -m pytest tests/test_b_reward.py
```

产物：

- `artifacts/b1_environment_audit.json`
- `src/dirty_da/b_env.py`
- `src/dirty_da/b_reward.py`

### 10.2 B3

```bash
python scripts/prepare_b3_tasks.py
CUDA_VISIBLE_DEVICES=0 python scripts/run_b3_rollouts.py \
  --shard-index 0 --shard-count 6 --resume \
  --model /root/autodl-tmp/Qwen3-1.7B
python scripts/merge_b3_rollouts.py --shards 6
```

产物：

- `configs/b3_tasks.json`
- `configs/b3_rollout.json`
- `artifacts/b3_tasks/`
- `artifacts/b3_rollouts/`

### 10.3 B4 训练

```bash
# 1) 在 base 环境生成 dirty cache（只需一次）
/root/miniconda3/bin/python scripts/prepare_b4_dirty_cache.py

# 2) 在 project-b 训练
/root/.venvs/project-b/bin/python scripts/run_b4_grpo.py \
  --config configs/b4_grpo_full2.json \
  --model /root/autodl-tmp/Qwen3-1.7B \
  --adapter /root/project-a/artifacts/a5_sft/qwen3_1_7b_main_14b_v1/final_adapter
```

### 10.4 B4 评测

```bash
/root/miniconda3/bin/python scripts/run_a3_qwen_acceptance.py \
  --config configs/a6_eval.json \
  --model /root/autodl-tmp/Qwen3-1.7B \
  --adapter artifacts/b4_grpo/qwen3_1_7b_full2/final_adapter \
  --output-dir artifacts/a6_eval/qwen3_1_7b_rl_full2_test_domain_v1 \
  --runtime-root .a6_runtime_rl_full2
```

### 10.5 关键文件

| 路径 | 内容 |
|---|---|
| `src/dirty_da/b_env.py` | B1 reset/step/replay 环境 |
| `src/dirty_da/b_reward.py` | B2 reward + anti-cheat |
| `src/dirty_da/b4_env.py` | B4 TRL environment adapter |
| `scripts/run_b1_environment_audit.py` | B1 1,000 次审计 |
| `scripts/run_b3_rollouts.py` | B3 rollout |
| `scripts/merge_b3_rollouts.py` | B3 合并/过滤 |
| `scripts/prepare_b4_dirty_cache.py` | 336 个 dirty CSV 缓存 |
| `scripts/run_b4_grpo.py` | TRL GRPO 训练 |
| `configs/b4_grpo_full1.json` | 42-step |
| `configs/b4_grpo_full2.json` | 336-step，format reward |
| `configs/b4_grpo_full3.json` | 336-step，outcome-only |
| `artifacts/b3_rollouts/` | 6,400 rollout + 336 题 |
| `artifacts/b4_grpo/` | B4 adapters + manifests |
| `artifacts/a6_eval/qwen3_1_7b_rl_*` | B4 120 题评测 |

本地归档：

```text
project-a/
├── artifacts\b3_rollouts\
├── artifacts\b4_grpo\
├── artifacts\a6_eval\qwen3_1_7b_rl_*\
└── docs\B_PLAN.md
```

---

## 11. 限制

1. **单模型**：只做了 Qwen3-1.7B，没有 4B RL。
2. **单 seed**：B3/B4 都是一次采样/训练，没有 seed 方差。
3. **评测小**：120 条 A6 test/domain，不是完整 benchmark。
4. **B4 full1 不是完整 epoch**：42 step、epoch=0.125；不能和 full2/full3 的 1 epoch 直接等同。
5. **训练/评测环境不同**：训练 transformers 5.2，评测 4.57.6；相对比较有效，但不是完全同环境。
6. **无 reward 曲线**：只记录了 train loss 和最终评测；没有策略熵、组内 reward 方差、anti-cheat 命中数。
7. **无人工 review**：B4 评测是自动 verifier；没有对失败轨迹做人工分类。
8. **B3 过滤只做一次**：没有在 B4 训练中动态重过滤全 0/全 1 组。
9. **没有 B5/B6**：没有 reward hacking 统计、课程消融、规模消融、最终 RL 报告。

这些限制决定了 B4 的正确表述是：

> 在当前环境、reward、数据和 1.7B 设置下，GRPO 没有超过 SFT；不能推广为“GRPO 对所有 agent 任务无效”。

---

## 12. 最终结论（B1–B4）

- B1：环境契约稳定；1,000/1,000 rollout 全 accepted，replay 一致。
- B2：reward/anti-cheat 可测试；13 个测试通过。
- B3：800 题 ×8 = 6,400 rollout；保留 336 题；pass@1 65.3%，保留题 56.0%。
- B4：13/42/336-step 全部低于 SFT；最好 67.5% vs 71.7%；336-step 没有继续提升。
- 负结果原因：工具格式噪声、terminal 后重复 submit、长轨迹行为漂移、MemoryError 增多、336 题数据不足、训练/评测环境不一致。
- 决策：B4 作为负结果关闭；B5/B6 不启动；不再为 B 租卡。
- 可以写进简历的版本：**“完成 SFT agent 的 RL 扩展评估；GRPO 未超过 SFT，定位失败原因并停止 B5。”** 不要写“RL 提升”。

---

## 13. 附录：关键数字速查

### B3

- 800 题，6,400 rollout
- accepted 4,177 / 65.3%
- invalid_action 1,244 / 19.4%
- rejected 937 / 14.6%
- max_steps 42 / 0.7%
- 0/8：130 题；8/8：334 题；保留：336 题
- kept accepted 1,505 / 56.0%
- kept invalid_action 953 / 35.5%
- kept pass@8 336/336
- all pass@8 670/800 = 83.75%

### B4

- SFT：86/120 = 71.7%
- train100：73/120 = 60.8%
- full1：81/120 = 67.5%
- full2：80/120 = 66.7%
- full3：79/120 = 65.8%
- 最好 B4 与 SFT 差：-4.2pp
- full1 42 step，epoch=0.125
- full2/full3 336 step，1 epoch
- full2 train_loss 0.05454，runtime 17,265.78s
- full3 train_loss 0.05022，runtime 15,154.33s
- B4 平均步数 4.38–4.46，SFT 2.86
- B4 run_python 307–314，SFT 121
- B4 MemoryError 203–225，SFT 0
- B4 max_steps 16，SFT 0
- B4 invalid_action 1–2，SFT 18

### 结论

```text
B1/B2/B3 完成。
B4 完成，但为负结果。
B5/B6 未启动。
B 分支关闭。
```
