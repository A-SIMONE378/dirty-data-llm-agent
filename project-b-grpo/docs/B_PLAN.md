# 项目 B 准备与实现计划

状态：B1/B2 代码与 CPU 审计已落地；B3 训练题目池已冻结；B4 框架尚未选定。当前 GPU 仍被 A6/A7 占用，B 的 GPU 阶段暂不启动。

## 1. 边界

B 复用 A 的：

- A2 训练任务（只用 train split，test/domain 继续封存）
- A3 三工具沙盒、轨迹 schema、严格 verifier
- A5 SFT adapter 作为 RL 初始策略

B 不重跑 A4 教师采样，不使用 test/domain 作为训练数据，不把 verifier 的 oracle 暴露给策略。

## 2. 已完成/可先做的 CPU 工作

### B1 环境封装

`src/dirty_da/b_env.py` 提供：

- `reset()`：新建一个 `AgentEpisode`，返回问题、公开工具 schema、最大步数、当前步数；不返回 expected answer。
- `step(tool, arguments)`：调用 A3 工具，直到 terminal 或总超时；terminal 时计算 B2 reward。
- `replay(actions)`：同一 task/actions 重放，返回 reward、terminal status、完整 public trajectory。
- 每步有 A3 的 sandbox timeout；B1 另有 rollout 总超时。
- `BEnvironment.from_task_record(...)` 和 `from_frame(...)` 两种构造方式。
- 多个环境对象相互独立，可在线程池中并行。

审计脚本：`scripts/run_b1_environment_audit.py`

- 默认 1,000 次 rollout，可指定 `--parallel`。
- 使用固定 fixture：inspect → run_python → submit。
- 检查 reward 全为 1、terminal 全 accepted、无异常、重放一致、observation 不含 `expected_answer`。
- 结果写 `artifacts/b1_environment_audit.json`。

### B2 奖励设计

`src/dirty_da/b_reward.py`：

- outcome：terminal accepted = 1，否则 0。
- format：accepted 且 submit_answer 结构合法时给很小 bonus（默认 0.05）。
- efficiency：可选 `-step_penalty * steps`（默认 0）。
- anti-cheat：以下任一命中，reward 直接归零：
  - 没有成功读取 `df`
  - 提交值不被工具输出支持
  - private field / 非法 trajectory
  - 代码里出现 `pd.DataFrame(` / `pd.read_csv(` 等硬编码表构造
- `RewardBreakdown` 保存 outcome/format/efficiency/anti_cheat/reasons，便于 B6 统计 reward hacking。

测试：`tests/test_b_reward.py`，覆盖：

- 正确数据框轨迹 = 1.0
- 未读 df = 0
- 硬编码 DataFrame = 0
- 答案无工具证据 = 0
- rejected = 0
- private field = 0
- efficiency 惩罚
- reward config 边界

### B3 题目池

`configs/b3_tasks.json` + `scripts/prepare_b3_tasks.py`：

- 从 A2 train tasks 选 800 个，按表均衡，确定性哈希。
- 每题计划 8 次采样。
- pass-rate 过滤区间 0.01–0.99：组内全 0/全 1 的题没有 GRPO 梯度，后续采样后剔除。
- 课程顺序：single_step → grouped → cleaning_required → judgment。
- 产物：`artifacts/b3_tasks/index.json`、`manifest.json`。
- 这里只冻结题目池；真正的 8 次 SFT-model rollout 需要 GPU（B3 下一步）。

## 3. 需要 GPU 的下一步

### B3 rollout 过滤

用 A5 SFT adapter 对 800 个训练题各采样 8 次：

- 记录 pass@1、pass@k、状态分布、步数。
- 只保留 `0 < pass_rate < 1` 的题，形成 GRPO 训练集。
- 800×8=6,400 条 rollout；一张 48GB 卡可跑，但需要先确认当前 A 队列结束。
- 采样可以用 HF 顺序 rollout；若要快，后续再评估 vLLM 批量。

### B4 GRPO 训练

当前环境 **没有安装 trl / verl / vllm**。不要边跑 A 边装，避免污染 A 环境。

框架决策顺序：

1. **TRL**：先做 API/多轮工具调用支持 spike；如果 GRPOTrainer 不能稳定处理每步 tool observation 和 token mask，则不用硬套。
2. **verl**：多轮 agent rollout 支持更完整，但安装/配置成本高；适合正式训练。
3. **自写 rollout + GRPO loss**：最可控，代码量最大；适合 1.7B、单卡 48GB、先跑通小规模。
4. vLLM 只作为采样加速，不作为训练框架。

训练协议（暂定）：

- 从 A5 1.7B SFT adapter 初始化。
- G=8（先），可扩 4/16。
- 整条多步轨迹只对策略生成的 token 算 logprob；tool/system/user mask。
- outcome reward 为主；format/efficiency 作为消融。
- KL 系数先做 0 和小值两组。
- 先小规模 50–100 题、1 个 epoch，检查奖励曲线和显存，再放大。
- 与“基座直接 GRPO”“只 SFT”对比；test/domain 仍只用于最终评测。

## 4. B5/B6 需要记录的指标

- reward 曲线、pass@1、pass@8、平均步数、回复长度、策略熵
- KL 系数、组大小 G、课程顺序
- 被 anti-cheat 规则拦截的轨迹数
- 至少 3 个真实 reward hacking/失败案例：硬编码、没读表猜答案、格式钻空子、重复提交、超步数

## 5. 当前建议顺序

1. 让 A6/A7 跑完并归档结果；不要同时启动 B4 训练。
2. 跑 B1 1,000 次 rollout 审计；这是纯 CPU/沙盒，不影响 GPU。
3. 用 B2 的人工作弊轨迹测试 reward；全部应归零。
4. A 的 GPU 空出来后，先跑 B3 6,400 次 SFT rollout，做 pass-rate 过滤。
5. 再做 B4 框架 spike；先 1.7B、G=8、小规模。
6. 不要在 A 还在跑时安装/升级 TRL、verl、vLLM 或 CUDA 依赖，避免破坏 A 的环境。

## 6. 已知风险

- 多轮工具调用的 GRPO 框架支持不确定，B4 是最大工程风险。
- 奖励若只用 outcome，模型可能学会乱提交/猜答案；必须保留 anti-cheat gate。
- 题组 pass rate 全 0/全 1 时无梯度；必须过滤。
- 用 SFT adapter 采样训练题可能产生与 A4 教师分布相近的轨迹；B 的增量要看相对 SFT 的 pass@1/pass@8。
- test/domain 必须继续封存；B3 只能使用 train tasks。

## 7. 2026-10-27 CPU 准备结果

- B1：1,000/1,000 rollouts 完成，parallel=4，全部 reward=1、terminal=accepted，重放 reward 一致，0 failure；`artifacts/b1_environment_audit.json` 为 PASS。
- B2：13 个测试通过；未读 df、硬编码 `pd.DataFrame`、无工具证据、private field 的轨迹 reward 均为 0；efficiency 惩罚测试通过。
- B3：800 个 train 任务、28 张表，`artifacts/b3_tasks/index.json` 和 manifest 已生成；尚未用 SFT 模型做 8 次 rollout 过滤。
- B4：当前环境未安装 trl / verl / vllm；等 A 队列结束再做框架 spike，避免污染 A 的环境。

## 8. 框架调研（2026-10-27）

- **TRL v1.16 文档**已经有 agent training、tools、`environment_factory`、`max_tool_calling_iterations` 和 environment-owned `get_reward`。但 `environment_factory` 要求 `transformers>=5.2.0`；当前 A5 环境是 `transformers 4.57.6`。不能为了 B 升级 A 的环境。
- 可行路线 A：为 B 建独立环境，升级到 `transformers>=5.2.0` + 新版 TRL，把 B1 的 reset/step 语义适配成 TRL environment（public methods + reset + get_reward）。优点：GRPO/vLLM/continuous batching 已有；风险：版本升级和多轮工具模板要重新验证。
- 可行路线 B：保持 `transformers 4.57.6`，用 HF 自写多轮 rollout + GRPO loss；最可控，但代码量最大，需要自己实现 logprob、mask、KL、组内 advantage。
- **verl** 文档明确有 Agentic RL、multi-turn、agent loop、reward loop、rollout correction；适合正式多轮 RL，但安装和配置比 TRL 重。若 TRL 的 environment 适配不顺，verl 是下一候选。
- vLLM 只用于采样加速；如果 TRL environment 路线走通，可用 `use_vllm=True`；不要在 A 运行时安装。
- 推荐顺序：A 结束后，先在**独立 venv** 做 50 题、G=8 的 TRL environment_factory spike；失败再评估 verl；最后才考虑自写 GRPO。
- 关键验收：同一轨迹重放得分一致；工具调用不超过 `max_tool_calling_iterations`；只对 assistant token 算 loss；reward 函数能拿到 B1 的 terminal reward；一组 G=8 内 reward 有差异（否则过滤该题）。

## 9. B3 rollout 实现（2026-10-27）

已新增：

- `configs/b3_rollout.json`：1.7B SFT、8 samples/task、temperature 0.8、top_p 0.95、max_steps 10。
- `scripts/run_b3_rollouts.py`：加载 A5 SFT adapter，对 B3 train tasks 逐题采样，保存 results + public trajectories；支持 `--shard-index` / `--shard-count` / `--resume`。
- `scripts/merge_b3_rollouts.py`：合并 shard，计算每题 accepted pass rate，保留 `0 < pass_rate < 1` 的题，输出 `tasks.json`、`filtered_tasks.json`、`summary.json`。

GPU 运行方式（一张 48GB 4090，两进程）：

```bash
# 进程 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_b3_rollouts.py \
  --shard-index 0 --shard-count 2 --resume \
  --model /path/to/Qwen3-1.7B

# 进程 1
CUDA_VISIBLE_DEVICES=0 python scripts/run_b3_rollouts.py \
  --shard-index 1 --shard-count 2 --resume \
  --model /path/to/Qwen3-1.7B

# 合并 + 过滤
python scripts/merge_b3_rollouts.py --shards 2
```

注意：两个进程共用同一张 48GB 卡时都写 `CUDA_VISIBLE_DEVICES=0`，各自输出到
`artifacts/b3_rollouts/shard0|1`；模型和 adapter 很小，显存足够，GPU 主要用在生成。

代价：800 题 × 8 = 6,400 rollouts。一张 48GB 卡、两个进程预计约 8–12 小时；
两张卡分片可约 4–6 小时。B3 只依赖 A5 1.7B adapter + base 1.7B，不需要 14B。

当前 SFT host `37670` 已不可连接，B3 尚未实际启动；需要一张 48GB GPU 才能跑。

