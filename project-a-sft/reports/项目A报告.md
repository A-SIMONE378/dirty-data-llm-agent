# 项目A报告

## 脏数据数据分析 Agent：可验证任务合成、A4 教师采样与故障排查

> 文档状态：2026-10-29。A0–A7 的实验和消融已完成；A8 失败分析/最终整理未完成。B1–B4 已完成，B4 为负结果并关闭；详见《项目B报告》。A4 已通过 19/19 机械审计和 50 条人工 review；A5/A6/A7 结果见下方项目报告。
> 适用读者：懂 SFT、Agent rollout、沙盒、评测、GPU 推理和后训练数据工程的技术人员。
> 结论先行：Qwen3-1.7B SFT 在 held-out test/domain 从 17.5% 提升到 71.7%；Qwen3-4B SFT 从 53.3% 提升到 85.0%；4B SFT 也已高于 8B base 的 76.7%（但 8B 未做 SFT）。数据量 1000 后接近饱和；success-only 90%、recovery-augmented 80% 与 main 85% 的差异只有 1–2 题，不能过度解读。B 的 GRPO 扩展已在 1.7B 上完成对照：B4 最好 67.5%，低于 SFT 71.7%，B5/B6 未启动。

## 当前最终状态（2026-10-27）

### A4

- 6,400/6,400 attempts，0 model_error
- accepted 3,834（59.9%），retained **3,133（49.0%）**
- 审计 **19/19 PASS**，人工 review **PASS**，`a4_complete=true`
- 类别：pollution_cleaning 1,813 / direct_success 1,310 / error_recovery 10
- 8B 对照：1,590 retained，其中 341 条 error_recovery 可补 14B 类别短板

### A5

- 基座 Qwen3-1.7B；LoRA r=16、all-linear、assistant-only loss、2 epochs、lr 1e-4
- 3,133 条 `main_14b`，4,096 context，392 steps，2,668s（44.5 分钟），train loss 0.07486
- 冻结 40 条 dev：base **4/40 = 10%**，SFT adapter **34/40 = 85%**
- 净提升 +30（要求 ≥ +4），恢复 30 条，回退 1 条，0 runtime integrity error
- adapter：`project-a/artifacts/a5_sft/qwen3_1_7b_main_14b_v1/final_adapter/`

### A6

120 条 test + domain_generalization：

| 模型 | accepted | accuracy |
| --- | ---: | ---: |
| 1.7B base | 21/120 | 17.5% |
| 1.7B SFT | 86/120 | 71.7% |
| 4B base | 64/120 | 53.3% |
| 4B SFT | 102/120 | **85.0%** |
| 8B base | 92/120 | 76.7% |

- 1.7B SFT：test 67.5%，domain 80.0%，net +65，CI [+42.5, +65.0]
- 4B SFT：test 81.2%，domain 92.5%，net +38，CI [+23.3, +40.0]；0 invalid / 0 max_steps
- 8B base：test 57/80（71.3%），domain 35/40（87.5%）；1 invalid、2 max_steps
- 4B SFT 高于 8B base 8.3 个百分点，但 8B 没有 SFT，不能直接宣称“4B 学生全面优于 8B”

### A7

1.7B dev（40 题，base=10%）：

| 训练集 | dev accepted | 净提升 |
| --- | ---: | ---: |
| success_only | 90.0% | +32 |
| main_14b | 85.0% | +30 |
| recovery_augmented | 80.0% | +28 |
| scale_1000 | 80.0% | +28 |
| scale_2000 | 75.0% | +26 |
| scale_500 | 55.0% | +18 |

- 4B main SFT train loss **0.0566**；4B SFT 在 A6 test/domain 为 **85.0%**，比 1.7B SFT 高 13.3 个百分点
- 数据量：500 明显不足；1000 后接近饱和，full 85% 与 1000 的 80% 只差 2 题
- success_only 90% 比 main 85% 高 2 题；recovery_augmented 80% 没有优于 main；40 题 dev 下不能当强结论
- 所有 A7 对比均通过最低 +4 gate；4B 没有单独跑 dev，A6 test/domain 是模型规模对照

下面的第 3–10 节保留 2026-10-25 A4 排查时的原始记录；其中标注的 `PENDING_HUMAN` 等状态是当时的快照，已被本节更新。


---


## A 项目报告（简历版）

### 一句话

从 45 张真实表出发，程序化注入脏数据、合成 4,360 条可验证任务，在受限工具沙盒里采样并过滤教师轨迹，完成 Qwen3-1.7B/4B LoRA SFT，并在 held-out test/domain 表上把 1.7B 从 17.5% 提升到 71.7%、4B 从 53.3% 提升到 85.0%。

### 最终结果：从多少到多少

“accuracy”均指严格 verifier 的 pass@1/accepted accuracy。

| 设置 | 起点 | 终点 | 提升 | 作用 |
|---|---:|---:|---:|---|
| 1.7B dev | 10.0% | **85.0%** | +75.0 | SFT 确实学会“检查/清洗/计算”的轨迹行为 |
| 1.7B test+domain | 17.5% | **71.7%** | +54.2 | 在未见表和 domain 表上仍然成立 |
| 4B test+domain | 53.3% | **85.0%** | +31.7 | 更大基座 + SFT 的组合最好 |
| 4B SFT vs 1.7B SFT | 71.7% | **85.0%** | +13.3 | 模型规模的额外收益 |
| 4B SFT vs 8B base | 76.7% | **85.0%** | +8.3 | SFT 后的 4B 超过未 SFT 的 8B（8B 未 SFT） |
| 8B teacher vs 14B teacher（A4 retained） | 1,590 | **3,133** | +1,543 | 教师能力提升直接扩大到高质量数据规模 |
| A7 data 500 → 1000 | 55.0% | **80.0%** | +25.0 | 小数据明显不足 |
| A7 data 1000 → full 3133 | 80.0% | **85.0%** | +5.0 | 1000 后接近饱和 |
| A7 success-only vs main | 85.0% | **90.0%** | +5.0 | 只有 2 题差异，不能当强结论 |
| A7 recovery-augmented vs main | 85.0% | 80.0% | -5.0 | 加 8B recovery 没有帮助，可能是来源混淆/噪声 |

A6 分项：

- 1.7B SFT：test 67.5%，domain 80.0%
- 4B SFT：test 81.2%，domain 92.5%
- 8B base：test 71.3%，domain 87.5%

### 这些数字说明什么

1. **SFT 是有效的，而且不是只记住训练表。** 1.7B 在 40 条 dev 上从 10% 到 85%，在 120 条 held-out test/domain 上从 17.5% 到 71.7%。dev 与 test 有差距，说明仍有泛化损失，但方向明确。
2. **4B 是明显更好的学生。** 4B SFT 在 test/domain 到 85.0%，比 1.7B SFT 高 13.3 个百分点；test 81.2%、domain 92.5%。
3. **SFT 后的 4B 超过了未 SFT 的 8B base。** 85.0% vs 76.7%。但不能反过来宣称“4B 学生全面优于 8B”，因为 8B 没有做同协议 SFT。
4. **数据量不是无限有用。** 500 题只有 55%；1000 到 2000 在 75–80%；full 3133 到 85%。这说明在 1.7B 上，数据量在约 1000 后开始饱和，后续更应关注数据质量和教师分布，而不是盲目加量。
5. **recovery 轨迹没有在这个 dev 上证明价值。** success-only 90% 比 main 85% 高 2 题，recovery-augmented 80% 反而更低。40 题 dev 的噪声很大，不能把 +5/-5 当成定论；但至少不能说“加 recovery 一定更好”。
6. **4B 的规模收益是真实的。** 4B main 的 train loss 0.0566，低于 1.7B 的 0.0749；A6 test/domain 85.0% 也高于 1.7B 的 71.7%。

### 项目展现了什么能力

| 能力 | 证据 |
|---|---|
| 数据工程 | 45 张真实表、28/4/8/5 split、4,360 条任务、6 种可控破坏、许可/隔离记录 |
| 可验证任务设计 | oracle 表、reference_clean、严格 verifier、dirty replay、A2 机械审计 + 100 题人工 review |
| 沙盒与工具链 | 三工具、断网/超时/限内存/只读 CSV、POSIX rlimit、1,000 次 B1 rollout、无泄漏 |
| Agent 轨迹工程 | 6,400 次教师采样、拒绝采样、证据链、去重、1,590/3,133 retained、8B→14B 教师升级 |
| SFT 训练 | Qwen3-1.7B/4B LoRA、assistant-only loss、4,096 context、bf16、gradient checkpointing、可复现 adapter |
| 评测设计 | dev/test/domain 分离、base/SFT 配对、bootstrap CI、4B/8B 对照、按 split 报告 |
| 消融分析 | 数据量 500/1000/2000/full、success-only、recovery-augmented、1.7B/4B |
| 故障诊断 | RLIMIT_AS、replace 误禁、parser、filter 同步、audit 去重、A5 eval invalid_action |
| 工程与可恢复性 | 双卡 shard、append-only results、resume、migration bundle、SHA256、19/19 audit |
| 研究判断 | 不盲目上 32B；先用 14B 解决质量，再用 4B 看学生规模；不放松过滤器换数量 |

### 简历 bullets（可直接改写）

- Built an end-to-end verifiable agent-data pipeline over **45 real-world tables** and **4,360 structured tasks**, with six programmatic corruption types, oracle answers, a reference cleaner, and a strict executor/verifier.
- Sampled **6,400** teacher attempts with Qwen3-8B/14B, retained **3,133** trajectories through execution-based verification, tool-output evidence checks, anti-hardcoding rules, and global deduplication; passed a **19/19** independent audit and 50-record human review.
- Trained Qwen3-1.7B and Qwen3-4B LoRA SFT with assistant-only loss; improved held-out test/domain pass@1 from **17.5% to 71.7%** (1.7B) and from **53.3% to 85.0%** (4B); the 4B SFT model also reached **85.0%** vs **76.7%** for the untuned 8B base.
- Ran data-scale and trajectory ablations: **55.0%** at 500 examples, **80.0%** at 1000, **85.0%** at full 3,133; success-only **90.0%**, recovery-augmented **80.0%** vs main **85.0%**, showing saturation after ~1k examples and no clear recovery-data gain on the 40-task dev set.
- Diagnosed and fixed sandbox/rollout failures (RLIMIT_AS below pandas baseline, over-broad `replace` ban, un-synced evidence filters, malformed Qwen tool calls, duplicate-blind audit), and built resumable sharded GPU runs with frozen manifests and SHA256-verifiable artifacts.

### 不能夸大的地方

- A5/A7 的 dev 只有 40 题、单 seed；A6 是 120 题的 test/domain 样本，不是完整 benchmark。
- 8B 没有做同协议 SFT；“4B SFT > 8B base”不能写成“4B 学生全面优于 8B”。
- A7 的 success-only/recovery 差异只有 1–2 题，不能当强结论。
- 外部基准、A8 失败分类仍未完成；B 的 GRPO 已完成，但 B4 是负结果（见《项目B报告》），B5/B6 未启动。
- 所有数字都来自本项目 verifier；不是公开 leaderboard 数字。

### 面试可讲的 3 个真实案例

1. **32.8 KiB 的 MemoryError**：`RLIMIT_AS=2GB` 低于 pandas/numpy 初始化后的约 2.7GB 虚拟内存基线；通过 `/proc/self/status` 和最小分配复现，改成 A4 专用 8GB，同时保留 CPU/文件/网络/进程限制。
2. **accepted 但 retained 为 0**：本地 evidence filter 已修，但 GPU 目录没同步；通过比较文件哈希和重跑 16 条轨迹发现，NumPy scalar / record 形状输出被误杀；同步过滤器并加标量证据回退，恢复 accepted 样本。
3. **8B 教师能力上限**：8B 只有 41.3% accepted、24.8% unique retained；不是继续调 prompt 能解决，而是换 14B。14B 到 49.0% retained，再配合 4B 学生到 85.0% test/domain。


## 0. 执行摘要

项目 A 的目标是训练一个小模型，在“脏表格”上完成多步数据分析：先检查、再清洗、再计算，答案由程序验证。整体链路是：

```text
A0 地基：脏数据注入 + oracle + 校验器
A1 选表：45 张真实表，按表划分 train/dev/test/domain
A2 任务合成：4,360 条结构化任务，题目、spec、答案、脏表指纹
A3 Agent 环境：inspect_table / run_python / submit_answer，沙盒执行
A4 教师采样：Qwen3-8B/14B 生成轨迹，拒绝采样、去重、质检、SFT 记录
A5–A7：SFT、test/domain 评测、消融
```

当前进度：

| 阶段 | 状态 | 关键结果 |
|---|---|---|
| A0 | 完成 | 6 种破坏、oracle、参考清洗器、测试 |
| A1 | 完成 | 45 张表：28 train / 4 dev / 8 test / 5 domain；9 张隔离 |
| A2 | 完成 | 4,360 条任务：3,000 train / 320 dev / 640 test / 400 domain；机械审计 + 100 题人工 review 通过 |
| A3 | 完成 | 三工具环境、Linux 沙盒、Qwen3-8B 50 题验收：24 accepted（48%），0 model_error |
| A4-8B | 完成但未达标 | 6,400 attempts；2,643 accepted（41.3%）；1,590 retained（24.8%）；目标 3,000 未达 |
| A4-14B | 完成 | 3,133 retained（49.0%）；audit 19/19 PASS；human review PASS；a4_complete=true |
| A5 | 完成 | 1.7B LoRA：dev 10% → 85%；test/domain 17.5% → 71.7% |
| A6 | 完成 | 4B SFT test/domain 85.0%；8B base 76.7%；4B base 53.3% |
| A7 | 完成 | success-only 90%、main 85%、recovery 80%、scale-500 55% |
| B1–B4 | 完成，负结果 | 336 题保留；B4 最好 67.5% vs SFT 71.7%；B5/B6 未启动 |
| 人工 review | 完成 | A4 50 条 owner review PASS |

这次工作真正花时间的不是“写模型”，而是把 A4 的工程链路从“能跑”修到“数据可信、可复现、可恢复”。中间遇到的主要问题包括：沙盒内存上限低于 pandas 启动基线、禁用 `replace` 误伤 pandas 清洗、质量过滤器未同步、Qwen 工具调用格式不合法、8B 教师能力不足、审计没有按全局去重口径复算、GPU 利用率和迁移管理问题。

---

## 1. 项目 A 的数据与任务契约

### 1.1 A0：脏数据与 oracle

A0 提供六类可程序化注入的破坏，例如缺失标记、哨兵值、类型错乱、重复行、格式不一致、类别拼写/空格污染。关键设计是：

- 破坏是确定性的、记录比例的、可重放的。
- 丢失信息的破坏不可逆，因此标准答案在“理想清洗后的表”上计算，题目明确写出缺失值排除、重复记录只算一次、类别规则等。
- `reference_clean` 负责把可恢复的破坏还原到 oracle；测试验证“直接算脏表”和“先清洗再算”的差距。
- 合成表 demo 中，轻/中/重三档的“不清洗直接算”准确率约 50%/17%/0%，先清洗都是 100%。这个差距是训练要提升的空间。

### 1.2 A1：真实表选择

- 45 张 UCI 表入选，全部记录来源和 CC BY 4.0 许可。
- 按表划分而不是按题划分，避免同一张表的题目泄漏到测试集。
- 28 张 train、4 张 dev、8 张 test、5 张 domain generalization。
- 9 张表被隔离：7 张许可未知的 pilot 表，2 张语义不符合要求的 UCI 表。
- 表级 catalog 记录 transformations、approval status、limitations；`run_a1_selection_audit.py` 会为每张表/每档破坏生成两次脏副本，验证可复现。

### 1.3 A2：结构化任务

- 4,360 条任务，训练 3,000、开发 320、测试 640、domain 400。
- 每条任务有结构化 spec、渲染后的自然语言问题、程序化答案、dirty profile、dirty fingerprint。
- 题目只从 spec 渲染；答案由 executor 在 canonical 表上计算，不由模型生成。
- 每条记录在 low/medium/high 脏表上按显式 cleaning rules 重放。
- A2 通过机械审计和独立 100 题人工 review。

### 1.4 A3：Agent 环境

固定三个工具：

1. `inspect_table`：有界列信息、dtype、缺失数、行数、head。
2. `run_python`：持久 Python 状态，围绕 `df`，有超时、输出截断、导入/文件/网络/进程限制，Linux 上启用 POSIX 资源限制。
3. `submit_answer`：严格 verifier，不返回 oracle 值。

模型只能看到问题、公开工具 schema、工具结果；轨迹不包含 task spec、expected answer、canonical fingerprint。A3 的 50 题 Qwen3-8B 验收结果是 24/50 accepted（48%），0 runtime model_error，轨迹和隐私检查通过。

---

## 2. A4 设计与不变量

### 2.1 采样计划

`configs/a4_teacher_sampling.json` 冻结：

| 项 | 值 |
|---|---:|
| train 表 | 28 |
| 选中任务 | 800（按表均衡、哈希排序） |
| 每题采样 | 8 |
| planned attempts | 6,400 |
| temperature / top_p | 0.8 / 0.95 |
| max_new_tokens | 256 |
| max_steps | 10 |
| invalid_action_retries | 1 |
| retained 目标 | 3,000–8,000 |
| A5 | 完成 | 1.7B LoRA：dev 10% → 85%；test/domain 17.5% → 71.7% |
| A6 | 完成 | 4B SFT test/domain 85.0%；8B base 76.7%；4B base 53.3% |
| A7 | 完成 | success-only 90%、main 85%、recovery 80%、scale-500 55% |
| 人工 review | 完成 | A4 50 条 owner review PASS |

14B 版本 `configs/a4_teacher_sampling_14b.json` 保持任务选择、attempt 计划、温度和工具协议不变，只把 `model_id` 改为 `Qwen/Qwen3-14B`，prompt_version 改为 `a4-qwen-teacher-v3-14b`。

### 2.2 确定性

- 任务选择：`sha256(seed|table)` 排序，表间均衡；`sha256(seed|task_id)` 在表内排序。
- attempt ID：`a4-{task_id}-s{sample_index:02d}`。
- 采样 seed：`sha256(seed|task_id|sample_index)[:8] % 2^31`。
- 14B 两个 shard 按 `attempt_index % 2` 分配；题目、seed、工具、verifier 完全不变。
- 结果 append-only；轨迹先写临时文件再原子替换；`results.jsonl` 每次 flush + fsync。

### 2.3 质量门禁

一条轨迹要 retained，必须同时满足：

1. `terminal_status == accepted`（提交答案通过严格 verifier）。
2. 至少一个成功的 `run_python` 调用读取了 `df`。
3. 提交值能由某个成功工具输出支持（先尝试结构化验证，再做标量 token 证据回退）。
4. 轨迹中没有 `"expected_answer"`、`"task_spec"`、`"canonical_fingerprint"`。
5. 全局轨迹指纹不重复。

轨迹指纹覆盖 question、events（assistant content、tool、arguments、ok、content、error_type）和 submitted answer。也就是说，只有公开轨迹几乎完全相同才会被判重；同一答案但步骤不同不会被误判。

retained 类别：

- `direct_success`：一次做对。
- `pollution_cleaning`：代码里出现 `dropna`、`fillna`、`drop_duplicates`、`replace`、`to_numeric`、`to_datetime`、`astype`、`.str.` 等清洗动作。
- `error_recovery`：前面有工具错误，后续修复并做对。

### 2.4 SFT 记录

`build_sft_record` 生成 chat 记录：

- system/user/tool 消息 `trainable=False`。
- assistant 工具调用消息 `trainable=True`。
- 每条 assistant 消息带一个 function tool call。
- 不写入 oracle 字段；写出前会再检查 forbidden tokens。

---

## 3. 问题、排查、修复：逐项复盘

### 3.1 沙盒 `RLIMIT_AS=2GB` 低于 pandas 启动基线

**现象**

8B pilot 中很多轨迹在 `run_python` 里出现：

```text
MemoryError: Unable to allocate 32.8 KiB for an array with shape (4192,) and data type object
```

看起来像“数据太大”，但 32.8 KiB 根本不该失败。相关轨迹大量变成 `max_steps` 或 `no_successful_dataframe_analysis`，pilot 保留率只有 25%–37.5%。

**排查**

我没有直接调大内存，而是先测进程虚拟内存。远程探针读取 `/proc/self/status`：

```text
VmPeak: 2735324 kB
VmSize: 2735324 kB
VmRSS:    65548 kB
VmData: 2657364 kB
```

在只 import pandas/numpy、还没读表的情况下，虚拟地址空间已经约 2.7GB；而 worker 设置的是：

```python
resource.setrlimit(resource.RLIMIT_AS, (2GB, 2GB))
```

`RLIMIT_AS` 限制的是虚拟地址空间，不是 RSS。pandas/NumPy/OpenBLAS 在启动时预留了大量 arena，2GB 上限低于基线，所以任何新分配都会失败。探针复现了同样的 32.8 KiB 报错。

**修复**

A4 通过 `replace(env_config, max_memory_mb=8192)` 把地址空间上限提高到 8GB，同时保留：

- `RLIMIT_CPU=120s`
- `RLIMIT_FSIZE=0`
- 15s per-call timeout
- import/file/network/process 限制

A3 的冻结配置没有被改写；A4 只在自己的启动路径覆盖资源上限，避免改动 A3 已提交证据。

**验证**

- 同样的 4,600 行 bank marketing 表可以完成 groupby。
- 后续 8B/14B 不再出现此类 32.8 KiB MemoryError。
- 资源限制仍报告 `posix_resource_limits_active=true`。

**为什么不选其他方案**

- 用 `RLIMIT_DATA` 限制堆：pandas/NumPy 的 mmap/arena 不一定计入 DATA，仍可能逃逸。
- 用 cgroups：更正确，但需要宿主机权限/容器配置；当前实例没有可控 cgroup 接口。
- 直接取消内存限制：会失去沙盒边界；8GB + 其他限制是可用折中。

### 3.2 禁用 `replace` 误伤 pandas 清洗

**现象**

模型写正常清洗：

```python
df["STG"] = df["STG"].replace("__MISSING__", pd.NA)
```

沙盒返回：

```text
Attribute 'replace' is not allowed in run_python.
```

模型反复重试同一写法，最终 max_steps。

**排查**

查看 `_sandbox_worker.py` 的 `FORBIDDEN_ATTRIBUTES`，发现 `replace` 与 `open`、`system`、`popen` 等放在一起。原本是想阻止 `os.replace`，但 AST 只按属性名匹配，把 pandas 的 `Series.replace` 和 `str.replace` 一并禁了。

**修复**

从属性黑名单中移除 `replace`，保留文件/进程相关属性；依赖以下边界：

- `open`、`os`、`subprocess`、`socket` 不可用。
- audit hook 仍然限制打开路径。
- `run_python` 没有 import os/subprocess 的能力（后来只放开 pandas/numpy import）。

**验证**

新增测试 `test_run_python_allows_pandas_replace_cleaning`，确认：

```python
df['group'].replace('a', 'z').str.replace('b', 'c')
```

可以执行；`import socket`、`open`、`pd.read_csv('../outside.csv')` 仍被拒绝。

### 3.3 质量过滤器“答案对了但不保留”

**现象**

8B 轨迹 `terminal_status=accepted`，但 `retained=False`，原因是：

```text
submitted_answer_not_supported_by_tool_output
```

例子：

- 工具输出 `{'Link': np.int64(2307)}`，提交 `{"Link": 2307}`。
- 工具输出 `[{'media_exposure': 0, 'wife_age_count': 342}, ...]`，提交 `{"0": 342, "1": 27}`。
- 工具输出 NumPy scalar 的 repr，提交普通 JSON 数字。

**排查**

1. 检查本地 `a4_teacher.py` 与 GPU 运行目录的 `a4_teacher.py` 哈希，发现不一致：本地已经有标量证据回退，GPU 目录仍在用旧版严格 `verify_result`。
2. 对已有 16 条轨迹本地重跑 `assess_teacher_trajectory`，retained 从 4 变为 6；两个 accepted 的误杀被找回。
3. 用真实轨迹测试 NumPy scalar、record 形状、Python mapping key 归一化。

**修复**

- 同步过滤器到 GPU 目录。
- 增加 `_normalize_json_keys`：Python dict 的 int key 按 JSON 语义转字符串。
- 增加标量证据回退：先尝试结构化 `verify_result`；失败则递归展开提交值，要求每个标量都在工具输出中出现。数字用词边界正则，避免 `1` 匹配到 `10`。
- 保留严格性：自由文本、无工具输出支持、硬编码答案仍然拒绝。

**验证**

- 16 条 v3 pilot 的 accepted 全部 retained。
- 8B 最终 accepted 2,643 中，1,590 retained；剩余 1,053 条是 1,018 个完全重复轨迹 + 35 个 unsupported，没有“本来正确但被证据过滤器误杀”的大批样本。

**为什么不直接放宽到“accepted 就 retained”**

那会把“没读数据就猜中”“硬编码答案”“工具调用和答案无关”的轨迹放进 SFT。项目目标是教“检查、清洗、再计算”，不是教猜答案。证据过滤器是数据质量的核心。

### 3.4 prompt 没有说明 `submit_answer` 是独立工具

**现象**

模型在 `run_python` 代码里调用：

```python
submit_answer({"answer": grouped})
```

沙盒返回 `NameError: name 'submit_answer' is not defined`，模型反复重试直到超步。

**排查**

查看完整轨迹和 assistant tool call，确认不是环境 bug，而是模型把 submit 当成 Python 函数。

**修复**

prompt v3 明确写：

- `submit_answer` 是独立工具；
- 必须在 `run_python` 的结果中出现计算值；
- 再用单独的 `submit_answer` 调用提交；
- 不要把 `submit_answer` 放进 Python 代码。

**验证**

后续轨迹中此类 NameError 明显减少；仍保留为失败，不重标为正确。

### 3.5 Qwen 工具调用格式不合法

**现象**

8B 的 invalid_action 一度占 30% 左右。诊断脚本抓取 raw generation，看到三类：

1. 完整 Hermes 格式：`<tool_call>{"name":..., "arguments":...}</tool_call>`，正常。
2. 省略 wrapper，直接输出 `{"name": "submit_answer", "arguments": {"answer": 379}}`。
3. 输出 Python literal：`<tool_call>{"name": "submit_answer", "arguments": {"answer": {4: 1.0}}}</tool_call>` 或 dict 用单引号。

**修复**

`parse_qwen_tool_call` 现在：

- 优先解析唯一的 `<tool_call>`。
- 如果没有 wrapper，只有整个 stripped output 是一个显式 `name/arguments` 对象时才接受。
- JSON 失败时用 `ast.literal_eval` 解析 Python literal（安全，不求值表达式）。
- arguments 是字符串时同样允许 JSON 或 Python literal。
- 仍然拒绝：自由文本、多个 tool call、未声明工具、缺 name/arguments、非 dict arguments。

**为什么不直接“模糊修复”**

自由文本可能包含模型猜测的最终答案；模糊匹配会把“猜的”当“工具调用”。只接受显式 `name/arguments` 是对轨迹真实性的最低要求。

### 3.6 invalid_action 重试

**现象**

即使修复 parser，8B 仍有约 10% 的 invalid_action；这些是格式失败，不是答案错误。

**修复**

对 `QwenOutputError` 做一次确定性格式重试：

```text
retry_seed = (base_seed + retry_index * 1_000_003) % 2^31
```

只重试 `QwenOutputError`，不重试 rejected、max_steps、sandbox_error、model_error。每次 attempt 最终只保留一条轨迹；result 记录 `sampling_seed_used` 和 `invalid_action_retries`。

**为什么是一次**

- 重试会消耗 GPU；一次能把多数格式失败救回，同时不让成本失控。
- 不重试 rejected/max_steps，避免把“能力失败”用多抽几次伪装成成功。
- 记录 retry 次数，审计可检查；manifest 记录 retry 策略。

### 3.7 8B 教师能力上限

**8B 最终结果（6,400 attempts）**

| 状态 | 数量 |
|---|---:|
| accepted | 2,643 |
| rejected | 1,858 |
| max_steps | 1,046 |
| invalid_action | 853 |
| model_error | 0 |

- 严格通过率：2,643 / 6,400 = **41.3%**
- retained：**1,590 / 6,400 = 24.8%**
- 目标 3,000：**未达**

质量原因（会重叠）：

| 原因 | 次数 |
|---|---:|
| answer_not_verified | 3,757 |
| submitted_answer_not_supported_by_tool_output | 2,501 |
| no_successful_dataframe_analysis | 1,793 |
| duplicate_trajectory | 1,018 |

关键发现：accepted 但未 retained 的 1,053 条中，**1,018 条是完全重复轨迹**，35 条是 unsupported。也就是说，8B 的主要损失不是过滤器误杀，而是：

1. 41.3% 的答案正确率；
2. 同一道题多次采样重复同一轨迹；
3. 大量 max_steps / invalid_action。

按 24.8% 的 unique retention，6,400 条只能到 1,590；即使把 1,016 条 import-blocked max_steps 以 43% 恢复，也只到约 2,027。8B 不可能仅靠“再跑同样的 6,400 条”达到 3,000。

### 3.8 沙盒禁止 import 导致 cleaning 轨迹失败

**发现**

对 1,793 条 `no_successful_dataframe_analysis` 轨迹做统计：

- 1,016 条是 max_steps，且轨迹中包含 `Import is not allowed`。
- 1,030 条代码里写了 `import pandas`。
- 1,024 条事件中出现 import 被拒绝。

模型很自然写：

```python
import pandas as pd
df = df.dropna()
```

但沙盒把所有 import 都禁了，代码在第一步就失败。

**修复**

只允许：

```python
import pandas as pd
import numpy as np
```

仍然禁止：

- 其他任何 module import；
- `from ... import ...`；
- 文件、网络、进程相关调用；
- audit hook 之外的路径打开。

为什么安全：pandas/numpy 本来就已经在 namespace 中，import 只是重新绑定已加载模块，不增加新能力；`os`、`subprocess`、`socket` 仍然不可达。

**验证**

- 100 条 targeted import-blocked max_steps 重跑：
  - 43 条 retained（43%）
  - 44 条 rejected（sandbox 通过了，但答案错）
  - 13 条仍 max/invalid
- 全量 1,016 条外推约 +437 retained；8B 总量约 2,027，仍不够 3,000。
- 因此这项修复保留，但单独无法救 8B。

### 3.9 审计没有按全局去重口径复算

**现象**

8B 运行完成后，审计 17/19，其中 `quality_assessments_recompute=false`。但 summary 和 trajectories 看起来都正常。

**排查**

审计最初对每个 trajectory 独立调用 `assess_teacher_trajectory`，没有模拟 sampler 的全局去重。sampler 在 `--refilter-existing` 中按 manifest 顺序维护 `seen_fingerprints`；第二条相同指纹会被标记 `duplicate_trajectory`。审计独立复算时却认为它 retained，于是不一致。

**修复**

- 审计按 manifest 顺序遍历 attempts。
- 维护 `seen_fingerprints`。
- 对 retained 且指纹已出现的轨迹，重写为 `QualityAssessment(retained=False, reasons=("duplicate_trajectory",))`。
- 再比较 latest result。
- CLI 增加 `--config`，支持 14B 配置。

**验证**

- 8B 审计从 17/19 变为 **18/19**；唯一失败是 `retained_trajectory_target`。
- 重复轨迹口径与 sampler 一致。
- 14B 审计会复用同一逻辑。

### 3.10 GPU 利用率、分片与“为什么不直接上更大模型”

**单卡 8B**

- 权重约 16GB；单进程显存约 19.4GB。
- `nvidia-smi` 快照利用率约 62%，但生成时接近 100%，低值多出现在工具执行/解析间隙。
- 单卡 6,400 条按 pilot 约 16–30 小时。

**双 8B 进程**

- 两份 8B 约 35GB，显存够。
- 两个进程共享同一张 4090 的 SM/带宽，每个进程变慢，总吞吐约 **1.2–1.3×**，不是 2×。
- 失败轨迹（尤其 max_steps）在争抢下从 40–60 秒变成 60–130 秒，尾部更差。
- 3 个 8B 进程约 52.5GB，超过 48GB，会 OOM。

**14B**

- 14B bf16 约 28GB，单卡 48GB 可以放一份。
- 14B 更强：invalid/max 明显少，答案正确率约 60%。
- 两张 48GB 4090 上每卡一个独立 14B worker，不做 tensor parallel：
  - 无 NVLink，TP 通信开销大；
  - 独立 shard 是 embarrassingly parallel，近似线性；
  - 每卡显存约 30–31GB，利用率 90%–97%，功率约 300W；
  - 两个进程合计约 60GB 权重，但分布在两张卡，不是一张卡 60GB。

**为什么不上 32B**

- 32B bf16 约 64GB，单张 48GB 放不下。
- 两张卡做 TP/量化：延迟、通信、工程复杂度都上升；质量收益不确定。
- 14B 最终 3,133 retained（49.0%），仍高于 3,000；早先按 60% 外推的 3,700–3,850 被后续重复轨迹拉低，但没有改变“14B 够用”的结论。
- 结论：不是“越大越好”，而是“在目标阈值上选成本最低的教师”。

**如果继续加卡**

- 任务级分片可以线性扩展；4×48GB 约 4 小时，8×48GB 约 2 小时。
- 但每张卡都要一份 14B 权重、一次模型加载；多机还要模型分发/合并，成本不只是 GPU 小时。
- 2×48GB 对当前目标已经够；4 张是性价比拐点；24GB 卡不适合 14B bf16。

### 3.11 备份、迁移和“主机不是同一台”

**8B 最终产物**

- 6,400 results / trajectories
- 1,590 SFT
- 50 条 review sample
- summary + audit
- 本地最终 tar：`backups\a4_final_20261024_203511\...`

**14B 迁移**

- 在旧 48GB 单卡跑到 585/6,400：351 retained（60.0%），0 max，0 duplicate。
- 停掉后做迁移包：代码、configs、tests、docs、data、14B 585 条 progress、8B final、A3 evidence、环境 freeze、模型 checksum、启动脚本。
- 迁移包 7.6MB，SHA256 `916e42f9...`；本地和 `/hy-tmp` 各一份。
- 模型权重 44GB 没有放入 tar：从 GPU 机器到本地实测约 100KB/s，16GB 都要数小时；模型公开，保留 SHA256 和 ModelScope 下载命令。
- 新主机 `/hy-tmp` 被清空，`/root` 项目还在；重新下载 Qwen3-14B 28GB，`sha256sum -c` 全部 OK。
- 新主机实际是两张 48GB 4090；把 585 条按 `attempt_index % 2` 拆成 293 + 292，两个 shard 各自 `CUDA_VISIBLE_DEVICES=0/1`。
- 曾遇到“终端 no such file”：用户终端在另一台 host（`I2bed85...`），任务在 `I2be719...`；不是代码问题，是连错机器。

**为什么备份策略这样设计**

- `results.jsonl` append-only + 原子轨迹写入：崩溃后最多丢当前一条。
- `--resume` 按 attempt_id 跳过已完成。
- 迁移包保留完整 manifest，恢复后不需要重新生成任务。
- 模型权重不备份：体积大、可公开重下、SHA256 可校验；真正不可再生的是任务/轨迹/配置。

---

## 4. 14B 最终结果（已完成）

2026-10-25 07:53，Qwen3-14B 双卡运行完成 6,400/6,400 条：

| 项 | 值 |
|---|---:|
| attempts | 6,400 |
| accepted | 3,834（59.9%） |
| rejected | 1,971 |
| invalid_action | 593 |
| max_steps | 2 |
| model_error | 0 |
| retained | **3,133（49.0%）** |
| duplicate | 693 |

质量拒绝原因（会重叠）：

| 原因 | 次数 |
|---|---:|
| answer_not_verified | 2,566 |
| duplicate_trajectory | 693 |
| submitted_answer_not_supported_by_tool_output | 778 |
| no_successful_dataframe_analysis | 545 |

retained 类别：

| 类别 | 数量 |
|---|---:|
| direct_success | 1,310 |
| pollution_cleaning | 1,813 |
| error_recovery | 10 |

运行信息：Qwen3-14B，2×48GB RTX 4090，temperature 0.8，top_p 0.95，max_steps 10，15 秒工具超时，8GB sandbox 地址空间，POSIX 限制有效。两个 shard 分别绑定 `CUDA_VISIBLE_DEVICES=0` 和 `1`，各约 30–31GB 显存，运行期间常见 90%–97% 利用率；从 2026-10-24 23:58 到 2026-10-25 07:53，约 8 小时完成。final 目录为 `qwen3_14b_train_v1/`，包含 6,400 条 results、6,400 条 trajectories、3,133 条 SFT、50 条 review sample。

审计 19/19 通过，`status=PASS`、`mechanical_target_met=true`；`a4_complete=false`，唯一阻塞项是 `human_review_50`，人工 review 状态为 `PENDING_HUMAN`。

## 5. 最终结果与预测偏差

### 5.1 实际结果

- 机械门禁：PASS（19/19）
- retained：3,133 / 6,400 = 48.95%
- 超出 3,000 目标：133 条（4.4% 余量）
- SFT：3,133 条，assistant-only trainable masks
- 50 条 review sample 已生成，等待人工 PASS

### 5.2 为什么比预测低

运行到 1,008 条时 retained rate 为 59.2%，按此线性外推会到约 3,800；最终是 49.0%，主要原因是后续 sample_index 2–7 的重复轨迹上升：

- 最终 `duplicate_trajectory`：693 条（占 attempts 的 10.8%）
- accepted 仍然稳定：3,834 / 6,400 = 59.9%
- 也就是说，答案正确率没有崩，掉的是 unique trajectory 数量

这是去重策略按设计生效，不是过滤器误杀。如果当时为了“预测数字好看”而保留 693 条重复轨迹，retained 会虚高到 3,826，但 SFT 数据质量会下降。

### 5.3 数据构成与短板

| 类别 | 最终数量 |
|---|---:|
| pollution_cleaning | 1,813 |
| direct_success | 1,310 |
| error_recovery | 10 |

**最大短板是 error_recovery 只有 10 条。** 14B 很少犯错，所以“报错后修复”轨迹自然稀少；8B 版本有 341 条 error_recovery。若项目 A 的消融要检验error-recovery 的价值，建议：

1. 把 8B 的 341 条 error_recovery 轨迹并入最终 SFT 数据；或
2. 对部分任务故意注入可恢复错误/让模型先执行错误代码；或
3. 单独采样一批高难度、多步任务，专门收集 recovery。

不能把 14B 的 10 条 error_recovery 当作“已覆盖”。

### 5.4 仍然不能声称的事

- 不能声称 A4 已完成：人工 review 未做。
- 不能声称 SFT 有效果：还没训练 1.7B/4B，也没有 dev pass@1。
- 不能声称优于 8B：14B 是教师采样质量，不是学生模型。
- 不能把 3,133 条当成外部基准泛化证明。

## 6. 方法选择：为什么这样做

### 6.1 先做小 pilot，再做全量

- 8B v1/v2/v3/t06 都是 8–32 条 pilot，用来看 invalid、max_steps、retention 和 GPU 时间。
- 14B 先 64 条 pilot（40/64 retained），再进行全量。
- 这样能在几十分钟内发现格式/沙盒问题，而不是在 6,400 条后才发现数据不可用。

### 6.2 用确定性哈希，而不是随机选择

- 任务选择、attempt seed、shard 分配、review sample 都由 seed + 内容哈希决定。
- 好处：可复现、可分片、可在新主机 resume。
- 代价：需要冻结 seed 和 manifest；任何配置/代码变化都可能改变 manifest hash。

### 6.3 拒绝采样而不是“生成后人工挑”

- 答案由 verifier 自动判定；质量由机械规则自动判定。
- 人工只做 50 条抽样复核，不参与全量筛选。
- 好处：规模可扩展、可审计；代价：规则可能误杀，所以要用真实轨迹做回归测试。

### 6.4 不放松过滤器换取数量

8B 到不了 3,000，最诱人的做法是“accepted 就 retained”或“去掉去重”。我拒绝：

- 1,018 条重复轨迹加入 SFT 不会增加信息，只会让模型记住同一路径。
- 35 条 unsupported 轨迹没有证据链。
- 项目的卖点是“可验证”，不是“数量”。

### 6.5 14B 双 shard，而不是 32B/TP

- 14B 单卡 28GB，48GB 卡放得下；两张卡独立跑，工程简单、扩展线性。
- 32B bf16 需要跨卡 TP/量化；在两张 4090 上通信和调度成本高，且没有证据表明 14B 不够。
- 结论：先让 14B 达到 3,000；如果不够再考虑更大教师或更多任务。

### 6.6 审计独立复算，而不是相信 summary

- `run_a4_teacher_audit.py` 重新生成 selection/attempts、重算 manifest hash、读取 latest result、验证轨迹、复算质量、检查 SFT mask、检查 review sample。
- 这样能发现“summary 说 PASS 但数据不一致”的情况；本次就抓到了审计自身的去重口径 bug。

---

## 7. 技术拷打：可能被追问的问题与回答

### 7.1 数据与任务

**Q1. 为什么按表划分，不按题划分？**  
同一张表的题目共享 schema、缺失模式、类别分布和清洗方式；按题划分会让测试题和训练题来自同一张表，答案/清洗规则泄漏。按表划分才是真正的未见表泛化。

**Q2. oracle 表怎么保证可达？**  
A0 只注入可恢复的破坏；丢失信息的破坏在题目里写明“排除缺失/重复只算一次”。标准答案在理想清洗后的表上计算，reference_clean 能还原到 oracle。A0 有合成表 demo 和测试，A2 有 dirty replay。

**Q3. 4,360 条题够吗？**  
A2 的目标是 3,000–5,000 训练题；当前 3,000 train，A4 再从中选 800 个任务采样轨迹。题目数量和轨迹数量是两层，不混淆。

**Q4. 800 个任务怎么选？**  
按表均衡，28 张表每张约 28–29 个任务；表内按 difficulty 分层，用哈希排序。不是随机抽，保证可复现和表覆盖。

**Q5. 为什么每题 8 条？**  
计划要求 3,000–8,000 retained；800×8=6,400 是成本与覆盖的折中。8B 的 unique retention 只有 24.8%，所以 6,400 不够；14B 约 60%，6,400 足够。

### 7.2 沙盒与安全

**Q6. 8GB `RLIMIT_AS` 会不会太大？**  
pandas/numpy 启动基线约 2.7GB；8GB 留出约 5GB 工作空间。还有 CPU 120s、单次 15s、文件大小 0、输出截断、audit hook、路径白名单、进程/网络禁用。真正危险的是文件/网络/进程，不是地址空间上限本身。

**Q7. 为什么不用 cgroups？**  
cgroups 更严格，但需要宿主机/容器权限；当前实例没有可控的 cgroup 写入接口。`RLIMIT_AS + audit hook + 禁用 import/文件/网络/进程` 是当前环境下可验证的折中。

**Q8. 允许 `import pandas/numpy` 会不会被绕过去读文件？**  
允许的是已加载模块的重新绑定，不是新 module。`from ... import ...` 仍禁止；`open`、`pd.read_csv` 的路径仍受 audit hook 限制；`os`、`subprocess`、`socket` 不可达。测试覆盖 `import os`、`from pandas import read_csv`、`pd.read_csv('../outside.csv')`。

**Q9. 为什么删掉 `replace` 黑名单不会放出 `os.replace`？**  
`os` 没有被导入，worker namespace 只有 `df`、`pd`、`np`；`os.replace` 不可达。audit hook 还拦截文件操作。黑名单按属性名匹配本身就会误伤 pandas，这是设计缺陷。

**Q10. 模型能不能通过 `pd.io.common.os.system` 执行命令？**  
A3 审计测试专门覆盖了这个表达式；`os.system` 被属性黑名单拦截，audit hook 也拦截 `os.system`/`subprocess`/`socket`。

### 7.3 采样与过滤

**Q11. 为什么 invalid_action 要重试？这不是 cherry-pick 吗？**  
只重试 `QwenOutputError`（格式/工具调用契约失败），不重试 rejected、max_steps、model_error。格式失败没有产生有效轨迹；重试是让模型有机会输出合法工具调用，不是把错误答案重抽到对。retry 次数和 seed 都记录。

**Q12. 重试 seed 怎么保证可复现？**  
`retry_seed = (base_seed + retry_index * 1_000_003) % 2^31`；base_seed 由任务/样本哈希决定。manifest 记录 retry 策略，result 记录 `sampling_seed_used` 和 `invalid_action_retries`。

**Q13. 为什么接受 bare JSON，而不要求严格的 `<tool_call>`？**  
Qwen 偶尔省略 wrapper，但输出仍是显式 `{"name":..., "arguments":...}`。这是在解析一个明确的工具调用，不是猜答案。仍拒绝自由文本、多调用、未声明工具。

**Q14. Python literal 解析安全吗？**  
`ast.literal_eval` 只接受字面量，不执行函数/表达式；payload 只用于提取 name/arguments，再走工具白名单和 schema 检查。

**Q15. 为什么证据过滤器这么严？**  
因为 accepted 只说明最终答案匹配，不说明答案来自数据。证据过滤器要求至少一次成功 `run_python` 读取 df，且提交值能在工具输出中找到；这是“可验证轨迹”的核心。

**Q16. 1,018 条 duplicate 都是完全相同的轨迹吗？**  
指纹包含 question、完整 events、submitted answer；同一道题的多条轨迹只有在公开行为几乎完全相同时才会判重。它们不增加 SFT 信息。不是按答案去重，答案相同但步骤不同会保留。

**Q17. 为什么不把 duplicate 留给 SFT 增加数据量？**  
SFT 不是越大越好；重复样本会放大同一路径，降低多样性，并让 retained 数字虚高。项目报告里的 retained 是去重后的数，不能混。

**Q18. 8B 的 1,590 条是否还能用？**  
可以作为一个 SFT pilot，但 A4 目标未达；不能声称 A4 complete。它的 error_recovery 341 条反而比 14B 丰富，可以补充 14B 的类别短板。

**Q19. 14B 的 error_recovery 为什么只有 3 条？**  
14B 强，工具错误少；error_recovery 是“前面有错误、后面修复”的轨迹，自然稀少。若实验要检验 recovery 的价值，需要合并 8B recovery 轨迹或主动构造可恢复错误，不能假装覆盖。

### 7.4 审计与结果

**Q20. 为什么 summary 说 `a4_complete=false`？**  
summary 里 A4 完成条件包含人工 review 和 retained 目标；脚本把 human gate 固定为 false，独立审计再计算。8B 的 summary 是 target fail；14B 完成后如果 target 达标，audit 会 PASS 机械项，但 `a4_complete` 仍需 human review PASS。

**Q21. 为什么 8B 审计是 18/19，不是 19/19？**  
唯一失败是 `retained_trajectory_target`：1,590 < 3,000。其他 18 项通过：manifest、attempt 覆盖、轨迹 schema、隐私、质量复算、SFT 覆盖、mask、review sample 等。

**Q22. `results.jsonl` 为什么 12,800 行？**  
append-only：6,400 条原始结果 + 6,400 条 `--refilter-existing` 更新。审计按 attempt_id 取 latest；旧行保留用于追溯。不是重复 attempt。

**Q23. 质量复算为什么一开始失败？**  
审计独立复算没有模拟 sampler 的全局去重，导致 duplicate 轨迹被审计认为 retained。修复为按 manifest 顺序维护 seen fingerprints，与 sampler/refilter 口径一致。

**Q24. 50 条人工 review 为什么不让 AI 代替？**  
这是设计中的独立人工门禁；AI 可以生成样本、检查泄漏，但不能把“AI 看过”写成 human PASS。报告会保留 `PENDING_HUMAN`。

**Q25. 14B 的 accepted 60% 会不会是 verifier 太松？**  
verifier 是 A2 的程序化答案检查，A3/A4 复用；8B 只有 41.3%，14B 60%，差异来自模型能力。严格类型/格式检查仍在；accepted 不等于 retained，retained 还要求证据和去重。

### 7.5 GPU 与工程

**Q26. 为什么两张 4090 不跑一个 14B 的 tensor parallel？**  
4090 无 NVLink；TP 跨卡通信在 PCIe 上开销大，且两张卡各跑独立 shard 可以近似线性扩展。独立 shard 不需要每步同步，工程更简单。

**Q27. 为什么不直接上 32B？**  
32B bf16 约 64GB，单卡放不下；TP/量化会引入复杂度和延迟。14B 已经 60% retained，按 6,400 条约 3,800，足够过 3,000。先满足目标，再谈更大教师。

**Q28. 为什么 8B 双进程只有 1.2–1.3×？**  
两张进程共享同一张卡的 SM/显存带宽；单进程在生成时已接近饱和，另一进程填的是工具执行间隙。显存翻倍不等于算力翻倍。14B 是每卡一份、两张独立卡，才是真正 2×。

**Q29. 3 个 8B 进程为什么不行？**  
3×约 17.5GB ≈ 52.5GB > 48GB，会 OOM。14B 两份 2×28GB 也不能放同一张 48GB 卡。

**Q30. 为什么模型权重没备份？**  
44GB，从 GPU 机器到本地实测约 100KB/s；模型公开、可重下，SHA256 已保存。不可再生的是任务、轨迹、配置、审计，这些都在迁移包。

**Q31. 如果训练中途崩了怎么办？**  
`--resume` 按 attempt_id 跳过；轨迹先写临时文件再原子替换；results append-only + fsync；shard 目录独立；迁移包可恢复 585 条；新主机重新下载模型后继续。

**Q32. 如果换卡后 `/hy-tmp` 被清空？**  
代码/进度在 `/root` 迁移包；模型在 `/hy-tmp` 会丢，但 ModelScope 可重下，SHA256 校验。本次实际发生了 `/hy-tmp` 清空，已重新下载并校验。

**Q33. 为什么终端 no such file？**  
用户终端在另一台 host（`I2bed85...`），任务在 `I2be719...`；不是文件消失。用正确 SSH 或在正确 host 上执行。

### 7.6 局限与下一步

**Q34. 这次结果能证明什么？**  
能证明：A0–A4 流水线可运行、可验证、可恢复；14B 教师能在该任务上产生约 60% unique retained 轨迹；A4 机械门禁大概率通过。不能证明：SFT 后学生模型效果、外部基准泛化、RL 收益。

**Q35. 最可能被质疑的弱点？**  
1) 14B error_recovery 类别太少；2) 人工 review 未做；3) 教师单一、只跑一次；4) 没有外部基准；5) 14B 全量还在运行，当前数字是快照；6) 采样仍有 CUDA 非确定性，seed 只能保证统计可复现，不是逐位一致。

**Q36. 下一步优先级？**  
1) 让 14B 跑完并审计；2) 完成 50 条人工 review；3) 把 8B 的 341 条 error_recovery 并入或单独做 recovery 采样；4) 再进入 A5 SFT；5) 如果时间允许，用 vLLM 批处理和多卡分片加速后续实验。

---

## 8. 复现与产物索引

### 8.1 关键路径

```text
/root/project-a/
  configs/a4_teacher_sampling.json
  configs/a4_teacher_sampling_14b.json
  scripts/run_a4_teacher_sampling.py
  scripts/run_a4_teacher_audit.py
  src/dirty_da/a4_teacher.py
  src/dirty_da/_sandbox_worker.py
  src/dirty_da/qwen_adapter.py

  artifacts/a4_teacher/qwen3_8b_train_v5/        # 8B final, 1,590
  artifacts/a4_teacher/qwen3_14b_train_v1_base585/  # 14B 585 base
  artifacts/a4_teacher/qwen3_14b_train_v1_s0/    # shard 0
  artifacts/a4_teacher/qwen3_14b_train_v1_s1/    # shard 1
  artifacts/a4_teacher/qwen3_14b_train_v1/       # final after merge
  artifacts/a4_teacher_audit.json
```

### 8.2 14B 双卡命令

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 /root/.venvs/project-a-gpu/bin/python scripts/run_a4_teacher_sampling.py \
  --config configs/a4_teacher_sampling_14b.json \
  --model /hy-tmp/Qwen3-14B \
  --cache-dir /hy-tmp/hf-cache \
  --runtime-root /hy-tmp/project-a-tmp \
  --output-dir artifacts/a4_teacher/qwen3_14b_train_v1_s0 \
  --shard-index 0 --shard-count 2 --resume

# GPU 1
CUDA_VISIBLE_DEVICES=1 /root/.venvs/project-a-gpu/bin/python scripts/run_a4_teacher_sampling.py \
  --config configs/a4_teacher_sampling_14b.json \
  --model /hy-tmp/Qwen3-14B \
  --cache-dir /hy-tmp/hf-cache \
  --runtime-root /hy-tmp/project-a-tmp \
  --output-dir artifacts/a4_teacher/qwen3_14b_train_v1_s1 \
  --shard-index 1 --shard-count 2 --resume
```

合并与审计：

```bash
# merge: concatenate disjoint shard results and trajectories
# then run the sampler with --resume --refilter-existing, then audit
python scripts/run_a4_teacher_audit.py \
  artifacts/a4_teacher/qwen3_14b_train_v1 \
  --config configs/a4_teacher_sampling_14b.json
```

### 8.3 本地归档

所有需要的内容已集中到桌面单一目录 `项目A_完整归档/`：

```text
项目A_完整归档/
  reports/项目A报告.md, 项目A报告.docx
  project-a/                          # source, configs, tests, docs, data
  project-a/artifacts/a4_teacher/qwen3_14b_train_v1/
  project-a/artifacts/a4_teacher/qwen3_8b_train_v5/
  project-a/artifacts/a4_teacher_audit.json          # 14B final audit
  project-a/artifacts/a4_teacher_audit_8b.json       # 8B audit
  evidence/environment/               # pip freeze, CUDA, model SHA256
  evidence/logs/                      # 8B/14B run and merge logs
  evidence/retry_import100/           # 8B import-recovery pilot
  evidence/pilots/14b_pilot/          # 14B 64-attempt pilot
  model/README_MODEL.md               # weights are not bundled
  SHA256SUMS.txt
```

模型权重（14B 28GB、8B 16GB）不放进归档，原因和重新下载命令见
`model/README_MODEL.md`；归档包含 SHA256 校验。

---

## 9. 附录：检查清单

### 9.1 8B 最终审计

| 检查 | 结果 |
|---|---|
| 计划 attempts = 6,400 | PASS |
| latest results 覆盖全部 attempts | PASS |
| trajectory 文件覆盖 latest results | PASS |
| 轨迹 schema | PASS |
| 隐私字段 | PASS |
| 质量复算（含全局去重） | PASS |
| SFT 覆盖 retained | PASS |
| SFT trainable mask | PASS |
| review sample 匹配 | PASS |
| retained target 3,000 | **FAIL（1,590）** |
| A5 | 完成 | 1.7B LoRA：dev 10% → 85%；test/domain 17.5% → 71.7% |
| A6 | 完成 | 4B SFT test/domain 85.0%；8B base 76.7%；4B base 53.3% |
| A7 | 完成 | success-only 90%、main 85%、recovery 80%、scale-500 55% |
| 人工 review | 完成 | A4 50 条 owner review PASS |

### 9.2 14B 最终审计

| 检查 | 结果 |
|---|---|
| attempts = 6,400 | PASS |
| latest results 覆盖全部 attempts | PASS |
| trajectory 文件覆盖 latest results | PASS |
| 轨迹 schema / 隐私字段 | PASS |
| 质量复算（含全局去重） | PASS |
| SFT 覆盖 / mask / 工具调用 | PASS |
| review sample 匹配 | PASS |
| retained target 3,000 | PASS（3,133） |
| A5 | 完成 | 1.7B LoRA：dev 10% → 85%；test/domain 17.5% → 71.7% |
| A6 | 完成 | 4B SFT test/domain 85.0%；8B base 76.7%；4B base 53.3% |
| A7 | 完成 | success-only 90%、main 85%、recovery 80%、scale-500 55% |
| 人工 review | 完成 | A4 50 条 owner review PASS |

---

## 10. 结论

A4 的工程排查结论不变：8B 的失败是教师能力上限；14B 通过双卡分片达到 3,133 retained，机械审计和 50 条人工 review 均 PASS。

A5 的第一轮 SFT 也完成了：Qwen3-1.7B + rank-16 LoRA 在 3,133 条 14B 轨迹上训练 2 个 epoch，冻结 40 条 dev 从 10% 提升到 85%，净提升 +30，满足 A5 gate。

仍需明确：A5 只是 40 条 dev、单 seed 的 gate；A6 的 test/domain 评测、A7 的消融、4B 对比和外部基准都未开始。不要把 85% 当成最终泛化结论。

## 11. A5 SFT 首轮附录

### 11.1 数据与训练

- `main_14b`：3,133 条（1,813 cleaning / 1,310 direct / 10 recovery），28 张训练表。
- 另外冻结 `success_only_14b` 3,123 条、`recovery_augmented` 3,474 条（含 341 条 8B recovery），仅用于后续消融。
- token 长度：min 1,007 / p50 1,619 / p95 2,229 / p99 2,429 / max 2,629；4,096 context 无溢出；禁止静默截断。
- LoRA：r=16、alpha=32、dropout=0.05、all-linear；2 epochs；lr 1e-4；effective batch 16；seed 20261025；bf16；gradient checkpointing。
- 392 steps；2,668.48s；train loss 0.07486。

### 11.2 开发集评测

| 模型 | accepted | accuracy |
| --- | ---: | ---: |
| base Qwen3-1.7B | 4/40 | 10% |
| SFT adapter | 34/40 | 85% |

`comparison_dev.json`：净 +30，最低要求 +4，30 recovered，1 regressed（`a2-dev-uci_212_vertebral_column-0005`），0 runtime integrity error。

### 11.3 评测路径事故

第一次 SFT 评测只得到 `invalid_action`（prepatch 12/40、随后 9/40，0 accepted）：评测入口尚未兼容 SFT adapter 的工具调用输出。`a5_eval_patch_20261026` 更新了 `qwen_adapter.py`、`run_a3_qwen_acceptance.py`、`compare_a5_dev_results.py`、配置和测试，随后得到 34/40。两次 pre-patch 结果保留在 `artifacts/a5_eval/_archive/`，不作为正式结果。

### 11.4 A5 的限制

- 只评估 40 条 dev，单 seed，无 bootstrap 置信区间。
- test 和 domain-generalization 表仍封存；A6/A7 未完成。
- 未做 4B、更多 seed、外部基准。
- A5 adapter 已保存并可 reload；A6 之前不应把 dev 85% 当成最终 benchmark。

## 12. 最终结论（A0–A7）

- A4：3,133 retained，19/19 audit，50 条 human review PASS。
- A5：1.7B dev 10% → 85%。
- A6：1.7B test/domain 17.5% → 71.7%；4B 53.3% → 85.0%；8B base 76.7%。
- A7：success-only 90%、main 85%、recovery 80%、scale-500 55%、scale-1000 80%、scale-2000 75%；4B SFT 85.0%。
- 最强结论：**4B SFT 在 test/domain 达到 85.0%，高于 1.7B SFT 的 71.7% 和 8B base 的 76.7%**。
- 最诚实的限制：dev 只有 40 题、单 seed；A6 只有 120 题；8B 未做 SFT；A7 的 success/recovery 差异很小；A8 未完成；B 的 GRPO 已完成，但 B4 为负结果，B5/B6 未启动。

