# A1 表级数据集与可复现轨迹审计

## A1 范围

A1 在 A0 的可信数据地基上增加：版本化表清单、严格按表 split、受约束题面、多轮工具轨迹、轨迹重建验证和 JSONL 数据集输出。A1 不下载模型、不调用 API、不训练 Qwen3，也不把 smoke 数据描述为真实实验数据。

## 数据流

```text
versioned DataFrames
  -> TableCatalog / manifest digest
  -> fingerprint-bound table split
  -> trusted TrajectoryScenario
       -> canonical oracle (TaskSpec + executor)
       -> dirty copy (ordered corruption requests)
       -> explicit reference cleaning
       -> cleaned result must equal oracle
  -> seven-message tool trajectory
  -> deterministic full-record rebuild verification
  -> train/dev/test JSONL + hashes
```

## 表清单与 split

`TableCatalog` 要求合法且唯一的 `table_id`、非空版本、字符串列名和 A0 支持的 canonical 值。catalog 不向调用方暴露内部 `CanonicalTable`；`get()` 返回重新封装的独立副本。

manifest 记录表 ID、版本、canonical fingerprint、行数、列名、dtype 和索引类型。manifest digest 是规范 JSON 的 SHA-256。

`split_tables()` 只处理完整表：排序键是 `SHA256(seed:table_id:fingerprint)`。默认比例为 0.8/0.1/0.1，小数据集会调整计数以保证 train/dev/test 均至少一张表。至少需要三张表。`SplitAssignment.validate()` 检查：

- manifest digest 完全一致；
- 三个 split 非空；
- table ID 无交集；
- 所有 manifest 表恰好覆盖一次。

不提供按行 split API，因此同一表不能通过正常构建路径泄漏到多个 split。

## 题面约束

`render_task_prompt()` 只有 variant 0 和 1。两种题面都逐项投影 `TaskSpec` 的 operation、target、grouping、missing、dedup 和 sentinel 语义，并明确 JSON 输出格式。variant 不是新的 ground truth；不支持自由改写或从自然语言反推 spec。

variant 1 可用于少量 held-out wording stress，但 A1 不批量扩展表达集合。

## 轨迹结构与 correctness gate

每条 `TrajectoryScenario` 绑定 table ID、TaskSpec、有序 corruption requests、CleanerSpec、seed 和 prompt variant。输出固定为 7 条消息：

1. system：限定数据分析与 JSON 输出；
2. user：dirty table 与受约束题面；
3. assistant：调用 `inspect_table`；
4. tool：返回真实 shape/schema；
5. assistant：调用 `clean_and_execute`，参数包含明确 cleaner/task spec；
6. tool：返回生产 cleaner/executor 的真实结果；
7. assistant：返回 oracle JSON answer。

构建时会执行以下硬检查：

```text
execute_frame(clean(reference_dirty, CleanerSpec), TaskSpec)
    == verify(execute(canonical, TaskSpec))
```

只要 cleaned dirty-table answer 与 canonical oracle 不一致，`TrajectoryBuildError` 会拒绝该样本。测试明确覆盖了“任务目标 cell 被 missing 覆盖且无法恢复”的拒绝路径。

`verify_trajectory()` 不只检查消息格式，而是从受信 scenario 与 catalog 完整重建 trajectory，并比较 ID、fingerprint、provenance、事件、全部消息和最终答案。修改最终答案即验证失败。

## 数据集输出

`build_dataset()` 根据 scenario 的 table ID 放入既定 split，拒绝重复 scenario ID、空 split、重复 trajectory ID 和跨 split 表泄漏。

`DatasetBundle.write_jsonl()` 输出：

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`
- `dataset_manifest.json`

dataset manifest 记录表 manifest digest、split、轨迹数量、轨迹 ID 和每个 JSONL 文件的 SHA-256。JSONL 使用固定 LF 字节，避免 Windows 换行转换导致哈希漂移。

## CPU smoke

运行：

```powershell
.\scripts\run_a1_smoke.ps1
```

该脚本使用 6 张独立的小型人工表，各生成 1 条轨迹，分别覆盖 missing、sentinel、duplicate、type、format 和 category。missing 只污染任务无关列；其余案例由显式可验证规则恢复。输出位于 `artifacts/a1_smoke/`，默认被 Git 忽略。

它只是端到端功能检查，不是 benchmark、模型实验或训练数据规模结论。

## 已知限制与进入 A2 前的缺口

- 当前只有内存 DataFrame catalog；尚未接入真实 CSV/Parquet 数据清单与许可记录。
- smoke 只有 6 张人工表，不能代表真实数据分布。
- 只提供两种受控题面，没有自由文本增强。
- 轨迹 schema 是项目内部 JSONL，尚未针对实际 Qwen3 tokenizer/chat template 做 tokenization 验证。
- A1 JSON 轨迹拒绝 `Decimal` 和非有限浮点，时间值转 ISO 字符串，timedelta 转确定性字符串。
- 没有模型推理、LoRA、GPU、训练指标或实验结论。

进入 A2 前需要真实表资产、来源/许可、版本与 schema 清单，以及最终 Qwen3 checkpoint/chat template 选择。之后才应实现 tokenizer-aware export 与 AutoDL 小规模训练 smoke。
