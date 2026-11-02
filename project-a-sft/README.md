# Project A — Verifiable Dirty-Table Agent

Building a small model that can inspect, clean, and analyze dirty tables with a strict, executable verifier.

This repository contains the code, configs, tests, docs, small result summaries, and final report for Project A. Large data packs, model weights, adapters, rollout trajectories, and GPU artifacts are intentionally excluded.

## Status

- **A0–A7 complete**
- **A8 failure analysis not included**
- **Project B (GRPO extension) tested separately and closed as a negative result**; see the companion `项目B` folder/report.

## Headline results

| Setting | Start | End | Gain |
|---|---:|---:|---:|
| Qwen3-1.7B, 40-task dev | 10.0% | **85.0%** | +75.0 |
| Qwen3-1.7B, 120-task test/domain | 17.5% | **71.7%** | +54.2 |
| Qwen3-4B, 120-task test/domain | 53.3% | **85.0%** | +31.7 |
| 4B SFT vs untuned 8B base | 76.7% | **85.0%** | +8.3 |

A6 test/domain details:

- 1.7B SFT: test 67.5%, domain 80.0%
- 4B SFT: test 81.2%, domain 92.5%
- 8B base: test 71.3%, domain 87.5%

The 8B model was not SFT-tuned, so “4B SFT > 8B base” must not be rewritten as “4B is universally better than 8B.”

## What is included

```text
src/dirty_da/       canonical data, corruption, task synthesis, sandbox,
                    agent environment, verifier, SFT/eval helpers
scripts/            A0–A7 audit, sampling, SFT, evaluation scripts
tests/              unit and integration tests
configs/            frozen task, corruption, environment, SFT, eval configs
docs/               A0–A5 status/audit notes
results/            selected small JSON summaries
reports/            项目A报告.md and .docx
pyproject.toml      package metadata and dependency groups
```

## What is deliberately excluded

- `data/tables/` and `data/tasks/` (raw UCI tables and full task/oracle records)
- `artifacts/` (teacher trajectories, rollout JSONL, adapters, checkpoints, eval trajectories)
- model weights and LoRA adapters
- SSH credentials, host details, and runtime state

Reasons: repository size, benchmark/test leakage, model-weight hosting, and license/provenance handling. The selected result summaries and the report retain the measured numbers.

## Setup

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
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

The command above runs the public, data-independent core suite. The full suite additionally verifies the separately generated data pack, 54 CSV tables, A2 task records, and teacher trajectories. UCI source tables are CC BY 4.0; see `docs/A1_STATUS.md` and the A report for provenance, transformations, and quarantine records.

## Project structure

```text
A0  Data audit, corruption operators, canonical snapshots
A1  45 UCI tables; table-level train/dev/test/domain split
A2  4,360 structured tasks; programmatic answers; dirty replay
A3  Three-tool sandbox: inspect_table / run_python / submit_answer
A4  Teacher sampling; 6,400 attempts; 3,133 retained SFT trajectories
A5  1.7B LoRA SFT; 40-task dev gate
A6  120-task test/domain evaluation; 1.7B/4B/8B comparison
A7  Data-scale and trajectory ablations
```

## Report

The full Chinese report is in `reports/项目A报告.md` / `.docx`.

It includes the resume-oriented summary, per-stage results, failure analysis, reproduction commands, and limitations.

## Companion project

Project B is the GRPO extension on top of the A5 adapter. It is documented separately because it is a negative result:

- B3: 6,400 SFT rollouts; 336 group-comparable tasks retained.
- B4: 13/42/336-step GRPO; best 67.5% vs SFT 71.7% on the same 120-task test/domain set.
- B5/B6 were not started.

See [`../project-b-grpo/reports/项目B报告.md`](../project-b-grpo/reports/项目B报告.md) for the full analysis.

## Scope and limitations

- The reported accuracy is strict verifier accepted accuracy, not a public leaderboard score.
- A5/A7 dev is 40 tasks and single-seed; A6 is 120 test/domain tasks.
- The 8B model was not SFT-tuned.
- No external benchmark is included.
- The repository is a code/architecture portfolio, not a one-command reproduction of the full GPU pipeline.
