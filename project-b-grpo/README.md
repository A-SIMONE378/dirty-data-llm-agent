# Project B — GRPO Extension (Negative Result)

This folder contains the **B-specific** code, configs, tests, docs, small result summaries, and report for the GRPO extension built on top of Project A.

## Headline

- B1: 1,000/1,000 environment rollouts passed; replay consistent.
- B2: reward + anti-cheat; 13 tests passed.
- B3: 800 train tasks × 8 rollouts = 6,400; **336** group-comparable tasks retained.
- B4: TRL GRPO on Qwen3-1.7B; 13/42/336-step runs.
- **B4 did not beat SFT.** Same 120-task test/domain set: best B4 **67.5%** vs SFT **71.7%**.
- B5/B6 were not started; the B branch is closed as a negative result.

This repository is not a claim that “GRPO improves agents.” It is a controlled experiment and a failure analysis.

## Relationship to Project A

B is an extension, not a standalone replacement:

- It reuses A2 train tasks, A3 sandbox/tools, and the A5 1.7B SFT adapter.
- It does not use the A6 test/domain data for training.
- It does not include A’s source tree or data pack.

To run B from this combined repository, start from `project-a-sft/` and overlay the B-specific files from `project-b-grpo/` into a disposable working copy:

```bash
# from the repository root, after copying project-a-sft to a disposable worktree
cp project-b-grpo/src/dirty_da/b*.py  project-a-sft/src/dirty_da/
cp project-b-grpo/scripts/*b*.py      project-a-sft/scripts/
cp project-b-grpo/tests/test_b*.py    project-a-sft/tests/
cp project-b-grpo/configs/b*.json     project-a-sft/configs/
cp project-b-grpo/docs/B_PLAN.md      project-a-sft/docs/
```

## B4 results

| Model | A6 120 tasks | Test 80 | Domain 40 |
|---|---:|---:|---:|
| A5 SFT | 86/120 = **71.7%** | 54/80 = 67.5% | 32/40 = 80.0% |
| B4 train100 (13 step) | 73/120 = 60.8% | 44/80 = 55.0% | 29/40 = 72.5% |
| B4 full1 (42 step) | 81/120 = **67.5%** | 51/80 = 63.7% | 30/40 = 75.0% |
| B4 full2 (336 step, format reward) | 80/120 = 66.7% | 51/80 = 63.7% | 29/40 = 72.5% |
| B4 full3 (336 step, outcome-only) | 79/120 = 65.8% | 49/80 = 61.3% | 30/40 = 75.0% |

All B4 adapters are below SFT. B4 full1 is 42 optimizer steps and `epoch=0.125`; full2/full3 are 1 epoch.

## B3 data

- 800 train tasks, 28 tables, 8 samples/task.
- accepted 4,177 / 6,400 = 65.3%.
- invalid_action 1,244 / 6,400 = 19.4%.
- 130 tasks 0/8, 334 tasks 8/8, **336 tasks kept**.
- Kept mean pass rate 56.0%; kept invalid_action 35.5%.

The high malformed-tool-call rate is part of the B4 failure analysis.

## What is included

```text
src/dirty_da/       b_env.py, b_reward.py, b4_env.py
scripts/            B1 audit, B3 rollout/merge, B4 cache/training
tests/              test_b_env.py, test_b_reward.py, test_b3_tasks.py
configs/            b1, b2, b3, b4 configs
docs/               B_PLAN.md
results/            B3 summary/tasks/filtered tasks, B3 task index; B4 manifests/eval summaries
reports/            项目B报告.md and .docx
requirements-b4.txt separate B4 training environment
```

## What is deliberately excluded

- A source tree, data pack, tests, and A reports (use Project A)
- model weights and LoRA adapters
- B3/B4 rollout trajectories (38 MB B3 + eval trajectories)
- B4 dirty cache (46 MB; regenerate)
- GPU logs, SSH credentials, and runtime state

## B4 environment

B4 training used a separate environment because TRL’s `environment_factory` requires `transformers>=5.2`:

```text
transformers==5.2.0
trl==1.16.0
peft==0.21.2
accelerate==1.15.0
torch==2.8.0+cu128
jmespath
```

The A/SFT evaluation env remains on `transformers 4.57.6`; this mismatch is documented as a limitation. B4 also uses a pre-generated dirty-table cache to avoid pandas/pyarrow fingerprint drift between environments.

## Reproduction outline

```bash
# B1 / B2
python scripts/run_b1_environment_audit.py --rollouts 1000 --parallel 4
python -m pytest tests/test_b_reward.py tests/test_b_env.py

# B3 (GPU)
python scripts/prepare_b3_tasks.py
CUDA_VISIBLE_DEVICES=0 python scripts/run_b3_rollouts.py \
  --shard-index 0 --shard-count 6 --resume \
  --model /path/to/Qwen3-1.7B
python scripts/merge_b3_rollouts.py --shards 6

# B4 cache (base env), then training (project-b env)
python scripts/prepare_b4_dirty_cache.py
python scripts/run_b4_grpo.py --config configs/b4_grpo_full2.json \
  --model /path/to/Qwen3-1.7B \
  --adapter /path/to/a5_sft/final_adapter
```

## Report

Full Chinese report: `reports/项目B报告.md` / `.docx`.

The report covers B1–B4, the negative result, failure analysis, exact metrics, reproduction commands, and the decision to stop B5/B6.

## Limitations

- One model size (1.7B), one seed, 120 test/domain tasks.
- B4 full1 is not a full epoch.
- Training/evaluation transformers versions differ.
- No reward curve, entropy, anti-cheat hit counts, or human review.
- Negative result is specific to this environment/reward/data setup, not a universal claim about GRPO.
