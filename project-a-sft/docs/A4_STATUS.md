# A4 Teacher Sampling and SFT Records

## Status

A4 is implemented. The frozen 8B run completed all 6,400 attempts but retained
only **1,590** trajectories, below the required 3,000. The Qwen3-14B rerun then
completed all 6,400 attempts and retained **3,133** trajectories. Its
independent audit passes **19/19** mechanical checks, including the 3,000
retained target. The project owner explicitly confirmed that all 50 frozen
human-review rows passed on 2026-10-25. That attestation is persisted alongside
the review sheet; the independent audit now reports `human_review_status=PASS`
and `a4_complete=true`.

## Implemented pipeline

`configs/a4_teacher_sampling.json` freezes the Qwen3-8B teacher plan:

- 800 deterministic training tasks, balanced across the 28 training tables.
- 8 sampled attempts per task, for 6,400 planned trajectories.
- Temperature 0.8, top-p 0.95, 256 new tokens, 10 tool calls per episode.
- One deterministic format retry for malformed or missing tool calls.
- Acceptance target of 3,000 to 8,000 retained trajectories.
- A deterministic 50-record human review sample.

`src/dirty_da/a4_teacher.py` provides deterministic task selection, attempt
seeding, mechanical rejection sampling, anti-hardcoding checks, trajectory
deduplication, quality categories, and leakage-safe SFT records with
assistant-only trainable masks. `scripts/run_a4_teacher_sampling.py` samples
with the local Qwen3-8B adapter, verifies every submitted answer through the A3
environment, persists append-only results and public trajectories, and rebuilds
`train_sft.jsonl` and the human review sample.

Retained categories are `direct_success`, `error_recovery`, and
`pollution_cleaning`. A trajectory is retained only when the answer passed the
strict verifier, at least one successful `run_python` call read `df`, the
submitted value is supported by tool output, no private oracle field leaked,
and the trajectory is not a duplicate.

## Pilot findings and corrections

Bounded pilots exposed defects that were not visible in the A3 acceptance run:

1. The Linux worker imposed `RLIMIT_AS=2 GiB`, but pandas and NumPy reserve
   about 2.7 GiB of virtual address space at import on this host. Any
   nontrivial allocation therefore raised `MemoryError`. A4 now uses an
   8 GiB address-space limit while keeping CPU, file-size, timeout, import,
   file, network, and process restrictions active.
2. The sandbox denied every attribute named `replace`, which also blocked
   legitimate pandas `Series.replace` and `str.replace` cleaning. The denial
   list now covers process and filesystem operations rather than pandas
   cleaning methods.
3. The teacher prompt did not state that `submit_answer` is a separate tool.
   Several pilots called it inside `run_python` and exhausted the step budget.
   Prompt version `a4-qwen-teacher-v3` makes the separate call and the required
   tool-output evidence explicit.
4. The stricter evidence matcher was present locally but had not been applied
   to the GPU run directory. It now accepts NumPy scalar representations and
   record-shaped Python output while still rejecting unsupported submissions.
5. Qwen occasionally omitted the Hermes wrapper or emitted a Python-literal
   mapping. The parser now accepts an explicit bare JSON/Python-literal tool
   call but still rejects free-form text, undeclared tools, and ambiguous
   multiple calls. One deterministic format retry is used when the output
   contract is violated; rejected and max-step outcomes are never retried.
6. The model sometimes tried to construct its own `df` from table rows. These
   truncated or malformed calls remain failures; the prompt tells the model to
   derive the answer from the provided `df` and to put the computed value in
   the tool result before submitting.

After these corrections, the 32-attempt v5 pilot produced 18 verified answers
and 18 retained trajectories (56.25%), with 3 invalid actions, 4 max-step
outcomes, and 7 rejected answers. The earlier 0.8 and 0.6 pilots showed that
lower temperature alone did not remove the format failures; parser recovery
and one retry did.

## Final 6,400-attempt result

The merged run contains 6,400 latest results and 6,400 public trajectories:

| Terminal status | Count |
| --- | ---: |
| `accepted` | 2,643 |
| `rejected` | 1,858 |
| `max_steps` | 1,046 |
| `invalid_action` | 853 |
| `model_error` | 0 |

Strict verified answer rate is 41.30%. Retention is lower because trajectories
must also show a successful dataframe analysis, submit a value supported by
tool output, contain no leakage, and survive global deduplication:

| Retained category | Count |
| --- | ---: |
| `direct_success` | 740 |
| `pollution_cleaning` | 509 |
| `error_recovery` | 341 |
| **Total retained** | **1,590** |

Quality rejection reasons overlap: `answer_not_verified` 3,757,
`submitted_answer_not_supported_by_tool_output` 2,501,
`no_successful_dataframe_analysis` 1,793, and `duplicate_trajectory` 1,018.
The 1,018 duplicate rejections are a major loss: many samples of the same task
repeat the same question, tool calls, and submitted value, so they add no new
SFT information.

The independent audit passes 18/19 checks. The only failed check is
`retained_trajectory_target`; all 6,400 results and trajectories are present,
all trajectories validate, no private field leaked, no runtime model error
occurred, POSIX limits were active, SFT records cover exactly the retained
trajectories, and the 50-record human review sample matches. The audit result
is in `artifacts/a4_teacher_audit.json`.

## Sandbox import correction and 14B rerun

The 8B run left 1,793 `no_successful_dataframe_analysis` failures. Of those,
1,016 were max-step trajectories whose code contained an `import pandas as pd`
or `import numpy as np`; the sandbox rejected every import before analysis. The
worker now allows only those two already-loaded modules and still rejects all
other imports, `from ... import ...`, files, network, and processes.

A 100-attempt pilot on those import-blocked max-step attempts recovered 43 to
retained status (43%), with 44 rejected, 13 still max-step/invalid, and no
leakage. Extrapolating evenly, the full 1,016 retries would add about 437
retained trajectories, taking the 8B dataset to roughly 2,000. That is still
below 3,000, so the 8B teacher cannot meet the target by this correction alone.

Qwen3-14B was downloaded to `/hy-tmp/Qwen3-14B` and piloted on the same first
64 attempts. It produced 40 accepted and 40 retained trajectories (62.5%), 19
rejected, 5 invalid, and no max-step outcomes.

At 585/6,400 attempts the 14B run was paused for a GPU migration: 351 retained
(60.0%), 352 accepted, 174 rejected, 59 invalid, zero max-step outcomes, and
zero duplicates. The replacement host had two 48 GB RTX 4090s; the 585
completed attempts were split by `attempt_index % 2` into two shards, one 14B
worker per GPU with `CUDA_VISIBLE_DEVICES=0` or `1`.

The merged final 14B run contains:

| Terminal status | Count |
| --- | ---: |
| `accepted` | 3,834 |
| `rejected` | 1,971 |
| `invalid_action` | 593 |
| `max_steps` | 2 |
| `model_error` | 0 |

Strict verified answer rate is 59.9%. The final retained count is **3,133
(49.0%)** after global deduplication:

| Retained category | Count |
| --- | ---: |
| `pollution_cleaning` | 1,813 |
| `direct_success` | 1,310 |
| `error_recovery` | 10 |
| **Total** | **3,133** |

Quality rejection reasons overlap: `answer_not_verified` 2,566,
`duplicate_trajectory` 693, `submitted_answer_not_supported_by_tool_output`
778, and `no_successful_dataframe_analysis` 545. The final run used 796 format
retries, produced 6,400 latest results and 6,400 public trajectories, 3,133 SFT
records, and a 50-record human-review sample.

The independent audit passes 19/19 checks. `status=PASS`,
`mechanical_target_met=true`, `retained_trajectory_target=true`, and
`run_is_complete=true`; the recorded human review also passes, so
`a4_complete=true`.

The 8B run and the 14B rerun remain separate teacher datasets. The 8B artifacts
are evidence for the first attempt and its post-hoc recovery analysis; the 14B
run is the production A4 candidate. The 14B data has very few
`error_recovery` trajectories (10), so the 8B run's 341 recovery trajectories
are the obvious supplement if later ablations need that category.

## Running and verifying

The replacement host exposed two 48 GB RTX 4090 devices and the run used two
deterministic shards, one worker per device. Shard membership was
`attempt_index % shard_count`; every attempt kept its original question,
sampling seed, tools, and verifier. The 60 attempts already completed by the
single-process run were redistributed before the shards resumed.

```bash
/root/.venvs/project-a-gpu/bin/python scripts/run_a4_teacher_sampling.py \
  --model /hy-tmp/Qwen3-8B \
  --cache-dir /hy-tmp/hf-cache \
  --runtime-root /hy-tmp/project-a-tmp \
  --output-dir artifacts/a4_teacher/qwen3_8b_train_v5_s0 \
  --shard-index 0 --shard-count 2 --resume
```

The second shard used `qwen3_8b_train_v5_s1` and `--shard-index 1`. When both
shards exited, `merge_a4_v5.sh` concatenated the disjoint results and
trajectories, ran the sampler once with `--refilter-existing` for global
duplicate removal and SFT rebuilding, and ran the audit.

The final `results.jsonl`, `trajectories/`, `train_sft.jsonl`,
`human_review_sample.jsonl`, `human_review_sample.csv`, and `summary.json` are
under `artifacts/a4_teacher/qwen3_8b_train_v5/`. The independent audit is:

```bash
python scripts/run_a4_teacher_audit.py \
  artifacts/a4_teacher/qwen3_8b_train_v5
```

The audit re-derives the frozen selection, attempt plan, manifest hash,
latest result per attempt, trajectory schema and privacy, quality decisions,
SFT coverage and masks, and human-review sample. It writes
`artifacts/a4_teacher_audit.json`.

## Remaining gates and next levers

- The 14B mechanical gate is complete: 6,400/6,400 attempts, 3,133 retained,
  19/19 audit checks passed, zero runtime model errors.
- The independent 50-record human review is recorded as PASS. A supplementary
  ten-record review covers every rare 14B error-recovery trajectory; it is a
  quality follow-up and does not retroactively change the frozen A4 gate.
- The main data limitation is category balance: 1,813 pollution-cleaning, 1,310
  direct-success, but only 10 error-recovery trajectories. The 8B run has 341
  error-recovery trajectories; merge or resample deliberately if A7 needs that
  ablation.
- The 8B import-recovery pilot (43/100 retained) and the 8B final run remain as
  supporting evidence; they do not replace the 14B production dataset.
- Do not treat the 3,133 records as evidence of student-model performance. A5
  data/tokenizer preflight is implemented, but GPU SFT and A6/A7 evaluation
  have not run.

