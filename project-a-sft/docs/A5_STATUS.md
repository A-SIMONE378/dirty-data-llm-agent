# A5 Qwen3-1.7B LoRA SFT Status

## Verdict

A5's first production LoRA run is complete and passes the frozen development-set
gate. This is a single-seed, 40-dev-task comparison, not the full A6 evaluation.
Test and domain-generalization tables remain sealed.

## Data

`artifacts/a5_data/manifest.json` freezes three selections from the A4 training
data only:

| Dataset | Records | Use |
| --- | ---: | --- |
| `main_14b` | 3,133 | production run |
| `success_only_14b` | 3,123 | recovery-category ablation |
| `recovery_augmented` | 3,474 | disclosed 8B-recovery ablation |

`main_14b` contains 1,813 pollution-cleaning, 1,310 direct-success, and 10
error-recovery trajectories over 28 training tables / 610 tasks. Source JSONL
SHA-256 values are verified when a selection is resolved; the main source hash
is `3e7dd525fa783e17764dc53591cd3d6398df1fc6d2ee01d709ddcea48fb744e8`.

The real Qwen3-1.7B tokenizer audit measured:

- min 1,007; p50 1,619; p95 2,229; p99 2,429; max 2,629 tokens
- 4,096-token context covers every record; 2,048 would truncate 340 records
- chat-template SHA-256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`
- silent truncation is forbidden

## Training

- Model: `Qwen/Qwen3-1.7B`
- Method: LoRA rank 16, alpha 32, dropout 0.05, all-linear targets, bias none
- Assistant-only loss; system/user/tool tokens are `-100`
- 2 epochs, learning rate 1e-4, micro-batch 1, gradient accumulation 16,
  effective batch 16, seed 20261025, bf16, gradient checkpointing
- Max sequence length 4,096
- 392 optimizer steps; train runtime 2,668.48s (44.5 minutes)
- Train loss 0.07486 (epoch 2.0)

Final adapter:

```text
artifacts/a5_sft/qwen3_1_7b_main_14b_v1/final_adapter/
```

The adapter is about 67MB; `adapter_model.safetensors` SHA-256 is
`e36296e1c053d8b6996f1dda3d71b6e8d43d67e487c9687f470bf62325408cfc`.
The final adapter was reloaded with the base model and generated successfully.
`run_manifest.json` records the frozen dataset hash, config hash, versions, and
train metrics; `checkpoint-392/trainer_state.json` contains the full loss curve.

## Development evaluation

Frozen `configs/a5_dev_eval.json`: 40 development tasks, greedy decoding,
maximum 768 new tokens, same manifest for both runs.

| Run | Accepted | Accuracy | Terminal status |
| --- | ---: | ---: | --- |
| base Qwen3-1.7B | 4/40 | 10% | accepted 4, rejected 22, max_steps 9, invalid 5 |
| SFT adapter | 34/40 | 85% | accepted 34, rejected 4, invalid 2 |

Paired comparison (`artifacts/a5_eval/comparison_dev.json`):

- net accepted gain: **+30**
- required minimum: +4
- `a5_dev_gate_pass: true`
- 30 recovered task IDs, 1 regression:
  `a2-dev-uci_212_vertebral_column-0005`
- 0 runtime integrity errors

The first evaluation attempt, before `a5_eval_patch_20261026`, produced only
`invalid_action` (12/40 then 9/40, 0 accepted): the evaluation path was not yet
compatible with the SFT adapter/tool-call output. The patch updates
`qwen_adapter.py`, `run_a3_qwen_acceptance.py`, `compare_a5_dev_results.py`,
the eval config, and tests. The two pre-patch runs are preserved under
`artifacts/a5_eval/_archive/`; they are not used as results.

## Scope and limitations

- Only dev is evaluated here; test and domain-generalization are sealed until A6.
- One seed, 40 tasks; no bootstrap confidence interval yet.
- This is an A5 student-model gate, not a full benchmark or scaling study.
- A4 is complete: 6,400/6,400 attempts, 3,133 retained, 19/19 audit, and the
  project owner's 50-record human review recorded as PASS.
- A6/A7 (test/domain evaluation, ablations, 4B comparison) are not complete.

## Artifacts

```text
artifacts/a5_data/manifest.json
artifacts/a5_data/main_14b.index.jsonl
artifacts/a5_data/success_only_14b.index.jsonl
artifacts/a5_data/recovery_augmented.index.jsonl
artifacts/a5_data/token_length_audit.json
artifacts/a5_sft/qwen3_1_7b_main_14b_v1/final_adapter/
artifacts/a5_sft/qwen3_1_7b_main_14b_v1/run_manifest.json
artifacts/a5_sft/qwen3_1_7b_main_14b_v1/checkpoint-392/trainer_state.json
artifacts/a5_eval/qwen3_1_7b_base_dev_v1/
artifacts/a5_eval/qwen3_1_7b_sft_dev_v1/
artifacts/a5_eval/comparison_dev.json
artifacts/a5_eval/_archive/
```
