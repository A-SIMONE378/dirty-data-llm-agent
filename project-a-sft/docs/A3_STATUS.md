# A3 Agent Environment and GPU Acceptance

## Verdict

A3 is complete. The bounded agent environment passes its Linux engineering
audit, and Qwen3-8B completed the frozen 50-task held-out acceptance run on an
NVIDIA GeForce RTX 4090. This is an environment and trajectory-integrity gate;
the observed answer accuracy is reported as measured and is not presented as a
trained-model result.

## Implemented boundary

The environment exposes exactly three tools:

1. `inspect_table`: bounded columns, dtypes, missing counts, row count, and head.
2. `run_python`: persistent Python state over the task dataframe with per-call
   timeout, output truncation, import/file/network/process restrictions, and
   Linux CPU/memory/file-size limits.
3. `submit_answer`: strict programmatic verification against a private expected
   answer without returning the oracle value.

Each episode receives only its dirty task CSV and constrained natural-language
question. The public trajectory excludes the structured spec, expected answer,
canonical fingerprint, and oracle table. The step budget is 12 and sandbox
output is capped at 8,000 characters. Malformed model tool calls are recorded as
`invalid_action`; they remain capability failures and never increase accuracy.
CUDA, model-loading, and other genuine inference exceptions remain
`model_error`.

## Frozen GPU acceptance

The deterministic manifest contains 50 unique A2 held-out tasks:

- 40 `test` tasks: five from each of eight held-out tables.
- 10 `domain_generalization` tasks: two from each of five held-out tables.
- No train or development task is included.

Runtime:

- Model: `Qwen/Qwen3-8B`
- GPU: NVIDIA GeForce RTX 4090
- Python: 3.11.11
- PyTorch: 2.5.1+cu124
- CUDA runtime reported by PyTorch: 12.4
- Transformers: 4.57.6

Observed results:

| Terminal status | Count | Meaning |
| --- | ---: | --- |
| `accepted` | 24 | Submitted answer passed the strict verifier |
| `rejected` | 6 | Submitted answer failed the strict verifier |
| `max_steps` | 18 | No final accepted submission within 12 tool calls |
| `invalid_action` | 2 | Inference completed but output violated the tool-call contract |
| `model_error` | 0 | No runtime/model-loading failure remained |

Strict accepted accuracy is **24/50 = 48%**. All 50 latest results have valid
trajectory files, the manifest and result task IDs match, private oracle fields
are absent, the manifest hash matches the summary, and POSIX resource limits
were active. The two `invalid_action` cases are retained as failures; they were
not relabeled as correct or silently discarded.

## Verification

Run the CPU/Linux environment audit and the complete test suite:

```bash
python scripts/run_a3_environment_audit.py
python -m pytest -v
```

The audit independently re-reads the GPU manifest, latest result per task,
summary, and all 50 trajectory files instead of trusting the summary pass flag.
On 2026-10-23, the environment audit passed 14/14 checks, the persisted GPU
evidence passed 12/12 checks, and the complete project suite passed 112 tests
with zero failures and zero skips.

To reproduce the frozen GPU run after installing `.[inference]` and making the
model available locally:

```bash
python scripts/run_a3_qwen_acceptance.py \
  --model /path/to/Qwen3-8B \
  --cache-dir /path/to/hf-cache \
  --runtime-root /path/to/runtime
```

Use `--resume` for an interrupted run. `--retry-model-errors` retries only
genuine prior `model_error` records; rejected, max-step, and invalid-action
outcomes are preserved.

## Artifacts and limitations

The formal evidence is under `artifacts/a3_gpu/qwen3_8b_50_v1/`:

- `manifest.json`: frozen selection and source/prompt hashes.
- `results.jsonl`: append-only attempts; audit uses the latest record per task.
- `trajectories/*.json`: one current public trajectory per task.
- `summary.json`: recomputed terminal counts and runtime metadata.

`artifacts/a3_environment_summary.json` is the combined environment/GPU audit.
Model weights and caches are deliberately not stored in the repository.

This is one deterministic base-model run, not a benchmark or scaling study.
The 48% accuracy and 18 max-step outcomes show substantial room for later SFT;
they must not be generalized beyond this frozen sample. A3 performs no SFT,
LoRA, GRPO, paid API call, or broad hyperparameter search.
