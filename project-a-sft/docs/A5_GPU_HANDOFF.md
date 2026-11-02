# A5 GPU Handoff

This handoff starts only the bounded Qwen3-1.7B LoRA stage. Do not rerun teacher
sampling and do not use development, test, or domain-generalization tables as
training data.

## Frozen local evidence

- A4 complete: 6,400 attempts, 3,133 retained, mechanical audit 19/19, human
  review PASS.
- A5 default dataset: `main_14b`, 3,133 records from 28 training tables.
- Qwen3-1.7B tokenizer revision:
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Chat-template SHA-256:
  `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
- Token lengths: min 1,007, p50 1,619, p95 2,229, p99 2,429, max 2,629.
- Frozen context: 4,096 tokens; silent truncation is forbidden.

## New-host sequence

After extracting the handoff archive to `/root/project-a`, create an isolated
Python 3.11 environment. Use the CUDA-enabled PyTorch supplied by the host or
install the wheel appropriate to that host; do not install a separate CUDA
Toolkit merely for this project. Then install the project and A5 libraries:

```bash
python -m pip install -e ".[dev]"
python -m pip install "transformers==4.57.6" "peft>=0.15,<1" \
  "accelerate>=1.2,<2" "safetensors>=0.4" "jinja2>=3.1"
```

Before loading the model, record `nvidia-smi`, Python, PyTorch, CUDA,
Transformers, PEFT, free disk space, and the SHA-256 of the migration archive.
Run the CPU/data gates:

```bash
python -m pytest -q
python scripts/run_a4_teacher_audit.py \
  artifacts/a4_teacher/qwen3_14b_train_v1 \
  --config configs/a4_teacher_sampling_14b.json
python scripts/prepare_a5_datasets.py
```

Place Qwen3-1.7B outside the repository and run the real local tokenizer audit:

```bash
python scripts/audit_a5_token_lengths.py --model /path/to/Qwen3-1.7B
```

Require 3,133 records, max length 2,629, zero 4,096-token overflow, and the same
chat-template hash before training. Any mismatch stops the run for diagnosis.

## Smoke gate

Run only 200 records first:

```bash
python scripts/run_a5_sft.py \
  --model /path/to/Qwen3-1.7B \
  --smoke-records 200
```

The smoke gate passes only if loss is finite, GPU memory is stable, a final
adapter and run manifest are written, and the saved adapter reloads for
inference. A failed smoke run is diagnosed; it is not bypassed by reducing
assertions or silently truncating data.

## Full run

After the smoke gate passes:

```bash
python scripts/run_a5_sft.py --model /path/to/Qwen3-1.7B
```

The first run stays on `main_14b`, rank 16, all-linear LoRA, learning rate
`1e-4`, two epochs, effective batch size 16, and one seed. Do not start the
recovery-augmented ablation until the base-vs-SFT development evaluation is
complete. Model checkpoints remain outside Git and must be separately archived
before the GPU instance is released.
