"""Run the bounded Qwen3-1.7B LoRA SFT after token-length preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a5_data import resolve_frozen_dataset, sha256_file  # noqa: E402
from dirty_da.a5_format import IGNORE_INDEX, encode_sft_record  # noqa: E402
from dirty_da.a5_train import (  # noqa: E402
    configure_gradient_checkpointing,
    validate_a5_training_config,
)


class EncodedDataset:
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "a5_sft.json")
    parser.add_argument("--model", default=None, help="Local model path; overrides model_id.")
    parser.add_argument("--smoke-records", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    try:
        import accelerate
        import peft
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        raise SystemExit('Install the project with the "training" extra on the GPU host.') from error

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_a5_training_config(config)
    if args.smoke_records is not None and args.smoke_records <= 0:
        raise SystemExit("--smoke-records must be positive.")

    manifest_path = ROOT / config["dataset_manifest"]
    records = resolve_frozen_dataset(
        ROOT, manifest_path, config["dataset_name"]
    )
    if args.smoke_records is not None:
        records = records[: args.smoke_records]
    model_path = args.model or config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    encoded = [
        encode_sft_record(
            record,
            tokenizer,
            max_length=int(config["max_seq_length"]),
        )
        for record in records
    ]
    training_config = config["training"]
    output_dir = ROOT / config["output_dir"]
    if args.smoke_records is not None:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if training_config["bf16"] else None,
        trust_remote_code=False,
    )
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            target_modules=str(lora["target_modules"]),
        ),
    )
    configure_gradient_checkpointing(
        model, enabled=bool(training_config["gradient_checkpointing"])
    )
    model.print_trainable_parameters()

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config["epochs"]),
        per_device_train_batch_size=int(training_config["micro_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=float(training_config["learning_rate"]),
        warmup_ratio=float(training_config["warmup_ratio"]),
        weight_decay=float(training_config["weight_decay"]),
        logging_steps=int(training_config["logging_steps"]),
        save_steps=int(training_config["save_steps"]),
        save_total_limit=int(training_config["save_total_limit"]),
        bf16=bool(training_config["bf16"]),
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        report_to=[],
        remove_unused_columns=False,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=EncodedDataset(encoded),
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    run_manifest = {
        "schema_version": 1,
        "config_sha256": sha256_file(args.config),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_name": config["dataset_name"],
        "record_count": len(records),
        "model": model_path,
        "smoke_records": args.smoke_records,
        "max_seq_length": config["max_seq_length"],
        "assistant_only_loss": True,
        "silent_truncation_allowed": False,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_metrics": result.metrics,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
