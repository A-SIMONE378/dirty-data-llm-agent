"""Run a bounded TRL GRPO smoke/full job on the A3 sandbox environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dirty_da.b4_env import B4EnvironmentFactory, load_b4_task_ids  # noqa: E402
from dirty_da.b_reward import RewardConfig  # noqa: E402
from run_a3_qwen_acceptance import MODEL_SYSTEM_PROMPT  # noqa: E402


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b4_grpo_smoke.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = _load_json(args.config)
    model_path = args.model or str(config["model_id"])
    adapter_path = args.adapter or str(config["adapter_path"])
    output_dir = Path(args.output_dir or config["output_dir"])
    task_limit = args.task_limit if args.task_limit is not None else config.get("task_limit")
    task_ids = load_b4_task_ids(ROOT, limit=int(task_limit) if task_limit is not None else None)
    records = B4EnvironmentFactory(
        project_root=ROOT,
        task_ids=task_ids,
        runtime_root=ROOT / ".b4_runtime",
    ).records

    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise SystemExit(
            "Run this script inside the project-b environment with TRL 1.16 "
            "and transformers>=5.2."
        ) from error

    max_tools = int(config["max_tool_calling_iterations"])
    dataset = Dataset.from_list(
        [
            {
                "task_id": task_id,
                "prompt": [
                    {"role": "system", "content": MODEL_SYSTEM_PROMPT + f" You may make at most {max_tools} tool calls. After submit_answer, stop calling tools."},
                    {"role": "user", "content": records[task_id]["question"]},
                ],
            }
            for task_id in task_ids
        ]
    )
    reward_config = RewardConfig(
        format_bonus=float(config.get("format_bonus", 0.05)),
        step_penalty=float(config.get("step_penalty", 0.0)),
    )
    factory = B4EnvironmentFactory(
        project_root=ROOT,
        task_ids=task_ids,
        runtime_root=ROOT / ".b4_runtime",
        reward_config=reward_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("Tokenizer has no pad or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)

    arguments = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=int(config["max_steps"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_generations=int(config["num_generations"]),
        max_completion_length=int(config["max_completion_length"]),
        max_tool_calling_iterations=int(config["max_tool_calling_iterations"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        beta=float(config["beta"]),
        learning_rate=float(config["learning_rate"]),
        logging_steps=int(config["logging_steps"]),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(config["seed"]),
        use_vllm=False,
        log_completions=bool(config.get("log_completions", False)),
    )
    trainer = GRPOTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=factory,
    )
    started = time.monotonic()
    train_result = trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    manifest = {
        "schema_version": 1,
        "config_sha256": __import__("hashlib").sha256(args.config.read_bytes()).hexdigest(),
        "model_path": model_path,
        "adapter_path": adapter_path,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "max_steps": int(config["max_steps"]),
        "num_generations": int(config["num_generations"]),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "train_metrics": train_result.metrics,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
