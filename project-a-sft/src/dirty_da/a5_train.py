"""Validation helpers for the bounded A5 LoRA training configuration."""

from __future__ import annotations

from typing import Any, Mapping


def validate_a5_training_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported A5 config schema version.")
    if config.get("assistant_only_loss") is not True:
        raise ValueError("A5 requires assistant_only_loss=true.")
    if config.get("overflow_policy") != "error":
        raise ValueError("A5 must reject over-length records instead of truncating them.")
    max_length = config.get("max_seq_length")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("Run the tokenizer audit and set a positive max_seq_length first.")
    lora = config.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("A5 LoRA config is missing.")
    rank = lora.get("r")
    if not isinstance(rank, int) or not 16 <= rank <= 64:
        raise ValueError("A5 LoRA rank must stay within the planned 16-64 range.")
    if lora.get("target_modules") != "all-linear":
        raise ValueError("A5 uses the frozen all-linear LoRA target policy.")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("A5 training config is missing.")
    learning_rate = training.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("A5 learning_rate must be positive.")
    if training.get("micro_batch_size") != 1:
        raise ValueError("The first A5 run freezes micro_batch_size=1 for 24 GB GPUs.")


def configure_gradient_checkpointing(model: Any, *, enabled: bool) -> None:
    """Make PEFT input embeddings participate in checkpointed backpropagation."""

    if not enabled:
        return
    if not hasattr(model, "enable_input_require_grads"):
        raise ValueError("Model does not support input gradients for checkpointing.")
    model.config.use_cache = False
    model.enable_input_require_grads()
