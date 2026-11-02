from __future__ import annotations

import pytest

from dirty_da.a5_train import (
    configure_gradient_checkpointing,
    validate_a5_training_config,
)


def _config():
    return {
        "schema_version": 1,
        "assistant_only_loss": True,
        "overflow_policy": "error",
        "max_seq_length": 4096,
        "lora": {"r": 16, "target_modules": "all-linear"},
        "training": {"learning_rate": 1e-4, "micro_batch_size": 1},
    }


def test_a5_config_accepts_bounded_first_run():
    validate_a5_training_config(_config())


def test_a5_config_blocks_training_before_token_audit():
    config = _config()
    config["max_seq_length"] = None
    with pytest.raises(ValueError, match="tokenizer audit"):
        validate_a5_training_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("assistant_only_loss", False, "assistant_only_loss"),
        ("overflow_policy", "truncate", "reject over-length"),
    ],
)
def test_a5_config_rejects_semantic_shortcuts(field, value, message):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_a5_training_config(config)


def test_gradient_checkpointing_enables_peft_input_gradients():
    class Config:
        use_cache = True

    class Model:
        config = Config()

        def __init__(self):
            self.input_grads_enabled = False

        def enable_input_require_grads(self):
            self.input_grads_enabled = True

    model = Model()
    configure_gradient_checkpointing(model, enabled=True)
    assert model.config.use_cache is False
    assert model.input_grads_enabled is True


def test_disabled_gradient_checkpointing_does_not_require_peft_hook():
    configure_gradient_checkpointing(object(), enabled=False)
