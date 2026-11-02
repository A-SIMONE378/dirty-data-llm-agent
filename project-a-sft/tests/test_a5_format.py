from __future__ import annotations

import pytest

from dirty_da.a5_format import IGNORE_INDEX, encode_sft_record


def _record():
    return {
        "split": "train",
        "category": "direct_success",
        "trajectory_id": "trajectory-1",
        "trajectory_fingerprint": "fingerprint-1",
        "task_id": "task-1",
        "table": "table.csv",
        "tools": [],
        "messages": [
            {"role": "system", "content": "system", "trainable": False},
            {"role": "user", "content": "question", "trainable": False},
            {"role": "assistant", "content": "answer", "trainable": True},
        ],
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        ids = []
        for message in messages:
            if message["role"] == "system":
                ids.extend([10, 11])
            elif message["role"] == "user":
                ids.extend([20, 21])
            elif message["role"] == "assistant":
                ids.extend([30, 31, 32])
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    def encode(self, value, add_special_tokens=False):
        assert value == "<|im_start|>assistant\n"
        return [30]

    def convert_tokens_to_ids(self, value):
        assert value == "<|im_end|>"
        return 32


def test_a5_encoding_masks_every_non_assistant_token():
    encoded = encode_sft_record(_record(), FakeTokenizer(), max_length=7)
    assert encoded["labels"] == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        31,
        32,
    ]


def test_a5_encoding_refuses_silent_truncation():
    with pytest.raises(ValueError, match="forbids silent truncation"):
        encode_sft_record(_record(), FakeTokenizer(), max_length=6)


def test_a5_encoding_rejects_missing_assistant_boundary():
    class MissingBoundaryTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            encoded = super().apply_chat_template(messages, **kwargs)
            encoded["input_ids"] = [value for value in encoded["input_ids"] if value != 30]
            encoded["attention_mask"] = [1] * len(encoded["input_ids"])
            return encoded

    with pytest.raises(ValueError, match="boundary count"):
        encode_sft_record(_record(), MissingBoundaryTokenizer(), max_length=7)


def test_a5_encoding_requires_at_least_one_trainable_token():
    record = _record()
    record["messages"] = record["messages"][:2]
    with pytest.raises(ValueError, match="no trainable assistant"):
        encode_sft_record(record, FakeTokenizer(), max_length=4)
