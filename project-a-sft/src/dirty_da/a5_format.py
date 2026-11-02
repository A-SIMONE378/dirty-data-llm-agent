"""Qwen chat-template encoding with strict assistant-only loss labels."""

from __future__ import annotations

from typing import Any, Mapping

from .a5_data import validate_sft_record


IGNORE_INDEX = -100


def _template_ids(
    tokenizer: Any,
    messages: list[Mapping[str, Any]],
    tools: Any,
) -> tuple[list[int], list[int]]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        enable_thinking=False,
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("Tokenizer did not return a mapping.")
    input_ids = list(encoded.get("input_ids", []))
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    if not input_ids or len(input_ids) != len(attention_mask):
        raise ValueError("Tokenizer returned inconsistent A5 encoding lengths.")
    return input_ids, attention_mask


def _subsequence_starts(values: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        raise ValueError("Assistant header token pattern is empty.")
    width = len(pattern)
    return [
        index
        for index in range(len(values) - width + 1)
        if values[index : index + width] == pattern
    ]


def encode_sft_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Encode one trajectory and fail rather than silently truncating it."""

    validate_sft_record(record)
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer.")
    messages = list(record["messages"])
    tools = record.get("tools")
    input_ids, attention_mask = _template_ids(tokenizer, messages, tools)
    assistant_mask = [0] * len(input_ids)
    assistant_count = sum(message["role"] == "assistant" for message in messages)
    header_ids = list(
        tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    )
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(end_id, int) or end_id < 0:
        raise ValueError("Qwen tokenizer does not expose the <|im_end|> token.")
    starts = _subsequence_starts(input_ids, header_ids)
    if len(starts) != assistant_count:
        raise ValueError(
            "Qwen rendered assistant boundary count does not match the trajectory."
        )
    prior_end = -1
    for start in starts:
        output_start = start + len(header_ids)
        try:
            output_end = input_ids.index(end_id, output_start)
        except ValueError as error:
            raise ValueError("Qwen assistant message has no <|im_end|> boundary.") from error
        if start <= prior_end:
            raise ValueError("Qwen assistant token boundaries overlap.")
        for position in range(output_start, output_end + 1):
            assistant_mask[position] = 1
        prior_end = output_end
    if len(input_ids) > max_length:
        raise ValueError(
            f"Encoded trajectory has {len(input_ids)} tokens, above max_length={max_length}; "
            "A5 forbids silent truncation."
        )
    labels = [
        token if bool(mask) else IGNORE_INDEX
        for token, mask in zip(input_ids, assistant_mask)
    ]
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("Encoded trajectory has no trainable assistant tokens.")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
