from collections import Counter
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dirty_da.a2_tasks import load_task_records
from dirty_da.qwen_adapter import (
    QwenAdapterError,
    QwenOutputError,
    QwenTransformersAdapter,
    normalize_messages,
    normalize_tool_schemas,
    parse_qwen_tool_call,
)
from scripts.run_a3_qwen_acceptance import select_frozen_records


ROOT = Path(__file__).resolve().parents[1]


TOOLS = [
    {
        "name": "run_python",
        "description": "Run code.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    }
]


def test_tool_schema_is_wrapped_for_transformers_without_mutating_source():
    normalized = normalize_tool_schemas(TOOLS)
    assert normalized[0]["type"] == "function"
    assert normalized[0]["function"]["name"] == "run_python"
    normalized[0]["function"]["parameters"]["required"].append("extra")
    assert TOOLS[0]["parameters"]["required"] == ["code"]


def test_internal_history_is_normalized_without_private_fields():
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Count rows."},
        {
            "role": "assistant",
            "content": "I will inspect.",
            "tool_call": {"name": "run_python", "arguments": {"code": "len(df)"}},
        },
        {
            "role": "tool",
            "name": "run_python",
            "ok": True,
            "content": "3",
            "error_type": None,
            "truncated": False,
        },
    ]
    normalized = normalize_messages(messages)
    assert normalized[2]["tool_calls"][0]["function"]["arguments"] == {
        "code": "len(df)"
    }
    assert normalized[3] == {"role": "tool", "name": "run_python", "content": "3"}
    assert "expected_answer" not in repr(normalized)


def test_parser_accepts_one_hermes_tool_call_and_json_string_arguments():
    action = parse_qwen_tool_call(
        '<think>brief</think>\n<tool_call>{"name":"run_python",'
        '"arguments":"{\\"code\\": \\"len(df)\\"}"}</tool_call><|im_end|>',
        {"run_python"},
    )
    assert action.tool == "run_python"
    assert action.arguments == {"code": "len(df)"}
    assert "brief" in action.content


def test_parser_accepts_bare_json_and_python_literal_tool_calls():
    bare = parse_qwen_tool_call(
        '{"name": "submit_answer", "arguments": {"answer": 379}}<|im_end|>',
        {"submit_answer"},
    )
    assert bare.tool == "submit_answer"
    assert bare.arguments == {"answer": 379}
    assert bare.content == ""

    python_literal = parse_qwen_tool_call(
        "<tool_call>{'name': 'submit_answer', 'arguments': {'answer': {4: 1.0}}}"
        "</tool_call><|im_end|>",
        {"submit_answer"},
    )
    assert python_literal.arguments == {"answer": {4: 1.0}}


@pytest.mark.parametrize(
    "text, message",
    [
        ("plain answer", "exactly one"),
        (
            '<tool_call>{"name":"bad","arguments":{}}</tool_call>',
            "undeclared",
        ),
        (
            '<tool_call>{"name":"run_python","arguments":{}}</tool_call>'
            '<tool_call>{"name":"run_python","arguments":{}}</tool_call>',
            "exactly one",
        ),
    ],
)
def test_parser_rejects_ambiguous_or_undeclared_output(text, message):
    with pytest.raises(QwenOutputError, match=message):
        parse_qwen_tool_call(text, {"run_python"})


def test_output_contract_errors_remain_adapter_errors():
    assert issubclass(QwenOutputError, QwenAdapterError)


def test_adapter_loads_saved_tokenizer_and_peft_weights(monkeypatch):
    calls = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["tokenizer"] = (path, kwargs)
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["base_model"] = (path, kwargs)
            return cls()

        def eval(self):
            calls["eval"] = True
            return self

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path):
            calls["adapter"] = (model, path)
            return model

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_peft = ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    adapter = QwenTransformersAdapter(
        "/models/base", adapter_name_or_path="/models/adapter"
    )

    assert calls["tokenizer"][0] == "/models/adapter"
    assert calls["base_model"][0] == "/models/base"
    assert calls["adapter"][1] == "/models/adapter"
    assert calls["eval"] is True
    assert adapter.model is calls["adapter"][0]


def test_frozen_acceptance_selection_is_deterministic_and_held_out():
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    config = json.loads(
        (ROOT / "configs" / "a3_qwen_acceptance.json").read_text(encoding="utf-8")
    )
    first = select_frozen_records(records, config)
    second = select_frozen_records(records, config)
    assert [r["task_id"] for r in first] == [r["task_id"] for r in second]
    assert len(first) == len({r["task_id"] for r in first}) == 50
    assert Counter(r["split"] for r in first) == {
        "test": 40,
        "domain_generalization": 10,
    }
    table_counts = Counter((r["split"], r["table"]) for r in first)
    assert all(
        count == (5 if split == "test" else 2)
        for (split, _table), count in table_counts.items()
    )
    assert {r["split"] for r in first}.isdisjoint({"train", "dev"})


def test_a5_dev_selection_is_fixed_and_never_uses_test_tables():
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    config = json.loads(
        (ROOT / "configs" / "a5_dev_eval.json").read_text(encoding="utf-8")
    )
    selected = select_frozen_records(records, config)
    assert len(selected) == len({record["task_id"] for record in selected}) == 40
    assert {record["split"] for record in selected} == {"dev"}
    assert set(Counter(record["table"] for record in selected).values()) == {10}
