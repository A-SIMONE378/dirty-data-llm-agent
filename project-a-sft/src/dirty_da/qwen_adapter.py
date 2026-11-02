"""Local Transformers adapter for the frozen Qwen3-8B A3 acceptance run.

Heavy inference libraries are imported only when the adapter is constructed so
the CPU-only A0--A3 test environment stays lightweight.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence

from .agent_loop import AssistantAction


class QwenAdapterError(RuntimeError):
    """Raised when model input or generated tool-call output is invalid."""


class QwenOutputError(QwenAdapterError):
    """Raised when inference succeeded but Qwen violated the tool-call contract."""


_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|(?:im_end|endoftext)\|>")


def normalize_tool_schemas(
    tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert the internal three-tool schema to Transformers JSON schemas."""

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if set(tool) != {"name", "description", "parameters"}:
            raise QwenAdapterError("Tool schema has unexpected fields.")
        name = tool["name"]
        if not isinstance(name, str) or not name:
            raise QwenAdapterError("Tool schema name must be a non-empty string.")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool["description"]),
                    "parameters": deepcopy(tool["parameters"]),
                },
            }
        )
    return normalized


def normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate the leakage-safe internal history to Qwen chat-template form."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "user"}:
            content = message.get("content")
            if not isinstance(content, str):
                raise QwenAdapterError(f"{role} message content must be a string.")
            normalized.append({"role": role, "content": content})
            continue
        if role == "assistant":
            call = message.get("tool_call")
            if not isinstance(call, Mapping) or set(call) != {"name", "arguments"}:
                raise QwenAdapterError("Assistant history has an invalid tool call.")
            normalized.append(
                {
                    "role": "assistant",
                    "content": str(message.get("content", "")),
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": deepcopy(call["arguments"]),
                            },
                        }
                    ],
                }
            )
            continue
        if role == "tool":
            content = message.get("content")
            name = message.get("name")
            if not isinstance(content, str) or not isinstance(name, str):
                raise QwenAdapterError("Tool history must contain string name/content.")
            if message.get("ok") is False:
                error_type = message.get("error_type") or "ToolError"
                content = f"ERROR [{error_type}]: {content}"
            normalized.append({"role": "tool", "name": name, "content": content})
            continue
        raise QwenAdapterError(f"Unsupported message role: {role!r}")
    return normalized


def _literal_payload(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            snippet = raw[:240].replace("\r", "\\r").replace("\n", "\\n")
            raise QwenOutputError(
                f"Qwen tool-call JSON is malformed; raw prefix={snippet!r}."
            ) from exc


def parse_qwen_tool_call(text: str, allowed_tools: set[str]) -> AssistantAction:
    """Parse one Qwen tool call without guessing free-form final answers.

    The model occasionally omits the Hermes ``<tool_call>`` wrapper or emits a
    Python-literal mapping instead of strict JSON.  Both forms still name an
    explicit declared tool and arguments, so they are parsed exactly rather
    than repaired or inferred.
    """

    if not isinstance(text, str) or not text.strip():
        raise QwenOutputError("Qwen returned empty output.")
    matches = list(_TOOL_CALL_PATTERN.finditer(text))
    if len(matches) > 1:
        raise QwenOutputError(
            f"Qwen must return exactly one tool call; found {len(matches)}."
        )
    if matches:
        raw_payload = matches[0].group(1)
        content = _SPECIAL_TOKEN_PATTERN.sub("", text[: matches[0].start()]).strip()
    else:
        stripped = _SPECIAL_TOKEN_PATTERN.sub("", text).strip()
        if stripped.startswith("{") and "</tool_call>" in stripped:
            stripped = stripped.split("</tool_call>", 1)[0].rstrip()
        if not stripped.startswith("{"):
            raise QwenOutputError("Qwen must return exactly one tool call; found 0.")
        raw_payload = stripped
        content = ""
    payload = _literal_payload(raw_payload)
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise QwenOutputError(
            "Qwen tool call must contain exactly name and arguments."
        )
    name = payload["name"]
    arguments = payload["arguments"]
    if name not in allowed_tools:
        raise QwenOutputError(f"Qwen requested an undeclared tool: {name!r}.")
    if isinstance(arguments, str):
        arguments = _literal_payload(arguments)
    if not isinstance(arguments, dict):
        raise QwenOutputError("Qwen tool arguments must be a JSON object.")
    return AssistantAction(content=content, tool=name, arguments=arguments)


class QwenTransformersAdapter:
    """One-GPU deterministic Qwen3 adapter implementing ``ModelAdapter``."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_name_or_path: str | None = None,
        cache_dir: str | None = None,
        max_new_tokens: int = 768,
        max_context_tokens: int = 28672,
        trust_remote_code: bool = False,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        if not model_name_or_path:
            raise ValueError("model_name_or_path must be non-empty.")
        if max_new_tokens < 64 or max_context_tokens <= max_new_tokens:
            raise ValueError("Invalid generation/context token limits.")
        if do_sample:
            if temperature is None or temperature <= 0:
                raise ValueError("Sampled generation requires a positive temperature.")
            if top_p is None or not 0 < top_p <= 1:
                raise ValueError("Sampled generation requires top_p in (0, 1].")
        elif temperature is not None or top_p is not None:
            raise ValueError("temperature/top_p require do_sample=True.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise QwenAdapterError(
                "Install the project inference extra before loading Qwen."
            ) from exc
        if not torch.cuda.is_available():
            raise QwenAdapterError("CUDA is required for the Qwen3-8B acceptance run.")
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        tokenizer_name_or_path = adapter_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        if adapter_name_or_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise QwenAdapterError(
                    "Install PEFT before loading a LoRA adapter."
                ) from exc
            model = PeftModel.from_pretrained(model, adapter_name_or_path)
        self.model = model.eval()

    def reset_sampling_seed(self, seed: int) -> None:
        """Reset RNG state once at a trajectory boundary for reproducible sampling."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Sampling seed must be a non-negative integer.")
        self._torch.manual_seed(seed)
        self._torch.cuda.manual_seed_all(seed)

    def next_action(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AssistantAction:
        normalized_tools = normalize_tool_schemas(tools)
        normalized_messages = normalize_messages(messages)
        encoded = self.tokenizer.apply_chat_template(
            normalized_messages,
            tools=normalized_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens + self.max_new_tokens > self.max_context_tokens:
            raise QwenAdapterError(
                f"Conversation requires {prompt_tokens} prompt tokens, exceeding the "
                "configured context budget."
            )
        encoded = encoded.to(self.model.device)
        generation_options: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_options.update(
                temperature=self.temperature,
                top_p=self.top_p,
            )
        with self._torch.inference_mode():
            output = self.model.generate(**encoded, **generation_options)
        generated_ids = output[0, prompt_tokens:]
        generated = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        allowed = {tool["name"] for tool in tools}
        return parse_qwen_tool_call(generated, allowed)
