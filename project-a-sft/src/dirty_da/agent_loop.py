"""Model-agnostic A3 loop over the bounded three-tool environment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .agent_env import AgentEpisode
from .trajectory import AgentTrajectory


DEFAULT_SYSTEM_PROMPT = (
    "Solve the dirty-table analysis task using only the declared tools. "
    "Follow the user's cleaning and aggregation rules exactly. Use inspect_table "
    "and persistent run_python as needed, then call submit_answer with one "
    "JSON-compatible final answer. Never invent a value lost to missingness."
)


class AgentLoopError(ValueError):
    """Raised when a model adapter returns a malformed action."""


@dataclass(frozen=True)
class AssistantAction:
    content: str
    tool: str
    arguments: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssistantAction":
        if not isinstance(value, Mapping) or set(value) != {
            "content",
            "tool",
            "arguments",
        }:
            raise AgentLoopError(
                "Model action must contain exactly content, tool, and arguments."
            )
        content = value["content"]
        tool = value["tool"]
        arguments = value["arguments"]
        if not isinstance(content, str):
            raise AgentLoopError("Model action content must be a string.")
        if not isinstance(tool, str) or not tool:
            raise AgentLoopError("Model action tool must be a non-empty string.")
        if not isinstance(arguments, dict):
            raise AgentLoopError("Model action arguments must be an object.")
        return cls(content=content, tool=tool, arguments=deepcopy(arguments))


class ModelAdapter(Protocol):
    """Minimal interface implemented later by a local Qwen inference adapter."""

    def next_action(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> AssistantAction | Mapping[str, Any]: ...


def run_agent_episode(
    episode: AgentEpisode,
    model: ModelAdapter,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentTrajectory:
    """Run until submit, sandbox failure, or the environment step limit."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise AgentLoopError("system_prompt must be a non-empty string.")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{system_prompt} You may make at most "
                f"{episode.config.max_steps} tool calls."
            ),
        },
        {"role": "user", "content": episode.question},
    ]
    tools = episode.tool_schemas()
    while not episode.terminal:
        raw_action = model.next_action(deepcopy(messages), deepcopy(tools))
        action = (
            raw_action
            if isinstance(raw_action, AssistantAction)
            else AssistantAction.from_mapping(raw_action)
        )
        result = episode.call_tool(
            action.tool,
            action.arguments,
            assistant_content=action.content,
        )
        messages.append(
            {
                "role": "assistant",
                "content": action.content,
                "tool_call": {
                    "name": action.tool,
                    "arguments": deepcopy(action.arguments),
                },
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": action.tool,
                "ok": result.ok,
                "content": result.content,
                "error_type": result.error_type,
                "truncated": result.truncated,
            }
        )
    return episode.trajectory
