"""Leakage-aware trajectory records for the bounded A3 agent environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


TRAJECTORY_SCHEMA_VERSION = 1
TERMINAL_STATUSES = {
    "running",
    "accepted",
    "rejected",
    "max_steps",
    "sandbox_error",
    "invalid_action",
    "model_error",
}


@dataclass(frozen=True)
class ToolEvent:
    step: int
    assistant_content: str
    tool: str
    arguments: dict[str, Any]
    ok: bool
    content: str
    truncated: bool
    error_type: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("ToolEvent.step must be a positive integer.")
        if not isinstance(self.assistant_content, str):
            raise TypeError("assistant_content must be a string.")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError("tool must be a non-empty string.")
        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dictionary.")
        if not isinstance(self.ok, bool) or not isinstance(self.truncated, bool):
            raise TypeError("ok and truncated must be boolean.")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string.")
        if self.error_type is not None and not isinstance(self.error_type, str):
            raise TypeError("error_type must be a string or None.")


@dataclass
class AgentTrajectory:
    task_id: str
    table: str
    split: str
    question: str
    dirty_profile: str
    max_steps: int
    events: list[ToolEvent] = field(default_factory=list)
    terminal_status: str = "running"
    submitted_answer: Any = None

    def __post_init__(self) -> None:
        for name in ("task_id", "table", "split", "question", "dirty_profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError("max_steps must be an integer.")
        if not 1 <= self.max_steps <= 100:
            raise ValueError("max_steps must be between 1 and 100.")
        if self.terminal_status not in TERMINAL_STATUSES:
            raise ValueError(f"Unsupported terminal status: {self.terminal_status}")

    def append(self, event: ToolEvent) -> None:
        if self.terminal_status != "running":
            raise RuntimeError("Cannot append to a terminal trajectory.")
        if event.step != len(self.events) + 1:
            raise ValueError("Tool event step is not consecutive.")
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        """Serialize without task spec, oracle data, or expected answer fields."""

        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "task_id": self.task_id,
            "table": self.table,
            "split": self.split,
            "question": self.question,
            "dirty_profile": self.dirty_profile,
            "max_steps": self.max_steps,
            "terminal_status": self.terminal_status,
            "submitted_answer": self.submitted_answer,
            "events": [asdict(event) for event in self.events],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def validate_trajectory_record(value: Mapping[str, Any]) -> None:
    """Validate the public, leakage-safe trajectory serialization."""

    allowed = {
        "schema_version",
        "task_id",
        "table",
        "split",
        "question",
        "dirty_profile",
        "max_steps",
        "terminal_status",
        "submitted_answer",
        "events",
    }
    if not isinstance(value, Mapping) or set(value) != allowed:
        raise ValueError("Trajectory fields differ from the A3 schema.")
    if value["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory schema version.")
    if value["terminal_status"] not in TERMINAL_STATUSES:
        raise ValueError("Unsupported trajectory terminal status.")
    events = value["events"]
    if not isinstance(events, list):
        raise TypeError("events must be a list.")
    for index, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or event.get("step") != index:
            raise ValueError("Trajectory event steps must be consecutive.")
    forbidden = {"answer", "expected_answer", "oracle", "spec", "task_spec"}
    if forbidden & set(value):
        raise ValueError("Trajectory exposes a forbidden oracle field.")
