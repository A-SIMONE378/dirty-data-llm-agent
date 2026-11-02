"""B2 deterministic verifiable reward with anti-hardcoding and efficiency terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .a4_teacher import assess_teacher_trajectory


ANTI_CHEAT_REASONS = {
    "invalid_trajectory_schema",
    "private_field_leakage",
    "no_successful_dataframe_analysis",
    "submitted_answer_not_supported_by_tool_output",
}
HARDCODED_MARKERS = ("pd.dataframe(", "pd.read_csv(", "dataframe(")


@dataclass(frozen=True)
class RewardConfig:
    format_bonus: float = 0.05
    step_penalty: float = 0.0
    max_reward: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.format_bonus <= 0.2:
            raise ValueError("format_bonus must be in [0, 0.2].")
        if not 0.0 <= self.step_penalty <= 0.1:
            raise ValueError("step_penalty must be in [0, 0.1].")
        if self.max_reward <= 0:
            raise ValueError("max_reward must be positive.")


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    outcome: float
    format: float
    efficiency: float
    anti_cheat: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "outcome": self.outcome,
            "format": self.format,
            "efficiency": self.efficiency,
            "anti_cheat": self.anti_cheat,
            "reasons": list(self.reasons),
        }


def _hardcoded_dataframe(trajectory: Mapping[str, Any]) -> bool:
    events = trajectory.get("events")
    if not isinstance(events, list):
        return False
    code = "\n".join(
        str(event.get("arguments", {}).get("code", ""))
        for event in events
        if isinstance(event, Mapping)
        and event.get("tool") == "run_python"
    ).lower()
    return any(marker in code for marker in HARDCODED_MARKERS)


def _valid_submit(trajectory: Mapping[str, Any]) -> bool:
    events = trajectory.get("events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, Mapping)
        and event.get("tool") == "submit_answer"
        and event.get("ok") is True
        and isinstance(event.get("arguments"), Mapping)
        and "answer" in event["arguments"]
        for event in events
    )


def compute_b_reward(
    trajectory: Mapping[str, Any], config: RewardConfig | None = None
) -> RewardBreakdown:
    """Return a bounded reward; any anti-cheat hit forces the total to zero."""

    config = config or RewardConfig()
    assessment = assess_teacher_trajectory({}, trajectory)
    reasons = list(assessment.reasons)
    hardcoded = _hardcoded_dataframe(trajectory)
    anti_cheat = hardcoded or bool(ANTI_CHEAT_REASONS.intersection(reasons))
    if hardcoded:
        reasons.append("hardcoded_dataframe")

    accepted = trajectory.get("terminal_status") == "accepted"
    outcome = 1.0 if accepted else 0.0
    format_reward = (
        config.format_bonus if accepted and _valid_submit(trajectory) else 0.0
    )
    events = trajectory.get("events")
    step_count = len(events) if isinstance(events, list) else 0
    efficiency = -config.step_penalty * step_count
    if anti_cheat:
        total = 0.0
    else:
        total = max(0.0, min(config.max_reward, outcome + format_reward + efficiency))
    return RewardBreakdown(
        total=total,
        outcome=outcome,
        format=format_reward,
        efficiency=efficiency,
        anti_cheat=anti_cheat,
        reasons=tuple(dict.fromkeys(reasons)),
    )
