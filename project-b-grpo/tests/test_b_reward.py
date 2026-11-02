from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.b_reward import RewardConfig, compute_b_reward  # noqa: E402


def _trajectory(
    *,
    code="len(df)",
    content="3",
    answer=3,
    status="accepted",
    submit=True,
    extra=None,
):
    events = [
        {
            "step": 1,
            "assistant_content": "Compute from data.",
            "tool": "run_python",
            "arguments": {"code": code},
            "ok": True,
            "content": content,
            "truncated": False,
            "error_type": None,
        },
    ]
    if submit:
        events.append(
            {
                "step": 2,
                "assistant_content": "Submit.",
                "tool": "submit_answer",
                "arguments": {"answer": answer},
                "ok": status == "accepted",
                "content": "accepted" if status == "accepted" else "rejected",
                "truncated": False,
                "error_type": None if status == "accepted" else "VerificationError",
            }
        )
    value = {
        "schema_version": 1,
        "task_id": "task-1",
        "table": "table.csv",
        "split": "train",
        "question": "Count rows.",
        "dirty_profile": "low",
        "max_steps": 10,
        "terminal_status": status,
        "submitted_answer": answer,
        "events": events,
    }
    if extra:
        value.update(extra)
    return value


def test_b2_verified_dataframe_trajectory_gets_full_reward():
    breakdown = compute_b_reward(_trajectory())
    assert breakdown.total == 1.0
    assert breakdown.outcome == 1.0
    assert breakdown.format == 0.05
    assert breakdown.anti_cheat is False


def test_b2_no_dataframe_analysis_is_zero():
    breakdown = compute_b_reward(_trajectory(code="3"))
    assert breakdown.total == 0.0
    assert breakdown.anti_cheat is True
    assert "no_successful_dataframe_analysis" in breakdown.reasons


def test_b2_hardcoded_dataframe_is_zero():
    breakdown = compute_b_reward(
        _trajectory(code='df = pd.DataFrame({"x": [1]})\nlen(df)')
    )
    assert breakdown.total == 0.0
    assert breakdown.anti_cheat is True
    assert "hardcoded_dataframe" in breakdown.reasons


def test_b2_unsupported_answer_is_zero():
    breakdown = compute_b_reward(_trajectory(content="4", answer=3))
    assert breakdown.total == 0.0
    assert breakdown.anti_cheat is True
    assert "submitted_answer_not_supported_by_tool_output" in breakdown.reasons


def test_b2_rejected_answer_is_zero():
    breakdown = compute_b_reward(_trajectory(status="rejected"))
    assert breakdown.total == 0.0
    assert breakdown.outcome == 0.0
    assert breakdown.anti_cheat is False


def test_b2_efficiency_penalty_is_optional():
    breakdown = compute_b_reward(
        _trajectory(), RewardConfig(step_penalty=0.05)
    )
    assert breakdown.efficiency == -0.1
    assert breakdown.total == pytest.approx(0.95)


def test_b2_private_field_forces_zero():
    breakdown = compute_b_reward(
        _trajectory(extra={"expected_answer": 3})
    )
    assert breakdown.total == 0.0
    assert breakdown.anti_cheat is True
    assert "private_field_leakage" in breakdown.reasons


def test_b2_config_is_bounded():
    with pytest.raises(ValueError):
        RewardConfig(format_bonus=0.5)
    with pytest.raises(ValueError):
        RewardConfig(step_penalty=0.5)
