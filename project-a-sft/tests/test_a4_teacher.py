from collections import Counter
import json
from pathlib import Path

from dirty_da.a2_tasks import load_task_records
from dirty_da.a4_teacher import (
    assess_teacher_trajectory,
    build_sampling_attempts,
    build_sft_record,
    select_teacher_tasks,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads(
        (ROOT / "configs" / "a4_teacher_sampling.json").read_text(encoding="utf-8")
    )


def _trajectory(*, code="len(df)", content="3", answer=3, failed_first=False):
    events = []
    if failed_first:
        events.append(
            {
                "step": 1,
                "assistant_content": "Try analysis.",
                "tool": "run_python",
                "arguments": {"code": "df['missing']"},
                "ok": False,
                "content": "missing",
                "truncated": False,
                "error_type": "KeyError",
            }
        )
    events.extend(
        [
            {
                "step": len(events) + 1,
                "assistant_content": "Compute from data.",
                "tool": "run_python",
                "arguments": {"code": code},
                "ok": True,
                "content": content,
                "truncated": False,
                "error_type": None,
            },
            {
                "step": len(events) + 2,
                "assistant_content": "Submit computed result.",
                "tool": "submit_answer",
                "arguments": {"answer": answer},
                "ok": True,
                "content": "accepted",
                "truncated": False,
                "error_type": None,
            },
        ]
    )
    return {
        "schema_version": 1,
        "task_id": "task-1",
        "table": "table.csv",
        "split": "train",
        "question": "Count rows.",
        "dirty_profile": "low",
        "max_steps": 12,
        "terminal_status": "accepted",
        "submitted_answer": answer,
        "events": events,
    }


def test_a4_selection_is_train_only_balanced_and_eight_samples_each():
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    config = _config()
    selected = select_teacher_tasks(records, config)
    attempts = build_sampling_attempts(selected, config)
    assert len(selected) == 800
    assert {record["split"] for record in selected} == {"train"}
    table_counts = Counter(record["table"] for record in selected)
    assert len(table_counts) == 28
    assert max(table_counts.values()) - min(table_counts.values()) <= 1
    assert len(attempts) == len({a["attempt_id"] for a in attempts}) == 6400
    assert set(Counter(a["task_id"] for a in attempts).values()) == {8}


def test_quality_filter_requires_dataframe_analysis_and_tool_supported_answer():
    accepted = assess_teacher_trajectory({}, _trajectory())
    assert accepted.retained
    assert accepted.category == "direct_success"
    assert accepted.uses_tool_feedback

    hardcoded = assess_teacher_trajectory({}, _trajectory(code="3"))
    assert not hardcoded.retained
    assert "no_successful_dataframe_analysis" in hardcoded.reasons

    unsupported = assess_teacher_trajectory({}, _trajectory(content="4"))
    assert not unsupported.retained
    assert "submitted_answer_not_supported_by_tool_output" in unsupported.reasons


def test_quality_categories_cleaning_and_error_recovery():
    cleaning = assess_teacher_trajectory(
        {}, _trajectory(code="df.dropna()['x'].sum()", content="3.0", answer=3.0)
    )
    assert cleaning.retained and cleaning.category == "pollution_cleaning"
    recovery = assess_teacher_trajectory({}, _trajectory(failed_first=True))
    assert recovery.retained and recovery.category == "error_recovery"


def test_tool_evidence_normalizes_python_mapping_keys_to_json_strings():
    grouped = assess_teacher_trajectory(
        {}, _trajectory(content="{0: 342, 1: 27}", answer={"0": 342, "1": 27})
    )
    assert grouped.retained
    assert grouped.uses_tool_feedback


def test_tool_evidence_accepts_traceable_record_reshape_and_numpy_repr():
    reshaped = assess_teacher_trajectory(
        {},
        _trajectory(
            content="[{'group': 0, 'count': 342}, {'group': 1, 'count': 27}]",
            answer={"0": 342, "1": 27},
        ),
    )
    assert reshaped.retained
    numpy_scalar = assess_teacher_trajectory(
        {}, _trajectory(content="{'Link': np.int64(2307)}", answer={"Link": 2307})
    )
    assert numpy_scalar.retained


def test_sft_record_masks_non_assistant_messages_and_contains_no_oracle():
    trajectory = _trajectory()
    assessment = assess_teacher_trajectory({}, trajectory)
    record = build_sft_record(
        attempt={"attempt_id": "attempt-1", "sample_index": 0, "sampling_seed": 7},
        trajectory=trajectory,
        assessment=assessment,
        system_prompt="Use tools.",
        tools=[],
        prompt_version="test-v1",
    )
    assert record["split"] == "train"
    assert all(
        message["trainable"] == (message["role"] == "assistant")
        for message in record["messages"]
    )
    serialized = json.dumps(record)
    assert "expected_answer" not in serialized
    assert "task_spec" not in serialized

