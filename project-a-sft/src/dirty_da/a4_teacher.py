"""Deterministic A4 teacher sampling, quality filtering, and SFT records."""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .trajectory import validate_trajectory_record
from .verifier import verify_result


FORBIDDEN_PRIVATE_FIELDS = (
    '"expected_answer"',
    '"task_spec"',
    '"canonical_fingerprint"',
)
CLEANING_MARKERS = (
    "dropna",
    "fillna",
    "drop_duplicates",
    "replace(",
    "to_numeric",
    "to_datetime",
    "astype(",
    ".str.",
)


class A4TeacherError(ValueError):
    """Raised when A4 configuration or persisted evidence is inconsistent."""


@dataclass(frozen=True)
class QualityAssessment:
    retained: bool
    category: str | None
    reasons: tuple[str, ...]
    uses_tool_feedback: bool
    trajectory_fingerprint: str


def _hash_rank(seed: str, *parts: str) -> str:
    payload = "|".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_from_table(
    records: Sequence[dict[str, Any]], count: int, seed: str, table: str
) -> list[dict[str, Any]]:
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_difficulty[record["difficulty"]].append(record)
    for difficulty in by_difficulty:
        by_difficulty[difficulty].sort(
            key=lambda record: _hash_rank(seed, record["task_id"])
        )
    difficulties = sorted(
        by_difficulty,
        key=lambda difficulty: _hash_rank(seed, table, difficulty),
    )
    selected: list[dict[str, Any]] = []
    offsets = {difficulty: 0 for difficulty in difficulties}
    while len(selected) < count:
        progressed = False
        for difficulty in difficulties:
            offset = offsets[difficulty]
            candidates = by_difficulty[difficulty]
            if offset < len(candidates):
                selected.append(candidates[offset])
                offsets[difficulty] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise A4TeacherError(f"Not enough tasks in training table {table}.")
    return selected


def select_teacher_tasks(
    records: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Select a deterministic, table-balanced subset from training tables only."""

    if config.get("schema_version") != 1 or config.get("split") != "train":
        raise A4TeacherError("A4 supports only schema v1 training-task sampling.")
    count = config.get("selected_task_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise A4TeacherError("selected_task_count must be a positive integer.")
    seed = config.get("selection_seed")
    if not isinstance(seed, str) or not seed:
        raise A4TeacherError("selection_seed must be a non-empty string.")
    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") == "train":
            by_table[record["table"]].append(record)
    if not by_table or count < len(by_table):
        raise A4TeacherError("Selection must cover every training table.")
    if count > sum(len(values) for values in by_table.values()):
        raise A4TeacherError("selected_task_count exceeds available training tasks.")

    tables = sorted(by_table, key=lambda table: _hash_rank(seed, table))
    base, remainder = divmod(count, len(tables))
    selected_by_table: dict[str, list[dict[str, Any]]] = {}
    for index, table in enumerate(tables):
        table_count = base + (1 if index < remainder else 0)
        selected_by_table[table] = _select_from_table(
            by_table[table], table_count, seed, table
        )

    # Interleave tables so a bounded pilot covers domains rather than exhausting
    # one table first.
    selected: list[dict[str, Any]] = []
    for offset in range(max(map(len, selected_by_table.values()))):
        for table in tables:
            table_records = selected_by_table[table]
            if offset < len(table_records):
                selected.append(table_records[offset])
    if len(selected) != count or len({r["task_id"] for r in selected}) != count:
        raise A4TeacherError("A4 selection is incomplete or contains duplicates.")
    return selected


def build_sampling_attempts(
    records: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    samples = config.get("samples_per_task")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise A4TeacherError("samples_per_task must be a positive integer.")
    seed = str(config["selection_seed"])
    attempts: list[dict[str, Any]] = []
    for sample_index in range(samples):
        for record in records:
            digest = hashlib.sha256(
                f"{seed}|{record['task_id']}|{sample_index}".encode("utf-8")
            ).digest()
            attempts.append(
                {
                    "attempt_id": f"a4-{record['task_id']}-s{sample_index:02d}",
                    "task_id": record["task_id"],
                    "sample_index": sample_index,
                    "sampling_seed": int.from_bytes(digest[:8], "big")
                    % (2**31),
                    "record": record,
                }
            )
    return attempts


def _parsed_tool_value(content: str) -> Any:
    try:
        return ast.literal_eval(content.strip())
    except (SyntaxError, ValueError):
        try:
            return json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return object()


def _normalize_json_keys(value: Any) -> Any:
    """Normalize mapping keys as JSON would when a Python result is submitted."""

    if isinstance(value, dict):
        return {
            str(key): _normalize_json_keys(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_json_keys(item) for item in value]
    return value


def _scalar_evidence_tokens(value: Any) -> list[Any]:
    if isinstance(value, dict):
        tokens: list[Any] = []
        for key, item in value.items():
            tokens.append(key)
            tokens.extend(_scalar_evidence_tokens(item))
        return tokens
    if isinstance(value, list):
        return [token for item in value for token in _scalar_evidence_tokens(item)]
    return [value]


def _content_contains_scalar(content: str, value: Any) -> bool:
    if value is None:
        return bool(re.search(r"(?<!\w)(?:None|null)(?!\w)", content))
    if isinstance(value, bool):
        options = ("True", "true") if value else ("False", "false")
        return any(option in content for option in options)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = re.escape(str(value))
        return bool(re.search(rf"(?<![\w.]){number}(?![\w.])", content))
    return str(value) in content


def _answer_supported_by_content(content: str, submitted: Any) -> bool:
    parsed = _normalize_json_keys(_parsed_tool_value(content))
    if verify_result(parsed, submitted).passed:
        return True
    tokens = _scalar_evidence_tokens(submitted)
    return bool(tokens) and all(
        _content_contains_scalar(content, token) for token in tokens
    )


def _code_reads_dataframe(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Name) and node.id == "df" for node in ast.walk(tree))


def _fingerprint_trajectory(trajectory: Mapping[str, Any]) -> str:
    material = {
        "question": trajectory.get("question"),
        "events": trajectory.get("events"),
        "submitted_answer": trajectory.get("submitted_answer"),
    }
    return hashlib.sha256(
        json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def assess_teacher_trajectory(
    task_record: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> QualityAssessment:
    """Apply mechanical rejection-sampling and anti-hardcoding checks."""

    reasons: list[str] = []
    try:
        validate_trajectory_record(trajectory)
    except (TypeError, ValueError):
        reasons.append("invalid_trajectory_schema")
    serialized = json.dumps(trajectory, ensure_ascii=False, sort_keys=True)
    if any(token in serialized for token in FORBIDDEN_PRIVATE_FIELDS):
        reasons.append("private_field_leakage")
    if trajectory.get("terminal_status") != "accepted":
        reasons.append("answer_not_verified")

    events = trajectory.get("events")
    if not isinstance(events, list):
        events = []
    successful_python = [
        event
        for event in events
        if event.get("tool") == "run_python" and event.get("ok") is True
    ]
    dataframe_reads = [
        event
        for event in successful_python
        if isinstance(event.get("arguments"), dict)
        and isinstance(event["arguments"].get("code"), str)
        and _code_reads_dataframe(event["arguments"]["code"])
    ]
    if not dataframe_reads:
        reasons.append("no_successful_dataframe_analysis")

    submitted = trajectory.get("submitted_answer")
    uses_tool_feedback = any(
        _answer_supported_by_content(str(event.get("content", "")), submitted)
        for event in successful_python
    )
    if not uses_tool_feedback:
        reasons.append("submitted_answer_not_supported_by_tool_output")

    prior_errors = any(
        event.get("tool") != "submit_answer" and event.get("ok") is False
        for event in events
    )
    code_text = "\n".join(
        str(event.get("arguments", {}).get("code", ""))
        for event in successful_python
    ).lower()
    if prior_errors:
        category = "error_recovery"
    elif any(marker in code_text for marker in CLEANING_MARKERS):
        category = "pollution_cleaning"
    else:
        category = "direct_success"

    return QualityAssessment(
        retained=not reasons,
        category=category if not reasons else None,
        reasons=tuple(reasons),
        uses_tool_feedback=uses_tool_feedback,
        trajectory_fingerprint=_fingerprint_trajectory(trajectory),
    )


def build_sft_record(
    *,
    attempt: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    assessment: QualityAssessment,
    system_prompt: str,
    tools: Sequence[Mapping[str, Any]],
    prompt_version: str,
) -> dict[str, Any]:
    """Build a leakage-safe chat record with explicit assistant-only loss masks."""

    if not assessment.retained or assessment.category is None:
        raise A4TeacherError("Only retained trajectories can become SFT records.")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt, "trainable": False},
        {
            "role": "user",
            "content": trajectory["question"],
            "trainable": False,
        },
    ]
    for event in trajectory["events"]:
        messages.append(
            {
                "role": "assistant",
                "content": event["assistant_content"],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": event["tool"],
                            "arguments": event["arguments"],
                        },
                    }
                ],
                "trainable": True,
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": event["tool"],
                "content": event["content"],
                "ok": event["ok"],
                "error_type": event.get("error_type"),
                "trainable": False,
            }
        )
    record = {
        "schema_version": 1,
        "trajectory_id": attempt["attempt_id"],
        "task_id": trajectory["task_id"],
        "table": trajectory["table"],
        "split": "train",
        "sample_index": attempt["sample_index"],
        "sampling_seed": attempt["sampling_seed"],
        "prompt_version": prompt_version,
        "category": assessment.category,
        "trajectory_fingerprint": assessment.trajectory_fingerprint,
        "tools": list(tools),
        "messages": messages,
    }
    serialized = json.dumps(record, ensure_ascii=False)
    if any(token in serialized for token in FORBIDDEN_PRIVATE_FIELDS):
        raise A4TeacherError("SFT record exposes a private oracle field.")
    return record
