"""A2 task rendering, serialization, and per-record correctness checks."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .executor import ExecutionError, execute_task
from .task_spec import TaskSpec, TaskSpecError
from .verifier import verify_result


A2_SCHEMA_VERSION = 1
DIFFICULTIES = {
    "single_step",
    "grouped",
    "cleaning_required",
    "judgment",
}
RENDER_STYLES = {"standard", "concise", "heldout"}
RECORD_FIELDS = {
    "schema_version",
    "task_id",
    "table",
    "split",
    "domain",
    "difficulty",
    "render_style",
    "question",
    "spec",
    "answer",
    "canonical_fingerprint",
    "dirty_profile",
    "dirty_fingerprint",
    "applied_corruptions",
    "inapplicable_corruptions",
    "needs",
}


class A2TaskError(ValueError):
    pass


def _format_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _filter_text(filter_spec: Mapping[str, Any]) -> str:
    operators = {
        "eq": "equals",
        "ne": "does not equal",
        "gt": "is greater than",
        "ge": "is at least",
        "lt": "is less than",
        "le": "is at most",
        "in": "is one of",
    }
    return (
        f"`{filter_spec['column']}` {operators[filter_spec['operator']]} "
        f"{_format_value(filter_spec['value'])}"
    )


def _cleaning_sentences(spec: TaskSpec) -> list[str]:
    sentences: list[str] = []
    if spec.drop_duplicates:
        if spec.dedup_subset:
            columns = ", ".join(f"`{column}`" for column in spec.dedup_subset)
            sentences.append(f"Deduplicate rows using {columns} as the key.")
        else:
            sentences.append("Remove exact duplicate rows.")
    for entry in spec.sentinel_map:
        sentences.append(
            f"Treat {_format_value(list(entry['values']))} in `{entry['column']}` as missing."
        )
    for column in spec.strip_columns:
        sentences.append(f"Trim surrounding whitespace in `{column}`.")
    for entry in spec.category_map:
        pairs = ", ".join(
            f"{_format_value(source)} to {_format_value(target)}"
            for source, target in sorted(
                entry["mapping"].items(), key=lambda item: str(item[0])
            )
        )
        sentences.append(f"In `{entry['column']}`, map {pairs}.")
    for column in spec.numeric_coerce:
        sentences.append(
            f"Parse `{column}` as numeric; values that cannot be parsed become missing."
        )
    if spec.missing_policy == "exclude":
        sentences.append("Exclude missing values from referenced aggregations and groups.")
    else:
        sentences.append("Reject the calculation if a referenced value is missing.")
    return sentences


def _operation_text(spec: TaskSpec) -> str:
    filters = ""
    if spec.filters:
        filters = " Keep only rows where " + " and ".join(
            _filter_text(filter_spec) for filter_spec in spec.filters
        ) + "."
    if spec.operation == "count_rows":
        request = "Return the number of remaining rows as an integer."
    elif spec.operation in {"sum", "mean", "min", "max"}:
        labels = {"sum": "sum", "mean": "mean", "min": "minimum", "max": "maximum"}
        request = f"Return the {labels[spec.operation]} of `{spec.target_column}`."
    elif spec.operation == "groupby_agg":
        request = (
            f"Group by `{spec.group_by}` and return a JSON object mapping each non-missing "
            f"group label to the {spec.aggregation} of `{spec.target_column}`."
        )
    elif spec.operation == "aggregate_compare":
        comparisons = {
            "eq": "equal to",
            "ne": "not equal to",
            "gt": "greater than",
            "ge": "at least",
            "lt": "less than",
            "le": "at most",
        }
        request = (
            f"Return true if the {spec.aggregation} of `{spec.target_column}` is "
            f"{comparisons[spec.comparison_operator]} {_format_value(spec.comparison_value)}; "
            "otherwise return false."
        )
    else:  # guarded by TaskSpec.validate
        raise A2TaskError(f"Unsupported operation: {spec.operation}")
    return filters + " " + request


def render_task_question(spec: TaskSpec, style: str) -> str:
    """Render natural language only from the structured task spec."""

    if style not in RENDER_STYLES:
        raise A2TaskError(f"Unsupported render style: {style}")
    cleaning = " ".join(_cleaning_sentences(spec))
    operation = _operation_text(spec).strip()
    if style == "standard":
        return f"Using the provided table, apply these rules: {cleaning} Then {operation}"
    if style == "concise":
        return f"Apply exactly the following data rules. {cleaning} Task: {operation}"
    return f"Question: {operation} Requirements: {cleaning}"


def load_task_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise A2TaskError(f"Line {line_number} is not a JSON object.")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise A2TaskError(f"Cannot load task records from {source}: {exc}") from exc
    return records


def answer_is_releasable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, list):
        return all(answer_is_releasable(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and answer_is_releasable(item)
            for key, item in value.items()
        )
    return False


def validate_task_record(
    record: Mapping[str, Any],
    canonical_frame: pd.DataFrame,
    dirty_frame: pd.DataFrame,
    *,
    role_by_column: Mapping[str, str],
    catalog_record: Mapping[str, Any],
    canonical_fingerprint: str,
    dirty_fingerprint: str,
) -> TaskSpec:
    if not isinstance(record, Mapping) or set(record) != RECORD_FIELDS:
        extra = set(record) - RECORD_FIELDS if isinstance(record, Mapping) else set()
        missing = RECORD_FIELDS - set(record) if isinstance(record, Mapping) else RECORD_FIELDS
        raise A2TaskError(
            f"Task record fields differ; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    if record["schema_version"] != A2_SCHEMA_VERSION:
        raise A2TaskError("Unsupported A2 task schema version.")
    if record["difficulty"] not in DIFFICULTIES:
        raise A2TaskError(f"Unsupported difficulty: {record['difficulty']}")
    if record["render_style"] not in RENDER_STYLES:
        raise A2TaskError(f"Unsupported render style: {record['render_style']}")
    if not isinstance(record["table"], str) or not record["table"]:
        raise A2TaskError("table must be a string.")
    if record["split"] != catalog_record.get("split"):
        raise A2TaskError("Task split differs from the frozen table split.")
    if record["domain"] != catalog_record.get("domain"):
        raise A2TaskError("Task domain differs from the frozen table domain.")
    if catalog_record.get("selection_status") != "selected" or not catalog_record.get(
        "approved_for_research"
    ):
        raise A2TaskError("A2 tasks may use only selected, research-approved tables.")
    try:
        spec = TaskSpec.from_dict(record["spec"])
        spec.validate(canonical_frame.columns)
    except TaskSpecError as exc:
        raise A2TaskError(f"Invalid structured spec: {exc}") from exc
    if spec.task_id != record["task_id"]:
        raise A2TaskError("Record task_id differs from structured spec task_id.")
    if spec.target_column is not None and role_by_column[spec.target_column] not in {
        "measure",
        "derived",
    }:
        raise A2TaskError("Aggregate targets must be declared measures or derived fields.")
    if spec.group_by is not None and role_by_column[spec.group_by] != "category":
        raise A2TaskError("group_by must be a declared category field.")
    if any(role_by_column[item["column"]] == "identifier" for item in spec.filters):
        raise A2TaskError("Identifier fields cannot be used as synthetic task filters.")
    if record["canonical_fingerprint"] != canonical_fingerprint:
        raise A2TaskError("Canonical fingerprint does not match the current table.")
    if record["dirty_fingerprint"] != dirty_fingerprint:
        raise A2TaskError("Dirty fingerprint does not match the declared profile.")
    if record["dirty_profile"] not in {"low", "medium", "high"}:
        raise A2TaskError("dirty_profile must be low, medium, or high.")
    applied = record["applied_corruptions"]
    needs = record["needs"]
    if not isinstance(applied, list) or not all(isinstance(item, str) for item in applied):
        raise A2TaskError("applied_corruptions must be a list of strings.")
    if not isinstance(needs, list) or not set(needs) <= set(applied):
        raise A2TaskError("needs must be a subset of applied_corruptions.")
    if not isinstance(record["inapplicable_corruptions"], dict):
        raise A2TaskError("inapplicable_corruptions must be an object.")
    if record["question"] != render_task_question(spec, record["render_style"]):
        raise A2TaskError("Natural-language question is not the exact spec rendering.")
    if not answer_is_releasable(record["answer"]):
        raise A2TaskError("Answer is missing, non-finite, or not JSON-releasable.")
    expected = execute_task(canonical_frame, spec)
    if not verify_result(expected, record["answer"]).passed:
        raise A2TaskError("Stored answer differs from canonical programmatic execution.")
    dirty_answer = execute_task(dirty_frame, spec)
    if not verify_result(expected, dirty_answer).passed:
        raise A2TaskError("Declared cleaning rules do not recover the canonical task answer.")
    return spec


def cleaning_fields_from_metadata(metadata: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build explicit, bounded cleaning declarations for known injector rules."""

    sentinel_values: dict[str, list[Any]] = {}
    strip_columns: list[str] = []
    category_mappings: dict[str, dict[Any, Any]] = {}
    numeric_columns: list[str] = []
    drop_duplicates = False
    for event in metadata:
        corruption_type = event["corruption_type"]
        column = event.get("column")
        if corruption_type == "duplicate_rows":
            drop_duplicates = True
        elif corruption_type == "sentinel_values" and column is not None:
            sentinel_values.setdefault(column, []).append(event["after"])
            before = event.get("before")
            if (
                isinstance(before, (int, float))
                and not isinstance(before, bool)
                and column not in numeric_columns
            ):
                numeric_columns.append(column)
        elif corruption_type == "type_corruption" and column is not None:
            if column not in numeric_columns:
                numeric_columns.append(column)
        elif corruption_type in {"format_inconsistency", "category_inconsistency"} and column is not None:
            if column not in strip_columns:
                strip_columns.append(column)
            after = event["after"]
            before = event["before"]
            if isinstance(after, str):
                category_mappings.setdefault(column, {})[after.strip()] = before
    sentinel_map = tuple(
        {"column": column, "values": tuple(values)}
        for column, values in sorted(sentinel_values.items())
    )
    category_map = tuple(
        {"column": column, "mapping": mapping}
        for column, mapping in sorted(category_mappings.items())
    )
    return {
        "drop_duplicates": drop_duplicates,
        "dedup_subset": (),
        "sentinel_policy": "map" if sentinel_map else "none",
        "sentinel_map": sentinel_map,
        "strip_columns": tuple(strip_columns),
        "category_map": category_map,
        "numeric_coerce": tuple(numeric_columns),
    }


def infer_needed_corruptions(
    dirty_frame: pd.DataFrame,
    spec: TaskSpec,
    metadata: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Identify corruption rules whose cleaning ablation changes execution."""

    baseline = execute_task(dirty_frame, spec)
    needed: list[str] = []
    events = list(metadata)
    for corruption_type in dict.fromkeys(event["corruption_type"] for event in events):
        columns = {
            event.get("column")
            for event in events
            if event["corruption_type"] == corruption_type and event.get("column") is not None
        }
        if corruption_type == "missing_values":
            continue
        if corruption_type == "duplicate_rows":
            ablated = replace(spec, drop_duplicates=False, dedup_subset=())
        elif corruption_type == "sentinel_values":
            remaining = tuple(
                entry for entry in spec.sentinel_map if entry["column"] not in columns
            )
            ablated = replace(
                spec,
                sentinel_map=remaining,
                sentinel_policy="map" if remaining else "none",
            )
        elif corruption_type == "type_corruption":
            ablated = replace(
                spec,
                numeric_coerce=tuple(
                    column for column in spec.numeric_coerce if column not in columns
                ),
            )
        elif corruption_type in {"format_inconsistency", "category_inconsistency"}:
            ablated = replace(
                spec,
                strip_columns=tuple(
                    column for column in spec.strip_columns if column not in columns
                ),
                category_map=tuple(
                    entry for entry in spec.category_map if entry["column"] not in columns
                ),
            )
        else:
            continue
        try:
            ablated_answer = execute_task(dirty_frame, ablated)
        except (ExecutionError, TypeError, ValueError):
            needed.append(corruption_type)
            continue
        if not verify_result(baseline, ablated_answer).passed:
            needed.append(corruption_type)
    return needed
