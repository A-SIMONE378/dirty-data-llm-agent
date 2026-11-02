"""Build the deterministic A2 task catalog from canonical tables and specs."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a1_catalog import load_a1_catalog  # noqa: E402
from dirty_da.a1_corruption import (  # noqa: E402
    generate_dirty_copy,
    load_corruption_profiles,
)
from dirty_da.a2_tasks import (  # noqa: E402
    answer_is_releasable,
    cleaning_fields_from_metadata,
    infer_needed_corruptions,
    render_task_question,
)
from dirty_da.canonical import CanonicalTable  # noqa: E402
from dirty_da.executor import ExecutionError, execute_task  # noqa: E402
from dirty_da.schema_semantics import (  # noqa: E402
    load_table_semantics,
    validate_table_semantics,
)
from dirty_da.task_spec import TaskSpec, TaskSpecError  # noqa: E402
from dirty_da.verifier import verify_result  # noqa: E402


TABLES = ROOT / "data" / "tables"
OUTPUT = ROOT / "data" / "tasks" / "a2_tasks.jsonl"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _load_config() -> dict[str, Any]:
    path = ROOT / "configs" / "a2_task_generation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported A2 generation config.")
    return value


def _stable_seed(base_seed: int, table_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{table_name}".encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def _pick_index(variant: int, label: str, size: int) -> int:
    """Choose independently from a bounded option set without modulo coupling."""
    if size <= 0:
        raise ValueError("Cannot choose from an empty option set.")
    digest = hashlib.sha256(f"{variant}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _nonmissing_values(frame: pd.DataFrame, column: str, limit: int = 16) -> list[Any]:
    values = [_json_scalar(value) for value in frame[column].dropna().unique().tolist()]
    values.sort(key=lambda value: (type(value).__name__, repr(value)))
    return values[:limit]


def _filter_variant(
    frame: pd.DataFrame,
    categories: list[str],
    variant: int,
    *,
    avoid: str | None = None,
) -> tuple[dict[str, Any], ...]:
    candidates = [column for column in categories if column != avoid] or categories
    if not candidates or _pick_index(variant, "filter-presence", 5) == 0:
        return ()
    column = candidates[_pick_index(variant, "filter-column", len(candidates))]
    values = _nonmissing_values(frame, column)
    if not values:
        return ()
    value = values[_pick_index(variant, "filter-value", len(values))]
    operator = ("eq", "ne", "in")[_pick_index(variant, "filter-operator", 3)]
    if operator == "in":
        second = values[_pick_index(variant, "filter-second-value", len(values))]
        selected = [value] if second == value else [value, second]
        filter_value: Any = selected
    else:
        filter_value = value
    return ({"column": column, "operator": operator, "value": filter_value},)


def _spec_signature(spec: TaskSpec) -> str:
    value = spec.to_dict()
    value.pop("task_id", None)
    value.pop("description", None)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _profile_for(config: Mapping[str, Any], difficulty: str, variant: int) -> str:
    cycle = config["profile_cycles"][difficulty]
    return cycle[variant % len(cycle)]


def _candidate_spec(
    *,
    task_id: str,
    difficulty: str,
    variant: int,
    frame: pd.DataFrame,
    numeric_targets: list[str],
    categories: list[str],
    group_categories: list[str],
    cleaning: Mapping[str, Any],
) -> TaskSpec:
    target = numeric_targets[
        _pick_index(variant, f"{difficulty}-target", len(numeric_targets))
    ]
    common = {
        "task_id": task_id,
        "missing_policy": "exclude",
        **cleaning,
    }
    if difficulty == "single_step":
        operation = ("count_rows", "sum", "mean", "min", "max")[
            _pick_index(variant, "single-operation", 5)
        ]
        filters = _filter_variant(frame, categories, variant)
        if operation == "count_rows":
            return TaskSpec(operation=operation, filters=filters, **common)
        return TaskSpec(
            operation=operation,
            target_column=target,
            filters=filters,
            **common,
        )
    if difficulty == "grouped":
        group_by = group_categories[
            _pick_index(variant, "group-column", len(group_categories))
        ]
        aggregation = ("count", "sum", "mean", "min", "max")[
            _pick_index(variant, "group-aggregation", 5)
        ]
        filters = _filter_variant(frame, categories, variant, avoid=group_by)
        return TaskSpec(
            operation="groupby_agg",
            target_column=target,
            group_by=group_by,
            aggregation=aggregation,
            filters=filters,
            **common,
        )
    if difficulty == "cleaning_required":
        if _pick_index(variant, "cleaning-shape", 3) == 0:
            group_by = group_categories[
                _pick_index(variant, "cleaning-group-column", len(group_categories))
            ]
            return TaskSpec(
                operation="groupby_agg",
                target_column=target,
                group_by=group_by,
                aggregation=("count", "sum", "mean")[
                    _pick_index(variant, "cleaning-group-aggregation", 3)
                ],
                filters=_filter_variant(frame, categories, variant, avoid=group_by),
                **common,
            )
        operation = ("count_rows", "sum", "min", "max")[
            _pick_index(variant, "cleaning-operation", 4)
        ]
        filters = _filter_variant(frame, categories, variant)
        if operation == "count_rows":
            return TaskSpec(operation=operation, filters=filters, **common)
        return TaskSpec(
            operation=operation,
            target_column=target,
            filters=filters,
            **common,
        )
    if difficulty != "judgment":
        raise ValueError(f"Unsupported difficulty: {difficulty}")

    aggregation = ("count", "sum", "mean", "min", "max")[
        _pick_index(variant, "judgment-aggregation", 5)
    ]
    filters = _filter_variant(frame, categories, variant)
    base_operation = "mean" if aggregation == "mean" else aggregation
    if aggregation == "count":
        base = TaskSpec(
            operation="groupby_agg",
            target_column=target,
            group_by=group_categories[
                _pick_index(variant, "judgment-group-column", len(group_categories))
            ],
            aggregation="count",
            filters=filters,
            **common,
        )
        grouped = execute_task(frame, base)
        aggregate_value = float(sum(grouped.values()))
    else:
        base = TaskSpec(
            operation=base_operation,
            target_column=target,
            filters=filters,
            **common,
        )
        aggregate_value = execute_task(frame, base)
    if aggregate_value is None or isinstance(aggregate_value, bool):
        raise ExecutionError("Cannot form a comparison from a missing aggregate.")
    numeric_value = float(aggregate_value)
    if not math.isfinite(numeric_value):
        raise ExecutionError("Cannot form a comparison from a non-finite aggregate.")
    delta_band = _pick_index(variant, "judgment-threshold-band", 5) + 1
    delta = max(abs(numeric_value) * delta_band * 0.01, 0.5)
    threshold = (
        numeric_value + delta
        if _pick_index(variant, "judgment-threshold-side", 2) == 0
        else numeric_value - delta
    )
    return TaskSpec(
        operation="aggregate_compare",
        target_column=target,
        aggregation=aggregation,
        comparison_operator=("gt", "ge", "lt", "le")[
            _pick_index(variant, "judgment-comparison", 4)
        ],
        comparison_value=round(threshold, 8),
        filters=filters,
        **common,
    )


def _target_count(
    table_name: str,
    split: str,
    train_names: list[str],
    config: Mapping[str, Any],
) -> int:
    if split != "train":
        return int(config["tasks_per_non_train_table"])
    total = int(config["train_task_total"])
    base, remainder = divmod(total, len(train_names))
    return base + (1 if train_names.index(table_name) < remainder else 0)


def _difficulty_targets(
    total: int, order: list[str], cleaning_fraction: float
) -> dict[str, int]:
    if not 0 < cleaning_fraction < 1:
        raise ValueError("cleaning_difficulty_fraction must be between zero and one.")
    cleaning_count = max(1, round(total * cleaning_fraction))
    other_difficulties = [item for item in order if item != "cleaning_required"]
    base, remainder = divmod(total - cleaning_count, len(other_difficulties))
    result = {"cleaning_required": cleaning_count}
    result.update(
        {
            difficulty: base + (1 if index < remainder else 0)
            for index, difficulty in enumerate(other_difficulties)
        }
    )
    return result


def _generate_for_table(
    table_name: str,
    catalog_record: Mapping[str, Any],
    semantics: Mapping[str, Any],
    profiles: Mapping[str, Any],
    config: Mapping[str, Any],
    target_count: int,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(TABLES / table_name, low_memory=False)
    canonical = CanonicalTable(frame, name=Path(table_name).stem)
    roles = validate_table_semantics(frame, table_name, semantics)
    numeric_targets = [
        str(column)
        for column in frame.columns
        if roles[str(column)] in {"measure", "derived"}
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().any()
    ]
    categories = [
        str(column)
        for column in frame.columns
        if roles[str(column)] == "category" and frame[column].notna().any()
    ]
    group_categories = [
        column
        for column in categories
        if 2 <= frame[column].nunique(dropna=True) <= 64
    ]
    if not numeric_targets or not categories or not group_categories:
        raise RuntimeError(f"{table_name}: insufficient bounded task candidates.")

    dirty_by_profile = {
        profile: generate_dirty_copy(
            canonical, table_name, semantics, profiles, profile
        )
        for profile in profiles["profiles"]
    }
    difficulty_targets = _difficulty_targets(
        target_count,
        list(config["difficulty_order"]),
        float(config["cleaning_difficulty_fraction"]),
    )
    styles = config["render_styles"][catalog_record["split"]]
    records: list[dict[str, Any]] = []
    signatures: set[str] = set()
    table_seed = _stable_seed(int(config["base_seed"]), table_name)

    for difficulty in config["difficulty_order"]:
        required = difficulty_targets[difficulty]
        accepted = 0
        attempts = 0
        while accepted < required and attempts < max(1000, required * 100):
            variant = table_seed + attempts
            profile_name = _profile_for(config, difficulty, variant)
            dirty = dirty_by_profile[profile_name]
            cleaning = cleaning_fields_from_metadata(dirty.result.metadata)
            task_number = len(records) + 1
            task_id = f"a2-{catalog_record['split']}-{Path(table_name).stem}-{task_number:04d}"
            attempts += 1
            try:
                spec = _candidate_spec(
                    task_id=task_id,
                    difficulty=difficulty,
                    variant=variant,
                    frame=frame,
                    numeric_targets=numeric_targets,
                    categories=categories,
                    group_categories=group_categories,
                    cleaning=cleaning,
                )
                spec.validate(frame.columns)
                signature = _spec_signature(spec)
                if signature in signatures:
                    continue
                answer = execute_task(frame, spec)
                if not answer_is_releasable(answer):
                    continue
                if isinstance(answer, dict) and len(answer) > 30:
                    continue
                dirty_answer = execute_task(dirty.result.dirty, spec)
                if not verify_result(answer, dirty_answer).passed:
                    continue
                needs = infer_needed_corruptions(
                    dirty.result.dirty, spec, dirty.result.metadata
                )
                if difficulty == "cleaning_required" and not needs:
                    continue
            except (ExecutionError, TaskSpecError, TypeError, ValueError):
                continue
            signatures.add(signature)
            style = styles[(task_number - 1) % len(styles)]
            records.append(
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "table": table_name,
                    "split": catalog_record["split"],
                    "domain": catalog_record["domain"],
                    "difficulty": difficulty,
                    "render_style": style,
                    "question": render_task_question(spec, style),
                    "spec": spec.to_dict(),
                    "answer": answer,
                    "canonical_fingerprint": canonical.fingerprint,
                    "dirty_profile": profile_name,
                    "dirty_fingerprint": dirty.fingerprint,
                    "applied_corruptions": list(dirty.applied_types),
                    "inapplicable_corruptions": dirty.inapplicable,
                    "needs": needs,
                }
            )
            accepted += 1
        if accepted != required:
            if difficulty == "cleaning_required":
                difficulty_targets["judgment"] += required - accepted
                print(
                    f"{table_name}: cleaning applicability capped at {accepted}/{required}; "
                    "shortfall reassigned to judgment"
                )
            else:
                raise RuntimeError(
                    f"{table_name}/{difficulty}: generated {accepted}/{required} unique valid tasks "
                    f"after {attempts} attempts."
                )
    return records


def main() -> None:
    config = _load_config()
    catalog = load_a1_catalog(ROOT / "configs" / "a1_table_catalog.json")
    semantics = load_table_semantics(ROOT / "configs" / "table_semantics.json")
    profiles = load_corruption_profiles(ROOT / "configs" / "a1_corruption_profiles.json")
    selected = {
        name: record
        for name, record in catalog["tables"].items()
        if record.get("selection_status") == "selected"
        and record.get("approved_for_research")
    }
    train_names = sorted(
        name for name, record in selected.items() if record["split"] == "train"
    )
    all_records: list[dict[str, Any]] = []
    for table_name in sorted(selected):
        record = selected[table_name]
        count = _target_count(table_name, record["split"], train_names, config)
        generated = _generate_for_table(
            table_name, record, semantics, profiles, config, count
        )
        all_records.extend(generated)
        print(
            f"{table_name}: {len(generated)} tasks "
            f"({Counter(item['difficulty'] for item in generated)})"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    split_counts = Counter(record["split"] for record in all_records)
    difficulty_counts = Counter(record["difficulty"] for record in all_records)
    print(json.dumps({
        "tasks": len(all_records),
        "split_counts": dict(sorted(split_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
