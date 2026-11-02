"""Full mechanical A2 catalog audit over canonical and regenerated dirty tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .a1_corruption import generate_dirty_copy
from .a2_tasks import DIFFICULTIES, validate_task_record
from .canonical import CanonicalTable
from .schema_semantics import validate_table_semantics


class A2AuditError(ValueError):
    pass


@dataclass(frozen=True)
class A2Audit:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _spec_signature(record: Mapping[str, Any]) -> str:
    spec = dict(record["spec"])
    spec.pop("task_id", None)
    spec.pop("description", None)
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_a2_catalog(
    tables_dir: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    a1_catalog: Mapping[str, Any],
    semantics_registry: Mapping[str, Any],
    profile_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> A2Audit:
    selected = {
        name: record
        for name, record in a1_catalog["tables"].items()
        if record.get("selection_status") == "selected"
        and record.get("approved_for_research")
    }
    declared_tables = {str(record.get("table")) for record in records}
    if declared_tables != set(selected):
        raise A2AuditError(
            "A2 table inventory differs from selected A1 inventory; "
            f"missing={sorted(set(selected) - declared_tables)}, "
            f"extra={sorted(declared_tables - set(selected))}."
        )
    task_ids = [record.get("task_id") for record in records]
    if not all(isinstance(task_id, str) and task_id for task_id in task_ids):
        raise A2AuditError("Every task requires a non-empty task_id.")
    if len(task_ids) != len(set(task_ids)):
        raise A2AuditError("Task ids are not unique.")

    root = Path(tables_dir)
    table_cache: dict[str, tuple[pd.DataFrame, CanonicalTable, dict[str, str]]] = {}
    dirty_cache: dict[tuple[str, str], Any] = {}
    for table_name in sorted(selected):
        frame = pd.read_csv(root / table_name, low_memory=False)
        canonical = CanonicalTable(frame, name=Path(table_name).stem)
        roles = validate_table_semantics(frame, table_name, semantics_registry)
        table_cache[table_name] = (frame, canonical, roles)
        for profile_name in profile_config["profiles"]:
            dirty_cache[(table_name, profile_name)] = generate_dirty_copy(
                canonical,
                table_name,
                semantics_registry,
                profile_config,
                profile_name,
            )

    rows: list[dict[str, Any]] = []
    signatures_by_table: dict[str, set[str]] = defaultdict(set)
    for record in records:
        table_name = record["table"]
        profile_name = record["dirty_profile"]
        if (table_name, profile_name) not in dirty_cache:
            raise A2AuditError(f"Unknown table/profile pair: {table_name}/{profile_name}")
        frame, canonical, roles = table_cache[table_name]
        dirty = dirty_cache[(table_name, profile_name)]
        if list(dirty.applied_types) != record["applied_corruptions"]:
            raise A2AuditError(f"{record['task_id']}: applied corruption list drifted.")
        if dirty.inapplicable != record["inapplicable_corruptions"]:
            raise A2AuditError(f"{record['task_id']}: inapplicability metadata drifted.")
        validate_task_record(
            record,
            frame,
            dirty.result.dirty,
            role_by_column=roles,
            catalog_record=selected[table_name],
            canonical_fingerprint=canonical.fingerprint,
            dirty_fingerprint=dirty.fingerprint,
        )
        signature = _spec_signature(record)
        if signature in signatures_by_table[table_name]:
            raise A2AuditError(f"{record['task_id']}: duplicate semantic task in table.")
        signatures_by_table[table_name].add(signature)
        rows.append(
            {
                "task_id": record["task_id"],
                "table": table_name,
                "split": record["split"],
                "domain": record["domain"],
                "difficulty": record["difficulty"],
                "render_style": record["render_style"],
                "dirty_profile": profile_name,
                "needs": ";".join(record["needs"]),
                "answer_type": type(record["answer"]).__name__,
                "mechanical_check": "PASS",
            }
        )

    split_counts = Counter(record["split"] for record in records)
    expected_split_counts = {
        "train": int(generation_config["train_task_total"]),
        "dev": sum(1 for value in selected.values() if value["split"] == "dev")
        * int(generation_config["tasks_per_non_train_table"]),
        "test": sum(1 for value in selected.values() if value["split"] == "test")
        * int(generation_config["tasks_per_non_train_table"]),
        "domain_generalization": sum(
            1 for value in selected.values() if value["split"] == "domain_generalization"
        )
        * int(generation_config["tasks_per_non_train_table"]),
    }
    if dict(split_counts) != expected_split_counts:
        raise A2AuditError(
            f"Split task counts differ: observed={dict(split_counts)}, "
            f"expected={expected_split_counts}."
        )
    table_counts = Counter(record["table"] for record in records)
    train_tables = sorted(
        name for name, value in selected.items() if value["split"] == "train"
    )
    train_base, train_remainder = divmod(
        int(generation_config["train_task_total"]), len(train_tables)
    )
    for index, table_name in enumerate(train_tables):
        expected = train_base + (1 if index < train_remainder else 0)
        if table_counts[table_name] != expected:
            raise A2AuditError(f"{table_name}: expected {expected} tasks.")
    for table_name, value in selected.items():
        if value["split"] != "train" and table_counts[table_name] != int(
            generation_config["tasks_per_non_train_table"]
        ):
            raise A2AuditError(f"{table_name}: unexpected non-train task count.")

    difficulty_counts = Counter(record["difficulty"] for record in records)
    if set(difficulty_counts) != DIFFICULTIES:
        raise A2AuditError("All four bounded difficulty classes must be present.")
    for split in expected_split_counts:
        observed = {record["difficulty"] for record in records if record["split"] == split}
        if observed != DIFFICULTIES:
            raise A2AuditError(f"Split {split} does not contain every difficulty.")
    judgment_answers = Counter(
        record["answer"] for record in records if record["difficulty"] == "judgment"
    )
    if set(judgment_answers) != {False, True}:
        raise A2AuditError("Judgment tasks must contain both true and false answers.")

    applied_counts = Counter(
        corruption
        for record in records
        for corruption in record["applied_corruptions"]
    )
    needs_counts = Counter(
        corruption for record in records for corruption in record["needs"]
    )
    profile_counts = Counter(record["dirty_profile"] for record in records)
    style_counts = Counter(record["render_style"] for record in records)
    if set(profile_counts) != {"low", "medium", "high"}:
        raise A2AuditError("Low, medium, and high dirty profiles must all be used.")

    summary = {
        "schema_version": 1,
        "tasks_checked": len(records),
        "tables_checked": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "expected_split_counts": expected_split_counts,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "profile_counts": dict(sorted(profile_counts.items())),
        "render_style_counts": dict(sorted(style_counts.items())),
        "applied_corruption_counts": dict(sorted(applied_counts.items())),
        "answer_critical_needs_counts": dict(sorted(needs_counts.items())),
        "judgment_answer_counts": {
            "false": judgment_answers[False],
            "true": judgment_answers[True],
        },
        "programmatic_answer_checks_passed": len(records),
        "dirty_replay_checks_passed": len(records),
        "question_spec_exact_render_checks_passed": len(records),
        "table_split_leakage_detected": False,
        "mechanical_gate_pass": True,
    }
    return A2Audit(rows=rows, summary=summary)
