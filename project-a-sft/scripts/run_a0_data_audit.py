"""Run A0 compatibility smoke checks without claiming Ground Truth correctness."""

from pathlib import Path
import json

import pandas as pd

from dirty_da.canonical import CanonicalTable
from dirty_da.corruption import apply_corruption
from dirty_da.executor import execute_task
from dirty_da.schema_semantics import (
    category_candidates,
    load_table_semantics,
    numeric_measure_candidates,
    validate_table_semantics,
)
from dirty_da.task_spec import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
ARTIFACTS = ROOT / "artifacts"
SEMANTICS_PATH = ROOT / "configs" / "table_semantics.json"
ARTIFACTS.mkdir(exist_ok=True)

CORRUPTIONS = [
    "missing_values",
    "sentinel_values",
    "duplicate_rows",
    "type_corruption",
    "format_inconsistency",
    "category_inconsistency",
]
REQUIRED_METADATA = {
    "event_id",
    "corruption_type",
    "severity",
    "seed",
    "column",
    "row_position",
    "row_index",
    "before",
    "after",
    "operation",
}

registry = load_table_semantics(SEMANTICS_PATH)
matrix: list[dict] = []
task_rows: list[dict] = []
semantic_rows: list[dict] = []

for path in sorted(TABLES.glob("*.csv")):
    df = pd.read_csv(path, low_memory=False)
    roles = validate_table_semantics(df, path.name, registry)
    numeric = numeric_measure_candidates(df, path.name, registry)
    strings = category_candidates(df, path.name, registry)
    corruption_numeric = [
        str(column)
        for column in df.columns
        if roles[str(column)] in {"measure", "derived"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    format_strings = strings + [
        str(column)
        for column in df.columns
        if roles[str(column)] == "date"
        and (
            pd.api.types.is_object_dtype(df[column].dtype)
            or pd.api.types.is_string_dtype(df[column].dtype)
        )
    ]
    semantic_rows.append(
        {
            "table": path.name,
            "status": "schema_pass",
            "columns": len(df.columns),
            "identifiers": sum(role == "identifier" for role in roles.values()),
            "categories": sum(role == "category" for role in roles.values()),
            "measures": sum(role == "measure" for role in roles.values()),
            "derived": sum(role == "derived" for role in roles.values()),
            "targets": len(registry["tables"][path.name].get("targets", [])),
            "replacement_chars": sum(
                int(
                    df[column]
                    .astype("string")
                    .str.contains("\ufffd", regex=False, na=False)
                    .sum()
                )
                for column in df.select_dtypes(include=["object", "string"]).columns
            ),
        }
    )

    canonical = CanonicalTable(df, name=path.stem)
    fingerprint = canonical.fingerprint
    numeric_column = numeric[0] if numeric else None
    corruption_numeric_column = corruption_numeric[0] if corruption_numeric else None
    string_column = strings[0] if strings else None
    format_column = format_strings[0] if format_strings else None
    candidates = {
        "missing_values": corruption_numeric_column or string_column,
        "sentinel_values": corruption_numeric_column or string_column,
        "duplicate_rows": None,
        "type_corruption": corruption_numeric_column,
        "format_inconsistency": format_column,
        "category_inconsistency": string_column,
    }
    for kind in CORRUPTIONS:
        column = candidates[kind]
        status = "inapplicable"
        detail = "No semantically suitable column detected."
        if kind != "duplicate_rows" and column is None:
            matrix.append(
                {
                    "table": path.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "corruption": kind,
                    "status": status,
                    "candidate_column": "",
                    "detail": detail,
                }
            )
            continue
        try:
            if kind == "duplicate_rows":
                result = apply_corruption(canonical, kind, row=0, seed=20261020)
            elif len(df):
                result = apply_corruption(
                    canonical,
                    kind,
                    column=column,
                    row=0,
                    seed=20261020,
                )
            metadata_keys = set(result.metadata[0])
            if not REQUIRED_METADATA <= metadata_keys:
                raise AssertionError(
                    f"metadata missing keys: {sorted(REQUIRED_METADATA - metadata_keys)}"
                )
            if canonical.fingerprint != fingerprint:
                raise AssertionError("canonical fingerprint changed")
            status = "execution_smoke_pass"
            detail = (
                f"column={column}; rows={len(result.dirty)}; "
                f"metadata_events={len(result.metadata)}"
            )
        except Exception as exc:
            status = "failed"
            detail = f"{type(exc).__name__}: {exc}"
        matrix.append(
            {
                "table": path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "corruption": kind,
                "status": status,
                "candidate_column": column or "",
                "detail": detail,
            }
        )

    # These rows prove only that schema-safe examples execute. Expected answers
    # are deliberately not generated here; independent handwritten fixtures in
    # pytest are the Ground Truth correctness gate.
    if numeric_column:
        spec = TaskSpec(
            task_id=f"{path.stem}-mean-smoke",
            operation="mean",
            target_column=numeric_column,
        )
        try:
            actual = execute_task(df, spec)
            task_rows.append(
                {
                    "table": path.name,
                    "task": "mean",
                    "target": numeric_column,
                    "status": "execution_smoke_pass",
                    "ground_truth": "not_checked_here",
                    "result_preview": repr(actual),
                }
            )
        except Exception as exc:
            task_rows.append(
                {
                    "table": path.name,
                    "task": "mean",
                    "target": numeric_column,
                    "status": "failed",
                    "ground_truth": "not_checked_here",
                    "result_preview": f"{type(exc).__name__}: {exc}",
                }
            )
        if string_column:
            grouped_spec = TaskSpec(
                task_id=f"{path.stem}-groupby-smoke",
                operation="groupby_agg",
                target_column=numeric_column,
                group_by=string_column,
                aggregation="mean",
            )
            try:
                actual = execute_task(df, grouped_spec)
                task_rows.append(
                    {
                        "table": path.name,
                        "task": "groupby_mean",
                        "target": numeric_column,
                        "group_by": string_column,
                        "status": "execution_smoke_pass",
                        "ground_truth": "not_checked_here",
                        "result_preview": f"{len(actual)} groups",
                    }
                )
            except Exception as exc:
                task_rows.append(
                    {
                        "table": path.name,
                        "task": "groupby_mean",
                        "target": numeric_column,
                        "group_by": string_column,
                        "status": "failed",
                        "ground_truth": "not_checked_here",
                        "result_preview": f"{type(exc).__name__}: {exc}",
                    }
                )

pd.DataFrame(matrix).to_csv(
    ARTIFACTS / "corruption_applicability_matrix.csv",
    index=False,
    encoding="utf-8-sig",
)
pd.DataFrame(task_rows).to_csv(
    ARTIFACTS / "task_smoke_tests.csv",
    index=False,
    encoding="utf-8-sig",
)
pd.DataFrame(semantic_rows).to_csv(
    ARTIFACTS / "table_semantics_audit.csv",
    index=False,
    encoding="utf-8-sig",
)
summary = {
    "tables_checked": len(semantic_rows),
    "semantic_schema_pass": sum(row["status"] == "schema_pass" for row in semantic_rows),
    "source_replacement_characters": sum(
        row["replacement_chars"] for row in semantic_rows
    ),
    "corruption_checks": len(matrix),
    "corruption_execution_smoke_pass": sum(
        row["status"] == "execution_smoke_pass" for row in matrix
    ),
    "corruption_inapplicable": sum(row["status"] == "inapplicable" for row in matrix),
    "corruption_failed": sum(row["status"] == "failed" for row in matrix),
    "task_execution_smoke_checks": len(task_rows),
    "task_execution_smoke_pass": sum(
        row["status"] == "execution_smoke_pass" for row in task_rows
    ),
    "task_execution_smoke_failed": sum(
        row["status"] != "execution_smoke_pass" for row in task_rows
    ),
    "ground_truth_correctness_gate": "pytest handwritten fixtures",
}
(ARTIFACTS / "data_audit_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
