"""Validation and safe task-candidate selection for table column semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PRIMARY_ROLES = {
    "identifier",
    "category",
    "date",
    "measure",
    "derived",
    "coordinate",
    "opaque",
}


class TableSemanticsError(ValueError):
    pass


def load_table_semantics(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TableSemanticsError(f"Cannot load table semantics from {source}: {exc}") from exc
    if value.get("schema_version") != 1 or not isinstance(value.get("tables"), dict):
        raise TableSemanticsError("Unsupported or malformed table semantics registry.")
    return value


def validate_table_semantics(
    df: pd.DataFrame,
    table_name: str,
    registry: Mapping[str, Any],
) -> dict[str, str]:
    try:
        spec = registry["tables"][table_name]
        role_groups = spec["roles"]
    except (KeyError, TypeError) as exc:
        raise TableSemanticsError(f"Missing semantics for table {table_name!r}.") from exc
    unknown_roles = set(role_groups) - PRIMARY_ROLES
    if unknown_roles:
        raise TableSemanticsError(f"{table_name}: unsupported roles {sorted(unknown_roles)}.")

    role_by_column: dict[str, str] = {}
    duplicates: list[str] = []
    for role, columns in role_groups.items():
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise TableSemanticsError(f"{table_name}: role {role!r} must contain a list of columns.")
        for column in columns:
            if column in role_by_column:
                duplicates.append(column)
            role_by_column[column] = role
    if duplicates:
        raise TableSemanticsError(
            f"{table_name}: columns assigned to multiple primary roles: {sorted(set(duplicates))}."
        )

    actual = set(map(str, df.columns))
    declared = set(role_by_column)
    if actual != declared:
        raise TableSemanticsError(
            f"{table_name}: semantic columns differ from CSV; "
            f"missing={sorted(actual - declared)}, extra={sorted(declared - actual)}."
        )

    targets = spec.get("targets", [])
    if not isinstance(targets, list) or not set(targets) <= actual:
        raise TableSemanticsError(f"{table_name}: targets must be declared CSV columns.")
    derived_from = spec.get("derived_from", {})
    if not isinstance(derived_from, dict):
        raise TableSemanticsError(f"{table_name}: derived_from must be a mapping.")
    for derived, sources in derived_from.items():
        if derived not in actual or role_by_column[derived] != "derived":
            raise TableSemanticsError(
                f"{table_name}: derived_from key {derived!r} must have primary role 'derived'."
            )
        if not isinstance(sources, list) or not set(sources) <= actual:
            raise TableSemanticsError(
                f"{table_name}: invalid source columns for derived field {derived!r}."
            )
    return role_by_column


def numeric_measure_candidates(
    df: pd.DataFrame,
    table_name: str,
    registry: Mapping[str, Any],
) -> list[str]:
    """Return non-target, non-derived numeric measures safe for generic mean smoke tasks."""

    roles = validate_table_semantics(df, table_name, registry)
    targets = set(registry["tables"][table_name].get("targets", []))
    return [
        str(column)
        for column in df.columns
        if roles[str(column)] == "measure"
        and str(column) not in targets
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def category_candidates(
    df: pd.DataFrame,
    table_name: str,
    registry: Mapping[str, Any],
) -> list[str]:
    roles = validate_table_semantics(df, table_name, registry)
    return [
        str(column)
        for column in df.columns
        if roles[str(column)] == "category"
        and (pd.api.types.is_object_dtype(df[column].dtype) or pd.api.types.is_string_dtype(df[column].dtype))
    ]
