"""Explicit reference cleaning rules that never guess unrecoverable values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .task_spec import TaskSpec


def _column_list(value: Any, rule: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{rule} must be a sequence of column names.")
    return list(value)


def clean_table(df: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    """Return a cleaned copy using only declared rules.

    Rule order is exact-duplicate removal, sentinel mapping, string stripping,
    category mapping, numeric coercion, then datetime parsing. Missing values
    introduced by corruption remain missing unless a task policy later drops or
    otherwise handles them; this cleaner never invents an original value.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_table requires a pandas DataFrame.")
    if not isinstance(rules, Mapping):
        raise TypeError("rules must be a mapping.")
    allowed = {
        "drop_exact_duplicates",
        "dedup_subset",
        "sentinel_map",
        "strip_strings",
        "category_map",
        "numeric_coerce",
        "datetime_parse",
    }
    unknown = set(rules) - allowed
    if unknown:
        raise ValueError(f"Unsupported cleaning rules: {sorted(unknown)}")
    if not isinstance(rules.get("drop_exact_duplicates", False), bool):
        raise ValueError("drop_exact_duplicates must be boolean.")

    dedup_subset = _column_list(rules.get("dedup_subset", []), "dedup_subset")
    unknown_dedup = set(dedup_subset) - set(df.columns)
    if unknown_dedup:
        raise ValueError(f"Unknown dedup columns: {sorted(unknown_dedup)}")
    if dedup_subset and not rules.get("drop_exact_duplicates", False):
        raise ValueError("dedup_subset requires drop_exact_duplicates=True.")

    out = df.copy(deep=True)
    if rules.get("drop_exact_duplicates", False):
        out = out.drop_duplicates(subset=dedup_subset or None).reset_index(drop=True)

    sentinel_map = rules.get("sentinel_map", {})
    if not isinstance(sentinel_map, Mapping):
        raise ValueError("sentinel_map must map columns to sentinel sequences.")
    for col, sentinels in sentinel_map.items():
        if col not in out.columns:
            raise ValueError(f"Unknown sentinel column: {col}")
        if isinstance(sentinels, (str, bytes)) or not isinstance(sentinels, Sequence):
            raise ValueError(f"Sentinels for {col!r} must be a non-string sequence.")
        out[col] = out[col].replace({sentinel: pd.NA for sentinel in sentinels})

    for col in _column_list(rules.get("strip_strings", []), "strip_strings"):
        if col not in out.columns:
            raise ValueError(f"Unknown string column: {col}")
        out[col] = out[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    category_map = rules.get("category_map", {})
    if not isinstance(category_map, Mapping):
        raise ValueError("category_map must map columns to explicit replacement mappings.")
    for col, mapping in category_map.items():
        if col not in out.columns:
            raise ValueError(f"Unknown category column: {col}")
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Category mapping for {col!r} must be a mapping.")
        out[col] = out[col].replace(dict(mapping))

    for col in _column_list(rules.get("numeric_coerce", []), "numeric_coerce"):
        if col not in out.columns:
            raise ValueError(f"Unknown numeric column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    datetime_parse = rules.get("datetime_parse", {})
    if not isinstance(datetime_parse, Mapping):
        raise ValueError("datetime_parse must map columns to explicit strptime formats.")
    for col, date_format in datetime_parse.items():
        if col not in out.columns:
            raise ValueError(f"Unknown datetime column: {col}")
        if not isinstance(date_format, str) or not date_format:
            raise ValueError(f"datetime_parse format for {col!r} must be a non-empty string.")
        try:
            out[col] = pd.to_datetime(out[col], format=date_format, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot parse datetime column {col!r}: {exc}") from exc

    return out


def rules_from_task_spec(spec: TaskSpec) -> dict[str, Any]:
    """Translate the cleaning portion of one validated task spec.

    The result contains only explicit rules. It never uses corruption metadata
    to restore an unknown original cell value.
    """

    if not isinstance(spec, TaskSpec):
        raise TypeError("rules_from_task_spec requires a TaskSpec.")
    sentinel_map = {
        entry["column"]: list(entry["values"]) for entry in spec.sentinel_map
    }
    category_map = {
        entry["column"]: dict(entry["mapping"]) for entry in spec.category_map
    }
    return {
        "drop_exact_duplicates": spec.drop_duplicates,
        "dedup_subset": list(spec.dedup_subset),
        "sentinel_map": sentinel_map,
        "strip_strings": list(spec.strip_columns),
        "category_map": category_map,
        "numeric_coerce": list(spec.numeric_coerce),
    }
