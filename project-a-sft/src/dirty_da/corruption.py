"""Deterministic corruptions applied only to defensive canonical copies."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .canonical import CanonicalTable

CORRUPTION_TYPES = {
    "missing_values",
    "sentinel_values",
    "duplicate_rows",
    "type_corruption",
    "format_inconsistency",
    "category_inconsistency",
}


@dataclass
class CorruptionResult:
    dirty: pd.DataFrame
    metadata: list[dict[str, Any]]


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _json_safe(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _row_position(df: pd.DataFrame, row: int) -> int:
    if isinstance(row, bool) or not isinstance(row, Integral):
        raise ValueError("row must be an integer position.")
    position = int(row)
    if position < 0 or position >= len(df):
        raise ValueError(f"Row out of range: {position}")
    return position


def _assign(df: pd.DataFrame, position: int, column: str, value: Any) -> None:
    column_position = int(df.columns.get_loc(column))
    # Always use object storage for a corrupted column so pandas cannot silently
    # parse a deliberately malformed string back into the original dtype.
    df[column] = df[column].astype("object")
    df.iat[position, column_position] = value


def _default_type_corruption(before: Any) -> Any:
    if _is_missing(before):
        raise ValueError("Cannot type-corrupt a missing value.")
    if isinstance(before, (bool, np.bool_)):
        return "true" if bool(before) else "false"
    if isinstance(before, (int, float, np.number)) and not isinstance(before, (bool, np.bool_)):
        return repr(before.item() if isinstance(before, np.generic) else before)
    if isinstance(before, pd.Timestamp):
        return before.isoformat()
    raise ValueError(
        "type_corruption requires a bool, numeric, or Timestamp cell unless an explicit "
        "replacement with a different type is supplied."
    )


def _apply_event(
    df: pd.DataFrame,
    corruption_type: str,
    *,
    column: str | None,
    row: int,
    seed: int,
    severity: str,
    value: Any,
    event_id: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"Unsupported corruption type: {corruption_type}")
    if not isinstance(severity, str) or not severity.strip():
        raise ValueError("severity must be a non-empty label.")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer.")
    if isinstance(event_id, bool) or not isinstance(event_id, Integral) or event_id < 0:
        raise ValueError("event_id must be a non-negative integer.")

    position = _row_position(df, row)
    event: dict[str, Any] = {
        "event_id": int(event_id),
        "corruption_type": corruption_type,
        "severity": severity,
        "seed": int(seed),
        "column": column,
        "row_position": position,
        "row_index": _json_safe(df.index[position]),
    }

    if corruption_type == "duplicate_rows":
        if column is not None or value is not None:
            raise ValueError("duplicate_rows does not accept column or value.")
        original = df.iloc[position].to_dict()
        output_position = len(df)
        df = pd.concat([df, df.iloc[[position]].copy(deep=True)], axis=0)
        event.update(
            before=_json_safe(original),
            after=_json_safe(original),
            output_row_position=output_position,
            operation="append_exact_duplicate",
        )
        return df, event

    if column is None:
        raise ValueError(f"{corruption_type} requires a column.")
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")

    before = df.iloc[position][column]
    if corruption_type == "missing_values":
        if _is_missing(before):
            raise ValueError("Cannot inject missingness into an already missing cell.")
        after = pd.NA
    elif corruption_type == "sentinel_values":
        if _is_missing(before):
            raise ValueError("Cannot replace a missing value with a sentinel corruption.")
        if value is not None:
            after = value
        elif pd.api.types.is_numeric_dtype(df[column].dtype) and not isinstance(
            before, (bool, np.bool_)
        ):
            after = -999999
        else:
            after = "__MISSING__"
        if _is_missing(after):
            raise ValueError("A sentinel corruption cannot itself be missing.")
        if after == before:
            raise ValueError("Sentinel value must differ from the original value.")
    elif corruption_type == "type_corruption":
        if _is_missing(before):
            raise ValueError("Cannot type-corrupt a missing value.")
        after = _default_type_corruption(before) if value is None else value
        if type(after) is type(before):
            raise ValueError("type_corruption replacement must have a different Python type.")
    elif corruption_type == "format_inconsistency":
        if _is_missing(before):
            raise ValueError("Cannot format-corrupt a missing value.")
        if isinstance(before, pd.Timestamp):
            after = before.strftime("%m/%d/%Y")
        elif isinstance(before, str):
            after = f"  {before.strip().upper()}  "
        else:
            raise ValueError("format_inconsistency requires a string or timestamp cell.")
        if after == before:
            raise ValueError("Format corruption produced no change.")
    else:
        if _is_missing(before):
            raise ValueError("Cannot category-corrupt a missing value.")
        if not isinstance(before, str):
            raise ValueError("category_inconsistency requires a string category cell.")
        after = value if value is not None else f"{before.strip().lower()} "
        if not isinstance(after, str) or after == before:
            raise ValueError("Category replacement must be a different string.")

    _assign(df, position, column, after)
    event.update(
        before=_json_safe(before),
        after=_json_safe(after),
        operation="replace_cell",
    )
    return df, event


def apply_corruptions(
    canonical: CanonicalTable,
    events: Iterable[Mapping[str, Any]],
) -> CorruptionResult:
    """Apply ordered event specifications to one fresh dirty copy.

    Each mapping must name corruption_type and may supply the same keyword
    fields as apply_corruption. Event ids are assigned by list order, making
    compound corruption order explicit and reproducible.
    """

    if not isinstance(canonical, CanonicalTable):
        raise TypeError("apply_corruptions requires a CanonicalTable.")
    dirty = canonical.copy_frame()
    metadata: list[dict[str, Any]] = []
    for event_id, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            raise TypeError("Each corruption event must be a mapping.")
        event = dict(raw_event)
        if "corruption_type" not in event:
            raise ValueError("Each corruption event requires corruption_type.")
        corruption_type = event.pop("corruption_type")
        unknown = set(event) - {"column", "row", "seed", "severity", "value"}
        if unknown:
            raise ValueError(f"Unsupported corruption event fields: {sorted(unknown)}")
        dirty, record = _apply_event(
            dirty,
            corruption_type,
            column=event.get("column"),
            row=event.get("row", 0),
            seed=event.get("seed", 0),
            severity=event.get("severity", "low"),
            value=event.get("value"),
            event_id=event_id,
        )
        metadata.append(record)
    canonical.assert_intact()
    return CorruptionResult(dirty=dirty, metadata=metadata)


def apply_corruption(
    canonical: CanonicalTable,
    corruption_type: str,
    *,
    column: str | None = None,
    row: int = 0,
    seed: int = 0,
    severity: str = "low",
    value: Any = None,
    event_id: int = 0,
) -> CorruptionResult:
    """Apply one explicit corruption to a defensive copy; invalid cases fail loudly."""

    if not isinstance(canonical, CanonicalTable):
        raise TypeError("apply_corruption requires a CanonicalTable.")
    dirty = canonical.copy_frame()
    dirty, record = _apply_event(
        dirty,
        corruption_type,
        column=column,
        row=row,
        seed=seed,
        severity=severity,
        value=value,
        event_id=event_id,
    )
    canonical.assert_intact()
    return CorruptionResult(dirty=dirty, metadata=[record])
