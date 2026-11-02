"""Programmatic execution of validated structured task specifications."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .task_spec import TaskSpec


class ExecutionError(ValueError):
    pass


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _apply_filters(df: pd.DataFrame, filters: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    out = df
    for filter_spec in filters:
        col, op, val = (
            filter_spec["column"],
            filter_spec["operator"],
            filter_spec["value"],
        )
        series = out[col]
        try:
            if op == "eq":
                mask = series == val
            elif op == "ne":
                mask = series != val
            elif op == "gt":
                mask = series > val
            elif op == "ge":
                mask = series >= val
            elif op == "lt":
                mask = series < val
            elif op == "le":
                mask = series <= val
            elif op == "in":
                mask = series.isin(val)
            else:  # guarded by TaskSpec.validate
                raise ExecutionError(f"Unsupported filter: {op}")
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                f"Filter {col} {op} {val!r} incompatible with dtype {series.dtype}"
            ) from exc
        mask = mask.fillna(False) & series.notna()
        out = out.loc[mask]
    return out


def _python_scalar(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _apply_declared_cleaning(work: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    if spec.drop_duplicates:
        subset = list(spec.dedup_subset) or None
        work = work.drop_duplicates(subset=subset).reset_index(drop=True)

    if spec.sentinel_policy == "map":
        for entry in spec.sentinel_map:
            col = entry["column"]
            values = list(entry["values"])
            if any(_is_missing(value) for value in values):
                raise ExecutionError("NaN/NA cannot be declared as a sentinel value.")
            mask = work[col].isin(values)
            work.loc[mask, col] = pd.NA

    for column in spec.strip_columns:
        work[column] = work[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    for entry in spec.category_map:
        work[entry["column"]] = work[entry["column"]].replace(dict(entry["mapping"]))

    for column in spec.numeric_coerce:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def _aggregate_target(series: pd.Series, aggregation: str) -> Any:
    if aggregation in {"sum", "mean"} and not pd.api.types.is_numeric_dtype(series):
        raise ExecutionError(f"{aggregation} requires a numeric target column.")
    try:
        if aggregation == "count":
            return int(series.count())
        if aggregation == "sum":
            value = _python_scalar(series.sum(skipna=True, min_count=1))
            return None if value is None else float(value)
        if aggregation == "mean":
            return _python_scalar(series.mean(skipna=True))
        if aggregation == "min":
            return _python_scalar(series.min(skipna=True))
        if aggregation == "max":
            return _python_scalar(series.max(skipna=True))
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"Cannot compute {aggregation}: {exc}") from exc
    raise ExecutionError(f"Unsupported aggregation: {aggregation}")


def _compare(value: Any, operator: str, threshold: Any) -> bool:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionError("aggregate_compare requires a non-missing numeric aggregate.")
    numeric_value = float(value)
    numeric_threshold = float(threshold)
    if not math.isfinite(numeric_value) or not math.isfinite(numeric_threshold):
        raise ExecutionError("aggregate_compare requires finite numeric values.")
    if operator == "eq":
        return numeric_value == numeric_threshold
    if operator == "ne":
        return numeric_value != numeric_threshold
    if operator == "gt":
        return numeric_value > numeric_threshold
    if operator == "ge":
        return numeric_value >= numeric_threshold
    if operator == "lt":
        return numeric_value < numeric_threshold
    if operator == "le":
        return numeric_value <= numeric_threshold
    raise ExecutionError(f"Unsupported comparison operator: {operator}")


def execute_task(df: pd.DataFrame, spec: TaskSpec) -> Any:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("execute_task expects a pandas DataFrame.")
    spec.validate(df.columns)
    work = _apply_declared_cleaning(df.copy(deep=True), spec)

    referenced = {filter_spec["column"] for filter_spec in spec.filters}
    if spec.target_column is not None:
        referenced.add(spec.target_column)
    if spec.group_by is not None:
        referenced.add(spec.group_by)
    if (
        spec.missing_policy == "error"
        and referenced
        and work.loc[:, list(referenced)].isna().any(axis=None)
    ):
        raise ExecutionError(
            "Task policy is missing_policy='error' but a referenced column contains missing values."
        )

    work = _apply_filters(work, spec.filters)
    operation = spec.operation
    if operation == "count_rows":
        return int(len(work))

    target = spec.target_column
    assert target is not None
    if operation in {"sum", "mean", "min", "max"}:
        return _aggregate_target(work[target], operation)

    if operation == "aggregate_compare":
        aggregation = spec.aggregation
        comparison_operator = spec.comparison_operator
        assert aggregation is not None and comparison_operator is not None
        value = _aggregate_target(work[target], aggregation)
        return _compare(value, comparison_operator, spec.comparison_value)

    if operation == "groupby_agg":
        group_column = spec.group_by
        aggregation = spec.aggregation
        assert group_column is not None and aggregation is not None
        if aggregation in {"sum", "mean"} and not pd.api.types.is_numeric_dtype(
            work[target]
        ):
            raise ExecutionError(f"{aggregation} requires numeric target column: {target}")
        grouped = work.groupby(group_column, dropna=True, sort=True)[target]
        try:
            if aggregation == "sum":
                values = grouped.sum(min_count=1)
            else:
                values = grouped.agg(aggregation)
        except (TypeError, ValueError) as exc:
            raise ExecutionError(
                f"Cannot compute grouped {aggregation} for target {target!r}: {exc}"
            ) from exc

        result: dict[str, Any] = {}
        for key, value in values.items():
            normalized = _python_scalar(value)
            if aggregation in {"sum", "mean"} and normalized is not None:
                normalized = float(normalized)
            result[str(_python_scalar(key))] = normalized
        return result
    raise ExecutionError(f"Unsupported operation: {operation}")
