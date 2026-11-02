"""Bounded, machine-validatable task semantics for Project A."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any


class TaskSpecError(ValueError):
    pass


ALLOWED_OPERATIONS = {
    "count_rows",
    "sum",
    "mean",
    "min",
    "max",
    "groupby_agg",
    "aggregate_compare",
}
ALLOWED_AGGS = {"count", "sum", "mean", "min", "max"}
ALLOWED_FILTER_OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le", "in"}
ALLOWED_COMPARISONS = {"eq", "ne", "gt", "ge", "lt", "le"}


def _column_sequence(value: Any, field_name: str, columns: set[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskSpecError(f"{field_name} must be a sequence of column names.")
    normalized = tuple(value)
    if len(normalized) != len(set(normalized)):
        raise TaskSpecError(f"{field_name} cannot contain duplicate columns.")
    if not all(isinstance(column, str) and column in columns for column in normalized):
        raise TaskSpecError(f"{field_name} contains an unknown or invalid column.")
    return normalized


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    operation: str
    target_column: str | None = None
    group_by: str | None = None
    aggregation: str | None = None
    comparison_operator: str | None = None
    comparison_value: Any = None
    filters: tuple[dict[str, Any], ...] = ()
    drop_duplicates: bool = False
    dedup_subset: tuple[str, ...] = ()
    missing_policy: str = "exclude"
    sentinel_policy: str = "none"
    sentinel_map: tuple[dict[str, Any], ...] = ()
    strip_columns: tuple[str, ...] = ()
    category_map: tuple[dict[str, Any], ...] = ()
    numeric_coerce: tuple[str, ...] = ()
    description: str = ""

    def validate(self, columns: Sequence[str]) -> TaskSpec:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise TaskSpecError("task_id must be a non-empty string.")
        cols = set(map(str, columns))
        if self.operation not in ALLOWED_OPERATIONS:
            raise TaskSpecError(f"Unsupported operation: {self.operation}")
        if not isinstance(self.drop_duplicates, bool):
            raise TaskSpecError("drop_duplicates must be boolean.")
        dedup_subset = _column_sequence(self.dedup_subset, "dedup_subset", cols)
        if dedup_subset and not self.drop_duplicates:
            raise TaskSpecError("dedup_subset requires drop_duplicates=True.")
        if self.missing_policy not in {"exclude", "error"}:
            raise TaskSpecError(f"Unsupported missing policy: {self.missing_policy}")
        if self.sentinel_policy not in {"none", "map"}:
            raise TaskSpecError(f"Unsupported sentinel policy: {self.sentinel_policy}")
        if self.sentinel_policy == "map":
            if not self.sentinel_map:
                raise TaskSpecError("sentinel_policy='map' requires sentinel_map entries.")
            seen_sentinel_columns: set[str] = set()
            for entry in self.sentinel_map:
                if not isinstance(entry, Mapping) or set(entry) != {"column", "values"}:
                    raise TaskSpecError("Each sentinel_map entry needs column and values.")
                column = entry["column"]
                if column not in cols:
                    raise TaskSpecError(f"Unknown sentinel column: {column}")
                if column in seen_sentinel_columns:
                    raise TaskSpecError(f"Duplicate sentinel_map column: {column}")
                seen_sentinel_columns.add(column)
                values = entry["values"]
                if (
                    isinstance(values, (str, bytes))
                    or not isinstance(values, Sequence)
                    or not values
                ):
                    raise TaskSpecError("sentinel_map values must be a non-empty sequence.")
        elif self.sentinel_map:
            raise TaskSpecError("sentinel_map is only valid with sentinel_policy='map'.")

        _column_sequence(self.strip_columns, "strip_columns", cols)
        _column_sequence(self.numeric_coerce, "numeric_coerce", cols)
        seen_category_columns: set[str] = set()
        for entry in self.category_map:
            if not isinstance(entry, Mapping) or set(entry) != {"column", "mapping"}:
                raise TaskSpecError("Each category_map entry needs column and mapping.")
            column = entry["column"]
            mapping = entry["mapping"]
            if column not in cols:
                raise TaskSpecError(f"Unknown category-map column: {column}")
            if column in seen_category_columns:
                raise TaskSpecError(f"Duplicate category_map column: {column}")
            if not isinstance(mapping, Mapping) or not mapping:
                raise TaskSpecError("category_map mapping must be a non-empty mapping.")
            seen_category_columns.add(column)

        requires_target = self.operation in {
            "sum",
            "mean",
            "min",
            "max",
            "groupby_agg",
            "aggregate_compare",
        }
        if requires_target and (
            not self.target_column or self.target_column not in cols
        ):
            raise TaskSpecError("A valid target_column is required.")
        if self.operation == "count_rows" and self.target_column is not None:
            raise TaskSpecError("count_rows does not accept target_column.")

        if self.operation == "groupby_agg":
            if not self.group_by or self.group_by not in cols:
                raise TaskSpecError("groupby_agg requires a valid group_by column.")
            if self.aggregation not in ALLOWED_AGGS:
                raise TaskSpecError("groupby_agg requires a supported aggregation.")
        elif self.group_by is not None:
            raise TaskSpecError(f"{self.operation} does not accept group_by.")

        if self.operation == "aggregate_compare":
            if self.aggregation not in ALLOWED_AGGS:
                raise TaskSpecError("aggregate_compare requires a supported aggregation.")
            if self.comparison_operator not in ALLOWED_COMPARISONS:
                raise TaskSpecError("aggregate_compare requires a supported comparison_operator.")
            if (
                self.comparison_value is None
                or isinstance(self.comparison_value, bool)
                or not isinstance(self.comparison_value, Real)
            ):
                raise TaskSpecError("aggregate_compare requires a numeric comparison_value.")
        else:
            if self.operation != "groupby_agg" and self.aggregation is not None:
                raise TaskSpecError(f"{self.operation} does not accept aggregation.")
            if self.comparison_operator is not None or self.comparison_value is not None:
                raise TaskSpecError(
                    f"{self.operation} does not accept comparison fields."
                )

        for filter_spec in self.filters:
            if not isinstance(filter_spec, Mapping) or set(filter_spec) != {
                "column",
                "operator",
                "value",
            }:
                raise TaskSpecError("Each filter must contain column, operator, and value.")
            if filter_spec["column"] not in cols:
                raise TaskSpecError(f"Unknown filter column: {filter_spec['column']}")
            if filter_spec["operator"] not in ALLOWED_FILTER_OPERATORS:
                raise TaskSpecError(
                    f"Unsupported filter operator: {filter_spec['operator']}"
                )
            if filter_spec["operator"] == "in":
                value = filter_spec["value"]
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                    raise TaskSpecError("The 'in' filter requires a non-string sequence.")
        return self

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["filters"] = list(self.filters)
        value["dedup_subset"] = list(self.dedup_subset)
        value["sentinel_map"] = list(self.sentinel_map)
        value["strip_columns"] = list(self.strip_columns)
        value["category_map"] = list(self.category_map)
        value["numeric_coerce"] = list(self.numeric_coerce)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSpec:
        if not isinstance(value, Mapping):
            raise TaskSpecError("Task spec must be an object.")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise TaskSpecError(f"Unsupported task spec fields: {sorted(unknown)}")
        normalized = dict(value)
        for field_name in (
            "filters",
            "dedup_subset",
            "sentinel_map",
            "strip_columns",
            "category_map",
            "numeric_coerce",
        ):
            if field_name in normalized:
                raw = normalized[field_name]
                if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                    raise TaskSpecError(f"{field_name} must be a sequence.")
                normalized[field_name] = tuple(raw)
        try:
            return cls(**normalized)
        except TypeError as exc:
            raise TaskSpecError(f"Malformed task spec: {exc}") from exc
