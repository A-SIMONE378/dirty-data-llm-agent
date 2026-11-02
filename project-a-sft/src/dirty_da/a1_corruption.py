"""Deterministic A1 corruption profiles built on the audited A0 injector."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .canonical import CanonicalTable, dataframe_fingerprint
from .corruption import CORRUPTION_TYPES, CorruptionResult, apply_corruptions
from .schema_semantics import validate_table_semantics


class A1CorruptionPlanError(ValueError):
    """Raised for invalid or inapplicable A1 profile declarations."""


@dataclass(frozen=True)
class PlannedDirtyCopy:
    result: CorruptionResult
    requested_types: tuple[str, ...]
    applied_types: tuple[str, ...]
    inapplicable: dict[str, str]
    fingerprint: str


def load_corruption_profiles(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A1CorruptionPlanError(f"Cannot load corruption profiles from {source}: {exc}") from exc
    if config.get("schema_version") != 1 or not isinstance(config.get("profiles"), dict):
        raise A1CorruptionPlanError("Unsupported or malformed corruption profile config.")
    seed = config.get("base_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise A1CorruptionPlanError("base_seed must be an integer.")
    for profile_name, profile in config["profiles"].items():
        if not isinstance(profile_name, str) or not profile_name:
            raise A1CorruptionPlanError("Profile names must be non-empty strings.")
        if not isinstance(profile, Mapping):
            raise A1CorruptionPlanError(f"Profile {profile_name!r} must be an object.")
        types = profile.get("corruption_types")
        if not isinstance(types, list) or not types:
            raise A1CorruptionPlanError(f"Profile {profile_name!r} must request corruptions.")
        if len(types) != len(set(types)) or not set(types) <= CORRUPTION_TYPES:
            raise A1CorruptionPlanError(
                f"Profile {profile_name!r} contains duplicate or unsupported corruptions."
            )
        allow = profile.get("allow_inapplicable", [])
        if not isinstance(allow, list) or not set(allow) <= set(types):
            raise A1CorruptionPlanError(
                f"Profile {profile_name!r} has invalid allow_inapplicable values."
            )
    return config


def _first_nonmissing_position(frame: pd.DataFrame, column: str, start: int) -> int:
    if frame.empty:
        raise A1CorruptionPlanError("Cannot corrupt an empty table.")
    for offset in range(len(frame)):
        position = (start + offset) % len(frame)
        if not pd.isna(frame.iloc[position][column]):
            return position
    raise A1CorruptionPlanError(f"Column {column!r} contains only missing values.")


def _candidate_columns(
    frame: pd.DataFrame,
    table_name: str,
    semantics_registry: Mapping[str, Any],
) -> dict[str, str | None]:
    roles = validate_table_semantics(frame, table_name, semantics_registry)
    numeric = [
        column
        for column in frame.columns
        if roles[str(column)] in {"measure", "derived"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    text_categories = [
        column
        for column in frame.columns
        if roles[str(column)] == "category"
        and (
            pd.api.types.is_object_dtype(frame[column].dtype)
            or pd.api.types.is_string_dtype(frame[column].dtype)
        )
    ]
    text_dates = [
        column
        for column in frame.columns
        if roles[str(column)] == "date"
        and (
            pd.api.types.is_object_dtype(frame[column].dtype)
            or pd.api.types.is_string_dtype(frame[column].dtype)
        )
    ]
    general = numeric or text_categories or text_dates
    return {
        "general": str(general[0]) if general else None,
        "numeric": str(numeric[0]) if numeric else None,
        "format": str((text_dates or text_categories)[0]) if (text_dates or text_categories) else None,
        "category": str(text_categories[0]) if text_categories else None,
    }


def _event_for_type(
    frame: pd.DataFrame,
    candidates: Mapping[str, str | None],
    corruption_type: str,
    event_position: int,
    seed: int,
    severity: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if corruption_type == "duplicate_rows":
        return {
            "corruption_type": corruption_type,
            "row": event_position % len(frame),
            "seed": seed,
            "severity": severity,
        }, None
    candidate_key = {
        "missing_values": "general",
        "sentinel_values": "general",
        "type_corruption": "numeric",
        "format_inconsistency": "format",
        "category_inconsistency": "category",
    }[corruption_type]
    column = candidates[candidate_key]
    if column is None:
        return None, f"no_semantically_suitable_{candidate_key}_column"
    row = _first_nonmissing_position(frame, column, event_position)
    return {
        "corruption_type": corruption_type,
        "column": column,
        "row": row,
        "seed": seed,
        "severity": severity,
    }, None


def generate_dirty_copy(
    canonical: CanonicalTable,
    table_name: str,
    semantics_registry: Mapping[str, Any],
    profile_config: Mapping[str, Any],
    profile_name: str,
) -> PlannedDirtyCopy:
    """Generate one deterministic dirty copy and record honest inapplicability."""

    try:
        profile = profile_config["profiles"][profile_name]
        base_seed = profile_config["base_seed"]
    except (KeyError, TypeError) as exc:
        raise A1CorruptionPlanError(f"Unknown profile {profile_name!r}.") from exc
    requested = tuple(profile["corruption_types"])
    allowed_inapplicable = set(profile.get("allow_inapplicable", []))
    frame = canonical.copy_frame()
    candidates = _candidate_columns(frame, table_name, semantics_registry)
    events: list[dict[str, Any]] = []
    inapplicable: dict[str, str] = {}
    for event_position, corruption_type in enumerate(requested):
        event, reason = _event_for_type(
            frame,
            candidates,
            corruption_type,
            event_position,
            base_seed + event_position,
            profile_name,
        )
        if event is None:
            inapplicable[corruption_type] = str(reason)
            continue
        events.append(event)
    unexpected = set(inapplicable) - allowed_inapplicable
    if unexpected:
        raise A1CorruptionPlanError(
            f"{table_name}/{profile_name}: unexpectedly inapplicable corruptions "
            f"{sorted(unexpected)}."
        )
    if not events:
        raise A1CorruptionPlanError(
            f"{table_name}/{profile_name}: no applicable corruption events."
        )
    result = apply_corruptions(canonical, events)
    canonical.assert_intact()
    applied = tuple(event["corruption_type"] for event in result.metadata)
    return PlannedDirtyCopy(
        result=result,
        requested_types=requested,
        applied_types=applied,
        inapplicable=inapplicable,
        fingerprint=dataframe_fingerprint(result.dirty),
    )
