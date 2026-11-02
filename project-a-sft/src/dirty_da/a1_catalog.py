"""A1 table catalog validation and completion accounting.

The catalog is deliberately explicit.  A numeric dtype does not establish that
a column is a measure, and a familiar file name does not establish provenance
or a reusable license.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .schema_semantics import validate_table_semantics


SPLITS = {"train", "dev", "test", "domain_generalization"}
SELECTION_STATUSES = {"selected", "quarantine"}
PROVENANCE_STATUSES = {"verified", "partial", "unknown"}
LICENSE_STATUSES = {"verified", "unknown", "restricted"}


class A1CatalogError(ValueError):
    """Raised when the committed A1 catalog is internally inconsistent."""


@dataclass(frozen=True)
class CatalogAudit:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def load_a1_catalog(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        catalog = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A1CatalogError(f"Cannot load A1 catalog from {source}: {exc}") from exc
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("tables"), dict):
        raise A1CatalogError("Unsupported or malformed A1 catalog.")
    return catalog


def _require_nonempty_string(value: Any, field: str, table_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A1CatalogError(f"{table_name}: {field} must be a non-empty string.")
    return value.strip()


def _validate_catalog_record(table_name: str, record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise A1CatalogError(f"{table_name}: catalog record must be an object.")
    _require_nonempty_string(record.get("domain"), "domain", table_name)
    selection_status = record.get("selection_status", "selected")
    if selection_status not in SELECTION_STATUSES:
        raise A1CatalogError(f"{table_name}: invalid selection_status.")
    split = record.get("split")
    if selection_status == "selected" and split not in SPLITS:
        raise A1CatalogError(f"{table_name}: unsupported split {split!r}.")
    if selection_status == "quarantine" and split is not None:
        raise A1CatalogError(f"{table_name}: quarantine records must have null split.")
    provenance = record.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise A1CatalogError(f"{table_name}: provenance must be an object.")
    _require_nonempty_string(provenance.get("source_name"), "source_name", table_name)
    if provenance.get("status") not in PROVENANCE_STATUSES:
        raise A1CatalogError(f"{table_name}: invalid provenance status.")
    source_url = provenance.get("source_url")
    if source_url is not None and (
        not isinstance(source_url, str) or not source_url.startswith(("https://", "http://"))
    ):
        raise A1CatalogError(f"{table_name}: source_url must be HTTP(S) or null.")
    transformations = provenance.get("transformations")
    if not isinstance(transformations, list) or not all(
        isinstance(item, str) and item.strip() for item in transformations
    ):
        raise A1CatalogError(f"{table_name}: transformations must be non-empty strings.")

    license_spec = record.get("license", {})
    if not isinstance(license_spec, Mapping):
        raise A1CatalogError(f"{table_name}: license must be an object.")
    license_status = license_spec.get("status")
    if license_status not in LICENSE_STATUSES:
        raise A1CatalogError(f"{table_name}: invalid license status.")
    _require_nonempty_string(license_spec.get("name"), "license.name", table_name)
    license_url = license_spec.get("url")
    if license_url is not None and (
        not isinstance(license_url, str)
        or not license_url.startswith(("https://", "http://"))
    ):
        raise A1CatalogError(f"{table_name}: license.url must be HTTP(S) or null.")

    approved = record.get("approved_for_research")
    if not isinstance(approved, bool):
        raise A1CatalogError(f"{table_name}: approved_for_research must be boolean.")
    if approved and (
        provenance.get("status") == "unknown" or license_status != "verified"
    ):
        raise A1CatalogError(
            f"{table_name}: research approval requires known provenance and verified license."
        )
    if selection_status == "quarantine" and approved:
        raise A1CatalogError(f"{table_name}: quarantine records cannot be research-approved.")
    notes = record.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise A1CatalogError(f"{table_name}: notes must be a list of strings.")


def audit_a1_catalog(
    tables_dir: str | Path,
    catalog: Mapping[str, Any],
    semantics_registry: Mapping[str, Any],
) -> CatalogAudit:
    """Validate the exact CSV inventory and return factual A1 readiness metrics."""

    root = Path(tables_dir)
    paths = sorted(root.glob("*.csv"))
    actual_names = {path.name for path in paths}
    catalog_tables = catalog.get("tables")
    if not isinstance(catalog_tables, Mapping):
        raise A1CatalogError("Catalog tables must be an object.")
    declared_names = set(catalog_tables)
    if actual_names != declared_names:
        raise A1CatalogError(
            "A1 catalog does not match CSV inventory; "
            f"missing={sorted(actual_names - declared_names)}, "
            f"extra={sorted(declared_names - actual_names)}."
        )

    criteria = catalog.get("selection_criteria", {})
    required_integer_fields = ("min_tables", "max_tables", "min_rows", "max_rows", "min_columns", "max_columns")
    if not isinstance(criteria, Mapping):
        raise A1CatalogError("selection_criteria must be an object.")
    for field in required_integer_fields:
        value = criteria.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise A1CatalogError(f"selection_criteria.{field} must be a non-negative integer.")
    if criteria["min_tables"] > criteria["max_tables"]:
        raise A1CatalogError("min_tables cannot exceed max_tables.")
    if criteria["min_rows"] > criteria["max_rows"]:
        raise A1CatalogError("min_rows cannot exceed max_rows.")
    if criteria["min_columns"] > criteria["max_columns"]:
        raise A1CatalogError("min_columns cannot exceed max_columns.")

    rows: list[dict[str, Any]] = []
    for path in paths:
        record = catalog_tables[path.name]
        _validate_catalog_record(path.name, record)
        frame = pd.read_csv(path, low_memory=False)
        role_by_column = validate_table_semantics(
            frame, path.name, semantics_registry
        )
        numeric_measures = [
            column
            for column, role in role_by_column.items()
            if role in {"measure", "derived"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        categories = [
            column for column, role in role_by_column.items() if role == "category"
        ]
        dates = [column for column, role in role_by_column.items() if role == "date"]
        row_count_ok = criteria["min_rows"] <= len(frame) <= criteria["max_rows"]
        column_count_ok = criteria["min_columns"] <= len(frame.columns) <= criteria["max_columns"]
        unique_rows = not bool(frame.duplicated().any())
        structural_eligible = bool(
            row_count_ok
            and column_count_ok
            and unique_rows
            and numeric_measures
            and categories
        )
        approved = bool(record["approved_for_research"])
        selection_status = record.get("selection_status", "selected")
        rows.append(
            {
                "table": path.name,
                "domain": record["domain"],
                "selection_status": selection_status,
                "split": record["split"],
                "rows": len(frame),
                "columns": len(frame.columns),
                "unique_rows": unique_rows,
                "numeric_measure_count": len(numeric_measures),
                "category_count": len(categories),
                "date_count": len(dates),
                "structural_eligible": structural_eligible,
                "provenance_status": record["provenance"]["status"],
                "license_status": record["license"]["status"],
                "approved_for_research": approved,
                "admitted": selection_status == "selected" and structural_eligible and approved,
            }
        )

    selected_rows = [row for row in rows if row["selection_status"] == "selected"]
    split_counts = {
        split: sum(row["split"] == split for row in selected_rows) for split in sorted(SPLITS)
    }
    core_count = sum(split_counts[split] for split in ("train", "dev", "test"))
    domains = {row["domain"] for row in selected_rows}
    admitted = [row for row in rows if row["admitted"]]
    min_domain_holdout = int(catalog.get("split_policy", {}).get("domain_generalization_min", 5))
    max_domain_holdout = int(catalog.get("split_policy", {}).get("domain_generalization_max", 8))
    domain_count = split_counts["domain_generalization"]
    blocking_reasons: list[str] = []
    if core_count < criteria["min_tables"]:
        blocking_reasons.append(
            f"core_table_count_below_minimum:{core_count}/{criteria['min_tables']}"
        )
    if core_count > criteria["max_tables"]:
        blocking_reasons.append(
            f"core_table_count_above_maximum:{core_count}/{criteria['max_tables']}"
        )
    if len(admitted) != len(selected_rows):
        blocking_reasons.append(
            f"selected_tables_not_admitted:{len(selected_rows) - len(admitted)}"
        )
    if not (min_domain_holdout <= domain_count <= max_domain_holdout):
        blocking_reasons.append(
            f"domain_generalization_count_out_of_range:{domain_count}/{min_domain_holdout}-{max_domain_holdout}"
        )
    for split in ("train", "dev", "test"):
        if split_counts[split] == 0:
            blocking_reasons.append(f"empty_split:{split}")

    split_policy = catalog.get("split_policy", {})
    try:
        train_ratio = float(split_policy["train_ratio"])
        dev_ratio = float(split_policy["dev_ratio"])
        test_ratio = float(split_policy["test_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise A1CatalogError("split_policy requires numeric train/dev/test ratios.") from exc
    if any(ratio < 0 for ratio in (train_ratio, dev_ratio, test_ratio)) or not abs(
        train_ratio + dev_ratio + test_ratio - 1.0
    ) < 1e-12:
        raise A1CatalogError("train/dev/test ratios must be non-negative and sum to 1.")
    expected_dev = round(core_count * dev_ratio)
    expected_test = round(core_count * test_ratio)
    expected_train = core_count - expected_dev - expected_test
    expected_counts = {
        "train": expected_train,
        "dev": expected_dev,
        "test": expected_test,
    }
    observed_core_counts = {split: split_counts[split] for split in expected_counts}
    if observed_core_counts != expected_counts:
        blocking_reasons.append(
            f"core_split_ratio_mismatch:observed={observed_core_counts},expected={expected_counts}"
        )

    summary = {
        "tables_cataloged": len(rows),
        "tables_selected": len(selected_rows),
        "tables_quarantined": len(rows) - len(selected_rows),
        "core_tables": core_count,
        "domain_generalization_tables": domain_count,
        "target_total_range": [
            criteria["min_tables"] + min_domain_holdout,
            criteria["max_tables"] + max_domain_holdout,
        ],
        "tables_structurally_eligible": sum(row["structural_eligible"] for row in rows),
        "tables_research_approved": sum(row["approved_for_research"] for row in rows),
        "tables_admitted": len(admitted),
        "domains": len(domains),
        "split_counts": split_counts,
        "expected_core_split_counts": expected_counts,
        "a1_complete": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }
    return CatalogAudit(rows=rows, summary=summary)
