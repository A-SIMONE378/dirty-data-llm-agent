"""Apply reviewed A1 semantic corrections discovered during the A2 audit.

UCI variable metadata is useful provenance, but it is not a substitute for
checking the actual values.  This script records the small, explicit set of
role corrections made after that check and quarantines tables that no longer
meet the A1 admission criteria.  It is intentionally idempotent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "configs" / "a1_table_catalog.json"
SEMANTICS_PATH = ROOT / "configs" / "table_semantics.json"
TABLES = ROOT / "data" / "tables"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def set_roles(semantics: dict, file_name: str, roles: dict[str, list[str]]) -> None:
    frame = pd.read_csv(TABLES / file_name)
    expected = list(map(str, frame.columns))
    assigned = [column for values in roles.values() for column in values]
    if len(assigned) != len(set(assigned)):
        raise RuntimeError(f"{file_name}: a column is assigned to multiple roles")
    if set(assigned) != set(expected):
        missing = sorted(set(expected) - set(assigned))
        extra = sorted(set(assigned) - set(expected))
        raise RuntimeError(f"{file_name}: role mismatch; missing={missing}, extra={extra}")
    semantics["tables"][file_name]["roles"] = roles


catalog = load(CATALOG_PATH)
semantics = load(SEMANTICS_PATH)

set_roles(
    semantics,
    "uci_247_istanbul_stock_exchange.csv",
    {
        "identifier": [],
        "category": [],
        "date": ["date"],
        "measure": ["ISE", "ISE.1", "SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"],
        "derived": [],
        "coordinate": [],
        "opaque": [],
    },
)
set_roles(
    semantics,
    "uci_536_pedestrians_traffic.csv",
    {
        "identifier": ["oid"],
        "category": [],
        "date": [],
        "measure": ["timestamp", "body_roll", "body_pitch", "body_yaw", "head_roll", "head_pitch", "head_yaw"],
        "derived": [],
        "coordinate": ["x", "y"],
        "opaque": ["other_oid", "other_class", "other_x", "other_y"],
    },
)
set_roles(
    semantics,
    "uci_029_computer_hardware.csv",
    {
        "identifier": ["ModelName"],
        "category": ["VendorName"],
        "date": [],
        "measure": ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"],
        "derived": [],
        "coordinate": [],
        "opaque": [],
    },
)
set_roles(
    semantics,
    "uci_312_dow_jones_index.csv",
    {
        "identifier": [],
        "category": ["stock", "quarter"],
        "date": ["date"],
        "measure": [
            "open", "high", "low", "close", "volume", "percent_change_price",
            "percent_change_volume_over_last_wk", "previous_weeks_volume",
            "next_weeks_open", "next_weeks_close", "percent_change_next_weeks_price",
            "days_to_next_dividend", "percent_return_next_dividend",
        ],
        "derived": [],
        "coordinate": [],
        "opaque": [],
    },
)
set_roles(
    semantics,
    "uci_390_stock_portfolio.csv",
    {
        "identifier": ["ID"],
        "category": [
            "period", "Large B/P", "Large ROE", "Large S/P",
            "Large Return Rate in the last quarter", "Large Market Value",
            "Small systematic Risk",
        ],
        "date": [],
        "measure": [
            "Annual Return", "Excess Return", "Systematic Risk", "Total Risk",
            "Abs. Win Rate", "Rel. Win Rate", "Annual Return.1", "Excess Return.1",
            "Systematic Risk.1", "Total Risk.1", "Abs. Win Rate.1", "Rel. Win Rate.1",
        ],
        "derived": [],
        "coordinate": [],
        "opaque": [],
    },
)
set_roles(
    semantics,
    "uci_461_drug_reviews.csv",
    {
        "identifier": ["reviewID"],
        "category": ["urlDrugName", "effectiveness", "sideEffects", "condition"],
        "date": [],
        "measure": ["rating"],
        "derived": [],
        "coordinate": [],
        "opaque": ["benefitsReview", "sideEffectsReview", "commentsReview"],
    },
)

quarantine_reasons = {
    "uci_247_istanbul_stock_exchange.csv": (
        "Quarantined after the A2 semantic audit: the table has continuous returns "
        "and a date, but no true categorical field; SP is a continuous S&P return."
    ),
    "uci_536_pedestrians_traffic.csv": (
        "Quarantined after the A2 semantic audit: the table has numeric trajectories, "
        "angles, and list-valued other-object fields, but no scalar categorical field."
    ),
}
for file_name, reason in quarantine_reasons.items():
    record = catalog["tables"][file_name]
    record["selection_status"] = "quarantine"
    record["split"] = None
    record["approved_for_research"] = False
    notes = [note for note in record.get("notes", []) if not note.startswith("Quarantined after the A2 semantic audit:")]
    notes.append(reason)
    record["notes"] = notes

atomic_write(SEMANTICS_PATH, semantics)
atomic_write(CATALOG_PATH, catalog)
print(json.dumps({"corrected_tables": 6, "newly_quarantined": 2}, indent=2))
