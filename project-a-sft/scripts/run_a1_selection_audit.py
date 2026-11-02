"""Audit the current A1 table-selection pilot and corruption reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dirty_da.a1_catalog import audit_a1_catalog, load_a1_catalog
from dirty_da.a1_corruption import generate_dirty_copy, load_corruption_profiles
from dirty_da.canonical import CanonicalTable
from dirty_da.schema_semantics import load_table_semantics


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

catalog = load_a1_catalog(CONFIGS / "a1_table_catalog.json")
semantics = load_table_semantics(CONFIGS / "table_semantics.json")
profiles = load_corruption_profiles(CONFIGS / "a1_corruption_profiles.json")
catalog_audit = audit_a1_catalog(TABLES, catalog, semantics)

reproducibility_rows: list[dict] = []
for path in sorted(TABLES.glob("*.csv")):
    frame = pd.read_csv(path, low_memory=False)
    canonical = CanonicalTable(frame, name=path.stem)
    canonical_fingerprint = canonical.fingerprint
    for profile_name in profiles["profiles"]:
        first = generate_dirty_copy(
            canonical, path.name, semantics, profiles, profile_name
        )
        second = generate_dirty_copy(
            canonical, path.name, semantics, profiles, profile_name
        )
        deterministic = bool(
            first.fingerprint == second.fingerprint
            and first.result.metadata == second.result.metadata
        )
        canonical_unchanged = canonical.fingerprint == canonical_fingerprint
        dirty_differs = first.fingerprint != canonical_fingerprint
        if not deterministic or not canonical_unchanged or not dirty_differs:
            raise RuntimeError(
                f"A1 dirty-copy invariant failed for {path.name}/{profile_name}."
            )
        reproducibility_rows.append(
            {
                "table": path.name,
                "profile": profile_name,
                "requested_event_count": len(first.requested_types),
                "applied_event_count": len(first.applied_types),
                "applied_types": ",".join(first.applied_types),
                "inapplicable": json.dumps(
                    first.inapplicable, sort_keys=True, ensure_ascii=False
                ),
                "deterministic": deterministic,
                "canonical_unchanged": canonical_unchanged,
                "dirty_differs": dirty_differs,
                "dirty_fingerprint": first.fingerprint,
            }
        )

pd.DataFrame(catalog_audit.rows).to_csv(
    ARTIFACTS / "a1_table_catalog_audit.csv",
    index=False,
    encoding="utf-8-sig",
)
pd.DataFrame(reproducibility_rows).to_csv(
    ARTIFACTS / "a1_corruption_reproducibility.csv",
    index=False,
    encoding="utf-8-sig",
)

summary = dict(catalog_audit.summary)
summary.update(
    {
        "corruption_profiles": len(profiles["profiles"]),
        "dirty_variants_checked": len(reproducibility_rows),
        "dirty_variants_reproducible": sum(
            row["deterministic"] for row in reproducibility_rows
        ),
        "canonical_invariants_passed": sum(
            row["canonical_unchanged"] for row in reproducibility_rows
        ),
        "engineering_gate_pass": all(
            row["deterministic"]
            and row["canonical_unchanged"]
            and row["dirty_differs"]
            for row in reproducibility_rows
        ),
        "status": "incomplete_selection" if not catalog_audit.summary["a1_complete"] else "complete",
    }
)
(ARTIFACTS / "a1_selection_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
