import copy
from pathlib import Path

import pandas as pd
import pytest

from dirty_da.a1_catalog import A1CatalogError, audit_a1_catalog, load_a1_catalog
from dirty_da.a1_corruption import (
    A1CorruptionPlanError,
    generate_dirty_copy,
    load_corruption_profiles,
)
from dirty_da.canonical import CanonicalTable
from dirty_da.schema_semantics import load_table_semantics


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def catalog():
    return load_a1_catalog(CONFIGS / "a1_table_catalog.json")


@pytest.fixture(scope="module")
def semantics():
    return load_table_semantics(CONFIGS / "table_semantics.json")


@pytest.fixture(scope="module")
def profiles():
    return load_corruption_profiles(CONFIGS / "a1_corruption_profiles.json")


def test_final_catalog_meets_a1_table_and_split_gate(catalog, semantics):
    audit = audit_a1_catalog(TABLES, catalog, semantics)
    assert audit.summary["tables_cataloged"] == 54
    assert audit.summary["tables_selected"] == 45
    assert audit.summary["tables_quarantined"] == 9
    assert audit.summary["core_tables"] == 40
    assert audit.summary["domain_generalization_tables"] == 5
    assert audit.summary["target_total_range"] == [45, 68]
    assert audit.summary["tables_structurally_eligible"] == 52
    assert audit.summary["tables_research_approved"] == 45
    assert audit.summary["tables_admitted"] == 45
    assert audit.summary["split_counts"] == {
        "dev": 4,
        "domain_generalization": 5,
        "test": 8,
        "train": 28,
    }
    assert audit.summary["a1_complete"] is True
    assert audit.summary["blocking_reasons"] == []


def test_semantically_ineligible_tables_are_quarantined_and_replaced(catalog):
    for name in (
        "uci_247_istanbul_stock_exchange.csv",
        "uci_536_pedestrians_traffic.csv",
    ):
        record = catalog["tables"][name]
        assert record["selection_status"] == "quarantine"
        assert record["split"] is None
        assert record["approved_for_research"] is False
    for name in ("uci_159_magic_gamma.csv", "uci_222_bank_marketing.csv"):
        record = catalog["tables"][name]
        assert record["selection_status"] == "selected"
        assert record["split"] == "train"
        assert record["approved_for_research"] is True


def test_only_selected_verified_license_records_are_research_approved(catalog):
    approved = {
        name
        for name, record in catalog["tables"].items()
        if record["approved_for_research"]
    }
    assert len(approved) == 45
    assert {
        "uci_adult.csv",
        "uci_bike_sharing.csv",
        "uci_seoul_bike.csv",
        "uci_wine_quality.csv",
    } <= approved
    selected = {
        name
        for name, record in catalog["tables"].items()
        if record.get("selection_status", "selected") == "selected"
    }
    assert approved == selected
    for name in approved:
        record = catalog["tables"][name]
        assert record["license"]["status"] == "verified"
        assert record["provenance"]["source_url"].startswith("https://archive.ics.uci.edu/")


def test_unknown_license_cannot_be_marked_approved(catalog, semantics):
    invalid = copy.deepcopy(catalog)
    invalid["tables"]["customer_master.csv"]["approved_for_research"] = True
    with pytest.raises(A1CatalogError, match="verified license"):
        audit_a1_catalog(TABLES, invalid, semantics)


def test_catalog_inventory_mismatch_fails(catalog, semantics):
    invalid = copy.deepcopy(catalog)
    invalid["tables"].pop("customer_master.csv")
    with pytest.raises(A1CatalogError, match="does not match CSV inventory"):
        audit_a1_catalog(TABLES, invalid, semantics)


def test_all_profiled_dirty_copies_are_reproducible(catalog, semantics, profiles):
    del catalog  # The exact catalog is validated separately; this test exercises every CSV.
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        canonical = CanonicalTable(frame, name=path.stem)
        fingerprint = canonical.fingerprint
        for profile_name in profiles["profiles"]:
            first = generate_dirty_copy(
                canonical, path.name, semantics, profiles, profile_name
            )
            second = generate_dirty_copy(
                canonical, path.name, semantics, profiles, profile_name
            )
            assert first.fingerprint == second.fingerprint
            assert first.result.metadata == second.result.metadata
            assert first.fingerprint != fingerprint
            assert canonical.fingerprint == fingerprint


def test_high_profile_records_known_inapplicability(semantics, profiles):
    expected = {
        "tianchi_baby_trade_history.csv": {
            "format_inconsistency",
            "category_inconsistency",
        },
        "uci_bike_sharing.csv": {"category_inconsistency"},
        "uci_015_breast_cancer_wisconsin.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_030_contraceptive_method.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_042_glass_identification.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_045_heart_disease.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_060_liver_disorders.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_143_australian_credit.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_145_statlog_heart.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_161_mammographic_mass.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_247_istanbul_stock_exchange.csv": {"category_inconsistency"},
        "uci_292_wholesale_customers.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_519_heart_failure.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_536_pedestrians_traffic.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_563_iranian_churn.csv": {
            "format_inconsistency", "category_inconsistency"
        },
        "uci_713_auction_verification.csv": {
            "format_inconsistency", "category_inconsistency"
        },
    }
    observed = {}
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        result = generate_dirty_copy(
            CanonicalTable(frame), path.name, semantics, profiles, "high"
        )
        if result.inapplicable:
            observed[path.name] = set(result.inapplicable)
    assert observed == expected


def test_unexpected_inapplicability_is_rejected(semantics, profiles):
    strict_profiles = copy.deepcopy(profiles)
    strict_profiles["profiles"]["high"]["allow_inapplicable"] = []
    path = TABLES / "tianchi_baby_trade_history.csv"
    frame = pd.read_csv(path, low_memory=False)
    with pytest.raises(A1CorruptionPlanError, match="unexpectedly inapplicable"):
        generate_dirty_copy(
            CanonicalTable(frame), path.name, semantics, strict_profiles, "high"
        )
