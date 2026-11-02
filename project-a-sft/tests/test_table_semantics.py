import json
from pathlib import Path

import pandas as pd

from dirty_da.schema_semantics import (
    load_table_semantics,
    numeric_measure_candidates,
    validate_table_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
SEMANTICS = ROOT / "configs" / "table_semantics.json"


def test_all_csv_columns_have_one_explicit_primary_role():
    registry = load_table_semantics(SEMANTICS)
    paths = sorted(TABLES.glob("*.csv"))
    assert len(paths) == 54
    assert set(registry["tables"]) == {path.name for path in paths}
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        roles = validate_table_semantics(frame, path.name, registry)
        assert len(roles) == len(frame.columns)


def test_generic_numeric_candidates_exclude_ids_targets_derived_and_coordinates():
    registry = load_table_semantics(SEMANTICS)
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        candidates = set(numeric_measure_candidates(frame, path.name, registry))
        spec = registry["tables"][path.name]
        excluded = set(spec["roles"]["identifier"])
        excluded.update(spec["roles"]["derived"])
        excluded.update(spec["roles"]["coordinate"])
        excluded.update(spec.get("targets", []))
        assert candidates.isdisjoint(excluded)
        assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in candidates)


def test_known_numeric_identifiers_and_derived_fields_are_never_generic_measures():
    registry = load_table_semantics(SEMANTICS)
    cases = {
        "customer_master.csv": {"customer_postal_code"},
        "tianchi_baby_trade_history.csv": {"user_id", "auction_id", "cat_id", "cat1"},
        "uci_bike_sharing.csv": {"total_rentals"},
        "zomato_restaurants.csv": {"Restaurant ID", "Country Code", "Longitude", "Latitude"},
    }
    for table_name, forbidden in cases.items():
        frame = pd.read_csv(TABLES / table_name, low_memory=False)
        candidates = set(numeric_measure_candidates(frame, table_name, registry))
        assert candidates.isdisjoint(forbidden)


def test_a2_reviewed_semantics_do_not_treat_ids_text_or_continuous_returns_as_categories():
    registry = load_table_semantics(SEMANTICS)["tables"]
    assert "SP" in registry["uci_247_istanbul_stock_exchange.csv"]["roles"]["measure"]
    assert registry["uci_247_istanbul_stock_exchange.csv"]["roles"]["category"] == []
    assert "oid" in registry["uci_536_pedestrians_traffic.csv"]["roles"]["identifier"]
    assert registry["uci_536_pedestrians_traffic.csv"]["roles"]["category"] == []
    assert "ModelName" in registry["uci_029_computer_hardware.csv"]["roles"]["identifier"]
    assert "period" in registry["uci_390_stock_portfolio.csv"]["roles"]["category"]
    assert {
        "benefitsReview",
        "sideEffectsReview",
        "commentsReview",
    } <= set(registry["uci_461_drug_reviews.csv"]["roles"]["opaque"])


def test_semantics_registry_is_json_serializable_without_runtime_objects():
    registry = load_table_semantics(SEMANTICS)
    json.dumps(registry, ensure_ascii=False)
