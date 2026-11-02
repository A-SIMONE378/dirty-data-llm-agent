import pandas as pd
import pytest
import json
from dirty_da.canonical import CanonicalTable
from dirty_da.corruption import apply_corruption, apply_corruptions

@pytest.mark.parametrize("kind,col,expected", [
    ("missing_values", "n", True),
    ("sentinel_values", "n", True),
    ("type_corruption", "n", True),
    ("format_inconsistency", "city", True),
    ("category_inconsistency", "city", True),
])
def test_cell_corruptions_are_deterministic_and_do_not_mutate_canonical(kind, col, expected):
    canonical = CanonicalTable(pd.DataFrame({"n": [10, 20], "city": ["Paris", "Rome"]}))
    fp = canonical.fingerprint
    a = apply_corruption(canonical, kind, column=col, row=0, seed=7)
    b = apply_corruption(canonical, kind, column=col, row=0, seed=7)
    assert a.dirty.equals(b.dirty)
    assert a.metadata == b.metadata
    assert canonical.fingerprint == fp

def test_duplicate_rows_appends_exact_copy():
    canonical = CanonicalTable(pd.DataFrame({"x": [1, 2]}))
    result = apply_corruption(canonical, "duplicate_rows", row=1, seed=3)
    assert len(result.dirty) == 3
    assert result.dirty.iloc[1].equals(result.dirty.iloc[2])
    assert result.metadata[0]["operation"] == "append_exact_duplicate"

def test_invalid_applicability_fails_loudly():
    canonical = CanonicalTable(pd.DataFrame({"n": [1, 2]}))
    with pytest.raises(ValueError):
        apply_corruption(canonical, "format_inconsistency", column="n", row=0)

def test_unknown_type_fails():
    canonical = CanonicalTable(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        apply_corruption(canonical, "unknown")

def test_numeric_default_sentinel_and_metadata_are_truthful_json():
    canonical = CanonicalTable(pd.DataFrame({"n": [10], "city": ["Paris"]}, index=[42]))
    result = apply_corruption(canonical, "sentinel_values", column="n", row=0, seed=17)
    assert result.dirty.iloc[0]["n"] == -999999
    event = result.metadata[0]
    assert event["row_position"] == 0
    assert event["row_index"] == 42
    assert event["before"] == 10
    assert event["after"] == -999999
    json.dumps(result.metadata)

def test_invalid_duplicate_row_is_rejected_instead_of_clamped():
    canonical = CanonicalTable(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(ValueError, match="Row out of range"):
        apply_corruption(canonical, "duplicate_rows", row=99)
    with pytest.raises(ValueError, match="Row out of range"):
        apply_corruption(canonical, "duplicate_rows", row=-1)

def test_compound_corruption_same_row_different_columns_records_order():
    canonical = CanonicalTable(pd.DataFrame({"amount": [10], "city": ["Paris"]}))
    result = apply_corruptions(canonical, [
        {"corruption_type": "missing_values", "column": "amount", "row": 0, "seed": 1},
        {"corruption_type": "format_inconsistency", "column": "city", "row": 0, "seed": 2},
    ])
    assert pd.isna(result.dirty.iloc[0]["amount"])
    assert result.dirty.iloc[0]["city"] == "  PARIS  "
    assert [event["event_id"] for event in result.metadata] == [0, 1]
    assert [event["seed"] for event in result.metadata] == [1, 2]

def test_compound_same_cell_incompatible_injection_fails_loudly():
    canonical = CanonicalTable(pd.DataFrame({"amount": [10]}))
    with pytest.raises(ValueError, match="missing"):
        apply_corruptions(canonical, [
            {"corruption_type": "missing_values", "column": "amount", "row": 0},
            {"corruption_type": "type_corruption", "column": "amount", "row": 0},
        ])

def test_compound_order_changes_result_and_metadata_order():
    canonical = CanonicalTable(pd.DataFrame({"city": ["Paris"]}))
    format_then_category = apply_corruptions(canonical, [
        {"corruption_type": "format_inconsistency", "column": "city", "row": 0},
        {"corruption_type": "category_inconsistency", "column": "city", "row": 0},
    ])
    category_then_format = apply_corruptions(canonical, [
        {"corruption_type": "category_inconsistency", "column": "city", "row": 0},
        {"corruption_type": "format_inconsistency", "column": "city", "row": 0},
    ])
    assert format_then_category.dirty.iloc[0]["city"] == "paris "
    assert category_then_format.dirty.iloc[0]["city"] == "  PARIS  "
    assert [event["corruption_type"] for event in format_then_category.metadata] == [
        "format_inconsistency", "category_inconsistency"
    ]
