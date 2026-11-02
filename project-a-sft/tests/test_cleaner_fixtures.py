import pandas as pd
import pytest
from dirty_da.reference_clean import clean_table
from dirty_da.canonical import CanonicalTable
from dirty_da.corruption import apply_corruption
from .fixtures.cleaner_fixtures import fixtures

def test_handwritten_cleaner_fixtures():
    for fixture in fixtures():
        actual = clean_table(fixture["input"], fixture["rules"])
        expected = fixture["expected"]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

def test_cleaner_does_not_mutate_input_and_parses_declared_datetime_format():
    source = pd.DataFrame({"date": ["01/31/2024", "02/01/2024"], "value": ["1", "bad"]})
    original = source.copy(deep=True)
    actual = clean_table(source, {
        "datetime_parse": {"date": "%m/%d/%Y"},
        "numeric_coerce": ["value"],
    })
    pd.testing.assert_frame_equal(source, original)
    assert actual["date"].tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-01")]
    assert actual.loc[0, "value"] == 1.0
    assert pd.isna(actual.loc[1, "value"])

def test_cleaner_rejects_ambiguous_rule_shapes():
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="non-string sequence"):
        clean_table(frame, {"sentinel_map": {"x": "NA"}})
    with pytest.raises(ValueError, match="sequence of column names"):
        clean_table(frame, {"numeric_coerce": "x"})

def test_datetime_format_corruption_is_real_and_explicitly_reversible():
    original = pd.DataFrame({"date": pd.to_datetime(["2024-01-31", "2024-02-01"])})
    dirty = apply_corruption(
        CanonicalTable(original),
        "format_inconsistency",
        column="date",
        row=0,
    ).dirty
    assert dirty.iloc[0]["date"] == "01/31/2024"
    cleaned = clean_table(dirty, {"datetime_parse": {"date": "%m/%d/%Y"}})
    pd.testing.assert_frame_equal(cleaned, original)
