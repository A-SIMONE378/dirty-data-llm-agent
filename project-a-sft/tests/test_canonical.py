import pandas as pd
import pytest
from dirty_da.canonical import CanonicalTable, CanonicalDataError, dataframe_fingerprint

def test_input_mutation_does_not_change_snapshot():
    source = pd.DataFrame({"x": [1, 2]})
    canonical = CanonicalTable(source)
    fp = canonical.fingerprint
    source.loc[0, "x"] = 99
    assert canonical.fingerprint == fp
    assert canonical.copy_frame().loc[0, "x"] == 1

def test_returned_copy_mutation_does_not_change_snapshot():
    canonical = CanonicalTable(pd.DataFrame({"x": [1, 2]}))
    fp = canonical.fingerprint
    copy = canonical.copy_frame()
    copy.loc[0, "x"] = 99
    assert canonical.fingerprint == fp

def test_schema_and_index_affect_fingerprint():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [1, 2]}).astype("float64")
    c = pd.DataFrame({"x": [1, 2]}, index=[5, 6])
    assert len({dataframe_fingerprint(x) for x in [a, b, c]}) == 3

def test_index_name_and_column_index_name_affect_fingerprint():
    base = pd.DataFrame({"x": [1, 2]})
    named_index = base.copy()
    named_index.index.name = "row_id"
    named_columns = base.copy()
    named_columns.columns.name = "fields"
    assert len({dataframe_fingerprint(x) for x in [base, named_index, named_columns]}) == 3

def test_unsupported_nested_values_and_duplicate_columns_fail_loudly():
    with pytest.raises(CanonicalDataError, match="Unsupported mutable or opaque"):
        CanonicalTable(pd.DataFrame({"x": [[1, 2]]}))
    duplicate_columns = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(CanonicalDataError, match="Duplicate column labels"):
        CanonicalTable(duplicate_columns)

def test_fingerprint_changes_if_private_snapshot_is_tampered(monkeypatch):
    canonical = CanonicalTable(pd.DataFrame({"x": [1, 2]}))
    canonical._snapshot.loc[0, "x"] = 9
    with pytest.raises(CanonicalDataError):
        canonical.assert_intact()
