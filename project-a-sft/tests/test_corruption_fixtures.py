import pandas as pd
import pytest
from dirty_da.canonical import CanonicalTable
from dirty_da.corruption import apply_corruption

@pytest.mark.parametrize("column,value,expected", [
    ("x", 1, pd.NA),
    ("x", 2, pd.NA),
])
def test_missing_value_fixture(column, value, expected):
    canonical = CanonicalTable(pd.DataFrame({"x": [1, 2]}))
    result = apply_corruption(canonical, "missing_values", column=column, row=value-1, seed=1)
    assert pd.isna(result.dirty.iloc[value-1]["x"])
    assert result.metadata[0]["before"] == value
    assert result.metadata[0]["after"] is None

@pytest.mark.parametrize("row,sentinel", [(0, -999), (1, -1)])
def test_sentinel_value_fixture(row, sentinel):
    canonical = CanonicalTable(pd.DataFrame({"x": [5, 8]}))
    result = apply_corruption(canonical, "sentinel_values", column="x", row=row, value=sentinel, seed=2)
    assert result.dirty.iloc[row]["x"] == sentinel
    assert result.metadata[0]["after"] == sentinel

@pytest.mark.parametrize("row", [0, 1])
def test_duplicate_row_fixture(row):
    canonical = CanonicalTable(pd.DataFrame({"x": [4, 9]}))
    result = apply_corruption(canonical, "duplicate_rows", row=row, seed=3)
    assert len(result.dirty) == 3
    assert result.dirty.iloc[row].to_dict() == result.dirty.iloc[-1].to_dict()

@pytest.mark.parametrize("row", [0, 1])
def test_type_corruption_fixture(row):
    canonical = CanonicalTable(pd.DataFrame({"x": [10, 20]}))
    result = apply_corruption(canonical, "type_corruption", column="x", row=row, seed=4)
    assert isinstance(result.dirty.iloc[row]["x"], str)
    assert result.metadata[0]["before"] == [10, 20][row]

@pytest.mark.parametrize("row,expected", [(0, "  PARIS  "), (1, "  ROME  ")])
def test_format_corruption_fixture(row, expected):
    canonical = CanonicalTable(pd.DataFrame({"city": ["Paris", "Rome"]}))
    result = apply_corruption(canonical, "format_inconsistency", column="city", row=row, seed=5)
    assert result.dirty.iloc[row]["city"] == expected

@pytest.mark.parametrize("row,expected", [(0, "paris "), (1, "rome ")])
def test_category_corruption_fixture(row, expected):
    canonical = CanonicalTable(pd.DataFrame({"city": ["Paris", "Rome"]}))
    result = apply_corruption(canonical, "category_inconsistency", column="city", row=row, seed=6)
    assert result.dirty.iloc[row]["city"] == expected

def test_task_spec_sentinel_map_is_applied():
    from dirty_da.task_spec import TaskSpec
    from dirty_da.executor import execute_task
    df = pd.DataFrame({"amount": [10, -999, 20]})
    spec = TaskSpec(task_id="sentinel-fixture", operation="mean", target_column="amount",
                    sentinel_policy="map", sentinel_map=({"column": "amount", "values": (-999,)},))
    assert execute_task(df, spec) == 15.0
