import pandas as pd
import pytest
from dirty_da.task_spec import TaskSpec, TaskSpecError
from dirty_da.executor import execute_task, ExecutionError
from dirty_da.verifier import verify_result

def test_hand_calculated_sum_fixture():
    # Independent handwritten answer: 10 + 20 + 30 = 60.
    df = pd.DataFrame({"amount": [10, 20, 30], "region": ["N", "S", "N"]})
    spec = TaskSpec(task_id="fixture-sum-1", operation="sum", target_column="amount")
    assert execute_task(df, spec) == 60.0
    assert verify_result(60.0, execute_task(df, spec)).passed

def test_hand_calculated_groupby_fixture():
    # Independent handwritten answer: N -> (10+30)/2=20, S -> 20.
    df = pd.DataFrame({"amount": [10, 20, 30], "region": ["N", "S", "N"]})
    spec = TaskSpec(task_id="fixture-group-1", operation="groupby_agg", target_column="amount",
                    group_by="region", aggregation="mean")
    actual = execute_task(df, spec)
    assert actual == {"N": 20.0, "S": 20.0}
    assert verify_result({"N": 20.0, "S": 20.0}, actual).passed

def test_wrong_answer_rejected():
    assert not verify_result(60.0, 61.0).passed
    assert not verify_result({"N": 20.0}, {"N": 21.0}).passed

def test_invalid_task_rejected():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(TaskSpecError):
        execute_task(df, TaskSpec(task_id="bad", operation="execute_arbitrary_code"))
    with pytest.raises(TaskSpecError):
        execute_task(df, TaskSpec(task_id="bad2", operation="mean", target_column="missing"))

def test_missing_policy_error():
    df = pd.DataFrame({"x": [1.0, float("nan")]})
    spec = TaskSpec(task_id="missing-error", operation="mean", target_column="x", missing_policy="error")
    with pytest.raises(ExecutionError):
        execute_task(df, spec)

def test_missing_policy_error_ignores_unreferenced_columns():
    df = pd.DataFrame({"amount": [1.0, 3.0], "unused": [None, "x"]})
    spec = TaskSpec(
        task_id="referenced-only",
        operation="mean",
        target_column="amount",
        missing_policy="error",
    )
    assert execute_task(df, spec) == 2.0

def test_all_missing_sum_is_missing_not_false_zero():
    df = pd.DataFrame({"amount": [float("nan"), float("nan")]})
    spec = TaskSpec(task_id="missing-sum", operation="sum", target_column="amount")
    assert execute_task(df, spec) is None

def test_groupby_excludes_missing_keys_and_preserves_missing_group_sum():
    df = pd.DataFrame({
        "region": ["N", None, "S"],
        "amount": [10.0, 99.0, float("nan")],
    })
    spec = TaskSpec(
        task_id="group-missing",
        operation="groupby_agg",
        target_column="amount",
        group_by="region",
        aggregation="sum",
    )
    assert execute_task(df, spec) == {"N": 10.0, "S": None}

def test_ne_filter_does_not_include_missing_operands():
    df = pd.DataFrame({"region": ["N", "S", None], "amount": [10, 20, 100]})
    spec = TaskSpec(
        task_id="filter-ne",
        operation="sum",
        target_column="amount",
        filters=({"column": "region", "operator": "ne", "value": "N"},),
    )
    assert execute_task(df, spec) == 20.0

def test_hand_calculated_dedup_sentinel_missing_fixture():
    # Independent answer: exact duplicate 10 is removed; -999 and None are missing => 10.
    df = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "amount": [10.0, 10.0, -999.0, float("nan")],
    })
    spec = TaskSpec(
        task_id="combined-policy",
        operation="sum",
        target_column="amount",
        drop_duplicates=True,
        sentinel_policy="map",
        sentinel_map=({"column": "amount", "values": (-999.0,)},),
    )
    original = df.copy(deep=True)
    assert execute_task(df, spec) == 10.0
    pd.testing.assert_frame_equal(df, original)

def test_task_spec_rejects_ignored_or_ambiguous_fields():
    columns = ["x", "region"]
    with pytest.raises(TaskSpecError, match="does not accept target_column"):
        TaskSpec(task_id="bad-count", operation="count_rows", target_column="x").validate(columns)
    with pytest.raises(TaskSpecError, match="does not accept group_by"):
        TaskSpec(task_id="bad-group", operation="mean", target_column="x", group_by="region").validate(columns)
    with pytest.raises(TaskSpecError, match="only valid"):
        TaskSpec(
            task_id="bad-sentinel",
            operation="mean",
            target_column="x",
            sentinel_map=({"column": "x", "values": (-1,)},),
        ).validate(columns)
    with pytest.raises(TaskSpecError, match="non-string sequence"):
        TaskSpec(
            task_id="bad-in",
            operation="count_rows",
            filters=({"column": "region", "operator": "in", "value": "NS"},),
        ).validate(columns)

def test_verifier_enforces_types_tolerance_and_missing_representation():
    assert not verify_result(1, 1.0).passed
    assert not verify_result(1.0, 1).passed
    assert verify_result(1.0, 1.0 + 1e-7, atol=1e-6, rtol=0).passed
    assert not verify_result(float("nan"), float("nan")).passed
    assert verify_result(None, None).passed
    with pytest.raises(ValueError, match="non-negative"):
        verify_result(1.0, 1.0, atol=-1)
