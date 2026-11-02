import json
from pathlib import Path

import pandas as pd
import pytest

from dirty_da.a1_catalog import load_a1_catalog
from dirty_da.a1_corruption import load_corruption_profiles
from dirty_da.a2_audit import audit_a2_catalog
from dirty_da.a2_tasks import load_task_records, render_task_question
from dirty_da.executor import execute_task
from dirty_da.reference_clean import clean_table, rules_from_task_spec
from dirty_da.schema_semantics import load_table_semantics
from dirty_da.task_spec import TaskSpec, TaskSpecError
from tests.fixtures.a2_domain_fixtures import A2_DOMAIN_FIXTURES


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fixture", A2_DOMAIN_FIXTURES, ids=lambda item: item.name)
def test_handwritten_a2_domain_answers(fixture):
    frame = fixture.frame_factory()
    original = frame.copy(deep=True)
    assert execute_task(frame, fixture.spec) == fixture.expected_answer
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fixture", A2_DOMAIN_FIXTURES, ids=lambda item: item.name)
def test_task_cleaning_rules_match_reference_cleaner(fixture):
    frame = fixture.frame_factory()
    cleaned = clean_table(frame, rules_from_task_spec(fixture.spec))
    assert execute_task(cleaned, fixture.spec) == fixture.expected_answer


def test_renderer_is_exactly_spec_derived_and_does_not_append_answer():
    spec = A2_DOMAIN_FIXTURES[0].spec
    expected = (
        "Using the provided table, apply these rules: Exclude missing values from "
        "referenced aggregations and groups. Then Keep only rows where `region` "
        "equals \"N\". Return the sum of `amount`."
    )
    assert render_task_question(spec, "standard") == expected
    assert "40" not in expected
    assert render_task_question(spec, "standard") == render_task_question(
        spec, "standard"
    )


def test_new_task_fields_reject_ambiguous_or_ignored_values():
    columns = ["x", "group"]
    with pytest.raises(TaskSpecError, match="requires drop_duplicates"):
        TaskSpec(
            task_id="bad-dedup",
            operation="sum",
            target_column="x",
            dedup_subset=("group",),
        ).validate(columns)
    with pytest.raises(TaskSpecError, match="comparison_value"):
        TaskSpec(
            task_id="bad-compare",
            operation="aggregate_compare",
            target_column="x",
            aggregation="mean",
            comparison_operator="gt",
        ).validate(columns)
    with pytest.raises(TaskSpecError, match="Unsupported task spec fields"):
        TaskSpec.from_dict(
            {"task_id": "bad-extra", "operation": "count_rows", "hidden_rule": True}
        )


def test_committed_a2_catalog_passes_every_mechanical_assertion():
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    generation = json.loads(
        (ROOT / "configs" / "a2_task_generation.json").read_text(encoding="utf-8")
    )
    audit = audit_a2_catalog(
        ROOT / "data" / "tables",
        records,
        a1_catalog=load_a1_catalog(ROOT / "configs" / "a1_table_catalog.json"),
        semantics_registry=load_table_semantics(ROOT / "configs" / "table_semantics.json"),
        profile_config=load_corruption_profiles(
            ROOT / "configs" / "a1_corruption_profiles.json"
        ),
        generation_config=generation,
    )
    assert audit.summary["tasks_checked"] == 4360
    assert audit.summary["split_counts"] == {
        "dev": 320,
        "domain_generalization": 400,
        "test": 640,
        "train": 3000,
    }
    assert audit.summary["programmatic_answer_checks_passed"] == 4360
    assert audit.summary["dirty_replay_checks_passed"] == 4360
    assert audit.summary["question_spec_exact_render_checks_passed"] == 4360
    assert audit.summary["table_split_leakage_detected"] is False
    assert audit.summary["mechanical_gate_pass"] is True
