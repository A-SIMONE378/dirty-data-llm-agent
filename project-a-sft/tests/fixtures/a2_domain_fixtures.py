"""Independent handwritten A2 fixtures.

Expected answers are literal values calculated by hand. They are never produced
by the executor or task generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from dirty_da.task_spec import TaskSpec


@dataclass(frozen=True)
class A2DomainFixture:
    name: str
    domain: str
    invariant: str
    frame_factory: Callable[[], pd.DataFrame]
    spec: TaskSpec
    expected_answer: Any


A2_DOMAIN_FIXTURES = (
    A2DomainFixture(
        name="commerce_filtered_sum",
        domain="commerce",
        invariant="A category filter is applied before a numeric sum; 10 + 30 = 40.",
        frame_factory=lambda: pd.DataFrame(
            {"amount": [10, 20, 30], "region": ["N", "S", "N"]}
        ),
        spec=TaskSpec(
            task_id="fixture-commerce-sum",
            operation="sum",
            target_column="amount",
            filters=({"column": "region", "operator": "eq", "value": "N"},),
        ),
        expected_answer=40.0,
    ),
    A2DomainFixture(
        name="health_grouped_mean",
        domain="health",
        invariant="Grouped means are keyed by category: A=(20+60)/2=40, B=40.",
        frame_factory=lambda: pd.DataFrame(
            {"age": [20, 40, 60], "outcome": ["A", "B", "A"]}
        ),
        spec=TaskSpec(
            task_id="fixture-health-group",
            operation="groupby_agg",
            target_column="age",
            group_by="outcome",
            aggregation="mean",
        ),
        expected_answer={"A": 40.0, "B": 40.0},
    ),
    A2DomainFixture(
        name="transport_type_and_duplicate_cleaning",
        domain="transport",
        invariant=(
            "An exact duplicate is removed, numeric text is parsed, and the "
            "unparseable token is excluded; the sum is 5."
        ),
        frame_factory=lambda: pd.DataFrame(
            {"route": ["A", "A", "B"], "delay": ["5", "5", "bad"]}
        ),
        spec=TaskSpec(
            task_id="fixture-transport-clean",
            operation="sum",
            target_column="delay",
            drop_duplicates=True,
            numeric_coerce=("delay",),
        ),
        expected_answer=5.0,
    ),
    A2DomainFixture(
        name="education_sentinel_mean",
        domain="education",
        invariant="The declared -999 sentinel is missing, so (80 + 90) / 2 = 85.",
        frame_factory=lambda: pd.DataFrame(
            {"score": [80, -999, 90], "class": ["X", "X", "Y"]}
        ),
        spec=TaskSpec(
            task_id="fixture-education-sentinel",
            operation="mean",
            target_column="score",
            sentinel_policy="map",
            sentinel_map=({"column": "score", "values": (-999,)},),
            numeric_coerce=("score",),
        ),
        expected_answer=85.0,
    ),
    A2DomainFixture(
        name="regional_category_normalization",
        domain="public_policy",
        invariant="Whitespace and an explicit alias map unify two north rows; 1 + 3 = 4.",
        frame_factory=lambda: pd.DataFrame(
            {"region": [" north ", "S", "north "], "value": [1, 2, 3]}
        ),
        spec=TaskSpec(
            task_id="fixture-policy-category",
            operation="sum",
            target_column="value",
            filters=({"column": "region", "operator": "eq", "value": "N"},),
            strip_columns=("region",),
            category_map=({"column": "region", "mapping": {"north": "N"}},),
        ),
        expected_answer=4.0,
    ),
    A2DomainFixture(
        name="energy_threshold_true",
        domain="energy",
        invariant="The mean is 20, so the explicit greater-than-15 judgment is true.",
        frame_factory=lambda: pd.DataFrame({"energy": [10, 20, 30]}),
        spec=TaskSpec(
            task_id="fixture-energy-judgment-true",
            operation="aggregate_compare",
            target_column="energy",
            aggregation="mean",
            comparison_operator="gt",
            comparison_value=15.0,
        ),
        expected_answer=True,
    ),
    A2DomainFixture(
        name="mobility_threshold_false",
        domain="mobility",
        invariant="Bus durations are 4 and 8; max=8 is not at most 7.",
        frame_factory=lambda: pd.DataFrame(
            {"duration": [4, 6, 8], "mode": ["bus", "rail", "bus"]}
        ),
        spec=TaskSpec(
            task_id="fixture-mobility-judgment-false",
            operation="aggregate_compare",
            target_column="duration",
            aggregation="max",
            comparison_operator="le",
            comparison_value=7.0,
            filters=({"column": "mode", "operator": "eq", "value": "bus"},),
        ),
        expected_answer=False,
    ),
    A2DomainFixture(
        name="survey_keyed_deduplication",
        domain="survey",
        invariant="Keyed dedup keeps the first id=1 row, so 10 + 5 = 15.",
        frame_factory=lambda: pd.DataFrame(
            {"respondent_id": [1, 1, 2], "amount": [10, 12, 5]}
        ),
        spec=TaskSpec(
            task_id="fixture-survey-dedup-key",
            operation="sum",
            target_column="amount",
            drop_duplicates=True,
            dedup_subset=("respondent_id",),
        ),
        expected_answer=15.0,
    ),
)
