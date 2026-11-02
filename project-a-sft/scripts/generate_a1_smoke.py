"""Generate a six-table deterministic A1 smoke dataset without model access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dirty_da.catalog import TableCatalog, split_tables
from dirty_da.corruption import CorruptionRequest
from dirty_da.dataset import build_dataset
from dirty_da.reference_clean import CleanerSpec
from dirty_da.task_spec import TaskSpec
from dirty_da.trajectory import TrajectoryScenario


def _catalog() -> TableCatalog:
    base = {
        "id": [1, 2],
        "region": ["east", "west"],
        "amount": [10, 20],
        "note": ["a", "b"],
    }
    frames = {
        "smoke_missing": pd.DataFrame(base),
        "smoke_sentinel": pd.DataFrame(base),
        "smoke_duplicate": pd.DataFrame(base),
        "smoke_type": pd.DataFrame(base),
        "smoke_format": pd.DataFrame({**base, "amount": [1200, 20]}),
        "smoke_category": pd.DataFrame(base),
    }
    return TableCatalog.from_frames(frames, {table_id: "smoke-v1" for table_id in frames})


def _sum(task_id: str) -> TaskSpec:
    return TaskSpec(task_id, "sum", target_column="amount")


def _scenarios() -> tuple[TrajectoryScenario, ...]:
    return (
        TrajectoryScenario(
            "smoke_missing",
            "smoke_missing",
            _sum("sum_missing"),
            (CorruptionRequest("missing", row=0, column="note"),),
            CleanerSpec(),
            seed=201,
        ),
        TrajectoryScenario(
            "smoke_sentinel",
            "smoke_sentinel",
            _sum("sum_sentinel"),
            (CorruptionRequest("sentinel", row=0, column="amount", value=-999),),
            CleanerSpec(
                sentinel_policy="replace",
                sentinel_values={"amount": (-999,)},
                sentinel_replacements={"amount": 10},
            ),
            seed=202,
        ),
        TrajectoryScenario(
            "smoke_duplicate",
            "smoke_duplicate",
            _sum("sum_duplicate"),
            (CorruptionRequest("duplicate_rows", row=0),),
            CleanerSpec(dedup_subset=("id", "region", "amount", "note")),
            seed=203,
        ),
        TrajectoryScenario(
            "smoke_type",
            "smoke_type",
            _sum("sum_type"),
            (CorruptionRequest("type_corruption", row=0, column="amount"),),
            CleanerSpec(type_coercions={"amount": "int"}),
            seed=204,
            prompt_variant=1,
        ),
        TrajectoryScenario(
            "smoke_format",
            "smoke_format",
            _sum("sum_format"),
            (CorruptionRequest("format_inconsistency", row=0, column="amount"),),
            CleanerSpec(
                format_rules={"amount": ("remove_commas",)},
                type_coercions={"amount": "int"},
            ),
            seed=205,
            prompt_variant=1,
        ),
        TrajectoryScenario(
            "smoke_category",
            "smoke_category",
            TaskSpec(
                "group_category",
                "group_sum",
                target_column="amount",
                group_by=("region",),
            ),
            (
                CorruptionRequest(
                    "category_inconsistency", row=0, column="region", value="EAST"
                ),
            ),
            CleanerSpec(category_mappings={"region": {"EAST": "east"}}),
            seed=206,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/a1_smoke"))
    args = parser.parse_args()

    catalog = _catalog()
    split = split_tables(catalog.manifest, seed=2026)
    bundle = build_dataset(catalog, split, _scenarios())
    args.output.mkdir(parents=True, exist_ok=True)
    catalog.manifest.write_json(args.output / "table_manifest.json")
    split.write_json(args.output / "split.json")
    paths = bundle.write_jsonl(args.output)
    summary = {
        "output": str(args.output.resolve()),
        "table_count": len(catalog.manifest.entries),
        "trajectory_count": sum(len(records) for records in bundle.records.values()),
        "split_counts": {name: len(bundle.records[name]) for name in bundle.records},
        "files": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
