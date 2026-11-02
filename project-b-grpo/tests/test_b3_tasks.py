from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.a4_teacher import select_teacher_tasks  # noqa: E402


def _config():
    return json.loads((ROOT / "configs" / "b3_tasks.json").read_text(encoding="utf-8"))


def test_b3_selection_is_train_only_unique_and_table_balanced():
    config = _config()
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    selected = select_teacher_tasks(
        records,
        {
            "schema_version": 1,
            "split": "train",
            "selected_task_count": int(config["task_count"]),
            "selection_seed": config["selection_seed"],
        },
    )
    assert len(selected) == config["task_count"] == 800
    assert {record["split"] for record in selected} == {"train"}
    assert len({record["task_id"] for record in selected}) == 800
    from collections import Counter

    table_counts = Counter(record["table"] for record in selected)
    assert len(table_counts) == 28
    assert max(table_counts.values()) - min(table_counts.values()) <= 1


def test_b3_manifest_is_explicit_about_pass_rate_gate_and_curriculum():
    config = _config()
    assert config["samples_per_task"] == 8
    assert 0.0 < config["pass_rate_min"] < config["pass_rate_max"] < 1.0
    assert config["curriculum"] == [
        "single_step",
        "grouped",
        "cleaning_required",
        "judgment",
    ]
