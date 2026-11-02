"""Prepare the frozen B3 training-task pool and curriculum metadata."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.a4_teacher import select_teacher_tasks  # noqa: E402


SOURCE = ROOT / "data" / "tasks" / "a2_tasks.jsonl"
CONFIG = ROOT / "configs" / "b3_tasks.json"
OUTPUT = ROOT / "artifacts" / "b3_tasks"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    records = load_task_records(SOURCE)
    selection_config = {
        "schema_version": 1,
        "split": config["split"],
        "selected_task_count": int(config["task_count"]),
        "selection_seed": config["selection_seed"],
    }
    selected = select_teacher_tasks(records, selection_config)
    if {record["split"] for record in selected} != {"train"}:
        raise SystemExit("B3 task pool must be train-only.")
    if len(selected) != int(config["task_count"]):
        raise SystemExit("B3 task selection count mismatch.")
    if len({record["task_id"] for record in selected}) != len(selected):
        raise SystemExit("B3 task selection contains duplicate task IDs.")

    index = [
        {
            "task_id": record["task_id"],
            "table": record["table"],
            "domain": record.get("domain"),
            "difficulty": record["difficulty"],
            "split": record["split"],
            "render_style": record.get("render_style"),
            "question": record["question"],
        }
        for record in selected
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "selection_seed": config["selection_seed"],
        "split": config["split"],
        "task_count": len(index),
        "samples_per_task": int(config["samples_per_task"]),
        "pass_rate_min": float(config["pass_rate_min"]),
        "pass_rate_max": float(config["pass_rate_max"]),
        "curriculum": list(config["curriculum"]),
        "table_counts": dict(sorted(Counter(row["table"] for row in index).items())),
        "difficulty_counts": dict(
            sorted(Counter(row["difficulty"] for row in index).items())
        ),
        "index_path": str((OUTPUT / "index.json").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
