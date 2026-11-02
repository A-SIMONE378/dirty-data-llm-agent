"""Freeze B4 dirty tables under the A4/B3 pandas environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dirty_da.a1_corruption import generate_dirty_copy, load_corruption_profiles  # noqa: E402
from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.canonical import CanonicalTable  # noqa: E402
from dirty_da.schema_semantics import load_table_semantics  # noqa: E402


def main() -> None:
    filtered_path = ROOT / "artifacts" / "b3_rollouts" / "filtered_tasks.json"
    rows = json.loads(filtered_path.read_text(encoding="utf-8"))
    records = {
        record["task_id"]: record
        for record in load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    }
    semantics = load_table_semantics(ROOT / "configs" / "table_semantics.json")
    profiles = load_corruption_profiles(ROOT / "configs" / "a1_corruption_profiles.json")
    cache = ROOT / "artifacts" / "b4_dirty_cache"
    cache.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        task_id = row["task_id"]
        record = records[task_id]
        frame = pd.read_csv(ROOT / "data" / "tables" / record["table"], low_memory=False)
        dirty = generate_dirty_copy(
            CanonicalTable(frame, name=Path(record["table"]).stem),
            record["table"],
            semantics,
            profiles,
            record["dirty_profile"],
        )
        dirty.result.dirty.to_csv(cache / f"{task_id}.csv", index=False)
        written += 1
    manifest = {
        "schema_version": 1,
        "source_filtered_tasks": str(filtered_path.relative_to(ROOT)).replace("\\", "/"),
        "source_filtered_sha256": hashlib.sha256(filtered_path.read_bytes()).hexdigest(),
        "task_count": written,
        "cache_files": written,
    }
    (cache / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
