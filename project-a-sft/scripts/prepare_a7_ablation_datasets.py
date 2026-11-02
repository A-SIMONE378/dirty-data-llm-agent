"""Freeze deterministic A7 data-scale subsets from the A5 main_14b index."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a5_data import (  # noqa: E402
    selection_summary,
    validate_unique_fingerprints,
    write_jsonl,
)

A5_MANIFEST = ROOT / "artifacts" / "a5_data" / "manifest.json"
A5_MAIN_INDEX = ROOT / "artifacts" / "a5_data" / "main_14b.index.jsonl"
OUTPUT = ROOT / "artifacts" / "a7_data"
SCALES = (500, 1000, 2000)
SEED = "project-a-a7-scale-v1"


def _rank(entry: dict) -> str:
    payload = f"{SEED}|{entry['source_run']}|{entry['trajectory_id']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def balanced_subset(entries: list[dict], count: int) -> list[dict]:
    by_table: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_table[entry["table"]].append(entry)
    for table in by_table:
        by_table[table].sort(key=_rank)
    tables = sorted(by_table)
    selected: list[dict] = []
    offsets = {table: 0 for table in tables}
    while len(selected) < count:
        progressed = False
        for table in tables:
            offset = offsets[table]
            candidates = by_table[table]
            if offset < len(candidates):
                selected.append(candidates[offset])
                offsets[table] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"Not enough entries for a {count}-record subset.")
    return selected


def main() -> None:
    manifest = json.loads(A5_MANIFEST.read_text(encoding="utf-8"))
    entries = [
        json.loads(line)
        for line in A5_MAIN_INDEX.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) != 3133:
        raise ValueError("Unexpected main_14b index size.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selections = {
        f"scale_{count}": balanced_subset(entries, count) for count in SCALES
    }
    for name, selected in selections.items():
        validate_unique_fingerprints(selected)
        write_jsonl(OUTPUT / f"{name}.index.jsonl", selected)

    a7_manifest = {
        "schema_version": 1,
        "default_dataset": "scale_2000",
        "selection_key": ["source_run", "trajectory_id"],
        "deduplication_key": "trajectory_fingerprint",
        "source_files": manifest["source_files"],
        "datasets": {
            name: {
                "index_path": str(
                    (OUTPUT / f"{name}.index.jsonl").relative_to(ROOT)
                ).replace("\\", "/"),
                **selection_summary(selected),
            }
            for name, selected in selections.items()
        },
        "notes": {
            name: (
                f"Deterministic table-balanced subset of main_14b, "
                f"{len(selected)} records."
            )
            for name, selected in selections.items()
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(a7_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(a7_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
