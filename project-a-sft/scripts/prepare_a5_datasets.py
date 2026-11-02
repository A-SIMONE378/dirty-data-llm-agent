"""Freeze A5 dataset indices without duplicating the large A4 JSONL files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a5_data import (  # noqa: E402
    load_sft_records,
    selection_entry,
    selection_summary,
    sha256_file,
    write_jsonl,
)


OUTPUT_DIR = ROOT / "artifacts" / "a5_data"
SOURCE_RUNS = {
    "qwen3_14b_train_v1": ROOT
    / "artifacts"
    / "a4_teacher"
    / "qwen3_14b_train_v1"
    / "train_sft.jsonl",
    "qwen3_8b_train_v5": ROOT
    / "artifacts"
    / "a4_teacher"
    / "qwen3_8b_train_v5"
    / "train_sft.jsonl",
}


def _review_markdown(records: list[dict]) -> str:
    lines = [
        "# A4 Error-Recovery Supplementary Review",
        "",
        "These are all ten Qwen3-14B retained error-recovery trajectories. Review",
        "each conversation for leakage, sample-specific hardcoding, correct use of tool",
        "feedback, and a tool-supported final submission. Record the verdicts in",
        "`recovery_review_supplement.csv`.",
        "",
    ]
    for index, record in enumerate(records, 1):
        lines.extend(
            [
                f"## {index}. {record['trajectory_id']}",
                "",
                f"- Table: `{record['table']}`",
                f"- Task: `{record['task_id']}`",
                "- Category: `error_recovery`",
                "",
                "```json",
                json.dumps(record["messages"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    records = {name: load_sft_records(path) for name, path in SOURCE_RUNS.items()}
    main_records = records["qwen3_14b_train_v1"]
    recovery_8b = [
        record
        for record in records["qwen3_8b_train_v5"]
        if record["category"] == "error_recovery"
    ]
    recovery_14b = [
        record for record in main_records if record["category"] == "error_recovery"
    ]

    main = [selection_entry(record, "qwen3_14b_train_v1") for record in main_records]
    success_only = [entry for entry in main if entry["category"] != "error_recovery"]
    recovery_augmented = main + [
        selection_entry(record, "qwen3_8b_train_v5") for record in recovery_8b
    ]
    selections = {
        "main_14b": main,
        "success_only_14b": success_only,
        "recovery_augmented": recovery_augmented,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, entries in selections.items():
        write_jsonl(OUTPUT_DIR / f"{name}.index.jsonl", entries)

    manifest = {
        "schema_version": 1,
        "default_dataset": "main_14b",
        "selection_key": ["source_run", "trajectory_id"],
        "deduplication_key": "trajectory_fingerprint",
        "source_files": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "record_count": len(records[name]),
            }
            for name, path in SOURCE_RUNS.items()
        },
        "datasets": {
            name: {
                "index_path": str(
                    (OUTPUT_DIR / f"{name}.index.jsonl").relative_to(ROOT)
                ).replace("\\", "/"),
                **selection_summary(entries),
            }
            for name, entries in selections.items()
        },
        "notes": {
            "main_14b": "Production A4 data; use for the first A5 training run.",
            "success_only_14b": "Ablation excluding the ten 14B recovery trajectories.",
            "recovery_augmented": (
                "Ablation only. Adds 341 recovery trajectories from the 8B teacher; "
                "source model is a disclosed confound."
            ),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_jsonl(OUTPUT_DIR / "recovery_review_supplement.jsonl", recovery_14b)
    review_csv = OUTPUT_DIR / "recovery_review_supplement.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "trajectory_id",
            "task_id",
            "table",
            "no_leakage_pass",
            "no_hardcoding_pass",
            "tool_feedback_used_pass",
            "overall_verdict",
            "review_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in recovery_14b:
            writer.writerow(
                {
                    "trajectory_id": record["trajectory_id"],
                    "task_id": record["task_id"],
                    "table": record["table"],
                    "no_leakage_pass": "",
                    "no_hardcoding_pass": "",
                    "tool_feedback_used_pass": "",
                    "overall_verdict": "",
                    "review_notes": "",
                }
            )
    (OUTPUT_DIR / "recovery_review_supplement.md").write_text(
        _review_markdown(recovery_14b), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
