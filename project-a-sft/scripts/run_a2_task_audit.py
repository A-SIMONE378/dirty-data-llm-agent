"""Run the full A2 task, answer, dirty replay, and leakage audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a1_catalog import load_a1_catalog  # noqa: E402
from dirty_da.a1_corruption import load_corruption_profiles  # noqa: E402
from dirty_da.a2_audit import audit_a2_catalog  # noqa: E402
from dirty_da.a2_tasks import DIFFICULTIES, load_task_records  # noqa: E402
from dirty_da.schema_semantics import load_table_semantics  # noqa: E402


REVIEW_CHECK_FIELDS = (
    "question_readable",
    "operation_filters_match_spec",
    "cleaning_instructions_match_spec",
    "answer_format_unambiguous",
)
REVIEW_FIELDS = (
    "task_id",
    "table",
    "split",
    "difficulty",
    "question",
    "spec_json",
    "expected_answer_json",
    *REVIEW_CHECK_FIELDS,
    "review_status",
    "reviewer_notes",
)


def _manual_review_sample(records, sample_size: int):
    per_difficulty, remainder = divmod(sample_size, len(DIFFICULTIES))
    selected = []
    for index, difficulty in enumerate(sorted(DIFFICULTIES)):
        count = per_difficulty + (1 if index < remainder else 0)
        candidates = [record for record in records if record["difficulty"] == difficulty]
        candidates.sort(
            key=lambda record: hashlib.sha256(record["task_id"].encode("utf-8")).hexdigest()
        )
        selected.extend(candidates[:count])
    selected.sort(key=lambda record: record["task_id"])
    return selected


def _load_existing_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviews: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = row.get("task_id", "")
        if not task_id or task_id in reviews:
            continue
        reviews[task_id] = {
            field: row.get(field, "")
            for field in (*REVIEW_CHECK_FIELDS, "review_status", "reviewer_notes")
        }
    return reviews


def _normalized_review(review: dict[str, str] | None) -> dict[str, str]:
    values = dict(review or {})
    for field in REVIEW_CHECK_FIELDS:
        value = values.get(field, "").strip().upper()
        values[field] = value if value in {"PASS", "FAIL"} else ""
    status = values.get("review_status", "").strip().upper()
    if status not in {"PASS", "FAIL", "PENDING_HUMAN"}:
        status = "PENDING_HUMAN"
    if any(values[field] == "FAIL" for field in REVIEW_CHECK_FIELDS):
        status = "FAIL"
    elif all(values[field] == "PASS" for field in REVIEW_CHECK_FIELDS):
        status = "PASS" if status == "PASS" else "PENDING_HUMAN"
    else:
        status = "PENDING_HUMAN"
    values["review_status"] = status
    values["reviewer_notes"] = values.get("reviewer_notes", "").strip()
    return values


def _manual_review_status(reviews: list[dict[str, str]]) -> str:
    if reviews and all(review["review_status"] == "PASS" for review in reviews):
        return "PASS"
    if any(review["review_status"] == "FAIL" for review in reviews):
        return "FAIL"
    return "PENDING_HUMAN"


def _write_review_artifacts(
    artifacts: Path,
    sample,
    existing_reviews: dict[str, dict[str, str]],
) -> str:
    normalized = {
        record["task_id"]: _normalized_review(existing_reviews.get(record["task_id"]))
        for record in sample
    }
    sample_path = artifacts / "a2_manual_review_sample.jsonl"
    with sample_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sample:
            review = normalized[record["task_id"]]
            review_record = {
                "task_id": record["task_id"],
                "table": record["table"],
                "split": record["split"],
                "difficulty": record["difficulty"],
                "question": record["question"],
                "spec": record["spec"],
                "expected_answer": record["answer"],
                "review_status": review["review_status"],
                "review_checks": [
                    "question is readable",
                    "question states the same operation and filters as spec",
                    "cleaning instructions match spec",
                    "answer format is unambiguous",
                ],
                "review_results": {
                    field: review[field] for field in REVIEW_CHECK_FIELDS
                },
                "reviewer_notes": review["reviewer_notes"],
            }
            handle.write(json.dumps(review_record, ensure_ascii=False, sort_keys=True) + "\n")

    review_csv_path = artifacts / "a2_manual_review_sample.csv"
    with review_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for record in sample:
            review = normalized[record["task_id"]]
            writer.writerow(
                {
                    "task_id": record["task_id"],
                    "table": record["table"],
                    "split": record["split"],
                    "difficulty": record["difficulty"],
                    "question": record["question"],
                    "spec_json": json.dumps(
                        record["spec"], ensure_ascii=False, sort_keys=True
                    ),
                    "expected_answer_json": json.dumps(
                        record["answer"], ensure_ascii=False, sort_keys=True
                    ),
                    **review,
                }
            )
    return _manual_review_status(list(normalized.values()))


def main() -> None:
    task_path = ROOT / "data" / "tasks" / "a2_tasks.jsonl"
    records = load_task_records(task_path)
    a1_catalog = load_a1_catalog(ROOT / "configs" / "a1_table_catalog.json")
    semantics = load_table_semantics(ROOT / "configs" / "table_semantics.json")
    profiles = load_corruption_profiles(ROOT / "configs" / "a1_corruption_profiles.json")
    generation = json.loads(
        (ROOT / "configs" / "a2_task_generation.json").read_text(encoding="utf-8")
    )
    audit = audit_a2_catalog(
        ROOT / "data" / "tables",
        records,
        a1_catalog=a1_catalog,
        semantics_registry=semantics,
        profile_config=profiles,
        generation_config=generation,
    )

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    audit_path = artifacts / "a2_task_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit.rows[0]))
        writer.writeheader()
        writer.writerows(audit.rows)

    sample = _manual_review_sample(
        records, int(generation["manual_review_sample_size"])
    )
    review_csv_path = artifacts / "a2_manual_review_sample.csv"
    existing_reviews = _load_existing_reviews(review_csv_path)
    manual_status = _write_review_artifacts(artifacts, sample, existing_reviews)

    summary = dict(audit.summary)
    summary.update(
        {
            "manual_review_sample_size": len(sample),
            "manual_review_status": manual_status,
            "a2_complete": manual_status == "PASS",
            "blocking_gate": (
                None
                if manual_status == "PASS"
                else "manual_question_review_failed"
                if manual_status == "FAIL"
                else "manual_question_review"
            ),
        }
    )
    summary_path = artifacts / "a2_task_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
