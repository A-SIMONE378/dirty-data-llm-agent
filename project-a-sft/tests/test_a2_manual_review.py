import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_a2_task_audit import (  # noqa: E402
    REVIEW_CHECK_FIELDS,
    _load_existing_reviews,
    _manual_review_status,
    _normalized_review,
)


def test_completed_manual_review_is_normalized_as_pass():
    review = {field: "pass" for field in REVIEW_CHECK_FIELDS}
    review["review_status"] = "PASS"
    normalized = _normalized_review(review)
    assert normalized["review_status"] == "PASS"
    assert _manual_review_status([normalized]) == "PASS"


def test_blank_or_inconsistent_manual_review_stays_pending():
    blank = _normalized_review(None)
    assert blank["review_status"] == "PENDING_HUMAN"
    inconsistent = {field: "PASS" for field in REVIEW_CHECK_FIELDS}
    inconsistent["review_status"] = "PENDING_HUMAN"
    assert _normalized_review(inconsistent)["review_status"] == "PENDING_HUMAN"


def test_any_failed_check_forces_failed_review():
    review = {field: "PASS" for field in REVIEW_CHECK_FIELDS}
    review[REVIEW_CHECK_FIELDS[1]] = "FAIL"
    review["review_status"] = "PASS"
    normalized = _normalized_review(review)
    assert normalized["review_status"] == "FAIL"
    assert _manual_review_status([normalized]) == "FAIL"


def test_existing_review_loader_preserves_only_review_fields(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", *REVIEW_CHECK_FIELDS, "review_status", "reviewer_notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "task-1",
                **{field: "PASS" for field in REVIEW_CHECK_FIELDS},
                "review_status": "PASS",
                "reviewer_notes": "checked",
            }
        )
    loaded = _load_existing_reviews(path)
    assert loaded["task-1"]["review_status"] == "PASS"
    assert loaded["task-1"]["reviewer_notes"] == "checked"
