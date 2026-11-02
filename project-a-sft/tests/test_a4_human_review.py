from __future__ import annotations

import csv

import pytest

from scripts.finalize_a4_human_review import REVIEW_FIELDS, record_all_pass


def _write_review(path, trajectory_ids):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        for index, trajectory_id in enumerate(trajectory_ids):
            writer.writerow(
                {
                    "trajectory_id": trajectory_id,
                    "task_id": f"task-{index}",
                    "table": "table.csv",
                    "category": "direct_success",
                    "message_count": "4",
                    "no_leakage_pass": "",
                    "no_hardcoding_pass": "",
                    "tool_feedback_used_pass": "",
                    "overall_verdict": "",
                    "review_notes": "",
                }
            )


def test_record_all_pass_requires_exact_frozen_sample(tmp_path):
    review_path = tmp_path / "review.csv"
    _write_review(review_path, ["trajectory-1", "trajectory-2"])
    with pytest.raises(ValueError, match="deterministic sample"):
        record_all_pass(
            review_path,
            ["trajectory-2", "trajectory-1"],
            reviewed_by="owner",
            review_date="2026-10-25",
        )


def test_record_all_pass_records_checks_and_attestation_note(tmp_path):
    review_path = tmp_path / "review.csv"
    ids = ["trajectory-1", "trajectory-2"]
    _write_review(review_path, ids)
    rows = record_all_pass(
        review_path,
        ids,
        reviewed_by="project owner",
        review_date="2026-10-25",
    )
    assert all(row["overall_verdict"] == "PASS" for row in rows)
    assert all(row["no_leakage_pass"] == "PASS" for row in rows)
    assert all(row["no_hardcoding_pass"] == "PASS" for row in rows)
    assert all(row["tool_feedback_used_pass"] == "PASS" for row in rows)
    assert all("project owner" in row["review_notes"] for row in rows)
