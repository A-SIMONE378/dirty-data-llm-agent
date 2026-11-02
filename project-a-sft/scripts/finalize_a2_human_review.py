"""Record and validate the independently completed A2 human review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dirty_da.a2_tasks import load_task_records  # noqa: E402
from run_a2_task_audit import (  # noqa: E402
    REVIEW_CHECK_FIELDS,
    REVIEW_FIELDS,
    _load_existing_reviews,
    _manual_review_sample,
    _write_review_artifacts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-all-pass",
        action="store_true",
        help="Record the project owner's explicit confirmation that every sampled task passed.",
    )
    parser.add_argument("--reviewed-by", default="project owner")
    parser.add_argument("--review-date", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.confirm_all_pass:
        raise SystemExit("Refusing to change review results without --confirm-all-pass.")

    generation = json.loads(
        (ROOT / "configs" / "a2_task_generation.json").read_text(encoding="utf-8")
    )
    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    expected = _manual_review_sample(
        records, int(generation["manual_review_sample_size"])
    )
    expected_ids = [record["task_id"] for record in expected]
    review_path = ROOT / "artifacts" / "a2_manual_review_sample.csv"
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual_ids = [row.get("task_id", "") for row in rows]
    if actual_ids != expected_ids:
        raise SystemExit("Review sheet does not match the deterministic A2 sample.")

    note = f"Human review confirmed by {args.reviewed_by} on {args.review_date}."
    for row in rows:
        for field in REVIEW_CHECK_FIELDS:
            row[field] = "PASS"
        row["review_status"] = "PASS"
        if not row.get("reviewer_notes", "").strip():
            row["reviewer_notes"] = note
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    artifacts = ROOT / "artifacts"
    status = _write_review_artifacts(
        artifacts,
        expected,
        _load_existing_reviews(review_path),
    )
    if status != "PASS":
        raise SystemExit("Recorded review did not normalize to PASS.")

    summary_path = artifacts / "a2_task_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mechanical_gate_pass") is not True:
        raise SystemExit("Cannot complete A2 because the mechanical gate is not PASS.")
    summary.update(
        {
            "manual_review_sample_size": len(expected),
            "manual_review_status": "PASS",
            "a2_complete": True,
            "blocking_gate": None,
            "manual_review_recorded_by": args.reviewed_by,
            "manual_review_recorded_date": args.review_date,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"a2_complete": True, "manual_review_status": "PASS"}))


if __name__ == "__main__":
    main()
