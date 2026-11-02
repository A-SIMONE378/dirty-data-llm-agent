"""Record an explicit human attestation for the frozen A4 review sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_a4_teacher_audit import (  # noqa: E402
    HUMAN_CHECK_FIELDS,
    _read_jsonl,
    _review_sample_ids,
    audit_teacher_run,
)


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "a4_teacher" / "qwen3_14b_train_v1"
DEFAULT_CONFIG = ROOT / "configs" / "a4_teacher_sampling_14b.json"
REVIEW_FIELDS = (
    "trajectory_id",
    "task_id",
    "table",
    "category",
    "message_count",
    *HUMAN_CHECK_FIELDS,
    "overall_verdict",
    "review_notes",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-all-pass",
        action="store_true",
        help="Record the project owner's explicit confirmation that all 50 rows passed.",
    )
    parser.add_argument("--reviewed-by", default="project owner")
    parser.add_argument("--review-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def record_all_pass(
    review_path: Path,
    expected_ids: list[str],
    *,
    reviewed_by: str,
    review_date: str,
) -> list[dict[str, str]]:
    """Validate the frozen sheet and record an already completed human review."""

    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError("A4 review sheet fields do not match the frozen schema.")
        rows = list(reader)
    actual_ids = [row.get("trajectory_id", "") for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("A4 review sheet does not match the deterministic sample.")

    note = f"Human review confirmed by {reviewed_by} on {review_date}."
    for row in rows:
        for field in HUMAN_CHECK_FIELDS:
            row[field] = "PASS"
        row["overall_verdict"] = "PASS"
        if not row.get("review_notes", "").strip():
            row["review_notes"] = note

    temporary = review_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(review_path)
    return rows


def _write_attestation(
    output_dir: Path,
    rows: list[Mapping[str, str]],
    *,
    reviewed_by: str,
    review_date: str,
) -> None:
    ids = [row["trajectory_id"] for row in rows]
    payload = {
        "schema_version": 1,
        "reviewed_by": reviewed_by,
        "review_date": review_date,
        "sample_size": len(rows),
        "overall_verdict": "PASS",
        "review_sample_ids_sha256": hashlib.sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "basis": "Explicit project-owner confirmation that every frozen review row passed.",
    }
    (output_dir / "human_review_attestation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    if not args.confirm_all_pass:
        raise SystemExit("Refusing to change review results without --confirm-all-pass.")

    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sft_records = _read_jsonl(output_dir / "train_sft.jsonl")
    expected_ids = _review_sample_ids(
        sft_records,
        int(config["human_review_sample_size"]),
        str(config["selection_seed"]),
    )
    rows = record_all_pass(
        output_dir / "human_review_sample.csv",
        expected_ids,
        reviewed_by=args.reviewed_by,
        review_date=args.review_date,
    )
    _write_attestation(
        output_dir,
        rows,
        reviewed_by=args.reviewed_by,
        review_date=args.review_date,
    )

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "human_review_status": "PASS",
            "a4_complete": True,
            "blocking_gate": None,
            "human_review_recorded_by": args.reviewed_by,
            "human_review_recorded_date": args.review_date,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = audit_teacher_run(output_dir, config_path=config_path)
    audit_path = ROOT / "artifacts" / "a4_teacher_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if audit.get("a4_complete") is not True:
        raise SystemExit("Recorded review did not close the A4 gate.")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
