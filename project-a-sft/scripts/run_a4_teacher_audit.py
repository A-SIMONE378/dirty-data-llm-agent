"""Independently audit persisted A4 teacher trajectories and SFT records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.a4_teacher import (  # noqa: E402
    QualityAssessment,
    assess_teacher_trajectory,
    build_sampling_attempts,
    select_teacher_tasks,
)
from dirty_da.trajectory import validate_trajectory_record  # noqa: E402
from run_a4_teacher_sampling import TEACHER_SYSTEM_PROMPT  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "a4_teacher" / "qwen3_8b_train_v5"
CONFIG_PATH = ROOT / "configs" / "a4_teacher_sampling.json"
SOURCE_PATH = ROOT / "data" / "tasks" / "a2_tasks.jsonl"
FORBIDDEN_PRIVATE_FIELDS = (
    '"expected_answer"',
    '"task_spec"',
    '"canonical_fingerprint"',
)
HUMAN_CHECK_FIELDS = (
    "no_leakage_pass",
    "no_hardcoding_pass",
    "tool_feedback_used_pass",
)
A4_TOOLS = ("inspect_table", "run_python", "submit_answer")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object on line {line_number} of {path}.")
        rows.append(value)
    return rows


def _latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for result in _read_jsonl(path):
        attempt_id = result.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("A4 result is missing attempt_id.")
        latest[attempt_id] = result
    return latest


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalized_human_review(row: Mapping[str, str]) -> dict[str, str]:
    values = {field: str(row.get(field, "")).strip().upper() for field in HUMAN_CHECK_FIELDS}
    for field in HUMAN_CHECK_FIELDS:
        if values[field] not in {"PASS", "FAIL"}:
            values[field] = ""
    verdict = str(row.get("overall_verdict", "")).strip().upper()
    values["overall_verdict"] = verdict if verdict in {"PASS", "FAIL"} else ""
    values["review_notes"] = str(row.get("review_notes", "")).strip()
    return values


def _human_review_status(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "MISSING"
    if any(
        row["overall_verdict"] == "FAIL"
        or any(row[field] == "FAIL" for field in HUMAN_CHECK_FIELDS)
        for row in rows
    ):
        return "FAIL"
    if all(
        row["overall_verdict"] == "PASS"
        and all(row[field] == "PASS" for field in HUMAN_CHECK_FIELDS)
        for row in rows
    ):
        return "PASS"
    return "PENDING_HUMAN"


def _review_sample_ids(sft_records: list[dict[str, Any]], sample_size: int, seed: str) -> list[str]:
    ranked = sorted(
        sft_records,
        key=lambda record: hashlib.sha256(
            f"{seed}|review|{record['trajectory_id']}".encode("utf-8")
        ).hexdigest(),
    )[:sample_size]
    return [record["trajectory_id"] for record in ranked]


def _sft_check(
    records: list[dict[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    retained_ids = {
        attempt_id for attempt_id, result in results.items() if result.get("retained")
    }
    record_ids = [str(record.get("trajectory_id")) for record in records]
    coverage = len(record_ids) == len(retained_ids) and set(record_ids) == retained_ids
    masks = True
    private = True
    tools_valid = True
    for record in records:
        attempt_id = str(record.get("trajectory_id"))
        trajectory = trajectories.get(attempt_id)
        if trajectory is None:
            coverage = False
            continue
        events = trajectory["events"]
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != len(events) * 2 + 2:
            masks = False
            continue
        if record.get("prompt_version") != config["prompt_version"]:
            coverage = False
        tool_names = [
            tool.get("name") for tool in record.get("tools", []) if isinstance(tool, dict)
        ]
        tools_valid &= tool_names == list(A4_TOOLS)
        if (
            messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
            or messages[0].get("trainable") is not False
            or messages[1].get("trainable") is not False
        ):
            masks = False
        for index, event in enumerate(events):
            assistant = messages[2 + index * 2]
            tool_message = messages[3 + index * 2]
            if (
                assistant.get("role") != "assistant"
                or assistant.get("trainable") is not True
                or tool_message.get("role") != "tool"
                or tool_message.get("trainable") is not False
                or tool_message.get("name") != event.get("tool")
                or tool_message.get("content") != event.get("content")
                or tool_message.get("ok") != event.get("ok")
            ):
                masks = False
            calls = assistant.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                masks = False
                continue
            function = calls[0].get("function") if isinstance(calls[0], dict) else None
            if (
                not isinstance(function, dict)
                or function.get("name") != event.get("tool")
                or function.get("arguments") != event.get("arguments")
            ):
                masks = False
        serialized = json.dumps(record, ensure_ascii=False)
        private &= all(token not in serialized for token in FORBIDDEN_PRIVATE_FIELDS)
    return coverage, masks, private and tools_valid


def audit_teacher_run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    config_path: Path = CONFIG_PATH,
    source_path: Path = SOURCE_PATH,
) -> dict[str, Any]:
    config = _read_json(config_path)
    records = load_task_records(source_path)
    selected = select_teacher_tasks(records, config)
    attempts = build_sampling_attempts(selected, config)
    attempts_by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    retained_target = int(config["minimum_retained_trajectories"])

    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    sft_path = output_dir / "train_sft.jsonl"
    trajectory_dir = output_dir / "trajectories"
    required = (manifest_path, results_path, summary_path, sft_path, trajectory_dir)
    if not all(path.exists() for path in required):
        return {
            "schema_version": 1,
            "status": "NOT_RUN",
            "passed": False,
            "checks": {},
            "blocking_gate": "sampling_incomplete",
        }

    checks: dict[str, bool] = {}
    try:
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        latest = _latest_results(results_path)
        sft_records = _read_jsonl(sft_path)
        trajectory_paths = sorted(trajectory_dir.glob("*.json"))
        trajectories = {path.stem: _read_json(path) for path in trajectory_paths}

        expected_ids = set(attempts_by_id)
        expected_attempt_count = (
            int(config["selected_task_count"]) * int(config["samples_per_task"])
        )
        checks["planned_attempts_match_frozen_selection"] = (
            manifest.get("planned_attempt_count")
            == len(attempts)
            == expected_attempt_count
        )
        checks["manifest_matches_frozen_selection"] = (
            manifest.get("selection_seed") == config["selection_seed"]
            and manifest.get("prompt_version") == config["prompt_version"]
            and manifest.get("selected_task_count")
            == len(selected)
            == int(config["selected_task_count"])
            and manifest.get("samples_per_task") == config["samples_per_task"]
            and manifest.get("generation", {}).get("invalid_action_retries")
            == config["invalid_action_retries"]
            and manifest.get("source_catalog_sha256")
            == hashlib.sha256(source_path.read_bytes()).hexdigest()
            and manifest.get("system_prompt_sha256")
            == hashlib.sha256(TEACHER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
            and [task.get("task_id") for task in manifest.get("tasks", [])]
            == [record["task_id"] for record in selected]
            and [attempt.get("attempt_id") for attempt in manifest.get("attempts", [])]
            == [attempt["attempt_id"] for attempt in attempts]
        )
        checks["summary_manifest_hash_matches"] = (
            summary.get("manifest_sha256") == _manifest_hash(manifest)
        )
        checks["latest_results_cover_planned_attempts"] = set(latest) == expected_ids
        checks["trajectory_files_cover_latest_results"] = (
            set(trajectories) == set(latest)
        )
        checks["result_attempt_metadata_matches_manifest"] = all(
            latest[attempt_id].get("task_id") == attempt["task_id"]
            and latest[attempt_id].get("sample_index") == attempt["sample_index"]
            and latest[attempt_id].get("sampling_seed") == attempt["sampling_seed"]
            for attempt_id, attempt in attempts_by_id.items()
            if attempt_id in latest
        )

        trajectories_valid = True
        trajectories_private = True
        for trajectory in trajectories.values():
            try:
                validate_trajectory_record(trajectory)
            except (TypeError, ValueError):
                trajectories_valid = False
                continue
            serialized = json.dumps(trajectory, ensure_ascii=False)
            trajectories_private &= all(
                token not in serialized for token in FORBIDDEN_PRIVATE_FIELDS
            )

        assessments_match = True
        seen_fingerprints: set[str] = set()
        for attempt in attempts:
            attempt_id = attempt["attempt_id"]
            trajectory = trajectories.get(attempt_id)
            result = latest.get(attempt_id)
            if trajectory is None or result is None:
                assessments_match = False
                continue
            assessment = assess_teacher_trajectory(attempt["record"], trajectory)
            if assessment.retained and (
                assessment.trajectory_fingerprint in seen_fingerprints
            ):
                assessment = QualityAssessment(
                    retained=False,
                    category=None,
                    reasons=("duplicate_trajectory",),
                    uses_tool_feedback=assessment.uses_tool_feedback,
                    trajectory_fingerprint=assessment.trajectory_fingerprint,
                )
            if assessment.retained:
                seen_fingerprints.add(assessment.trajectory_fingerprint)
            assessments_match &= (
                result.get("retained") == assessment.retained
                and result.get("category") == assessment.category
                and result.get("quality_reasons") == list(assessment.reasons)
                and result.get("uses_tool_feedback") == assessment.uses_tool_feedback
                and result.get("trajectory_fingerprint")
                == assessment.trajectory_fingerprint
                and result.get("terminal_status") == trajectory.get("terminal_status")
                and result.get("steps") == len(trajectory.get("events", []))
                and trajectory.get("task_id") == attempt["task_id"]
            )
        checks["all_trajectories_validate"] = trajectories_valid
        checks["private_fields_absent"] = trajectories_private
        checks["quality_assessments_recompute"] = assessments_match
        checks["run_is_complete"] = (
            summary.get("run_complete") is True
            and summary.get("completed_attempts") == len(attempts)
            and len(latest) == len(attempts)
        )

        statuses = Counter(result["terminal_status"] for result in latest.values())
        retained = [result for result in latest.values() if result.get("retained")]
        categories = Counter(result["category"] for result in retained)
        rejection_reasons = Counter(
            reason for result in latest.values() for reason in result.get("quality_reasons", [])
        )
        checks["summary_counts_match_latest_results"] = (
            summary.get("planned_attempts") == len(attempts)
            and summary.get("completed_attempts") == len(latest)
            and summary.get("terminal_status_counts") == dict(sorted(statuses.items()))
            and summary.get("retained_trajectory_count") == len(retained)
            and summary.get("retained_category_counts") == dict(sorted(categories.items()))
            and summary.get("quality_rejection_reasons")
            == dict(sorted(rejection_reasons.items()))
            and summary.get("invalid_action_retries_used")
            == sum(
                int(result.get("invalid_action_retries", 0))
                for result in latest.values()
            )
        )
        retained_count = len(retained)
        checks["retained_trajectory_target"] = (
            retained_target <= retained_count <= int(config["maximum_retained_trajectories"])
        )
        checks["no_runtime_model_errors"] = statuses.get("model_error", 0) == 0
        checks["posix_resource_limits_active"] = (
            summary.get("runtime", {}).get("posix_resource_limits_active") is True
        )
        checks["sandbox_limits_match_config"] = (
            summary.get("runtime", {}).get("sandbox_memory_mb")
            == config["max_memory_mb"]
            and summary.get("runtime", {}).get("sandbox_timeout_seconds")
            == config["run_timeout_seconds"]
            and summary.get("runtime", {}).get("sandbox_cpu_seconds")
            == config["max_cpu_seconds"]
        )

        sft_coverage, sft_masks, sft_private = _sft_check(
            sft_records, trajectories, latest, config
        )
        checks["sft_covers_retained_trajectories"] = sft_coverage
        checks["sft_trainable_masks_and_tool_calls"] = sft_masks
        checks["sft_private_fields_absent"] = sft_private

        review_csv_path = output_dir / "human_review_sample.csv"
        review_jsonl_path = output_dir / "human_review_sample.jsonl"
        review_rows: list[dict[str, str]] = []
        review_ids: list[str] = []
        if review_csv_path.is_file() and review_jsonl_path.is_file():
            with review_csv_path.open(encoding="utf-8-sig", newline="") as handle:
                review_rows = list(csv.DictReader(handle))
            review_ids = [str(row.get("trajectory_id")) for row in review_rows]
        expected_review_ids = _review_sample_ids(
            sft_records, int(config["human_review_sample_size"]), str(config["selection_seed"])
        )
        checks["human_review_sample_matches_retained_sft"] = (
            len(review_rows) == int(config["human_review_sample_size"])
            and review_ids == expected_review_ids
        )
        human_status = _human_review_status(review_rows)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "schema_version": 1,
            "status": "FAIL",
            "passed": False,
            "checks": checks,
            "blocking_gate": "audit_error",
        }

    mechanical_pass = all(checks.values())
    a4_complete = mechanical_pass and human_status == "PASS"
    blocking_gate = (
        None
        if a4_complete
        else "sampling_incomplete"
        if not checks.get("run_is_complete")
        else "retained_trajectory_target"
        if not checks.get("retained_trajectory_target")
        else "human_review_50"
        if human_status != "PASS"
        else "audit_failure"
    )
    return {
        "schema_version": 1,
        "status": "PASS" if mechanical_pass else "FAIL",
        "passed": mechanical_pass,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "mechanical_target_met": checks.get("retained_trajectory_target", False),
        "retained_trajectory_count": len(retained),
        "completed_attempts": len(latest),
        "terminal_status_counts": dict(sorted(statuses.items())),
        "human_review_status": human_status,
        "a4_complete": a4_complete,
        "blocking_gate": blocking_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    audit = audit_teacher_run(args.output_dir, config_path=args.config)
    destination = ROOT / "artifacts" / "a4_teacher_audit.json"
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if audit["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

