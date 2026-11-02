import csv
import json
from pathlib import Path
import sys

from dirty_da.a2_tasks import load_task_records
from dirty_da.a4_teacher import (
    assess_teacher_trajectory,
    build_sampling_attempts,
    select_teacher_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_a4_teacher_audit import audit_teacher_run  # noqa: E402
from run_a4_teacher_sampling import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    _manifest,
    _rebuild_sft_dataset,
    _summary,
    _write_human_review_sample,
    _write_json,
)


SOURCE = ROOT / "data" / "tasks" / "a2_tasks.jsonl"
TOOLS = [
    {"name": name, "description": "test", "parameters": {"type": "object"}}
    for name in ("inspect_table", "run_python", "submit_answer")
]


def _trajectory(record):
    return {
        "schema_version": 1,
        "task_id": record["task_id"],
        "table": record["table"],
        "split": record["split"],
        "question": record["question"],
        "dirty_profile": record["dirty_profile"],
        "max_steps": 10,
        "terminal_status": "accepted",
        "submitted_answer": 3,
        "events": [
            {
                "step": 1,
                "assistant_content": "Count rows.",
                "tool": "run_python",
                "arguments": {"code": "len(df)"},
                "ok": True,
                "content": "3",
                "truncated": False,
                "error_type": None,
            },
            {
                "step": 2,
                "assistant_content": "Submit.",
                "tool": "submit_answer",
                "arguments": {"answer": 3},
                "ok": True,
                "content": "Answer accepted.",
                "truncated": False,
                "error_type": None,
            },
        ],
    }


def _complete_evidence(tmp_path):
    config = json.loads(
        (ROOT / "configs" / "a4_teacher_sampling.json").read_text(encoding="utf-8")
    )
    config.update(
        selected_task_count=28,
        samples_per_task=1,
        minimum_retained_trajectories=1,
        maximum_retained_trajectories=28,
        human_review_sample_size=5,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    records = load_task_records(SOURCE)
    selected = select_teacher_tasks(records, config)
    attempts = build_sampling_attempts(selected, config)
    attempts_by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    manifest = _manifest(
        selected=selected,
        attempts=attempts,
        source_path=SOURCE,
        config=config,
    )
    output_dir = tmp_path / "run"
    _write_json(output_dir / "manifest.json", manifest)

    results = {}
    for attempt in attempts:
        trajectory = _trajectory(attempt["record"])
        _write_json(
            output_dir / "trajectories" / f"{attempt['attempt_id']}.json",
            trajectory,
        )
        assessment = assess_teacher_trajectory(attempt["record"], trajectory)
        result = {
            "attempt_id": attempt["attempt_id"],
            "task_id": attempt["task_id"],
            "table": attempt["record"]["table"],
            "difficulty": attempt["record"]["difficulty"],
            "dirty_profile": attempt["record"]["dirty_profile"],
            "sample_index": attempt["sample_index"],
            "sampling_seed": attempt["sampling_seed"],
            "terminal_status": trajectory["terminal_status"],
            "steps": len(trajectory["events"]),
            "elapsed_seconds": 1.0,
            "retained": assessment.retained,
            "category": assessment.category,
            "quality_reasons": list(assessment.reasons),
            "uses_tool_feedback": assessment.uses_tool_feedback,
            "trajectory_fingerprint": assessment.trajectory_fingerprint,
            "error_type": None,
            "error_message": None,
        }
        results[attempt["attempt_id"]] = result
        with (output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    _rebuild_sft_dataset(
        output_dir=output_dir,
        attempts_by_id=attempts_by_id,
        results=results,
        tools=TOOLS,
        prompt_version=config["prompt_version"],
    )
    _write_human_review_sample(
        output_dir,
        _read_sft(output_dir),
        config["human_review_sample_size"],
        config["selection_seed"],
    )
    runtime = {
        "posix_resource_limits_active": True,
        "sandbox_memory_mb": config["max_memory_mb"],
        "sandbox_timeout_seconds": config["run_timeout_seconds"],
        "sandbox_cpu_seconds": config["max_cpu_seconds"],
    }
    _write_json(
        output_dir / "summary.json",
        _summary(results=results, manifest=manifest, config=config, runtime=runtime),
    )
    return output_dir, config_path, output_dir / "human_review_sample.csv"


def _read_sft(output_dir):
    return [
        json.loads(line)
        for line in (output_dir / "train_sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _set_human_review(path, verdict):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        row["no_leakage_pass"] = verdict
        row["no_hardcoding_pass"] = verdict
        row["tool_feedback_used_pass"] = verdict
        row["overall_verdict"] = verdict
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_a4_audit_recomputes_evidence_and_keeps_human_gate_pending(tmp_path):
    output_dir, config_path, review_path = _complete_evidence(tmp_path)
    audit = audit_teacher_run(
        output_dir, config_path=config_path, source_path=SOURCE
    )
    assert audit["status"] == "PASS"
    assert audit["mechanical_target_met"] is True
    assert audit["human_review_status"] == "PENDING_HUMAN"
    assert audit["a4_complete"] is False
    assert audit["blocking_gate"] == "human_review_50"
    _set_human_review(review_path, "PASS")
    completed = audit_teacher_run(
        output_dir, config_path=config_path, source_path=SOURCE
    )
    assert completed["a4_complete"] is True
    assert completed["blocking_gate"] is None


def test_a4_audit_rejects_tampered_quality_result(tmp_path):
    output_dir, config_path, _ = _complete_evidence(tmp_path)
    results_path = output_dir / "results.jsonl"
    rows = [
        json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["retained"] = False
    results_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    audit = audit_teacher_run(
        output_dir, config_path=config_path, source_path=SOURCE
    )
    assert audit["status"] == "FAIL"
    assert not audit["checks"]["quality_assessments_recompute"]
