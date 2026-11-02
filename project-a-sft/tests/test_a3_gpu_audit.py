import hashlib
import json

from scripts.run_a3_environment_audit import audit_gpu_acceptance


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _complete_gpu_evidence(tmp_path):
    task_ids = [f"task-{index:02d}" for index in range(50)]
    manifest = {
        "schema_version": 1,
        "task_count": 50,
        "tasks": [{"task_id": task_id} for task_id in task_ids],
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(tmp_path / "manifest.json", manifest)

    result_lines = []
    for task_id in task_ids:
        result_lines.append(
            json.dumps({"task_id": task_id, "terminal_status": "accepted"})
        )
        _write_json(
            tmp_path / "trajectories" / f"{task_id}.json",
            {
                "schema_version": 1,
                "task_id": task_id,
                "table": "fixture.csv",
                "split": "test",
                "question": "Count rows.",
                "dirty_profile": "low",
                "max_steps": 12,
                "terminal_status": "accepted",
                "submitted_answer": 3,
                "events": [],
            },
        )
    (tmp_path / "results.jsonl").write_text(
        "\n".join(result_lines) + "\n", encoding="utf-8"
    )
    _write_json(
        tmp_path / "summary.json",
        {
            "a3_gpu_acceptance_pass": True,
            "accepted_accuracy": 1.0,
            "manifest_sha256": manifest_hash,
            "model_id": "Qwen/Qwen3-8B",
            "posix_resource_limits_active": True,
            "private_fields_absent": True,
            "run_complete": True,
            "runtime": {"gpu": "fixture"},
            "tasks_attempted": 50,
            "tasks_required": 50,
            "terminal_status_counts": {"accepted": 50},
            "trajectory_schema_valid_count": 50,
        },
    )


def test_gpu_audit_requires_independent_complete_evidence(tmp_path):
    assert audit_gpu_acceptance(tmp_path)["status"] == "NOT_RUN"
    _complete_gpu_evidence(tmp_path)
    audit = audit_gpu_acceptance(tmp_path)
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())


def test_gpu_audit_rejects_summary_that_disagrees_with_results(tmp_path):
    _complete_gpu_evidence(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["terminal_status_counts"] = {"accepted": 49, "model_error": 1}
    _write_json(summary_path, summary)
    audit = audit_gpu_acceptance(tmp_path)
    assert audit["status"] == "FAIL"
    assert not audit["checks"]["summary_counts_match_latest_results"]

