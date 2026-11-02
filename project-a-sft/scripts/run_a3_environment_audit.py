"""Exercise A3 and incorporate a completed, independently checked GPU run."""

from __future__ import annotations

import json
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a1_corruption import load_corruption_profiles  # noqa: E402
from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.agent_env import AgentEnvironmentConfig, AgentEpisode  # noqa: E402
from dirty_da.agent_loop import AssistantAction, run_agent_episode  # noqa: E402
from dirty_da.schema_semantics import load_table_semantics  # noqa: E402
from dirty_da.trajectory import validate_trajectory_record  # noqa: E402


GPU_RESULT_DIR = ROOT / "artifacts" / "a3_gpu" / "qwen3_8b_50_v1"
FORBIDDEN_PRIVATE_FIELDS = (
    '"expected_answer"',
    '"task_spec"',
    '"canonical_fingerprint"',
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def audit_gpu_acceptance(result_dir: Path = GPU_RESULT_DIR) -> dict:
    """Validate persisted GPU evidence instead of trusting its summary flag."""

    required = (
        result_dir / "manifest.json",
        result_dir / "results.jsonl",
        result_dir / "summary.json",
        result_dir / "trajectories",
    )
    if not all(path.exists() for path in required):
        return {"status": "NOT_RUN", "checks": {}, "passed": False}

    try:
        manifest = _read_json(required[0])
        summary = _read_json(required[2])
        latest_results: dict[str, dict] = {}
        for line in required[1].read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest_results[record["task_id"]] = record

        tasks = manifest["tasks"]
        manifest_ids = [task["task_id"] for task in tasks]
        trajectory_paths = sorted(required[3].glob("*.json"))
        trajectory_ids = [path.stem for path in trajectory_paths]
        trajectories_valid = True
        trajectories_private = True
        for path in trajectory_paths:
            trajectory = _read_json(path)
            validate_trajectory_record(trajectory)
            serialized = json.dumps(trajectory, ensure_ascii=False)
            trajectories_private &= all(
                token not in serialized for token in FORBIDDEN_PRIVATE_FIELDS
            )

        statuses = Counter(
            result["terminal_status"] for result in latest_results.values()
        )
        manifest_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        checks = {
            "summary_declares_pass": summary.get("a3_gpu_acceptance_pass") is True,
            "frozen_manifest_is_50_unique_tasks": (
                manifest.get("schema_version") == 1
                and manifest.get("task_count") == 50
                and len(manifest_ids) == len(set(manifest_ids)) == 50
            ),
            "manifest_hash_matches_summary": (
                summary.get("manifest_sha256") == manifest_hash
            ),
            "latest_results_cover_manifest": (
                set(latest_results) == set(manifest_ids)
            ),
            "trajectory_files_cover_manifest": (
                len(trajectory_ids) == 50
                and set(trajectory_ids) == set(manifest_ids)
            ),
            "all_trajectories_validate": trajectories_valid,
            "private_fields_absent": (
                trajectories_private
                and summary.get("private_fields_absent") is True
            ),
            "summary_counts_match_latest_results": (
                summary.get("tasks_required") == 50
                and summary.get("tasks_attempted") == 50
                and summary.get("terminal_status_counts")
                == dict(sorted(statuses.items()))
                and summary.get("trajectory_schema_valid_count") == 50
            ),
            "accuracy_is_recomputed": (
                summary.get("accepted_accuracy") == statuses.get("accepted", 0) / 50
            ),
            "no_runtime_model_errors": statuses.get("model_error", 0) == 0,
            "posix_limits_active_in_gpu_run": (
                summary.get("posix_resource_limits_active") is True
            ),
            "run_is_complete": summary.get("run_complete") is True,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"status": "FAIL", "checks": {}, "passed": False}
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "passed": passed,
        "accepted_accuracy": summary.get("accepted_accuracy"),
        "terminal_status_counts": summary.get("terminal_status_counts"),
        "model_id": summary.get("model_id"),
        "runtime": summary.get("runtime"),
    }


def _fixture_episode(
    *,
    expected_answer=3,
    config: AgentEnvironmentConfig | None = None,
) -> AgentEpisode:
    return AgentEpisode(
        task_id="a3-environment-audit",
        table="handwritten_fixture.csv",
        split="dev",
        question="Return the number of rows.",
        dirty_profile="low",
        dirty_frame=pd.DataFrame(
            {"group": ["a", "a", "b"], "value": [1.0, 2.0, 3.0]}
        ),
        expected_answer=expected_answer,
        config=config,
        runtime_root=ROOT / ".a3_runtime",
    )


class _ScriptedAdapter:
    def __init__(self, actions: list[AssistantAction]) -> None:
        self.actions = list(actions)
        self.contexts = []

    def next_action(self, messages, tools):
        self.contexts.append((messages, tools))
        return self.actions.pop(0)


def main() -> None:
    checks: dict[str, bool] = {}
    config = AgentEnvironmentConfig.from_json(ROOT / "configs" / "a3_agent_env.json")

    records = load_task_records(ROOT / "data" / "tasks" / "a2_tasks.jsonl")
    with AgentEpisode.from_task_record(
        records[0],
        tables_dir=ROOT / "data" / "tables",
        semantics_registry=load_table_semantics(ROOT / "configs" / "table_semantics.json"),
        profile_config=load_corruption_profiles(
            ROOT / "configs" / "a1_corruption_profiles.json"
        ),
        config=config,
        runtime_root=ROOT / ".a3_runtime",
    ) as episode:
        checks["exact_three_tools"] = episode.tools == (
            "inspect_table",
            "run_python",
            "submit_answer",
        )
        inspection = episode.call_tool("inspect_table", {"head_rows": 2})
        checks["real_a2_dirty_replay_loads"] = inspection.ok
        checks["inspect_has_no_oracle_fields"] = all(
            token not in inspection.content
            for token in ("expected_answer", "task_spec", "canonical_fingerprint")
        )

    with _fixture_episode(config=config) as episode:
        first = episode.call_tool("run_python", {"code": "subtotal = df['value'].sum()"})
        second = episode.call_tool("run_python", {"code": "subtotal + len(df)"})
        checks["python_state_persists"] = (
            first.ok and second.ok and second.content.strip() == "9.0"
        )
        checks["platform_resource_limits_reported"] = (
            episode.sandbox_resource_limits_active == (os.name == "posix")
        )
        blocked_import = episode.call_tool("run_python", {"code": "import socket"})
        checks["imports_blocked"] = (
            not blocked_import.ok and blocked_import.error_type == "PermissionError"
        )
        blocked_file = episode.call_tool(
            "run_python", {"code": "pd.read_csv('../outside.csv')"}
        )
        checks["file_escape_blocked"] = (
            not blocked_file.ok
            and blocked_file.error_type in {"PermissionError", "FileNotFoundError"}
        )

    with _fixture_episode(expected_answer=3, config=config) as episode:
        accepted = episode.call_tool("submit_answer", {"answer": 3})
        trajectory = episode.trajectory.to_dict()
        validate_trajectory_record(trajectory)
        checks["strict_correct_answer_accepted"] = accepted.ok
        checks["trajectory_schema_valid"] = trajectory["terminal_status"] == "accepted"
        serialized = json.dumps(trajectory, ensure_ascii=False)
        checks["trajectory_has_no_oracle_fields"] = all(
            token not in serialized for token in ('"expected_answer"', '"task_spec"')
        )

    adapter = _ScriptedAdapter(
        [
            AssistantAction("Inspect.", "inspect_table", {"head_rows": 1}),
            AssistantAction("Count.", "run_python", {"code": "len(df)"}),
            AssistantAction("Submit.", "submit_answer", {"answer": 3}),
        ]
    )
    with _fixture_episode(expected_answer=3, config=config) as episode:
        loop_trajectory = run_agent_episode(episode, adapter)
        checks["agent_loop_completes"] = (
            loop_trajectory.terminal_status == "accepted"
            and len(loop_trajectory.events) == 3
        )
        model_context = repr(adapter.contexts)
        checks["model_context_has_no_private_oracle"] = all(
            token not in model_context
            for token in ("expected_answer", "canonical_fingerprint", "task_spec")
        )

    with _fixture_episode(expected_answer=3, config=config) as episode:
        rejected = episode.call_tool("submit_answer", {"answer": 3.0})
        checks["strict_type_mismatch_rejected"] = (
            not rejected.ok and rejected.error_type == "VerificationError"
        )

    truncation_config = AgentEnvironmentConfig(max_output_chars=200)
    with _fixture_episode(config=truncation_config) as episode:
        truncated = episode.call_tool("run_python", {"code": "print('x' * 2000)"})
        checks["tool_output_truncated"] = (
            truncated.ok and truncated.truncated and len(truncated.content) == 200
        )

    passed = sum(checks.values())
    gpu_acceptance = audit_gpu_acceptance()
    environment_pass = passed == len(checks)
    a3_complete = environment_pass and gpu_acceptance["passed"]
    summary = {
        "schema_version": 1,
        "checks": checks,
        "checks_passed": passed,
        "checks_total": len(checks),
        "environment_engineering_pass": environment_pass,
        "platform": os.name,
        "posix_resource_limits_active": os.name == "posix"
        and checks["platform_resource_limits_reported"],
        "a2_source_tasks": len(records),
        "gpu_acceptance": gpu_acceptance,
        "qwen3_8b_50_task_acceptance": gpu_acceptance["status"],
        "a3_complete": a3_complete,
        "blocking_gate": (
            None if a3_complete else "qwen3_8b_50_task_end_to_end_acceptance"
        ),
    }
    output = ROOT / "artifacts" / "a3_environment_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["environment_engineering_pass"] or gpu_acceptance["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
