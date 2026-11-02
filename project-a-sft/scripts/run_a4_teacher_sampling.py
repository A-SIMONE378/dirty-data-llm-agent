"""Sample, verify, filter, and persist A4 Qwen3-8B teacher trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a1_corruption import load_corruption_profiles  # noqa: E402
from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.a4_teacher import (  # noqa: E402
    QualityAssessment,
    assess_teacher_trajectory,
    build_sampling_attempts,
    build_sft_record,
    select_teacher_tasks,
)
from dirty_da.agent_env import AgentEnvironmentConfig, AgentEpisode  # noqa: E402
from dirty_da.agent_loop import DEFAULT_SYSTEM_PROMPT, run_agent_episode  # noqa: E402
from dirty_da.qwen_adapter import QwenOutputError, QwenTransformersAdapter  # noqa: E402
from dirty_da.schema_semantics import load_table_semantics  # noqa: E402
from dirty_da.trajectory import validate_trajectory_record  # noqa: E402


TEACHER_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " Return exactly one declared tool call per turn and no free-form final "
    "answer. Inspect the table before analysis. Use run_python to derive the "
    "answer from df, ensure the computed value appears in its returned result, "
    "then submit it with a separate submit_answer call. Never call submit_answer "
    "inside run_python. Follow every explicit missing, sentinel, deduplication, "
    "type, format, and category rule. If a tool fails, inspect the error and "
    "recover rather than guessing. Match the output shape exactly: scalar for one "
    "aggregate or count, boolean for yes/no, and an object only for a requested "
    "grouped mapping. Never hardcode or guess an answer. /no_think"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("attempt_id"), str):
            raise ValueError(f"Invalid A4 result on line {line_number}.")
        results[value["attempt_id"]] = value
    return results


def _manifest(
    *,
    selected: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    source_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_seed": config["selection_seed"],
        "prompt_version": config["prompt_version"],
        "system_prompt_sha256": hashlib.sha256(
            TEACHER_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "source_catalog": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_catalog_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "selected_task_count": len(selected),
        "samples_per_task": config["samples_per_task"],
        "generation": {
            "model_id": config["model_id"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_new_tokens": config["max_new_tokens"],
            "max_context_tokens": config["max_context_tokens"],
            "max_steps": config["max_steps"],
            "invalid_action_retries": config["invalid_action_retries"],
        },
        "planned_attempt_count": len(attempts),
        "tasks": [
            {
                "task_id": record["task_id"],
                "table": record["table"],
                "difficulty": record["difficulty"],
                "dirty_profile": record["dirty_profile"],
                "dirty_fingerprint": record["dirty_fingerprint"],
            }
            for record in selected
        ],
        "attempts": [
            {
                "attempt_id": attempt["attempt_id"],
                "task_id": attempt["task_id"],
                "sample_index": attempt["sample_index"],
                "sampling_seed": attempt["sampling_seed"],
            }
            for attempt in attempts
        ],
    }


def _assessment_from_result(result: Mapping[str, Any]) -> QualityAssessment:
    return QualityAssessment(
        retained=bool(result["retained"]),
        category=result.get("category"),
        reasons=tuple(result.get("quality_reasons", [])),
        uses_tool_feedback=bool(result.get("uses_tool_feedback")),
        trajectory_fingerprint=str(result["trajectory_fingerprint"]),
    )


def _rebuild_sft_dataset(
    *,
    output_dir: Path,
    attempts_by_id: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    tools: list[dict[str, Any]],
    prompt_version: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attempt_id in sorted(results):
        result = results[attempt_id]
        if not result.get("retained"):
            continue
        trajectory = _load_json(
            output_dir / "trajectories" / f"{attempt_id}.json"
        )
        attempt = dict(attempts_by_id[attempt_id])
        attempt["sampling_seed"] = int(
            result.get("sampling_seed_used", attempt["sampling_seed"])
        )
        records.append(
            build_sft_record(
                attempt=attempt,
                trajectory=trajectory,
                assessment=_assessment_from_result(result),
                system_prompt=TEACHER_SYSTEM_PROMPT,
                tools=tools,
                prompt_version=prompt_version,
            )
        )
    destination = output_dir / "train_sft.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return records


def _write_human_review_sample(
    output_dir: Path,
    sft_records: list[dict[str, Any]],
    sample_size: int,
    selection_seed: str,
) -> None:
    if len(sft_records) < sample_size:
        return
    ranked = sorted(
        sft_records,
        key=lambda record: hashlib.sha256(
            f"{selection_seed}|review|{record['trajectory_id']}".encode("utf-8")
        ).hexdigest(),
    )[:sample_size]
    jsonl_path = output_dir / "human_review_sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ranked:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    csv_path = output_dir / "human_review_sample.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trajectory_id",
                "task_id",
                "table",
                "category",
                "message_count",
                "no_leakage_pass",
                "no_hardcoding_pass",
                "tool_feedback_used_pass",
                "overall_verdict",
                "review_notes",
            ],
        )
        writer.writeheader()
        for record in ranked:
            writer.writerow(
                {
                    "trajectory_id": record["trajectory_id"],
                    "task_id": record["task_id"],
                    "table": record["table"],
                    "category": record["category"],
                    "message_count": len(record["messages"]),
                    "no_leakage_pass": "",
                    "no_hardcoding_pass": "",
                    "tool_feedback_used_pass": "",
                    "overall_verdict": "",
                    "review_notes": "",
                }
            )


def _summary(
    *,
    results: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = Counter(result["terminal_status"] for result in results.values())
    retained = [result for result in results.values() if result.get("retained")]
    categories = Counter(result["category"] for result in retained)
    rejection_reasons = Counter(
        reason
        for result in results.values()
        for reason in result.get("quality_reasons", [])
    )
    completed = len(results)
    planned = int(manifest["planned_attempt_count"])
    minimum = int(config["minimum_retained_trajectories"])
    maximum = int(config["maximum_retained_trajectories"])
    retained_count = len(retained)
    return {
        "schema_version": 1,
        "model_id": config["model_id"],
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "planned_attempts": planned,
        "completed_attempts": completed,
        "run_complete": completed == planned,
        "terminal_status_counts": dict(sorted(statuses.items())),
        "verified_answer_pass_rate": (
            statuses.get("accepted", 0) / completed if completed else 0.0
        ),
        "retained_trajectory_count": retained_count,
        "retained_rate": retained_count / completed if completed else 0.0,
        "retained_category_counts": dict(sorted(categories.items())),
        "mean_retained_steps": (
            sum(result["steps"] for result in retained) / retained_count
            if retained_count
            else None
        ),
        "quality_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "invalid_action_retries_used": sum(
            int(result.get("invalid_action_retries", 0))
            for result in results.values()
        ),
        "human_review_required": int(config["human_review_sample_size"]),
        "human_review_status": "PENDING",
        "runtime": dict(runtime),
        "platform": platform.platform(),
        "mechanical_target_met": minimum <= retained_count <= maximum,
        "a4_complete": (
            completed == planned
            and minimum <= retained_count <= maximum
            and statuses.get("model_error", 0) == 0
            and False  # independent human review must be finalized separately
        ),
        "blocking_gate": (
            "sampling_incomplete"
            if completed != planned
            else "retained_trajectory_target"
            if not minimum <= retained_count <= maximum
            else "human_review_50"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "a4_teacher_sampling.json"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "a4_teacher" / "qwen3_8b_train_v5",
    )
    parser.add_argument("--runtime-root", type=Path, default=ROOT / ".a4_runtime")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-model-errors", action="store_true")
    parser.add_argument("--retry-attempts-file", type=Path, default=None)
    parser.add_argument("--refilter-existing", action="store_true")
    args = parser.parse_args()

    config = _load_json(args.config)
    if (
        isinstance(config.get("invalid_action_retries"), bool)
        or not isinstance(config.get("invalid_action_retries"), int)
        or config["invalid_action_retries"] < 0
    ):
        raise SystemExit("invalid_action_retries must be a non-negative integer.")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count).")
    source_path = ROOT / "data" / "tasks" / "a2_tasks.jsonl"
    task_records = load_task_records(source_path)
    selected = select_teacher_tasks(task_records, config)
    all_attempts = build_sampling_attempts(selected, config)
    attempts = [
        attempt
        for index, attempt in enumerate(all_attempts)
        if index % args.shard_count == args.shard_index
    ]
    attempts_by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    manifest = _manifest(
        selected=selected,
        attempts=all_attempts,
        source_path=source_path,
        config=config,
    )
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise SystemExit("Existing A4 manifest differs from the frozen sampling plan.")
    _write_json(manifest_path, manifest)

    results_path = args.output_dir / "results.jsonl"
    prior_results = _load_latest_results(results_path)
    if prior_results and not args.resume:
        raise SystemExit("A4 results already exist; use --resume.")
    retry_ids: set[str] = set()
    if args.retry_attempts_file is not None:
        payload = _load_json(args.retry_attempts_file)
        attempt_ids = payload.get("attempt_ids")
        if not isinstance(attempt_ids, list) or not all(
            isinstance(attempt_id, str) for attempt_id in attempt_ids
        ):
            raise SystemExit("retry-attempts-file must contain a string attempt_ids list.")
        retry_ids = set(attempt_ids)
        unknown = retry_ids - {attempt["attempt_id"] for attempt in all_attempts}
        if unknown:
            raise SystemExit(f"Unknown retry attempts: {sorted(unknown)[:5]}")

    semantics = load_table_semantics(ROOT / "configs" / "table_semantics.json")
    profiles = load_corruption_profiles(ROOT / "configs" / "a1_corruption_profiles.json")
    env_config = AgentEnvironmentConfig.from_json(ROOT / "configs" / "a3_agent_env.json")
    env_config = replace(
        env_config,
        max_steps=int(config["max_steps"]),
        run_timeout_seconds=float(config["run_timeout_seconds"]),
        max_memory_mb=int(config["max_memory_mb"]),
        max_cpu_seconds=int(config["max_cpu_seconds"]),
    )
    with AgentEpisode.from_task_record(
        selected[0],
        tables_dir=ROOT / "data" / "tables",
        semantics_registry=semantics,
        profile_config=profiles,
        config=env_config,
        runtime_root=args.runtime_root,
    ) as preflight:
        preflight_result = preflight.call_tool("run_python", {"code": "len(df)"})
        if os.name != "posix" or not (
            preflight_result.ok and preflight.sandbox_resource_limits_active
        ):
            raise SystemExit("Verified POSIX sandbox resource limits are required.")
        public_tools = preflight.tool_schemas()
        posix_limits_active = preflight.sandbox_resource_limits_active

    model_path = args.model or config["model_id"]
    adapter = QwenTransformersAdapter(
        model_path,
        cache_dir=args.cache_dir,
        max_new_tokens=config["max_new_tokens"],
        max_context_tokens=config["max_context_tokens"],
        trust_remote_code=config["trust_remote_code"],
        do_sample=True,
        temperature=config["temperature"],
        top_p=config["top_p"],
    )
    import torch
    import transformers

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model_path": model_path,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "posix_resource_limits_active": posix_limits_active,
        "sandbox_memory_mb": int(config["max_memory_mb"]),
        "sandbox_timeout_seconds": float(config["run_timeout_seconds"]),
        "sandbox_cpu_seconds": int(config["max_cpu_seconds"]),
    }
    if args.refilter_existing:
        refiltered_fingerprints: set[str] = set()
        for attempt in attempts:
            previous = prior_results.get(attempt["attempt_id"])
            if previous is None:
                continue
            trajectory_record = _load_json(
                args.output_dir
                / "trajectories"
                / f"{attempt['attempt_id']}.json"
            )
            assessment = assess_teacher_trajectory(
                attempt["record"], trajectory_record
            )
            if assessment.retained and (
                assessment.trajectory_fingerprint in refiltered_fingerprints
            ):
                assessment = QualityAssessment(
                    retained=False,
                    category=None,
                    reasons=("duplicate_trajectory",),
                    uses_tool_feedback=assessment.uses_tool_feedback,
                    trajectory_fingerprint=assessment.trajectory_fingerprint,
                )
            if assessment.retained:
                refiltered_fingerprints.add(assessment.trajectory_fingerprint)
            updated = dict(previous)
            updated.update(
                retained=assessment.retained,
                category=assessment.category,
                quality_reasons=list(assessment.reasons),
                uses_tool_feedback=assessment.uses_tool_feedback,
                trajectory_fingerprint=assessment.trajectory_fingerprint,
            )
            _append_jsonl(results_path, updated)
            prior_results[attempt["attempt_id"]] = updated
    pending = []
    for attempt in attempts:
        previous = prior_results.get(attempt["attempt_id"])
        if previous is not None and not (
            attempt["attempt_id"] in retry_ids
            or (
                args.retry_model_errors
                and previous["terminal_status"] == "model_error"
            )
        ):
            continue
        pending.append(attempt)
    if args.max_attempts is not None:
        if args.max_attempts < 1:
            raise SystemExit("--max-attempts must be positive.")
        pending = pending[: args.max_attempts]

    retained_fingerprints = {
        str(result["trajectory_fingerprint"])
        for result in prior_results.values()
        if result.get("retained")
    }
    for index, attempt in enumerate(pending, 1):
        record = attempt["record"]
        started = time.monotonic()
        retries = int(config["invalid_action_retries"])
        sampling_seeds: list[int] = []
        trajectory = None
        error_type = None
        error_message = None
        for retry_index in range(retries + 1):
            sampling_seed = attempt["sampling_seed"]
            if retry_index:
                sampling_seed = (
                    attempt["sampling_seed"] + retry_index * 1_000_003
                ) % (2**31)
            sampling_seeds.append(sampling_seed)
            error_type = None
            error_message = None
            adapter.reset_sampling_seed(sampling_seed)
            with AgentEpisode.from_task_record(
                record,
                tables_dir=ROOT / "data" / "tables",
                semantics_registry=semantics,
                profile_config=profiles,
                config=env_config,
                runtime_root=args.runtime_root,
            ) as episode:
                try:
                    trajectory = run_agent_episode(
                        episode, adapter, system_prompt=TEACHER_SYSTEM_PROMPT
                    )
                except QwenOutputError as exc:
                    episode.trajectory.terminal_status = "invalid_action"
                    trajectory = episode.trajectory
                    error_type = type(exc).__name__
                    error_message = str(exc)[:500]
                except Exception as exc:
                    episode.trajectory.terminal_status = "model_error"
                    trajectory = episode.trajectory
                    error_type = type(exc).__name__
                    error_message = str(exc)[:500]
                trajectory_record = trajectory.to_dict()
            if error_type != "QwenOutputError":
                break

        assessment = assess_teacher_trajectory(record, trajectory_record)
        if assessment.retained and (
            assessment.trajectory_fingerprint in retained_fingerprints
        ):
            assessment = QualityAssessment(
                retained=False,
                category=None,
                reasons=("duplicate_trajectory",),
                uses_tool_feedback=assessment.uses_tool_feedback,
                trajectory_fingerprint=assessment.trajectory_fingerprint,
            )
        if assessment.retained:
            retained_fingerprints.add(assessment.trajectory_fingerprint)
        trajectory_path = (
            args.output_dir / "trajectories" / f"{attempt['attempt_id']}.json"
        )
        _write_json(trajectory_path, trajectory_record)
        result = {
            "attempt_id": attempt["attempt_id"],
            "task_id": attempt["task_id"],
            "table": record["table"],
            "difficulty": record["difficulty"],
            "dirty_profile": record["dirty_profile"],
            "sample_index": attempt["sample_index"],
            "sampling_seed": attempt["sampling_seed"],
            "sampling_seed_used": sampling_seeds[-1],
            "invalid_action_retries": len(sampling_seeds) - 1,
            "terminal_status": trajectory.terminal_status,
            "steps": len(trajectory.events),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "retained": assessment.retained,
            "category": assessment.category,
            "quality_reasons": list(assessment.reasons),
            "uses_tool_feedback": assessment.uses_tool_feedback,
            "trajectory_fingerprint": assessment.trajectory_fingerprint,
            "error_type": error_type,
            "error_message": error_message,
        }
        _append_jsonl(results_path, result)
        prior_results[attempt["attempt_id"]] = result
        summary = _summary(
            results=prior_results,
            manifest=manifest,
            config=config,
            runtime=runtime,
        )
        _write_json(args.output_dir / "summary.json", summary)
        print(
            f"[{index}/{len(pending)}] {attempt['attempt_id']}: "
            f"{trajectory.terminal_status}, retained={assessment.retained} "
            f"({result['elapsed_seconds']}s)",
            flush=True,
        )

    sft_records = _rebuild_sft_dataset(
        output_dir=args.output_dir,
        attempts_by_id=attempts_by_id,
        results=prior_results,
        tools=public_tools,
        prompt_version=config["prompt_version"],
    )
    _write_human_review_sample(
        args.output_dir,
        sft_records,
        int(config["human_review_sample_size"]),
        str(config["selection_seed"]),
    )
    summary = _summary(
        results=prior_results,
        manifest=manifest,
        config=config,
        runtime=runtime,
    )
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

