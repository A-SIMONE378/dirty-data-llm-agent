"""Run the frozen 50-task Qwen3-8B A3 GPU acceptance gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from dirty_da.agent_env import AgentEnvironmentConfig, AgentEpisode  # noqa: E402
from dirty_da.agent_loop import DEFAULT_SYSTEM_PROMPT, run_agent_episode  # noqa: E402
from dirty_da.qwen_adapter import QwenOutputError, QwenTransformersAdapter  # noqa: E402
from dirty_da.schema_semantics import load_table_semantics  # noqa: E402
from dirty_da.trajectory import validate_trajectory_record  # noqa: E402


MODEL_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " Return exactly one declared tool call per turn and no free-form final "
    "answer. Inspect the table before analysis. Perform calculations with "
    "run_python; when ready, call submit_answer. Match the requested output "
    "shape exactly: submit a scalar for a single aggregate or count, a boolean "
    "for a yes/no judgment, and a JSON object only when the question explicitly "
    "requests a grouped mapping. Never add a label around a scalar. /no_think"
)
FORBIDDEN_PRIVATE_FIELDS = (
    '"expected_answer"',
    '"task_spec"',
    '"canonical_fingerprint"',
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _hash_rank(seed: str, *parts: str) -> str:
    payload = "|".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_frozen_records(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Select fixed held-out tasks with every held-out table represented."""

    if config.get("schema_version") != 1:
        raise ValueError("Unsupported A3 acceptance config schema version.")
    seed = config["selection_seed"]
    selected: list[dict[str, Any]] = []
    for split, required_total in config["tasks_per_split"].items():
        per_table = config["tasks_per_table"][split]
        candidates = [record for record in records if record["split"] == split]
        by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in candidates:
            by_table[record["table"]].append(record)
        if len(by_table) * per_table != required_total:
            raise ValueError(
                f"{split}: {len(by_table)} tables * {per_table} does not equal "
                f"the required {required_total} tasks."
            )
        for table in sorted(by_table):
            table_records = by_table[table]
            by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in table_records:
                by_difficulty[record["difficulty"]].append(record)
            difficulty_order = sorted(
                by_difficulty,
                key=lambda value: _hash_rank(seed, split, table, value),
            )
            table_selected: list[dict[str, Any]] = []
            for difficulty in difficulty_order:
                ranked = sorted(
                    by_difficulty[difficulty],
                    key=lambda record: _hash_rank(seed, record["task_id"]),
                )
                table_selected.append(ranked[0])
                if len(table_selected) == per_table:
                    break
            if len(table_selected) < per_table:
                used = {record["task_id"] for record in table_selected}
                remaining = sorted(
                    (r for r in table_records if r["task_id"] not in used),
                    key=lambda record: _hash_rank(seed, record["task_id"]),
                )
                table_selected.extend(remaining[: per_table - len(table_selected)])
            if len(table_selected) != per_table:
                raise ValueError(f"Not enough frozen candidates for {split}/{table}.")
            selected.extend(table_selected)
    expected = sum(config["tasks_per_split"].values())
    if len(selected) != expected or len({r["task_id"] for r in selected}) != expected:
        raise ValueError("Frozen A3 selection is incomplete or contains duplicates.")
    return selected


def _manifest(
    selected: list[dict[str, Any]], source_path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_seed": config["selection_seed"],
        "prompt_version": config["prompt_version"],
        "system_prompt_sha256": hashlib.sha256(
            MODEL_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "source_catalog": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_catalog_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "task_count": len(selected),
        "tasks": [
            {
                "task_id": record["task_id"],
                "table": record["table"],
                "split": record["split"],
                "difficulty": record["difficulty"],
                "dirty_profile": record["dirty_profile"],
                "dirty_fingerprint": record["dirty_fingerprint"],
            }
            for record in selected
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("task_id"), str):
            raise ValueError(f"Invalid result record on line {line_number}.")
        results[value["task_id"]] = value
    return results


def _append_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _build_summary(
    *,
    results: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    model_id: str,
    resource_limits_active: bool,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = Counter(result["terminal_status"] for result in results.values())
    task_count = int(manifest["task_count"])
    completed = len(results)
    valid_trajectories = sum(bool(r["trajectory_schema_valid"]) for r in results.values())
    leakage_safe = all(bool(r["private_fields_absent"]) for r in results.values())
    model_errors = statuses.get("model_error", 0)
    accepted = statuses.get("accepted", 0)
    return {
        "schema_version": 1,
        "model_id": model_id,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "tasks_required": task_count,
        "tasks_attempted": completed,
        "terminal_status_counts": dict(sorted(statuses.items())),
        "accepted_accuracy": accepted / task_count if task_count else 0.0,
        "trajectory_schema_valid_count": valid_trajectories,
        "private_fields_absent": leakage_safe,
        "posix_resource_limits_active": resource_limits_active,
        "platform": platform.platform(),
        "runtime": dict(runtime),
        "run_complete": completed == task_count,
        "a3_gpu_acceptance_pass": (
            completed == task_count
            and valid_trajectories == task_count
            and leakage_safe
            and resource_limits_active
            and model_errors == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "a3_qwen_acceptance.json"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT adapter path loaded on top of --model.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "a3_gpu" / "qwen3_8b_50_v1",
    )
    parser.add_argument("--runtime-root", type=Path, default=ROOT / ".a3_runtime")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-model-errors", action="store_true")
    args = parser.parse_args()

    config = _load_json(args.config)
    source_path = ROOT / "data" / "tasks" / "a2_tasks.jsonl"
    records = load_task_records(source_path)
    selected = select_frozen_records(records, config)
    manifest = _manifest(selected, source_path, config)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise SystemExit("Existing output manifest differs from the frozen selection.")
    _write_json(manifest_path, manifest)

    results_path = args.output_dir / "results.jsonl"
    prior_results = _load_results(results_path)
    if prior_results and not args.resume:
        raise SystemExit("Results already exist; use --resume to continue safely.")

    semantics = load_table_semantics(ROOT / "configs" / "table_semantics.json")
    profiles = load_corruption_profiles(ROOT / "configs" / "a1_corruption_profiles.json")
    env_config = AgentEnvironmentConfig.from_json(ROOT / "configs" / "a3_agent_env.json")

    with AgentEpisode.from_task_record(
        selected[0],
        tables_dir=ROOT / "data" / "tables",
        semantics_registry=semantics,
        profile_config=profiles,
        config=env_config,
        runtime_root=args.runtime_root,
    ) as preflight:
        preflight_result = preflight.call_tool("run_python", {"code": "len(df)"})
        resource_limits_active = bool(
            preflight_result.ok and preflight.sandbox_resource_limits_active
        )
    if os.name != "posix" or not resource_limits_active:
        raise SystemExit("Verified POSIX sandbox resource limits are required.")

    model_path = args.model or config["model_id"]
    adapter = QwenTransformersAdapter(
        model_path,
        adapter_name_or_path=args.adapter,
        cache_dir=args.cache_dir,
        max_new_tokens=config["max_new_tokens"],
        max_context_tokens=config["max_context_tokens"],
        trust_remote_code=config["trust_remote_code"],
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
        "adapter_path": args.adapter,
    }
    pending: list[dict[str, Any]] = []
    for record in selected:
        previous = prior_results.get(record["task_id"])
        if previous is not None and not (
            args.retry_model_errors and previous["terminal_status"] == "model_error"
        ):
            continue
        pending.append(record)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive.")
        pending = pending[: args.limit]

    for index, record in enumerate(pending, 1):
        started = time.monotonic()
        error_type = None
        error_message = None
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
                    episode, adapter, system_prompt=MODEL_SYSTEM_PROMPT
                )
            except QwenOutputError as exc:
                # Inference completed, but the model violated the declared action
                # protocol.  This is an auditable capability failure, not an
                # infrastructure/runtime failure.
                episode.trajectory.terminal_status = "invalid_action"
                trajectory = episode.trajectory
                error_type = type(exc).__name__
                error_message = str(exc)[:500]
            except Exception as exc:  # persist genuine runtime failures for audit
                episode.trajectory.terminal_status = "model_error"
                trajectory = episode.trajectory
                error_type = type(exc).__name__
                error_message = str(exc)[:500]
            trajectory_record = trajectory.to_dict()
            trajectory_schema_valid = True
            try:
                validate_trajectory_record(trajectory_record)
            except (TypeError, ValueError):
                trajectory_schema_valid = False
            serialized = json.dumps(trajectory_record, ensure_ascii=False)
            private_fields_absent = all(
                token not in serialized for token in FORBIDDEN_PRIVATE_FIELDS
            )
            trajectory_path = args.output_dir / "trajectories" / f"{record['task_id']}.json"
            _write_json(trajectory_path, trajectory_record)
            result = {
                "task_id": record["task_id"],
                "table": record["table"],
                "split": record["split"],
                "difficulty": record["difficulty"],
                "terminal_status": trajectory.terminal_status,
                "steps": len(trajectory.events),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "trajectory_schema_valid": trajectory_schema_valid,
                "private_fields_absent": private_fields_absent,
                "error_type": error_type,
                "error_message": error_message,
            }
            _append_result(results_path, result)
            prior_results[record["task_id"]] = result
        summary = _build_summary(
            results=prior_results,
            manifest=manifest,
            model_id=config["model_id"],
            resource_limits_active=resource_limits_active,
            runtime=runtime,
        )
        _write_json(args.output_dir / "summary.json", summary)
        print(
            f"[{index}/{len(pending)}] {record['task_id']}: "
            f"{result['terminal_status']} ({result['elapsed_seconds']}s)",
            flush=True,
        )

    summary = _build_summary(
        results=prior_results,
        manifest=manifest,
        model_id=config["model_id"],
        resource_limits_active=resource_limits_active,
        runtime=runtime,
    )
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["run_complete"] and not summary["a3_gpu_acceptance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
