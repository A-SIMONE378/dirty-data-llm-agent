"""Run B3 SFT-model rollouts and persist per-task pass-rate evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dirty_da.a1_corruption import load_corruption_profiles  # noqa: E402
from dirty_da.a2_tasks import load_task_records  # noqa: E402
from dirty_da.agent_env import AgentEnvironmentConfig, AgentEpisode  # noqa: E402
from dirty_da.agent_loop import run_agent_episode  # noqa: E402
from dirty_da.qwen_adapter import QwenOutputError, QwenTransformersAdapter  # noqa: E402
from dirty_da.schema_semantics import load_table_semantics  # noqa: E402
from run_a3_qwen_acceptance import MODEL_SYSTEM_PROMPT  # noqa: E402


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_results(path: Path) -> dict[tuple[str, int], dict]:
    results: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return results
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid B3 result on line {line_number}.")
        key = (value.get("task_id"), value.get("sample_index"))
        if not isinstance(key[0], str) or not isinstance(key[1], int):
            raise ValueError(f"Invalid B3 result key on line {line_number}.")
        results[key] = value
    return results


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b3_rollout.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=ROOT / ".b3_runtime")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("Invalid B3 shard index/count.")
    config = _load_json(args.config)
    samples = int(config["samples_per_task"])
    if samples < 2:
        raise SystemExit("B3 needs at least two samples per task.")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise SystemExit("--max-tasks must be positive.")

    model_path = args.model or str(config["model_id"])
    adapter_path = args.adapter or str(config["adapter_path"])
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "artifacts" / "b3_rollouts"
        if args.shard_count > 1:
            output_dir = output_dir / f"shard{args.shard_index}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((ROOT / config["task_index"]).read_text(encoding="utf-8"))
    if not isinstance(index, list) or not index:
        raise SystemExit("B3 task index is empty.")
    selected = [
        row for position, row in enumerate(index)
        if position % args.shard_count == args.shard_index
    ]
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]
    records = {record["task_id"]: record for record in load_task_records(ROOT / config["source_tasks"])}
    missing = [row["task_id"] for row in selected if row["task_id"] not in records]
    if missing:
        raise SystemExit(f"B3 task index references unknown tasks: {missing[:5]}")
    if any(records[row["task_id"]]["split"] != "train" for row in selected):
        raise SystemExit("B3 selected task index contains non-train tasks.")

    semantics = load_table_semantics(ROOT / "configs" / "table_semantics.json")
    profiles = load_corruption_profiles(ROOT / "configs" / "a1_corruption_profiles.json")
    env_config = AgentEnvironmentConfig.from_json(ROOT / "configs" / "a3_agent_env.json")
    env_config = replace(
        env_config,
        max_steps=int(config["max_steps"]),
        run_timeout_seconds=15.0,
        max_memory_mb=8192,
        max_cpu_seconds=120,
    )
    adapter = QwenTransformersAdapter(
        model_path,
        adapter_name_or_path=adapter_path,
        cache_dir=args.cache_dir,
        max_new_tokens=int(config["max_new_tokens"]),
        max_context_tokens=int(config["max_context_tokens"]),
        do_sample=True,
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
    )

    results_path = output_dir / "results.jsonl"
    trajectories_dir = output_dir / "trajectories"
    prior = _load_results(results_path) if args.resume else {}
    completed = 0
    started = time.monotonic()
    for task_position, row in enumerate(selected, 1):
        task_id = row["task_id"]
        record = records[task_id]
        for sample_index in range(samples):
            if (task_id, sample_index) in prior:
                continue
            digest = hashlib.sha256(
                f"{config['selection_seed']}|{task_id}|{sample_index}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(digest[:8], "big") % (2**31)
            attempt_started = time.monotonic()
            adapter.reset_sampling_seed(seed)
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
            trajectory_path = trajectories_dir / f"{task_id}-s{sample_index:02d}.json"
            _write_json(trajectory_path, trajectory_record)
            result = {
                "task_id": task_id,
                "sample_index": sample_index,
                "sampling_seed": seed,
                "terminal_status": trajectory.terminal_status,
                "accepted": trajectory.terminal_status == "accepted",
                "steps": len(trajectory.events),
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "error_type": error_type,
                "error_message": error_message,
            }
            _append_jsonl(results_path, result)
            prior[(task_id, sample_index)] = result
            completed += 1
            print(
                f"[{task_position}/{len(selected)}] {task_id} s{sample_index:02d}: "
                f"{trajectory.terminal_status} ({result['elapsed_seconds']}s)",
                flush=True,
            )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tasks": len(selected),
                "samples_per_task": samples,
                "new_rollouts": completed,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
