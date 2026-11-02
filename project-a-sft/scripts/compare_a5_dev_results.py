"""Compare paired base and LoRA runs on the frozen A5 development tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        task_id = value.get("task_id") if isinstance(value, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Invalid result on line {line_number} of {path}.")
        if task_id in results:
            raise ValueError(f"Duplicate result for {task_id} in {path}.")
        results[task_id] = value
    return results


def compare_dev_runs(
    base_dir: Path,
    sft_dir: Path,
    *,
    minimum_net_gain: int,
) -> dict[str, Any]:
    if minimum_net_gain < 1:
        raise ValueError("minimum_net_gain must be positive.")
    base_manifest = _load_json(base_dir / "manifest.json")
    sft_manifest = _load_json(sft_dir / "manifest.json")
    if base_manifest != sft_manifest:
        raise ValueError("Base and SFT manifests differ; paired comparison is invalid.")
    tasks = base_manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Evaluation manifest has no tasks.")
    if {task.get("split") for task in tasks if isinstance(task, Mapping)} != {"dev"}:
        raise ValueError("A5 comparison is restricted to the development split.")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) for task_id in task_ids):
        raise ValueError("Evaluation manifest contains an invalid task ID.")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Evaluation manifest contains duplicate task IDs.")

    base = _load_results(base_dir / "results.jsonl")
    sft = _load_results(sft_dir / "results.jsonl")
    expected = set(task_ids)
    if set(base) != expected or set(sft) != expected:
        raise ValueError("Both runs must contain exactly the frozen development tasks.")

    invalid_runtime = []
    for label, results in (("base", base), ("sft", sft)):
        for task_id, result in results.items():
            if result.get("terminal_status") == "model_error":
                invalid_runtime.append(f"{label}:{task_id}:model_error")
            if result.get("trajectory_schema_valid") is not True:
                invalid_runtime.append(f"{label}:{task_id}:invalid_schema")
            if result.get("private_fields_absent") is not True:
                invalid_runtime.append(f"{label}:{task_id}:private_field_leak")

    base_accepted = {task_id for task_id, r in base.items() if r["terminal_status"] == "accepted"}
    sft_accepted = {task_id for task_id, r in sft.items() if r["terminal_status"] == "accepted"}
    recovered = sorted(sft_accepted - base_accepted)
    regressed = sorted(base_accepted - sft_accepted)
    net_gain = len(sft_accepted) - len(base_accepted)
    total = len(task_ids)
    return {
        "schema_version": 1,
        "task_count": total,
        "split": "dev",
        "base_accepted": len(base_accepted),
        "base_accuracy": len(base_accepted) / total,
        "sft_accepted": len(sft_accepted),
        "sft_accuracy": len(sft_accepted) / total,
        "net_accepted_gain": net_gain,
        "minimum_net_accepted_gain": minimum_net_gain,
        "recovered_task_ids": recovered,
        "regressed_task_ids": regressed,
        "runtime_integrity_errors": invalid_runtime,
        "a5_dev_gate_pass": not invalid_runtime and net_gain >= minimum_net_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("sft_dir", type=Path)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "a5_dev_eval.json"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = _load_json(args.config)
    comparison = compare_dev_runs(
        args.base_dir,
        args.sft_dir,
        minimum_net_gain=int(config["minimum_net_accepted_gain"]),
    )
    rendered = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not comparison["a5_dev_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
