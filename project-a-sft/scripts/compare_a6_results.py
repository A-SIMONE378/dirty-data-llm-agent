"""Paired A6 test/domain comparison with task-level bootstrap confidence intervals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
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


def _bootstrap_ci(
    base: Mapping[str, bool],
    sft: Mapping[str, bool],
    task_ids: list[str],
    *,
    iterations: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    rng = random.Random(seed)
    base_accuracy: list[float] = []
    sft_accuracy: list[float] = []
    difference: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(task_ids) for _ in task_ids]
        base_score = sum(base[task_id] for task_id in sample) / len(sample)
        sft_score = sum(sft[task_id] for task_id in sample) / len(sample)
        base_accuracy.append(base_score)
        sft_accuracy.append(sft_score)
        difference.append(sft_score - base_score)

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]

    return interval(base_accuracy), interval(sft_accuracy), interval(difference)


def compare_test_domain(
    base_dir: Path,
    sft_dir: Path,
    *,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20261026,
) -> dict[str, Any]:
    base_manifest = _load_json(base_dir / "manifest.json")
    sft_manifest = _load_json(sft_dir / "manifest.json")
    if base_manifest != sft_manifest:
        raise ValueError("Base and SFT manifests differ; paired comparison is invalid.")
    tasks = base_manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A6 manifest has no tasks.")
    splits = {task.get("split") for task in tasks if isinstance(task, Mapping)}
    if splits != {"test", "domain_generalization"}:
        raise ValueError("A6 comparison requires exactly test and domain_generalization.")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) for task_id in task_ids):
        raise ValueError("A6 manifest contains an invalid task ID.")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("A6 manifest contains duplicate task IDs.")

    base = _load_results(base_dir / "results.jsonl")
    sft = _load_results(sft_dir / "results.jsonl")
    expected = set(task_ids)
    if set(base) != expected or set(sft) != expected:
        raise ValueError("Both runs must contain exactly the frozen A6 tasks.")

    invalid_runtime: list[str] = []
    for label, results in (("base", base), ("sft", sft)):
        for task_id, result in results.items():
            if result.get("terminal_status") == "model_error":
                invalid_runtime.append(f"{label}:{task_id}:model_error")
            if result.get("trajectory_schema_valid") is not True:
                invalid_runtime.append(f"{label}:{task_id}:invalid_schema")
            if result.get("private_fields_absent") is not True:
                invalid_runtime.append(f"{label}:{task_id}:private_field_leak")

    base_accepted = {
        task_id: base[task_id]["terminal_status"] == "accepted" for task_id in task_ids
    }
    sft_accepted = {
        task_id: sft[task_id]["terminal_status"] == "accepted" for task_id in task_ids
    }
    base_count = sum(base_accepted.values())
    sft_count = sum(sft_accepted.values())
    total = len(task_ids)
    base_ci, sft_ci, difference_ci = _bootstrap_ci(
        base_accepted,
        sft_accepted,
        task_ids,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    by_split: dict[str, Any] = {}
    for split in ("test", "domain_generalization"):
        split_ids = [
            task["task_id"] for task in tasks if task.get("split") == split
        ]
        base_split = sum(base_accepted[task_id] for task_id in split_ids)
        sft_split = sum(sft_accepted[task_id] for task_id in split_ids)
        by_split[split] = {
            "task_count": len(split_ids),
            "base_accepted": base_split,
            "base_accuracy": base_split / len(split_ids),
            "sft_accepted": sft_split,
            "sft_accuracy": sft_split / len(split_ids),
            "net_accepted_gain": sft_split - base_split,
            "base_status_counts": dict(
                sorted(Counter(base[task_id]["terminal_status"] for task_id in split_ids).items())
            ),
            "sft_status_counts": dict(
                sorted(Counter(sft[task_id]["terminal_status"] for task_id in split_ids).items())
            ),
        }

    return {
        "schema_version": 1,
        "task_count": total,
        "splits": sorted(splits),
        "base_accepted": base_count,
        "base_accuracy": base_count / total,
        "sft_accepted": sft_count,
        "sft_accuracy": sft_count / total,
        "net_accepted_gain": sft_count - base_count,
        "recovered_task_ids": sorted(
            task_id for task_id in task_ids if sft_accepted[task_id] and not base_accepted[task_id]
        ),
        "regressed_task_ids": sorted(
            task_id for task_id in task_ids if base_accepted[task_id] and not sft_accepted[task_id]
        ),
        "base_accuracy_bootstrap_95": base_ci,
        "sft_accuracy_bootstrap_95": sft_ci,
        "accuracy_difference_bootstrap_95": difference_ci,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "by_split": by_split,
        "runtime_integrity_errors": invalid_runtime,
        "a6_integrity_pass": not invalid_runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("sft_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261026)
    args = parser.parse_args()
    comparison = compare_test_domain(
        args.base_dir,
        args.sft_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not comparison["a6_integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
