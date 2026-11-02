"""Merge B3 shard rollouts and apply the GRPO pass-rate filter."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid result on line {line_number} of {path}.")
        rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "b3_rollouts")
    parser.add_argument("--task-index", type=Path, default=ROOT / "artifacts" / "b3_tasks" / "index.json")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--pass-rate-min", type=float, default=0.01)
    parser.add_argument("--pass-rate-max", type=float, default=0.99)
    args = parser.parse_args()
    if args.shards < 1 or args.samples_per_task < 2:
        raise SystemExit("Invalid shard or sample count.")

    task_rows = json.loads(args.task_index.read_text(encoding="utf-8"))
    if not isinstance(task_rows, list) or not task_rows:
        raise SystemExit("Empty B3 task index.")
    expected = {row["task_id"]: row for row in task_rows}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for shard in range(args.shards):
        path = args.output_dir / f"shard{shard}" / "results.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing B3 shard results: {path}")
        for row in _read_jsonl(path):
            by_task[row["task_id"]].append(row)

    if set(by_task) != set(expected):
        missing = sorted(set(expected) - set(by_task))
        extra = sorted(set(by_task) - set(expected))
        raise SystemExit(f"B3 task coverage mismatch; missing={missing[:5]} extra={extra[:5]}")

    tasks = []
    for task_id in sorted(expected):
        rows = by_task[task_id]
        if len(rows) != args.samples_per_task:
            raise SystemExit(
                f"{task_id}: expected {args.samples_per_task} rollouts, found {len(rows)}"
            )
        samples = sorted(rows, key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in samples] != list(
            range(args.samples_per_task)
        ):
            raise SystemExit(f"{task_id}: sample indices are incomplete.")
        accepted = sum(bool(row.get("accepted")) for row in samples)
        pass_rate = accepted / len(samples)
        keep = args.pass_rate_min < pass_rate < args.pass_rate_max
        tasks.append(
            {
                **expected[task_id],
                "samples": len(samples),
                "accepted": accepted,
                "pass_rate": pass_rate,
                "keep": keep,
                "terminal_status_counts": dict(
                    sorted(Counter(row["terminal_status"] for row in samples).items())
                ),
            }
        )

    filtered = [row for row in tasks if row["keep"]]
    excluded_zero = sum(row["accepted"] == 0 for row in tasks)
    excluded_full = sum(row["accepted"] == args.samples_per_task for row in tasks)
    summary = {
        "schema_version": 1,
        "task_count": len(tasks),
        "samples_per_task": args.samples_per_task,
        "pass_rate_min": args.pass_rate_min,
        "pass_rate_max": args.pass_rate_max,
        "kept_task_count": len(filtered),
        "excluded_zero_pass": excluded_zero,
        "excluded_full_pass": excluded_full,
        "kept_table_counts": dict(sorted(Counter(row["table"] for row in filtered).items())),
        "kept_difficulty_counts": dict(
            sorted(Counter(row["difficulty"] for row in filtered).items())
        ),
        "mean_pass_rate_all": sum(row["pass_rate"] for row in tasks) / len(tasks),
        "mean_pass_rate_kept": (
            sum(row["pass_rate"] for row in filtered) / len(filtered) if filtered else None
        ),
        "b3_filter_pass": len(filtered) > 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "filtered_tasks.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["b3_filter_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
