import json
from pathlib import Path

import pytest

from scripts.compare_a5_dev_results import compare_dev_runs


def _write_run(path: Path, statuses: list[str], *, split: str = "dev") -> None:
    path.mkdir()
    tasks = [
        {"task_id": f"task-{index}", "split": split}
        for index in range(len(statuses))
    ]
    (path / "manifest.json").write_text(
        json.dumps({"tasks": tasks}), encoding="utf-8"
    )
    lines = [
        json.dumps(
            {
                "task_id": f"task-{index}",
                "terminal_status": status,
                "trajectory_schema_valid": True,
                "private_fields_absent": True,
            }
        )
        for index, status in enumerate(statuses)
    ]
    (path / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_paired_dev_comparison_requires_material_net_gain(tmp_path):
    base = tmp_path / "base"
    sft = tmp_path / "sft"
    _write_run(base, ["accepted", "rejected", "rejected", "rejected", "rejected"])
    _write_run(sft, ["accepted", "accepted", "accepted", "accepted", "rejected"])
    comparison = compare_dev_runs(base, sft, minimum_net_gain=3)
    assert comparison["net_accepted_gain"] == 3
    assert comparison["a5_dev_gate_pass"] is True
    assert comparison["recovered_task_ids"] == ["task-1", "task-2", "task-3"]


def test_comparison_rejects_non_dev_or_unpaired_runs(tmp_path):
    base = tmp_path / "base"
    sft = tmp_path / "sft"
    _write_run(base, ["accepted"], split="test")
    _write_run(sft, ["accepted"], split="test")
    with pytest.raises(ValueError, match="development split"):
        compare_dev_runs(base, sft, minimum_net_gain=1)

    sft_manifest = json.loads((sft / "manifest.json").read_text(encoding="utf-8"))
    sft_manifest["tasks"][0]["split"] = "dev"
    (sft / "manifest.json").write_text(json.dumps(sft_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifests differ"):
        compare_dev_runs(base, sft, minimum_net_gain=1)


def test_runtime_error_prevents_gate_pass(tmp_path):
    base = tmp_path / "base"
    sft = tmp_path / "sft"
    _write_run(base, ["rejected", "rejected"])
    _write_run(sft, ["accepted", "model_error"])
    comparison = compare_dev_runs(base, sft, minimum_net_gain=1)
    assert comparison["net_accepted_gain"] == 1
    assert comparison["runtime_integrity_errors"]
    assert comparison["a5_dev_gate_pass"] is False
