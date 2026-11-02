from __future__ import annotations

import json

import pytest

from dirty_da.a5_data import (
    resolve_frozen_dataset,
    selection_entry,
    selection_summary,
    sha256_file,
    validate_sft_record,
    write_jsonl,
)


def _record(**updates):
    value = {
        "split": "train",
        "category": "direct_success",
        "trajectory_id": "trajectory-1",
        "trajectory_fingerprint": "fingerprint-1",
        "task_id": "task-1",
        "table": "table.csv",
        "messages": [
            {"role": "system", "content": "system", "trainable": False},
            {"role": "user", "content": "question", "trainable": False},
            {"role": "assistant", "content": "answer", "trainable": True},
        ],
    }
    value.update(updates)
    return value


def test_a5_record_requires_train_split_and_assistant_only_masks():
    validate_sft_record(_record())
    with pytest.raises(ValueError, match="train-split"):
        validate_sft_record(_record(split="test"))
    bad = _record()
    bad["messages"][1]["trainable"] = True
    with pytest.raises(ValueError, match="Only assistant"):
        validate_sft_record(bad)


def test_a5_record_rejects_private_oracle_fields():
    with pytest.raises(ValueError, match="Private oracle"):
        validate_sft_record(_record(expected_answer=42))


def test_selection_uses_source_run_to_disambiguate_ids_but_deduplicates_content():
    first = selection_entry(_record(), "run-14b")
    second_record = _record(trajectory_fingerprint="fingerprint-2")
    second = selection_entry(second_record, "run-8b")
    summary = selection_summary([first, second])
    assert summary["count"] == 2
    assert summary["source_counts"] == {"run-14b": 1, "run-8b": 1}

    second["trajectory_fingerprint"] = first["trajectory_fingerprint"]
    with pytest.raises(ValueError, match="Duplicate trajectory fingerprint"):
        selection_summary([first, second])


def test_frozen_dataset_resolver_rejects_changed_source(tmp_path):
    source = tmp_path / "source.jsonl"
    index = tmp_path / "index.jsonl"
    manifest_path = tmp_path / "manifest.json"
    record = _record()
    entry = selection_entry(record, "teacher")
    write_jsonl(source, [record])
    write_jsonl(index, [entry])
    manifest = {
        "source_files": {
            "teacher": {
                "path": "source.jsonl",
                "sha256": sha256_file(source),
                "record_count": 1,
            }
        },
        "datasets": {
            "main": {"index_path": "index.jsonl", "count": 1}
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(resolve_frozen_dataset(tmp_path, manifest_path, "main")) == 1

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash changed"):
        resolve_frozen_dataset(tmp_path, manifest_path, "main")
