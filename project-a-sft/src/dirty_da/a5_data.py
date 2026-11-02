"""Freeze and validate the exact A5 SFT dataset selections."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ALLOWED_CATEGORIES = {
    "direct_success",
    "pollution_cleaning",
    "error_recovery",
}
PRIVATE_FIELDS = {"expected_answer", "task_spec", "canonical_fingerprint"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sft_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}.")
            validate_sft_record(value)
            records.append(value)
    if not records:
        raise ValueError(f"No SFT records found in {path}.")
    return records


def validate_sft_record(record: Mapping[str, Any]) -> None:
    if record.get("split") != "train":
        raise ValueError("A5 SFT data must contain train-split records only.")
    if record.get("category") not in ALLOWED_CATEGORIES:
        raise ValueError("Unsupported A5 trajectory category.")
    for field in ("trajectory_id", "trajectory_fingerprint", "task_id", "table"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"SFT record has invalid {field}.")
    if PRIVATE_FIELDS.intersection(record):
        raise ValueError("Private oracle fields are forbidden in A5 SFT records.")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT record must contain messages.")
    trainable_messages = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Every SFT message must be an object.")
        role = message.get("role")
        trainable = message.get("trainable")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported SFT message role: {role!r}.")
        if not isinstance(trainable, bool):
            raise ValueError("Every SFT message must declare a boolean trainable mask.")
        if trainable != (role == "assistant"):
            raise ValueError("Only assistant messages may be trainable in A5.")
        trainable_messages += int(trainable)
    if trainable_messages == 0:
        raise ValueError("SFT record has no trainable assistant message.")


def selection_entry(record: Mapping[str, Any], source_run: str) -> dict[str, str]:
    return {
        "source_run": source_run,
        "trajectory_id": str(record["trajectory_id"]),
        "trajectory_fingerprint": str(record["trajectory_fingerprint"]),
        "task_id": str(record["task_id"]),
        "table": str(record["table"]),
        "category": str(record["category"]),
    }


def validate_unique_fingerprints(entries: Iterable[Mapping[str, str]]) -> None:
    seen: set[str] = set()
    for entry in entries:
        fingerprint = entry["trajectory_fingerprint"]
        if fingerprint in seen:
            raise ValueError(f"Duplicate trajectory fingerprint in A5 selection: {fingerprint}")
        seen.add(fingerprint)


def selection_summary(entries: list[Mapping[str, str]]) -> dict[str, Any]:
    validate_unique_fingerprints(entries)
    return {
        "count": len(entries),
        "source_counts": dict(sorted(Counter(x["source_run"] for x in entries).items())),
        "category_counts": dict(sorted(Counter(x["category"] for x in entries).items())),
        "table_count": len({x["table"] for x in entries}),
        "task_count": len({x["task_id"] for x in entries}),
    }


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_frozen_dataset(
    root: Path, manifest_path: Path, dataset_name: str
) -> list[dict[str, Any]]:
    """Resolve one frozen index and reject changed sources or index metadata."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"Unknown frozen A5 dataset: {dataset_name}")
    sources: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for source_run, source in manifest["source_files"].items():
        path = root / source["path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Frozen A5 source hash changed: {source_run}")
        records = load_sft_records(path)
        if len(records) != int(source["record_count"]):
            raise ValueError(f"Frozen A5 source count changed: {source_run}")
        by_id: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            key = (source_run, record["trajectory_id"])
            if key in by_id:
                raise ValueError(f"Duplicate trajectory ID inside source run: {key}")
            by_id[key] = record
        sources[source_run] = by_id

    index_path = root / datasets[dataset_name]["index_path"]
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        source_run = entry["source_run"]
        key = (source_run, entry["trajectory_id"])
        try:
            record = sources[source_run][key]
        except KeyError as error:
            raise ValueError(f"Frozen A5 selection cannot resolve {key}.") from error
        for field in ("trajectory_fingerprint", "task_id", "table", "category"):
            if record[field] != entry[field]:
                raise ValueError(f"Frozen A5 index mismatch for {key}: {field}")
        resolved.append(dict(record))
    if len(resolved) != int(datasets[dataset_name]["count"]):
        raise ValueError("Frozen A5 index count does not match its manifest.")
    validate_unique_fingerprints(
        selection_entry(record, entry["source_run"])
        for record, entry in zip(resolved, entries, strict=True)
    )
    return resolved
