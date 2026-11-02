"""Import the reviewed UCI A1 plan into deterministic canonical CSV files.

The plan is explicit and versioned. This command never discovers or silently
approves extra datasets, and it refuses to overwrite an existing CSV with
different bytes.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
from urllib.request import urlopen

import pandas as pd
from ucimlrepo import fetch_ucirepo

from dirty_da.canonical import CanonicalTable


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
CONFIGS = ROOT / "configs"
PLAN_PATH = CONFIGS / "a1_uci_import_plan.json"
CATALOG_PATH = CONFIGS / "a1_table_catalog.json"
SEMANTICS_PATH = CONFIGS / "table_semantics.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def deterministic_sample(frame: pd.DataFrame, uci_id: int, limit: int) -> tuple[pd.DataFrame, list[int]]:
    source_positions = list(range(len(frame)))
    if len(frame) <= limit:
        return frame.reset_index(drop=True), source_positions
    ranked = sorted(
        source_positions,
        key=lambda position: hashlib.sha256(
            f"{uci_id}:{position}".encode("ascii")
        ).digest(),
    )[:limit]
    selected = sorted(ranked)
    return frame.iloc[selected].reset_index(drop=True), selected


def csv_text(frame: pd.DataFrame) -> str:
    buffer = StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def verify_cc_by_4(repository_url: str) -> None:
    with urlopen(repository_url, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    if not (
        "CC BY 4.0" in html
        or "Creative Commons Attribution 4.0" in html
    ):
        raise RuntimeError(
            f"Could not verify CC BY 4.0 text on official page {repository_url}."
        )


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def build_semantics(
    frame: pd.DataFrame,
    variables: pd.DataFrame,
    added_row_id: bool,
) -> dict:
    metadata_by_name: dict[str, list[pd.Series]] = defaultdict(list)
    for _, row in variables.iterrows():
        metadata_by_name[normalize(row["name"])].append(row)
    roles: dict[str, list[str]] = {
        "identifier": [],
        "category": [],
        "date": [],
        "measure": [],
        "derived": [],
        "coordinate": [],
        "opaque": [],
    }
    targets: list[str] = []
    for column in map(str, frame.columns):
        if column == "row_id" and added_row_id:
            roles["identifier"].append(column)
            continue
        normalized_column = normalize(column)
        suffix_match = re.fullmatch(r"(.+)\.(\d+)", normalized_column)
        if suffix_match and suffix_match.group(1) in metadata_by_name:
            base_name = suffix_match.group(1)
            occurrence = int(suffix_match.group(2))
        else:
            base_name = normalized_column
            occurrence = 0
        matches = metadata_by_name.get(base_name, [])
        if occurrence >= len(matches):
            raise RuntimeError(f"No UCI variable metadata found for column {column!r}.")
        metadata = matches[occurrence]
        uci_role = normalize(metadata.get("role"))
        uci_type = normalize(metadata.get("type"))
        if uci_role == "target":
            targets.append(column)
        if uci_role in {"id", "identifier"}:
            primary = "identifier"
        elif normalize(column) in {"latitude", "longitude", "lat", "lon", "lng"}:
            primary = "coordinate"
        elif uci_type in {"categorical", "binary"}:
            primary = "category"
        elif uci_type in {"date", "datetime"}:
            primary = "date"
        elif uci_type in {"integer", "continuous", "real"}:
            primary = "measure"
        elif uci_type in {"text", "string"}:
            primary = "opaque"
        else:
            raise RuntimeError(
                f"Unsupported UCI type {metadata.get('type')!r} for column {column!r}."
            )
        roles[primary].append(column)
    return {"roles": roles, "targets": targets, "derived_from": {}}


plan = load_json(PLAN_PATH)
catalog = load_json(CATALOG_PATH)
semantics = load_json(SEMANTICS_PATH)
if plan.get("schema_version") != 1 or len(plan.get("datasets", [])) != 41:
    raise RuntimeError("The reviewed UCI import plan must contain exactly 41 datasets.")
if len({item["uci_id"] for item in plan["datasets"]}) != 41:
    raise RuntimeError("UCI ids in the import plan must be unique.")
if len({item["file"] for item in plan["datasets"]}) != 41:
    raise RuntimeError("Output file names in the import plan must be unique.")

TABLES.mkdir(parents=True, exist_ok=True)
import_rows: list[dict] = []
for item in plan["datasets"]:
    uci_id = int(item["uci_id"])
    dataset = fetch_ucirepo(id=uci_id)
    metadata = dataset.metadata
    frame = dataset.data.original.copy(deep=True)
    if frame.columns.duplicated().any():
        raise RuntimeError(f"UCI {uci_id}: duplicate column labels are unsupported.")
    frame.columns = [str(column) for column in frame.columns]
    sampled, source_positions = deterministic_sample(frame, uci_id, int(plan["max_rows"]))
    added_row_id = bool(sampled.duplicated().any())
    if added_row_id:
        if "row_id" in sampled.columns:
            raise RuntimeError(f"UCI {uci_id}: cannot add row_id because it already exists.")
        sampled.insert(0, "row_id", [f"uci{uci_id}-{position}" for position in source_positions])
    if not 200 <= len(sampled) <= int(plan["max_rows"]):
        raise RuntimeError(f"UCI {uci_id}: row count outside the A1 range.")
    if not 5 <= len(sampled.columns) <= 20:
        raise RuntimeError(
            f"UCI {uci_id}: {len(sampled.columns)} columns outside the A1 range."
        )
    if sampled.duplicated().any():
        raise RuntimeError(f"UCI {uci_id}: rows are not unique after row-id handling.")
    semantic_spec = build_semantics(sampled, dataset.variables, added_row_id)
    if not semantic_spec["roles"]["category"]:
        raise RuntimeError(f"UCI {uci_id}: no semantic category column.")
    numeric_measures = [
        column
        for column in semantic_spec["roles"]["measure"]
        if pd.api.types.is_numeric_dtype(sampled[column])
    ]
    if not numeric_measures:
        raise RuntimeError(f"UCI {uci_id}: no numeric measure column.")

    verify_cc_by_4(metadata.repository_url)
    CanonicalTable(sampled, name=Path(item["file"]).stem)
    output = TABLES / item["file"]
    serialized = csv_text(sampled)
    if output.exists():
        if output.read_bytes() != serialized.encode("utf-8"):
            raise RuntimeError(f"Refusing to overwrite different existing file {output}.")
    else:
        temporary = output.with_suffix(".csv.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, output)

    transformations = [
        "Fetched from the official UCI data URL through ucimlrepo 0.0.7.",
        "Preserved UCI column names and values in CSV form.",
    ]
    if len(frame) > len(sampled):
        transformations.append(
            f"Selected the 5000 source row positions with the smallest SHA-256 rank of '{uci_id}:row_position', then restored source order (seed label {plan['sample_seed']})."
        )
    if added_row_id:
        transformations.append(
            "Added row_id from the UCI id and original source row position because source rows were not unique."
        )
    catalog["tables"][item["file"]] = {
        "domain": item["domain"],
        "selection_status": "selected",
        "split": item["split"],
        "provenance": {
            "status": "verified",
            "source_name": f"UCI {metadata.name}",
            "source_url": metadata.repository_url,
            "transformations": transformations,
        },
        "license": {
            "status": "verified",
            "name": plan["license"]["name"],
            "url": plan["license"]["url"],
        },
        "approved_for_research": True,
        "notes": [
            f"Attribution must cite UCI DOI {metadata.dataset_doi}."
            if metadata.dataset_doi
            else "Use the citation shown on the official UCI dataset page."
        ],
    }
    semantics["tables"][item["file"]] = semantic_spec
    import_rows.append(
        {
            "uci_id": uci_id,
            "file": item["file"],
            "source_name": metadata.name,
            "rows": len(sampled),
            "columns": len(sampled.columns),
            "source_rows": len(frame),
            "sampled": len(frame) > len(sampled),
            "row_id_added": added_row_id,
            "split": item["split"],
            "domain": item["domain"],
            "doi": metadata.dataset_doi,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )

atomic_json(CATALOG_PATH, catalog)
atomic_json(SEMANTICS_PATH, semantics)
pd.DataFrame(import_rows).to_csv(
    ROOT / "artifacts" / "a1_uci_import_manifest.csv",
    index=False,
    encoding="utf-8-sig",
)
split_counts = defaultdict(int)
for row in import_rows:
    split_counts[row["split"]] += 1
print(
    json.dumps(
        {
            "imported_or_verified": len(import_rows),
            "split_counts": dict(sorted(split_counts.items())),
            "sampled_tables": sum(row["sampled"] for row in import_rows),
            "row_id_added_tables": sum(row["row_id_added"] for row in import_rows),
        },
        indent=2,
    )
)
