"""Discover UCI tables matching the documented A1 structural screen.

This script downloads metadata only. It does not import or approve datasets.
Candidate rows still require column-level semantic review before selection.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "a1_uci_candidates.csv"
LIST_URL = "https://archive.ics.uci.edu/api/datasets/list?filter=python"
DETAIL_URL = "https://archive.ics.uci.edu/api/dataset"


def fetch_json(url: str, attempts: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=30) as response:
                value = json.load(response)
            if value.get("status") != 200 or not isinstance(value.get("data"), (dict, list)):
                raise RuntimeError(f"UCI API returned a non-success payload for {url}.")
            return value
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Unable to fetch UCI metadata from {url}: {last_error}")


def fetch_detail(item: dict) -> tuple[dict | None, dict | None]:
    url = f"{DETAIL_URL}?{urlencode({'id': item['id']})}"
    try:
        return fetch_json(url)["data"], None
    except RuntimeError as exc:
        return None, {
            "uci_id": item["id"],
            "name": item["name"],
            "url": url,
            "error": str(exc),
        }


items = fetch_json(LIST_URL)["data"]
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_detail, items))
metadata = [detail for detail, error in results if detail is not None]
errors = [error for detail, error in results if error is not None]

rows: list[dict] = []
for item in metadata:
    instances = item.get("num_instances")
    features = item.get("num_features")
    feature_types = set(item.get("feature_types") or [])
    variables = item.get("variables") or []
    variable_types = {str(variable.get("type")) for variable in variables}
    categorical = bool(feature_types & {"Categorical"}) or bool(
        variable_types & {"Categorical", "Binary"}
    )
    numeric = bool(feature_types & {"Integer", "Real"}) or bool(
        variable_types & {"Integer", "Continuous", "Real"}
    )
    if not isinstance(instances, int) or instances < 200:
        continue
    if not isinstance(features, int) or not 4 <= features <= 19:
        continue
    if not categorical or not numeric:
        continue
    rows.append(
        {
            "uci_id": item["uci_id"],
            "name": item["name"],
            "area": item.get("area"),
            "instances": instances,
            "features": features,
            "target_columns": ",".join(item.get("target_col") or []),
            "has_missing_values": item.get("has_missing_values"),
            "repository_url": item["repository_url"],
            "data_url": item["data_url"],
            "dataset_doi": item.get("dataset_doi"),
            "candidate_only": True,
        }
    )

frame = pd.DataFrame(rows).sort_values(["area", "uci_id"], na_position="last")
OUTPUT.parent.mkdir(exist_ok=True)
frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
error_output = ROOT / "artifacts" / "a1_uci_discovery_errors.csv"
pd.DataFrame(errors, columns=["uci_id", "name", "url", "error"]).to_csv(
    error_output, index=False, encoding="utf-8-sig"
)
print(
    f"Wrote {len(frame)} metadata candidates to {OUTPUT}; "
    f"recorded {len(errors)} inaccessible metadata entries in {error_output}."
)
