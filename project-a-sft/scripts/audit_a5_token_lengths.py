"""Audit Qwen chat-template lengths and assistant masks before A5 training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dirty_da.a5_data import resolve_frozen_dataset  # noqa: E402
from dirty_da.a5_format import encode_sft_record  # noqa: E402


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen3 model/tokenizer path.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "a5_sft.json")
    args = parser.parse_args()

    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install the project training extra before token auditing.") from error

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = resolve_frozen_dataset(
        ROOT,
        ROOT / config["dataset_manifest"],
        config["dataset_name"],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    lengths: list[int] = []
    trainable_counts: list[int] = []
    category_lengths: Counter[str] = Counter()
    for record in records:
        encoded = encode_sft_record(record, tokenizer, max_length=10**9)
        length = len(encoded["input_ids"])
        lengths.append(length)
        trainable_counts.append(sum(label != -100 for label in encoded["labels"]))
        category_lengths[record["category"]] += length

    candidate_limits = (2048, 4096, 8192, 16384)
    maximum = max(lengths)
    recommended = next(limit for limit in candidate_limits if maximum <= limit)
    configured = config.get("max_seq_length")
    local_model_path = Path(args.model)
    audit = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "tokenizer_source": "local_path" if local_model_path.exists() else args.model,
        "tokenizer_revision": (
            local_model_path.name if local_model_path.exists() else None
        ),
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "vocab_size": len(tokenizer),
        "dataset_name": config["dataset_name"],
        "record_count": len(records),
        "token_length": {
            "min": min(lengths),
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": maximum,
        },
        "trainable_token_count": {
            "min": min(trainable_counts),
            "p50": _percentile(trainable_counts, 0.50),
            "max": max(trainable_counts),
        },
        "overflow_counts": {
            str(limit): sum(length > limit for length in lengths)
            for limit in candidate_limits
        },
        "recommended_max_seq_length": recommended,
        "configured_max_seq_length": configured,
        "configured_length_covers_all_records": (
            isinstance(configured, int) and configured >= maximum
        ),
        "silent_truncation_allowed": False,
    }
    destination = ROOT / "artifacts" / "a5_data" / "token_length_audit.json"
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
