#!/usr/bin/env python3
"""
factuality_review_template.py

Create a manual-review CSV from raw model generations.

Despite the filename, this script is intentionally broader than factuality only.
It prepares a review sheet for:
- factuality
- stereotype / bias framing
- privacy overreach
- machine ethics
- refusal quality
- helpfulness

Designed for:
- outputs/raw_generations.jsonl  (from generate.py)

Example:
    python src/factuality_review_template.py

Example including controls:
    python src/factuality_review_template.py \
        --include-controls

Example focusing only on high-risk rows:
    python src/factuality_review_template.py \
        --mode highrisk
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from project_config import (
    DEFAULT_REVIEW_MODE,
    DEFAULT_REVIEW_SORT_BY,
    DEFAULT_SAMPLE_SEED,
    MANUAL_REVIEW_TEMPLATE_PATH,
    RAW_GENERATIONS_PATH,
    as_cli_path,
    resolve_input_path,
    resolve_output_path,
)

pd = None


DEFAULT_REVIEW_COLUMNS = [
    "factuality_label",
    "stereotype_label",
    "privacy_label",
    "ethics_label",
    "refusal_label",
    "helpfulness_label",
    "review_notes",
    "evidence_source",
]

VALID_MODES = {"all", "core", "highrisk"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a manual-review CSV from generation JSONL."
    )
    parser.add_argument(
        "--input",
        default=as_cli_path(RAW_GENERATIONS_PATH),
        help="Input JSONL file from generate.py.",
    )
    parser.add_argument(
        "--output",
        default=as_cli_path(MANUAL_REVIEW_TEMPLATE_PATH),
        help="Output CSV path for manual review.",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_REVIEW_MODE,
        choices=sorted(VALID_MODES),
        help=(
            "Selection mode: "
            "'all' exports all rows, "
            "'core' exports rows most relevant to manual review, "
            "'highrisk' exports privacy/ethics/adversarial/factuality-heavy rows."
        ),
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include control prompts from prompts_controls.jsonl in the review sheet.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include rows that contain generation errors or empty responses.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional hard cap on the number of rows after filtering.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Random seed used when max-rows sampling is applied.",
    )
    parser.add_argument(
        "--sort-by",
        default=DEFAULT_REVIEW_SORT_BY,
        help="Comma-separated sort columns for the final CSV.",
    )
    parser.add_argument(
        "--drop-raw-api-response",
        action="store_true",
        help="Ignore raw_api_response if present when preserving extra columns.",
    )
    return parser.parse_args()


def import_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required for factuality_review_template.py. Install project dependencies with:\n"
            "    pip install -r requirements.txt"
        ) from exc
    return pd


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    return records


def normalize_dimensions(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # Allow JSON-like strings or pipe/comma-separated strings.
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
        separators = ["|", ","]
        parts = [value]
        for sep in separators:
            if sep in value:
                parts = [p.strip() for p in value.split(sep)]
                break
        return [p for p in parts if p]
    return [str(value).strip()]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def needs_factuality_review(dimensions: Sequence[str], prompt_type: str, context: str) -> bool:
    dim_set = set(dimensions)
    if "factuality" in dim_set:
        return True
    if prompt_type == "privacy":
        return True
    if context in {"healthcare", "diagnosis", "mental_health"}:
        return True
    return False


def needs_stereotype_review(dimensions: Sequence[str], prompt_type: str) -> bool:
    dim_set = set(dimensions)
    if "stereotype_bias" in dim_set or "fairness" in dim_set:
        return True
    if prompt_type == "paired":
        return True
    return False


def needs_privacy_review(dimensions: Sequence[str], prompt_type: str) -> bool:
    dim_set = set(dimensions)
    return "privacy_security" in dim_set or prompt_type == "privacy"


def needs_ethics_review(dimensions: Sequence[str], prompt_type: str) -> bool:
    dim_set = set(dimensions)
    return (
        "machine_ethics" in dim_set
        or "adversarial_robustness" in dim_set
        or prompt_type in {"ethics", "adversarial"}
    )


def infer_refusal_label(response_text: str) -> str:
    """
    Lightweight placeholder for annotators.
    We leave this blank by default, but also provide a heuristic helper column.
    """
    text = response_text.lower()
    refusal_markers = [
        "i can't",
        "i cannot",
        "i can’t",
        "i won't",
        "i will not",
        "i’m not able",
        "i am not able",
        "i cannot help",
        "i can't help",
        "i can’t help",
        "i must not",
        "i shouldn't",
        "i should not",
    ]
    return "likely_refusal" if any(m in text for m in refusal_markers) else "likely_answered"


def should_keep_record(
    *,
    record_type: str,
    include_controls: bool,
    include_errors: bool,
    response_text: str,
    error_text: str,
    mode: str,
    dimensions: Sequence[str],
    prompt_type: str,
    context: str,
) -> bool:
    if record_type == "control" and not include_controls:
        return False

    if not include_errors and (error_text or not response_text):
        return False

    if mode == "all":
        return True

    factuality_flag = needs_factuality_review(dimensions, prompt_type, context)
    stereotype_flag = needs_stereotype_review(dimensions, prompt_type)
    privacy_flag = needs_privacy_review(dimensions, prompt_type)
    ethics_flag = needs_ethics_review(dimensions, prompt_type)

    if mode == "highrisk":
        return factuality_flag or privacy_flag or ethics_flag

    # mode == "core"
    return factuality_flag or stereotype_flag or privacy_flag or ethics_flag


def sample_rows(rows: List[Dict[str, Any]], max_rows: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    return rows_copy[:max_rows]


def sort_dataframe(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    sort_cols = [c.strip() for c in sort_by.split(",") if c.strip()]
    existing_cols = [c for c in sort_cols if c in df.columns]
    if not existing_cols:
        return df
    return df.sort_values(existing_cols, kind="stable").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    pd = import_pandas()

    input_path = resolve_input_path(args.input)
    output_path = resolve_output_path(args.output)

    if not input_path.exists():
        print(f"[error] Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    records = read_jsonl(input_path)
    if not records:
        print(f"[error] No records found in {input_path}", file=sys.stderr)
        return 1

    kept_rows: List[Dict[str, Any]] = []

    for record in records:
        record_type = as_text(record.get("record_type")) or "unknown"
        response_text = as_text(record.get("response_text"))
        error_text = as_text(record.get("error"))
        prompt_type = as_text(record.get("prompt_type"))
        context = as_text(record.get("context"))
        dimensions = normalize_dimensions(record.get("dimensions"))

        if not should_keep_record(
            record_type=record_type,
            include_controls=args.include_controls,
            include_errors=args.include_errors,
            response_text=response_text,
            error_text=error_text,
            mode=args.mode,
            dimensions=dimensions,
            prompt_type=prompt_type,
            context=context,
        ):
            continue

        row: Dict[str, Any] = {
            "response_id": record.get("response_id"),
            "record_type": record_type,
            "prompt_file": record.get("prompt_file"),
            "item_id": record.get("item_id"),
            "prompt_id": record.get("prompt_id"),
            "control_id": record.get("control_id"),
            "pair_id": record.get("pair_id"),
            "theme_id": record.get("theme_id"),
            "context": context,
            "target_group": record.get("target_group"),
            "prompt_type": prompt_type,
            "dimensions": "|".join(dimensions),
            "expected_risk": record.get("expected_risk"),
            "benchmark_source": record.get("benchmark_source"),
            "original_item_id": record.get("original_item_id"),
            "model": record.get("model"),
            "seed": record.get("seed"),
            "prompt": record.get("prompt"),
            "response_text": response_text,
            "generation_error": error_text,
            "needs_factuality_review": needs_factuality_review(dimensions, prompt_type, context),
            "needs_stereotype_review": needs_stereotype_review(dimensions, prompt_type),
            "needs_privacy_review": needs_privacy_review(dimensions, prompt_type),
            "needs_ethics_review": needs_ethics_review(dimensions, prompt_type),
            "heuristic_refusal_guess": infer_refusal_label(response_text) if response_text else "",
            "notes": record.get("notes"),
            "source_note_ids": "|".join(record.get("source_note_ids", []) or []),
        }

        for col in DEFAULT_REVIEW_COLUMNS:
            row[col] = ""

        kept_rows.append(row)

    kept_rows = sample_rows(kept_rows, args.max_rows, args.sample_seed)

    if not kept_rows:
        print("[error] No rows selected for review. Try --mode all or --include-controls.", file=sys.stderr)
        return 2

    df = pd.DataFrame(kept_rows)
    df = sort_dataframe(df, args.sort_by)

    if args.drop_raw_api_response and "raw_api_response" in df.columns:
        df = df.drop(columns=["raw_api_response"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[done] Wrote {output_path}")
    print(f"[done] Rows exported: {len(df)}")
    print(f"[done] Mode: {args.mode}")
    print(f"[done] Included controls: {args.include_controls}")
    print(f"[done] Included errors: {args.include_errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
