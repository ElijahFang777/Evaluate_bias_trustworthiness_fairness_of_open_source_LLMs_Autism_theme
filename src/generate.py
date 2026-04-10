#!/usr/bin/env python3
"""
generate.py

Run prompt datasets against one or more OpenAI-compatible chat endpoints
(such as a local Ollama server) and save one JSON object per response.

Designed for:
- data/prompts_asc.jsonl
- data/prompts_controls.jsonl

Example:
    python src/generate.py \
        --prompts data/prompts_asc.jsonl \
        --models llama3.1:8b mistral:7b qwen2.5:7b \
        --out outputs/raw_generations.jsonl \
        --seeds 42 43

    python src/generate.py \
        --prompts data/prompts_controls.jsonl \
        --models llama3.1:8b mistral:7b qwen2.5:7b \
        --out outputs/raw_generations.jsonl \
        --append \
        --seeds 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model outputs for JSONL prompt datasets."
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="One or more JSONL prompt files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model names to query.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL file to write generated responses to.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing output file and skip already completed items.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "ollama"),
        help="API key for the endpoint. Ollama usually ignores this.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt sent with every request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="One or more integer seeds.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per failed request.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Base retry backoff in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between successful requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of prompts read from each file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API; just print planned work.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl_line(path: Path, record: Dict[str, Any], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            response_id = obj.get("response_id")
            if response_id:
                ids.add(response_id)
    return ids


def normalize_prompt_record(
    record: Dict[str, Any], prompt_file: Path
) -> Dict[str, Any]:
    item_id = record.get("prompt_id") or record.get("control_id")
    if not item_id:
        raise ValueError(
            f"Record in {prompt_file} is missing both 'prompt_id' and 'control_id'."
        )
    if "prompt" not in record:
        raise ValueError(f"Record {item_id} in {prompt_file} is missing 'prompt'.")

    normalized = {
        "item_id": item_id,
        "record_type": "asc" if "prompt_id" in record else "control",
        "prompt_file": prompt_file.name,
        "prompt_path": str(prompt_file),
        "prompt": record["prompt"],
        "pair_id": record.get("pair_id"),
        "theme_id": record.get("theme_id"),
        "context": record.get("context"),
        "target_group": record.get("target_group"),
        "prompt_type": record.get("prompt_type"),
        "dimensions": record.get("dimensions", []),
        "expected_risk": record.get("expected_risk"),
        "source_note_ids": record.get("source_note_ids", []),
        "notes": record.get("notes"),
        "benchmark_source": record.get("benchmark_source"),
        "original_item_id": record.get("original_item_id"),
        "prompt_format": record.get("prompt_format"),
        "gold_label": record.get("gold_label"),
    }

    # Preserve the original input record for traceability.
    normalized["source_record"] = record
    return normalized


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def make_response_id(
    prompt_file: str,
    item_id: str,
    model: str,
    seed: int,
    prompt: str,
) -> str:
    raw = f"{prompt_file}|{item_id}|{model}|{seed}|{prompt}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{item_id}__{model.replace(':', '_')}__s{seed}__{digest}"


def extract_text_from_response(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some APIs may return structured message content.
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_finish_reason(payload: Dict[str, Any]) -> Optional[str]:
    choices = payload.get("choices", [])
    if not choices:
        return None
    return choices[0].get("finish_reason")


def safe_get_usage(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    usage = payload.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return prompt_tokens, completion_tokens, total_tokens


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=request_body,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # broad for practical retry handling
            last_error = exc
            if attempt == retries:
                break
            sleep_time = retry_backoff * attempt
            print(
                f"[warn] Request failed for model={model}, attempt={attempt}/{retries}: {exc}. "
                f"Retrying in {sleep_time:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_time)

    assert last_error is not None
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}") from last_error


def iter_jobs(
    prompt_files: Iterable[Path],
    models: Iterable[str],
    seeds: Iterable[int],
    limit: Optional[int],
) -> Iterable[Tuple[Path, Dict[str, Any], str, int]]:
    for prompt_file in prompt_files:
        records = read_jsonl(prompt_file, limit=limit)
        for raw_record in records:
            normalized = normalize_prompt_record(raw_record, prompt_file)
            for model in models:
                for seed in seeds:
                    yield prompt_file, normalized, model, seed


def main() -> int:
    args = parse_args()

    prompt_files = [Path(p) for p in args.prompts]
    out_path = Path(args.out)

    if not args.append and out_path.exists():
        print(
            f"[error] Output file already exists: {out_path}. "
            f"Use --append to continue writing to it.",
            file=sys.stderr,
        )
        return 1

    existing_ids: Set[str] = load_existing_ids(out_path) if args.append else set()

    jobs = list(iter_jobs(prompt_files, args.models, args.seeds, args.limit))
    total_jobs = len(jobs)

    print(f"[info] Loaded {len(prompt_files)} prompt file(s).")
    print(f"[info] Planned request count: {total_jobs}")
    print(f"[info] Output file: {out_path}")

    if args.dry_run:
        preview = min(5, total_jobs)
        print(f"[dry-run] Previewing first {preview} job(s):")
        for prompt_file, rec, model, seed in jobs[:preview]:
            response_id = make_response_id(
                prompt_file=prompt_file.name,
                item_id=rec["item_id"],
                model=model,
                seed=seed,
                prompt=rec["prompt"],
            )
            print(
                f"  - response_id={response_id} | prompt_file={prompt_file.name} | "
                f"item_id={rec['item_id']} | model={model} | seed={seed}"
            )
        return 0

    completed = 0
    skipped = 0
    failed = 0

    for idx, (prompt_file, rec, model, seed) in enumerate(jobs, start=1):
        response_id = make_response_id(
            prompt_file=prompt_file.name,
            item_id=rec["item_id"],
            model=model,
            seed=seed,
            prompt=rec["prompt"],
        )

        if response_id in existing_ids:
            skipped += 1
            print(
                f"[skip {idx}/{total_jobs}] {response_id} already exists in output."
            )
            continue

        print(
            f"[run  {idx}/{total_jobs}] prompt_file={prompt_file.name} "
            f"item_id={rec['item_id']} model={model} seed={seed}"
        )

        messages = build_messages(args.system_prompt, rec["prompt"])
        started_at = time.perf_counter()
        wallclock_started = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            payload = call_chat_completion(
                base_url=args.base_url,
                api_key=args.api_key,
                model=model,
                messages=messages,
                seed=seed,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
            )
            elapsed_sec = time.perf_counter() - started_at

            response_text = extract_text_from_response(payload)
            finish_reason = extract_finish_reason(payload)
            prompt_tokens, completion_tokens, total_tokens = safe_get_usage(payload)

            output_record: Dict[str, Any] = {
                "response_id": response_id,
                "run_started_at": wallclock_started,
                "elapsed_sec": round(elapsed_sec, 4),
                "prompt_file": rec["prompt_file"],
                "prompt_path": rec["prompt_path"],
                "record_type": rec["record_type"],
                "item_id": rec["item_id"],
                "prompt_id": rec["source_record"].get("prompt_id"),
                "control_id": rec["source_record"].get("control_id"),
                "pair_id": rec.get("pair_id"),
                "theme_id": rec.get("theme_id"),
                "context": rec.get("context"),
                "target_group": rec.get("target_group"),
                "prompt_type": rec.get("prompt_type"),
                "dimensions": rec.get("dimensions", []),
                "expected_risk": rec.get("expected_risk"),
                "source_note_ids": rec.get("source_note_ids", []),
                "notes": rec.get("notes"),
                "benchmark_source": rec.get("benchmark_source"),
                "original_item_id": rec.get("original_item_id"),
                "prompt_format": rec.get("prompt_format"),
                "gold_label": rec.get("gold_label"),
                "model": model,
                "seed": seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "system_prompt": args.system_prompt,
                "prompt": rec["prompt"],
                "response_text": response_text,
                "finish_reason": finish_reason,
                "usage_prompt_tokens": prompt_tokens,
                "usage_completion_tokens": completion_tokens,
                "usage_total_tokens": total_tokens,
                "raw_api_response": payload,
            }

            write_jsonl_line(out_path, output_record, append=True)
            existing_ids.add(response_id)
            completed += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as exc:
            elapsed_sec = time.perf_counter() - started_at
            failed += 1
            error_record: Dict[str, Any] = {
                "response_id": response_id,
                "run_started_at": wallclock_started,
                "elapsed_sec": round(elapsed_sec, 4),
                "prompt_file": rec["prompt_file"],
                "prompt_path": rec["prompt_path"],
                "record_type": rec["record_type"],
                "item_id": rec["item_id"],
                "prompt_id": rec["source_record"].get("prompt_id"),
                "control_id": rec["source_record"].get("control_id"),
                "pair_id": rec.get("pair_id"),
                "theme_id": rec.get("theme_id"),
                "context": rec.get("context"),
                "target_group": rec.get("target_group"),
                "prompt_type": rec.get("prompt_type"),
                "dimensions": rec.get("dimensions", []),
                "model": model,
                "seed": seed,
                "prompt": rec["prompt"],
                "error": str(exc),
            }
            write_jsonl_line(out_path, error_record, append=True)
            print(
                f"[fail {idx}/{total_jobs}] prompt_file={prompt_file.name} "
                f"item_id={rec['item_id']} model={model} seed={seed} error={exc}",
                file=sys.stderr,
            )

    print(
        f"[done] completed={completed} skipped={skipped} failed={failed} "
        f"total={total_jobs}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
