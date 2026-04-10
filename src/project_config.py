#!/usr/bin/env python3
"""
Shared project configuration for CLI scripts.

Edit this file when you want to change global defaults such as model names,
seed values, or the standard input/output locations used across the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


PROMPTS_ASC_PATH = DATA_DIR / "prompts_asc.jsonl"
PROMPTS_CONTROLS_PATH = DATA_DIR / "prompts_controls.jsonl"

RAW_GENERATIONS_PATH = OUTPUT_DIR / "raw_generations.jsonl"
TOXICITY_SCORES_PATH = OUTPUT_DIR / "toxicity_scores.csv"
MANUAL_REVIEW_TEMPLATE_PATH = OUTPUT_DIR / "manual_review_template.csv"
MANUAL_REVIEW_COMPLETED_PATH = OUTPUT_DIR / "manual_review_completed.csv"
FAIRNESS_METRICS_PATH = OUTPUT_DIR / "fairness_metrics.csv"
PAIRWISE_DIFFERENCES_PATH = OUTPUT_DIR / "pairwise_differences.csv"
RESULTS_MASTER_PATH = OUTPUT_DIR / "results_master.csv"
SUMMARY_BY_MODEL_PATH = OUTPUT_DIR / "summary_by_model.csv"
SUMMARY_BY_CONTEXT_PATH = OUTPUT_DIR / "summary_by_context.csv"
SUMMARY_BY_DIMENSION_PATH = OUTPUT_DIR / "summary_by_dimension.csv"


DEFAULT_PROMPT_SET = "asc"
DEFAULT_MODELS = ["llama3.1:8b", "deepseek-r1:8b", "qwen2.5:7b"]
DEFAULT_SEEDS = [42, 43]
DEFAULT_CONTROL_SEEDS = [42]

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_SLEEP = 0.0

DEFAULT_TOXICITY_MODEL = "unbiased"
DEFAULT_TOXICITY_DEVICE = "auto"
DEFAULT_TOXICITY_BATCH_SIZE = 32

DEFAULT_REVIEW_MODE = "core"
DEFAULT_SAMPLE_SEED = 42
DEFAULT_REVIEW_SORT_BY = "model,context,item_id,seed"

DEFAULT_TARGET_GROUPS = ["autistic", "neurotypical"]


PROMPT_SET_FILES = {
    "asc": [PROMPTS_ASC_PATH],
    "controls": [PROMPTS_CONTROLS_PATH],
    "all": [PROMPTS_ASC_PATH, PROMPTS_CONTROLS_PATH],
}


def prompt_files_for_set(prompt_set: str) -> List[Path]:
    try:
        return list(PROMPT_SET_FILES[prompt_set])
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_SET_FILES))
        raise ValueError(f"Unknown prompt set '{prompt_set}'. Expected one of: {valid}.") from exc


def generation_seeds_for_set(prompt_set: str) -> List[int]:
    if prompt_set == "controls":
        return list(DEFAULT_CONTROL_SEEDS)
    return list(DEFAULT_SEEDS)


def resolve_input_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / path).resolve()


def resolve_output_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def as_cli_path(path: Path) -> str:
    return str(path)
