# ASC LLM Evaluation Pipeline

This repository evaluates open-source LLMs for **safety, trustworthiness, and fairness** in **ASC-sensitive contexts**.

## Structure

```text
data/
  prompts_asc.jsonl
  prompts_controls.jsonl

src/
  generate.py
  score_toxicity.py
  factuality_review_template.py
  score_fairness.py
  analyze.py
  plot_results.ipynb
  project_config.py

outputs/
  raw_generations.jsonl
  toxicity_scores.csv
  manual_review_completed.csv
  fairness_metrics.csv
  pairwise_differences.csv
  results_master.csv
  summary_by_model.csv
  summary_by_context.csv
  summary_by_dimension.csv
  figures/
```

## Setup

Recommended: Python `3.10` to `3.12`.

```bash
pip install -r requirements.txt
pip install seaborn
```

If you use Python `3.14`, upgrade Matplotlib before plotting:

```bash
pip install --upgrade matplotlib
```

## Pipeline

Run in this order:

```bash
python src/generate.py
python src/generate.py --prompt-set controls --append
python src/score_toxicity.py --verbose
python src/factuality_review_template.py --include-controls
python src/score_fairness.py --verbose
python src/analyze.py --verbose
```

Then open:

```text
src/plot_results.ipynb
```

All generated files should go to the top-level `outputs/` directory. The notebook is configured to save figures to:

```text
outputs/figures/
```

## Endpoint

`src/generate.py` uses an **OpenAI-compatible chat endpoint**. Ollama is one option, but not required.

Default endpoint:

```text
http://localhost:11434/v1
```

Example with another compatible endpoint:

```bash
python src/generate.py --base-url YOUR_ENDPOINT --api-key YOUR_KEY --models YOUR_MODEL_NAME
```

## Current Summary

From the checked-in analysis outputs:

| model | mean_toxicity | mean_helpfulness | mean_factuality | mean_stereotype |
| --- | ---: | ---: | ---: | ---: |
| deepseek-r1:8b | 0.0030 | 0.9615 | 1.0 | 0.0000 |
| qwen2.5:7b | 0.0041 | 0.9231 | 0.5 | 0.0435 |
| llama3.1:8b | 0.0055 | 0.7885 | 0.5 | 0.1522 |

## Notes

- Shared defaults live in `src/project_config.py`.
- The checked-in `outputs/` files are enough to run analysis and plotting without regenerating everything.
- `response_id` is the main join key across the pipeline.
- `pair_id` is required for paired fairness comparisons.
