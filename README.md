# ASC LLM Evaluation Pipeline

This repository evaluates the **bias, trustworthiness, safety, and fairness** of open-source LLMs in **autism-spectrum-condition (ASC) sensitive contexts**.

The project is organized as a small, reproducible pipeline:

1. load ASC-focused and control prompts
2. generate model responses through an OpenAI-compatible chat endpoint
3. score toxicity automatically with Detoxify
4. export a manual review sheet for factuality, stereotype framing, privacy, ethics, refusal, and helpfulness
5. compute paired fairness gaps
6. aggregate analysis tables
7. create presentation-ready figures in a notebook

This README is updated to match the current repo state, including the checked-in analysis outputs and the seaborn plotting notebook.

---

## 1. Project structure

```text
data/
  README.md
  reddit_raw_paraphrased_notes.md
  reddit_theme_notes.md
  prompts_asc.jsonl
  prompts_controls.jsonl

src/
  analyze.py
  factuality_review_template.py
  generate.py
  plot_results.ipynb
  project_config.py
  score_fairness.py
  score_toxicity.py

outputs/
  raw_generations.jsonl
  toxicity_scores.csv
  manual_review_template.csv
  manual_review_completed.csv
  fairness_metrics.csv
  pairwise_differences.csv
  results_master.csv
  summary_by_model.csv
  summary_by_context.csv
  summary_by_dimension.csv

src/outputs/
  ...links to analysis CSVs for notebook compatibility...
  figures/
```

Notes:
- `src/project_config.py` is the shared source of truth for default paths, models, seeds, and API endpoint settings.
- The checked-in `outputs/` files let you inspect analysis and plotting without rerunning the full generation pipeline.
- `src/outputs/` exists mainly to keep the notebook working when Jupyter starts from `src/` instead of the repo root.

---

## 2. Quick start

### Option A: use the checked-in outputs

If you want to inspect results quickly without rerunning model generation:

```bash
pip install -r requirements.txt
pip install seaborn
```

Then open:

```text
src/plot_results.ipynb
```

and run the notebook from the first cell.

This is the fastest way to reproduce the tables and figures already bundled with the repo.

### Option B: run the full pipeline yourself

Install dependencies:

```bash
pip install -r requirements.txt
pip install seaborn
```

If you are using Python 3.14, upgrade Matplotlib before running the plotting notebook:

```bash
pip install --upgrade matplotlib
```

Then run the pipeline in order:

```bash
python src/generate.py
python src/generate.py --prompt-set controls --append
python src/score_toxicity.py --verbose
python src/factuality_review_template.py --include-controls
python src/score_fairness.py --verbose
python src/analyze.py --verbose
```

Finally open:

```text
src/plot_results.ipynb
```

---

## 3. Environment notes

### Recommended Python version

Use Python **3.10 to 3.12** when possible.

Python 3.14 can work for the scripts, but older Matplotlib builds such as `3.9.0` may trigger `RecursionError` in seaborn plots. If you stay on Python 3.14, use a newer Matplotlib build.

### Suggested environment setup

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install seaborn
```

Using conda:

```bash
conda create -n asc_eval python=3.10 -y
conda activate asc_eval
pip install -r requirements.txt
pip install seaborn
```

### LLM endpoint requirement

`src/generate.py` expects an **OpenAI-compatible chat endpoint**. Ollama is one option, but it is not the only option.

Default endpoint:

```text
http://localhost:11434/v1
```

If your endpoint differs, pass:

```bash
python src/generate.py --base-url YOUR_ENDPOINT --api-key YOUR_KEY --models YOUR_MODEL_NAME
```

---

## 4. Data overview

### `data/prompts_asc.jsonl`

Main ASC-sensitive prompt dataset. Typical fields include:

- `prompt_id`
- `pair_id`
- `theme_id`
- `context`
- `target_group`
- `prompt_type`
- `dimensions`
- `prompt`
- `expected_risk`
- `source_note_ids`
- `notes`

### `data/prompts_controls.jsonl`

Small control/reference prompt set. Typical fields include:

- `control_id`
- `benchmark_source`
- `original_item_id`
- `context`
- `dimensions`
- `prompt_format`
- `prompt`
- `gold_label`
- `notes`

### `data/reddit_raw_paraphrased_notes.md`

Working evidence notebook with de-identified, paraphrased Reddit notes and reference links.

### `data/reddit_theme_notes.md`

Theme synthesis derived from the Reddit notes and used to guide prompt design.

---

## 5. Execution order

The intended end-to-end order is:

1. `python src/generate.py`
2. `python src/generate.py --prompt-set controls --append`
3. `python src/score_toxicity.py --verbose`
4. `python src/factuality_review_template.py --include-controls`
5. manually complete `outputs/manual_review_completed.csv`
6. `python src/score_fairness.py --verbose`
7. `python src/analyze.py --verbose`
8. run `src/plot_results.ipynb`

Shared defaults live in:

```text
src/project_config.py
```

That file controls:

- input and output paths
- default model names
- default seeds
- default prompt set
- default OpenAI-compatible endpoint

---

## 6. Checked-in outputs

The repository currently includes populated analysis artifacts in `outputs/`. That means you can run the notebook even if you do not regenerate model outputs locally.

Important files:

- `outputs/raw_generations.jsonl`
- `outputs/toxicity_scores.csv`
- `outputs/manual_review_template.csv`
- `outputs/manual_review_completed.csv`
- `outputs/fairness_metrics.csv`
- `outputs/pairwise_differences.csv`
- `outputs/results_master.csv`
- `outputs/summary_by_model.csv`
- `outputs/summary_by_context.csv`
- `outputs/summary_by_dimension.csv`

These are useful for:

- notebook development
- debugging downstream scripts
- validating file formats and joins
- preparing figures quickly for presentation

---

## 7. Current model summary

The plotting notebook derives a compact presentation table called `current_model_summary` from `outputs/summary_by_model.csv` using the ASC rows.

Current summary:

| model | mean_toxicity | mean_helpfulness | mean_factuality | mean_stereotype |
| --- | ---: | ---: | ---: | ---: |
| deepseek-r1:8b | 0.0030 | 0.9615 | 1.0 | 0.0000 |
| qwen2.5:7b | 0.0041 | 0.9231 | 0.5 | 0.0435 |
| llama3.1:8b | 0.0055 | 0.7885 | 0.5 | 0.1522 |

Interpretation:

- `deepseek-r1:8b` appears safest and most factually reliable in the current checked-in outputs.
- `qwen2.5:7b` is close on helpfulness, with slightly higher toxicity and stereotype score.
- `llama3.1:8b` is the most variable of the three, with the highest toxicity and stereotype score in the current summary.

If you are writing this up, describe these as the **current repository results**. If you did not run live models yourself, do not overclaim them as fresh empirical findings.

---

## 8. Main outputs and file contracts

### `outputs/raw_generations.jsonl`

One JSON object per response from `src/generate.py`.

Important fields include:

- `response_id`
- `record_type`
- `item_id`
- `prompt_id` or `control_id`
- `pair_id`
- `theme_id`
- `context`
- `target_group`
- `prompt_type`
- `dimensions`
- `model`
- `seed`
- `prompt`
- `response_text`

### `outputs/toxicity_scores.csv`

One row per `response_id` with Detoxify scores such as:

- `toxicity`
- `identity_attack`
- `insult`
- `sexual_explicit`
- `score_status`

### `outputs/manual_review_template.csv`

Exported review sheet before annotation.

### `outputs/manual_review_completed.csv`

Completed review sheet used by `score_fairness.py` and `analyze.py`.

Required review columns include:

- `factuality_label`
- `stereotype_label`
- `privacy_label`
- `ethics_label`
- `refusal_label`
- `helpfulness_label`
- `review_notes`
- `evidence_source`

### `outputs/fairness_metrics.csv`

Aggregated fairness summaries by model, context, theme, and group.

### `outputs/pairwise_differences.csv`

Pair-level autistic-vs-neurotypical comparisons keyed by `pair_id`.

### `outputs/results_master.csv`

Merged row-level analysis table used for debugging and plotting.

### `outputs/summary_by_model.csv`

Model-level aggregates. Raw columns in this file use names such as:

- `mean_toxicity`
- `mean_helpfulness_score`
- `mean_factuality_score`
- `mean_stereotype_score`

The notebook renames a subset of those into a cleaner display table.

### `outputs/summary_by_context.csv`

Context-level grouped analysis.

### `outputs/summary_by_dimension.csv`

Dimension-level grouped analysis.

---

## 9. Plotting notebook

The main presentation notebook is:

```text
src/plot_results.ipynb
```

It currently:

- loads the aggregated CSV outputs
- builds `current_model_summary`
- displays the summary table
- plots average toxicity by model
- plots a model profile heatmap
- plots core quality metrics
- plots fairness gap summaries
- plots context-level toxicity heatmaps
- plots dimension-level toxicity rankings

The notebook uses **seaborn** for presentation quality.

### Figure output location

The notebook resolves its base directory from the current Jupyter working directory.

Typical behavior:

- if Jupyter starts from the repo root, figures go to `outputs/figures/`
- if Jupyter starts from `src/`, figures may go to `src/outputs/figures/`

In this repo, `src/outputs/` is present to make both launch styles work.

---

## 10. What each script does

### `src/generate.py`

- reads prompt JSONL files
- calls an OpenAI-compatible chat API
- writes one JSON record per response
- supports multiple models and seeds
- supports append mode and skip-by-`response_id`
- supports `--dry-run` and `--limit`

### `src/score_toxicity.py`

- reads `raw_generations.jsonl`
- scores `response_text` with Detoxify
- preserves rows with missing text or errors using `score_status`

### `src/factuality_review_template.py`

- exports a manual review CSV
- supports review modes `all`, `core`, and `highrisk`
- can include controls with `--include-controls`
- does not perform the human review itself

### `src/score_fairness.py`

- merges generations, toxicity, and manual review data
- filters to paired ASC rows
- computes fairness metrics and pairwise differences

### `src/analyze.py`

- merges all main pipeline outputs
- normalizes manual labels into numeric analysis fields
- writes row-level and grouped summary tables

---

## 11. Common debugging checks

### A. `generate.py` produced no usable responses

Check:

- the endpoint is reachable
- model names are valid
- `response_text` exists in successful rows
- failed rows may contain only `error`

### B. there is no time to install Ollama

You can still use another OpenAI-compatible endpoint:

```bash
python src/generate.py --base-url YOUR_ENDPOINT --api-key YOUR_KEY --models YOUR_MODEL_NAME
```

If you only need a quick pipeline check:

```bash
python src/generate.py --models YOUR_MODEL_NAME --seeds 42 --limit 5
```

If generation is not possible, the checked-in `outputs/` files are enough to run analysis and plotting.

### C. `score_toxicity.py` fails

Check:

- `torch` is installed
- `detoxify` is installed
- the selected device is valid
- the input JSONL has `response_text`

### D. `score_fairness.py` says `manual_review_completed.csv` is missing

Create it by:

1. running `python src/factuality_review_template.py --include-controls`
2. filling the review columns
3. saving the completed file to `outputs/manual_review_completed.csv`

### E. notebook cannot find analysis CSVs

If the notebook searches under `src/outputs/`, that usually means Jupyter started from `src/` instead of the repo root.

This repo includes `src/outputs/` links to keep the notebook working in either case.

### F. seaborn or Matplotlib raises `RecursionError`

This is usually an environment issue, especially with Python 3.14 plus older Matplotlib builds.

Recommended fix:

```bash
pip install --upgrade matplotlib
```

### G. summary tables look too uniform

If you are using prebuilt or synthetic outputs, expect smoother metrics and smaller differences than a full live evaluation would normally show.

---

## 12. File contracts for future edits

When modifying code, preserve these contracts unless you update the whole pipeline together.

### Contract 1: `response_id` is the primary join key

All downstream scripts depend on it.

### Contract 2: `pair_id` defines fairness pairs

Do not rename or remove it without updating `score_fairness.py`.

### Contract 3: `manual_review_completed.csv` must retain `response_id`

Without it, merges fail.

### Contract 4: `dimensions` must remain available

It may be stored as a list or serialized string, but downstream scripts expect the field to exist.

### Contract 5: output schema changes must be propagated

If you rename or remove output columns, update every downstream consumer, especially:

- `score_fairness.py`
- `analyze.py`
- `plot_results.ipynb`

---

## 13. Minimal reproducibility checklist

Before calling the run complete, verify:

- `outputs/raw_generations.jsonl` exists
- `outputs/toxicity_scores.csv` exists
- `outputs/manual_review_completed.csv` exists
- `outputs/fairness_metrics.csv` exists
- `outputs/pairwise_differences.csv` exists
- `outputs/results_master.csv` exists
- `outputs/summary_by_model.csv` exists
- `outputs/summary_by_context.csv` exists
- `outputs/summary_by_dimension.csv` exists
- the notebook runs and saves figure PNGs

---

## 14. Recommended extension path

If you continue improving the project, validate the row-level pipeline first:

1. `generate.py`
2. `score_toxicity.py`
3. `manual_review_completed.csv`

Once those are reliable, fairness aggregation and plotting become much easier to trust.
