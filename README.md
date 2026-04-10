# Project README

This repository evaluates the **bias, trustworthiness, and fairness** of open-source LLMs in **ASC-sensitive contexts**. The project uses a compact, reproducible pipeline:

1. prepare Reddit-informed ASC prompts and small control prompts  
2. generate model responses through an API-compatible local LLM server  
3. score toxicity automatically  
4. export a manual-review sheet for factuality, stereotype framing, privacy, ethics, refusal, and helpfulness  
5. compute fairness gaps on paired prompts  
6. aggregate results  
7. generate report figures

This README is written to help both human readers and AI coding assistants understand the project quickly for debugging or extension.

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
  generate.py
  score_toxicity.py
  factuality_review_template.py
  score_fairness.py
  analyze.py
  plot_results.ipynb

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
  figures/
```

---

## 2. Core design assumptions

- The main evaluation dataset is `data/prompts_asc.jsonl`.
- The control/reference dataset is `data/prompts_controls.jsonl`.
- `generate.py` is the source of truth for response-level records.
- All later scripts join on **`response_id`**.
- `pair_id` links autistic vs neurotypical counterfactual prompts.
- `target_group` is expected to contain values such as `autistic` and `neurotypical`.
- `manual_review_completed.csv` is created by a human reviewer after filling the template exported by `factuality_review_template.py`.
- The plotting notebook assumes all earlier scripts have already run successfully.

If debugging, always verify these assumptions first.

---

## 3. Environment setup

### Recommended Python version
Use Python **3.10+**.

### Recommended setup
Create and activate a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or with conda:

```bash
conda create -n asc_eval python=3.10 -y
conda activate asc_eval
```

### Install Python packages
Install the main runtime packages:

```bash
pip install pandas requests matplotlib jupyter notebook
pip install detoxify torch
```

If using CUDA, install a CUDA-compatible PyTorch build appropriate for your system.

### Local LLM server
This project assumes an **OpenAI-compatible chat endpoint**. It was designed for **Ollama**.

Typical setup:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull deepseek-r1:8b
ollama pull qwen2.5:7b
```

Default endpoint expected by `generate.py`:

```text
http://localhost:11434/v1
```

If the endpoint differs, pass `--base-url`.

---

## 4. Data overview

### `data/reddit_raw_paraphrased_notes.md`
Working evidence notebook with de-identified, paraphrased Reddit notes and reference links.

### `data/reddit_theme_notes.md`
Theme synthesis derived from the raw Reddit notes. Explains:
- real-world context
- challenge
- consequence
- risk for LLM evaluation
- prompt idea

### `data/prompts_asc.jsonl`
Main custom ASC prompt dataset.

Each line is one JSON object with fields such as:
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
Small control/reference prompt set.

Each line is one JSON object with fields such as:
- `control_id`
- `benchmark_source`
- `original_item_id`
- `context`
- `dimensions`
- `prompt_format`
- `prompt`
- `gold_label`
- `notes`

---

## 5. Execution order

Run the project in this order.

Global defaults for paths, models, and seeds now live in:

```text
src/project_config.py
```

Edit that file once when you want to change shared settings such as:
- output directories
- default models
- default seeds
- default prompt set
- API endpoint defaults

### Step 1: Generate responses

ASC prompts:

```bash
python src/generate.py
```

Control prompts:

```bash
python src/generate.py \
  --prompt-set controls \
  --append
```

### Step 2: Score toxicity

```bash
python src/score_toxicity.py --verbose
```

### Step 3: Export manual review template

```bash
python src/factuality_review_template.py --include-controls
```

### Step 4: Manually review and save completed file

Create:

```text
outputs/manual_review_completed.csv
```

This file should preserve the exported rows and fill the annotation columns:
- `factuality_label`
- `stereotype_label`
- `privacy_label`
- `ethics_label`
- `refusal_label`
- `helpfulness_label`
- `review_notes`
- `evidence_source`

### Step 5: Compute fairness metrics

```bash
python src/score_fairness.py --verbose
```

### Step 6: Aggregate analysis tables

```bash
python src/analyze.py --verbose
```

### Step 7: Plot results

Open and run:

```text
src/plot_results.ipynb
```

This notebook reads the output CSV files and writes figures into:

```text
outputs/figures/
```

---

## 6. Expected outputs

### `outputs/raw_generations.jsonl`
One JSON object per response. This is the main row-level log.

Important fields:
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
Toxicity scores per `response_id`.

Important fields:
- `response_id`
- `toxicity`
- `identity_attack`
- `insult`
- `score_status`

### `outputs/manual_review_template.csv`
Exported review sheet before annotation.

### `outputs/manual_review_completed.csv`
Human-completed review sheet after annotation.

### `outputs/fairness_metrics.csv`
Aggregated fairness summaries by model/context/theme/group.

### `outputs/pairwise_differences.csv`
Pair-level autistic-vs-neurotypical differences.

### `outputs/results_master.csv`
Merged row-level table used for downstream analysis and debugging.

### `outputs/summary_by_model.csv`
Model-level averages and counts.

### `outputs/summary_by_context.csv`
Context-level averages and counts.

### `outputs/summary_by_dimension.csv`
Dimension-level averages and counts.

### `outputs/figures/`
PNG figures generated by the notebook.

---

## 7. What each script is responsible for

### `src/generate.py`
- reads prompt JSONL files
- calls the OpenAI-compatible chat API
- writes one response per line to JSONL
- supports multiple models and seeds
- supports append mode and skip-by-`response_id`

### `src/score_toxicity.py`
- reads the generation JSONL
- scores `response_text` with Detoxify
- writes one CSV row per `response_id`

### `src/factuality_review_template.py`
- exports a manual-review CSV
- flags rows needing factuality, stereotype, privacy, or ethics review
- does not perform the manual review itself

### `src/score_fairness.py`
- merges generation, toxicity, and manual-review data
- filters to paired ASC prompts
- computes group-level metrics and pairwise autistic-vs-neurotypical gaps

### `src/analyze.py`
- merges all major outputs
- creates analysis-ready summary tables

### `src/plot_results.ipynb`
- loads summary outputs
- creates report-ready figures
- saves figures to `outputs/figures/`

---

## 8. Common debugging checks

If something breaks, check these in order.

### A. `generate.py` produced no usable text
Check:
- Ollama server is running
- model name is correct
- endpoint is correct
- output JSONL contains `response_text`
- rows with failures may only contain `error`

### B. `score_toxicity.py` fails
Check:
- `detoxify` is installed
- `torch` is installed
- CUDA is available if using `--device cuda`
- input file contains `response_text`

### C. Manual review merge looks wrong
Check:
- `manual_review_completed.csv` still contains `response_id`
- reviewer did not delete or reorder required columns incorrectly
- `response_id` values exactly match those in `raw_generations.jsonl`

### D. `score_fairness.py` finds no pairs
Check:
- `record_type == "asc"`
- `prompt_type == "paired"`
- `pair_id` is filled
- `target_group` contains both comparison groups
- no paired rows were lost during generation or manual review

### E. `analyze.py` outputs empty summaries
Check:
- earlier CSV files were created successfully
- numeric columns are present
- `results_master.csv` has non-empty `response_text`
- manual labels were actually filled for reviewed rows

### F. plotting notebook fails
Check:
- all expected CSV files exist in `outputs/`
- column names were not changed manually
- summary files are not empty

---

## 9. File contracts for AI assistants

When modifying code, preserve these contracts unless the whole pipeline is updated together.

### Contract 1: `response_id` is the primary join key
All downstream scripts depend on it.

### Contract 2: `pair_id` defines fairness pairs
Do not rename or remove it without updating `score_fairness.py`.

### Contract 3: `dimensions` may be stored as a JSON list or a delimited string
`analyze.py` normalizes it, but the field must still exist.

### Contract 4: `manual_review_completed.csv` must keep `response_id`
Without it, merging fails.

### Contract 5: do not change output schemas casually
If a script changes columns, update every downstream consumer.

---

## 10. Suggested minimal reproducibility checklist

Before considering the run complete, verify:

- `outputs/raw_generations.jsonl` exists and contains non-empty `response_text`
- `outputs/toxicity_scores.csv` exists and has toxicity scores
- `outputs/manual_review_completed.csv` exists and contains reviewer labels
- `outputs/fairness_metrics.csv` and `outputs/pairwise_differences.csv` exist
- `outputs/results_master.csv` exists
- `outputs/summary_by_model.csv`, `outputs/summary_by_context.csv`, and `outputs/summary_by_dimension.csv` exist
- `outputs/figures/` contains PNG plots

---

## 11. Recommended next debugging target

If extending the project, start by validating the **row-level pipeline** first:

1. `generate.py`
2. `score_toxicity.py`
3. `manual_review_completed.csv`

Once those are correct, fairness and aggregation become much easier to debug.
