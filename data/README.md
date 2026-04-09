# Data Folder README

This folder contains the data files used to build and evaluate the ASC-focused trustworthiness benchmark for open-source LLMs.

## Files

### `reddit_raw_paraphrased_notes.md`
Working evidence file derived from Reddit reading.  
It contains short, de-identified paraphrased notes from r/autism discussions, with one note per item and a reference link for traceability.

**Purpose:**  
Used as the low-level evidence source for theme extraction.

**Structure:**  
Each note records:
- note ID
- broad context
- short paraphrased observation
- reference link

---

### `reddit_theme_notes.md`
Clean research notebook that summarises recurring themes from the raw paraphrased notes.

**Purpose:**  
Used to explain how Reddit observations were converted into prompt design choices.

**Structure:**  
Each theme section records:
- social context
- common challenge
- likely consequence
- emotional tone
- harmful stereotype or trustworthiness risk
- possible prompt idea

---

### `prompts_asc.jsonl`
Main custom evaluation dataset for ASC-sensitive contexts.

**Purpose:**  
Used to query LLMs with prompts derived from Reddit-informed themes.

**Structure:**  
One JSON object per line. Each prompt includes:
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

This file contains paired prompts for fairness comparison and extra probes for privacy, ethics, and adversarial robustness.

---

### `prompts_controls.jsonl`
Small benchmark-style control set.

**Purpose:**  
Used as a reference set beside the ASC prompts, so model behaviour can be compared against non-Reddit control prompts.

**Structure:**  
One JSON object per line. Each prompt includes:
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

## How to check whether the data is reasonable

A quick sanity check for this folder is:

1. `reddit_raw_paraphrased_notes.md` should provide enough diverse evidence across multiple real-world ASC contexts.
2. `reddit_theme_notes.md` should clearly summarise those notes into themes and explain why each theme matters for evaluation.
3. `prompts_asc.jsonl` should trace back to the themes and include paired prompts that test fairness and stereotype bias.
4. `prompts_controls.jsonl` should remain separate from the ASC prompts and act as a small reference benchmark.
5. All files should be de-identified and should not expose usernames or long raw Reddit quotations.
