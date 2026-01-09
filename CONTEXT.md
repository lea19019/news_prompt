# Project Context for LLM Assistants

This document provides complete context for an LLM to understand and assist with this codebase. Copy this entire file when asking another LLM for help.

---

## What This Project Does

A Python pipeline for evaluating **prompt engineering techniques** on **news classification** tasks using **local LLMs via Ollama**.

**Research Question**: Which prompting technique (zero-shot, few-shot, chain-of-thought, etc.) works best for classifying news articles with open-source LLMs?

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Package Manager | **uv** (not pip) |
| LLM Backend | **Ollama** (local inference) |
| Data Processing | pandas, scikit-learn |
| HTTP Requests | requests |
| Progress Bars | tqdm |

**Important**: All Python commands use `uv run python ...`, not `python ...`.

---

## Project Structure

```
news_classification_prompts/
├── experiments.py          # CLI entry point, orchestrates everything
├── dataset_loader.py       # Downloads/caches datasets as CSV
├── prompt_loader.py        # Loads prompt templates from prompts/
├── model_inference.py      # Ollama API wrapper, response parsing
├── evaluation.py           # Computes accuracy, F1, precision, recall
├── check_status.py         # Verifies system readiness
├── setup.sh                # Automated setup script
├── requirements.txt        # pandas, scikit-learn, requests, tqdm, ollama, numpy
│
├── prompts/                # Editable prompt templates (plain text)
│   ├── zero_shot.txt       # No examples
│   ├── few_shot.txt        # 3 examples included
│   ├── constrained.txt     # Explicit output format
│   ├── chain_of_thought.txt # Step-by-step reasoning
│   └── self_consistency.txt # Same prompt, 5 samples, majority vote
│
├── data/                   # Auto-downloaded datasets
│   ├── ag_news_{train,test}.csv
│   ├── bbc_news_{train,test}.csv
│   └── 20newsgroups_{train,test}.csv
│
└── results/                # Experiment outputs
    └── run_YYYYMMDD_HHMMSS[_name]/
        ├── run_config.json     # What was run
        ├── results.json        # Full results
        ├── summary.csv         # Quick overview
        └── run_summary.json    # Duration, best/worst
```

---

## Key Classes & Methods

### `ExperimentPipeline` (experiments.py)

Main orchestrator. Coordinates dataset loading, prompt formatting, inference, and evaluation.

```python
pipeline = ExperimentPipeline()

# Flexible experiment (any combination)
pipeline.run_custom_experiment(
    models=['gemma3:12b', 'phi4:14b'],
    datasets=['ag_news', 'bbc_news'],
    techniques=['zero_shot', 'few_shot'],
    limit=100,          # samples per experiment
    run_name='my_run'   # folder name suffix
)

# Full pipeline (all combinations)
pipeline.run_full_experiment(limit=50, run_name='full_test')

# Single experiment
result = pipeline.run_single_experiment('ag_news', 'gemma3:12b', 'zero_shot', limit=10)
```

**Internal methods**:
- `_start_run(run_name)` - Creates `results/run_YYYYMMDD_HHMMSS[_name]/`
- `_save_run_config()` - Saves configuration JSON
- `_save_results()` - Saves results.json and summary.csv
- `_save_run_summary()` - Saves run_summary.json with stats

---

### `DatasetLoader` (dataset_loader.py)

Downloads and caches datasets. All datasets normalized to CSV with columns: `text`, `label`, `label_text`.

```python
loader = DatasetLoader(data_dir="data")

# Load data (auto-downloads if missing)
df = loader.load_dataset('ag_news', split='test')  # or 'train'

# Get categories
categories = loader.get_categories('ag_news')
# → ['Business', 'Sci/Tech', 'Sports', 'World']

# Get few-shot examples
examples = loader.get_few_shot_examples('ag_news', n=3)
# → [('Article text...', 'Sports'), ...]
```

**Datasets**:
- `ag_news` - 4 classes, 7.6K test samples
- `bbc_news` - 5 classes, ~445 test samples
- `20newsgroups` - 20 classes, ~7.5K test samples

---

### `PromptLoader` (prompt_loader.py)

Loads templates from `prompts/*.txt` and formats with data.

```python
loader = PromptLoader(prompts_dir="prompts")

# Available techniques
techniques = loader.get_available_techniques()
# → ['zero_shot', 'few_shot', 'constrained', 'chain_of_thought', 'self_consistency']

# Format prompt
prompt = loader.format_prompt(
    technique='few_shot',
    text='Article text here...',
    categories=['Sports', 'Business', 'World', 'Sci/Tech'],
    examples=[('Example text', 'Sports'), ...]
)
```

**Placeholders**:
- `{text}` → Article content
- `{categories}` → Comma-separated category list
- `{examples}` → Formatted as `- "text" → category`

---

### `ModelInference` (model_inference.py)

Wraps Ollama API. Handles prediction and response parsing.

```python
inference = ModelInference()

# Check models
models = inference.get_available_models()
available = inference.is_model_available('gemma3:12b')

# Get prediction
result = inference.predict(
    model_name='gemma3:12b',
    prompt='...',
    categories=['Sports', 'Business', 'World', 'Sci/Tech'],
    technique='zero_shot'
)
# → {'prediction': 'Sports', 'raw_response': 'Sports', ...}

# Self-consistency (5 samples, majority vote)
prediction = inference.predict_with_self_consistency(
    model_name='gemma3:12b', prompt='...', n_samples=5, temperature=0.7, categories=[...]
)
```

**Response parsing** tries:
1. Exact match (case-insensitive)
2. Category substring in response
3. First word/line extraction
4. Fuzzy matching for variations

---

### `ClassificationMetrics` (evaluation.py)

Computes metrics.

```python
calc = ClassificationMetrics()

metrics = calc.compute_metrics(
    y_true=['Sports', 'Business', ...],
    y_pred=['Sports', 'World', ...],
    categories=['Sports', 'Business', 'World', 'Sci/Tech']
)
# → {accuracy, precision, recall, f1_score, per_class: {...}}

calc.print_metrics(metrics, 'ag_news', 'gemma3:12b', 'zero_shot')
```

---

## CLI Usage

```bash
# Flexible experiment (comma-separated or "all")
uv run python experiments.py \
  --models gemma3:12b,phi4:14b \
  --datasets ag_news,bbc_news \
  --techniques zero_shot,few_shot \
  --limit 100 \
  --run-name my_experiment

# Single model on everything
uv run python experiments.py --models gemma3:12b --datasets all --techniques all

# Full pipeline
uv run python experiments.py --full --run-name baseline

# Legacy single experiment
uv run python experiments.py --dataset ag_news --model gemma3:12b --technique zero_shot
```

---

## Adding New Components

### New Prompting Technique

1. Create `prompts/my_technique.txt` with placeholders `{text}`, `{categories}`, `{examples}`
2. Add to `prompt_loader.py`: `self.prompt_templates['my_technique'] = 'my_technique.txt'`
3. Add to `experiments.py`: `self.techniques = [..., 'my_technique']`
4. (Optional) Add special handling in `model_inference.py`

### New Dataset

1. Add config to `dataset_loader.py`:
   ```python
   self.dataset_configs['my_dataset'] = {
       'categories': [...],
       'train_file': 'my_dataset_train.csv',
       'test_file': 'my_dataset_test.csv'
   }
   ```
2. Add download method `_process_my_dataset()` that creates CSV with columns: `text`, `label`, `label_text`
3. Add to `_download_and_preprocess()` switch statement
4. Add to `experiments.py`: `self.datasets = [..., 'my_dataset']`

### New Model

Just install via Ollama: `ollama pull model:size`. Pipeline auto-detects available models.

---

## Common Tasks

### Run Quick Test
```bash
uv run python experiments.py --models gemma3:4b --datasets ag_news --techniques zero_shot --limit 10
```

### Compare Techniques
```bash
uv run python experiments.py --models gemma3:12b --datasets ag_news --techniques all --run-name technique_comparison
```

### Analyze Results
```python
import pandas as pd
df = pd.read_csv('results/run_XXXXXX_my_run/summary.csv')
print(df.groupby('technique')['accuracy'].mean().sort_values(ascending=False))
```

---

## File Formats

### Prompt Template (`prompts/*.txt`)
```
Classify this article: {categories}
Article: {text}
Category:
```

### Dataset CSV (`data/*.csv`)
```csv
text,label,label_text
"Article content...",0,Sports
"Another article...",1,Business
```

### Results (`results/run_*/summary.csv`)
```csv
dataset,model,technique,accuracy,precision,recall,f1_score,parse_failure_rate,n_samples,timestamp
ag_news,gemma3:12b,zero_shot,0.85,0.82,0.80,0.81,0.02,100,2025-12-03T15:20:30
```

---

## Key Design Decisions

1. **Prompts as files**: Easy to edit without code changes
2. **CSV caching**: Download once, reuse forever
3. **Run folders**: Each experiment isolated with full config
4. **Fault tolerance**: Results saved after each experiment
5. **Flexible CLI**: Comma-separated lists or "all" keyword
6. **Local inference**: Ollama for reproducible, offline experiments

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | `ollama pull model:size` |
| Dataset error | `uv run python dataset_loader.py` |
| High parse failures | Edit prompts to be more explicit |
| Out of memory | Use smaller model or `--limit N` |

---

## Current State

- **Working**: All core functionality, flexible CLI, run folders
- **Models available**: Check with `ollama list`
- **Recent changes**: Added `--models/--datasets/--techniques` with comma-separated and "all" support
