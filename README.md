
# Prompt Enrichment in News Classification: Shifting the LLM Performance Frontier

## Abstract

News classification is a canonical challenge for evaluating the reasoning and generalization capabilities of Large Language Models (LLMs). While prompt engineering has become a standard tool, most approaches treat category labels as atomic tokens—missing the rich semantic context that real-world categories encode. This project investigates a new direction: **Prompt Enrichment**—the systematic augmentation of category labels with descriptive, contextual, or semantic information. The result is a modular, research-grade framework that demonstrates how enrichment can shift the performance frontier in zero-shot and few-shot news classification.

---

## Hypothesis

> **Does augmenting category labels with semantic context—"enrichment"—improve LLM zero-shot and few-shot accuracy in news classification?**

The core research question: Can LLMs leverage richer, more descriptive prompts to better map ambiguous news articles to the correct category, especially in the absence of fine-tuning?

---

## Methodology & Implementation

This project is built on a modular, SWE-principled framework that abstracts datasets, prompt templates, model APIs, and evaluation metrics. This architecture enables systematic experimentation across:

- **3 Datasets:** AG News, BBC News, 20 Newsgroups
- **Multiple LLMs:** (e.g., gemma3:12b, llama3.2:1b, gpt-oss:20b)
- **Prompting Techniques:** Baseline (atomic labels), Enriched (semantic labels), Zero-shot, Few-shot, Chain-of-Thought, and their enriched variants

The framework orchestrates full matrix sweeps (model × dataset × technique), ensuring reproducibility and extensibility for future research.

---

## The Enrichment Strategy: Intellectual Approach

Traditional prompts present category labels as mere tokens (e.g., "World", "Sports"). The **Enrichment Strategy** expands each label into a semantically loaded description (e.g., "World: International news, global events, and geopolitical developments"). This approach is inspired by the hypothesis that LLMs, when given richer context, can better disambiguate and generalize—especially in zero-shot and few-shot settings. Enrichment is applied systematically across all datasets and techniques, enabling direct comparison with baseline prompts.

---

## Breakthrough Results: Where Vibe Meets Data

**Key Findings (see [statistical_summary.txt](results/run_20251204_083400_baseline_vs_enriched_v2/statistical_summary.txt) and [main.pdf](project_report/main.pdf) for full analysis):**

- **Performance Frontier:** The best F1 score (95.43%) was achieved by gemma3:12b on BBC News with the enriched few-shot technique, demonstrating that prompt enrichment can unlock near-supervised performance in select settings.
- **Robustness:** Enrichment reduced parse failure rates (from 16.43% to 15.62% on average), indicating improved model alignment with task constraints.
- **Metric Distributions:** Radar plots and metric distribution figures ([fig2_model_radar.png](project_report/fig2_model_radar.png), [fig8_metric_distributions.png](project_report/fig8_metric_distributions.png)) reveal that enrichment narrows the gap between baseline and advanced prompting, especially in complex, multi-class settings (e.g., 20 Newsgroups).
- **Nuanced Impact:** While mean F1 improvement was modest (+0.04 percentage points), enrichment consistently improved precision and reduced failure rates, with the most pronounced gains in the hardest dataset (20 Newsgroups).
- **Technique Ranking:** Few-shot and enriched variants dominate the top ranks, with chain-of-thought enrichment showing the most variance—highlighting the importance of prompt design for LLM reliability.

---

## Conclusion: What This Proves

This project demonstrates that **Prompt Enrichment** is a viable, scalable strategy for improving LLM performance in news classification—especially in zero- and few-shot regimes. The results suggest that LLMs are sensitive to the semantic richness of prompts, and that careful prompt engineering can substitute for (or amplify) the effects of data-driven fine-tuning. For practitioners and researchers, this work provides both a blueprint for modular experimentation and a proof point for the power of prompt-centric research.

---

## Technical Deep Dive

For a comprehensive breakdown of methodology, results, and figures, see the [Technical Deep Dive (main.pdf)](project_report/main.pdf).

---

## Project Structure


---

## Project Structure

```
news_classification_prompts/
├── experiments.py          # Main orchestrator - CLI entry point
├── dataset_loader.py       # Dataset download, preprocessing, caching
├── prompt_loader.py        # Prompt template loading and formatting
├── model_inference.py      # Ollama API wrapper with response parsing
├── evaluation.py           # Metrics computation (accuracy, F1, etc.)
├── check_status.py         # System readiness verification
├── setup.sh                # Automated setup script
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
│
├── prompts/                # Editable prompt templates (plain text)
│   ├── zero_shot.txt       # No examples, direct classification
│   ├── few_shot.txt        # 3 labeled examples included
│   ├── constrained.txt     # Few-shot + strict output format
│   ├── chain_of_thought.txt # Few-shot + step-by-step reasoning
│   └── self_consistency.txt # Same as constrained, but sampled 5x
│
├── data/                   # Auto-downloaded datasets (CSV format)
│   ├── ag_news_train.csv   # 120,000 samples
│   ├── ag_news_test.csv    # 7,600 samples
│   ├── bbc_news_train.csv  # ~1,780 samples
│   ├── bbc_news_test.csv   # ~445 samples
│   ├── 20newsgroups_train.csv  # ~11,300 samples
│   └── 20newsgroups_test.csv   # ~7,500 samples
│
└── results/                # Experiment outputs (auto-created)
    └── run_YYYYMMDD_HHMMSS[_name]/
        ├── run_config.json     # What was run (models, datasets, techniques)
        ├── results.json        # Full results with per-class metrics
        ├── summary.csv         # Quick tabular overview
        └── run_summary.json    # High-level stats (duration, best/worst)
```

---

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **[uv](https://docs.astral.sh/uv/)** package manager:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **[Ollama](https://ollama.ai/)** with at least one model installed:
   ```bash
   ollama pull gemma3:4b
   ```

### Setup

```bash
# Clone/navigate to project
cd news_classification_prompts

# Automated setup (installs deps + downloads datasets)
./setup.sh

# Or manual setup
uv pip install -r requirements.txt
uv run python dataset_loader.py
```

### Verify Installation

```bash
uv run python check_status.py
```

### Run Your First Experiment

```bash
# Quick test (10 samples)
uv run python experiments.py \
  --models gemma3:4b \
  --datasets ag_news \
  --techniques zero_shot \
  --limit 10 \
  --run-name first_test
```

---

## Usage

### Flexible Experiment Configuration

The `experiments.py` CLI supports flexible combinations with comma-separated lists or `all`:

```bash
# Single model on all datasets and techniques
uv run python experiments.py --models gemma3:12b --datasets all --techniques all

# Multiple models on one dataset
uv run python experiments.py --models gemma3:12b,phi4:14b --datasets ag_news --techniques all

# Compare specific techniques
uv run python experiments.py --models gemma3:12b --datasets ag_news,bbc_news --techniques zero_shot,few_shot

# Full pipeline (all available models × all datasets × all techniques)
uv run python experiments.py --full --run-name baseline_v1

# Quick test with sample limit
uv run python experiments.py --full --limit 50 --run-name quick_test
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--models` | Comma-separated model names or `all` |
| `--datasets` | Comma-separated dataset names or `all` (ag_news, bbc_news, 20newsgroups) |
| `--techniques` | Comma-separated techniques or `all` |
| `--limit N` | Test on first N samples only (for quick testing) |
| `--run-name NAME` | Custom name for this experiment run |
| `--full` | Run all combinations (shorthand for `--models all --datasets all --techniques all`) |

### Legacy Single-Experiment Syntax

```bash
uv run python experiments.py --dataset ag_news --model gemma3:4b --technique zero_shot
```

---

## Datasets

All datasets are automatically downloaded and preprocessed to a consistent CSV format with columns: `text`, `label`, `label_text`.

| Dataset | Classes | Train Samples | Test Samples | Categories |
|---------|---------|---------------|--------------|------------|
| **AG News** | 4 | 120,000 | 7,600 | World, Sports, Business, Sci/Tech |
| **BBC News** | 5 | ~1,780 | ~445 | business, entertainment, politics, sport, tech |
| **20 Newsgroups** | 20 | ~11,300 | ~7,500 | Various newsgroup topics |

---

## Prompting Techniques

Each technique is stored as an editable text file in `prompts/`:

### 1. Zero-Shot (`zero_shot.txt`)
Direct classification with no examples.
```
Classify this news article into one category: {categories}
Article: {text}
Category:
```

### 2. Few-Shot (`few_shot.txt`)
Includes 3 labeled examples from the training set.
```
Classify news articles into categories.
Examples:
{examples}
Article: {text}
Category:
```

### 3. Constrained (`constrained.txt`)
Few-shot with explicit output format constraint.
```
...
Answer with ONLY ONE WORD from: {categories}
```

### 4. Chain-of-Thought (`chain_of_thought.txt`)
Few-shot with step-by-step reasoning instructions.
```
Let's think step by step:
1. Identify the main topic
2. Determine the best category
3. Provide your answer
```

### 5. Self-Consistency (`self_consistency.txt`)
Same prompt as constrained, but generates 5 predictions at temperature=0.7 and takes majority vote.

### Template Placeholders

- `{text}` → The article text to classify
- `{categories}` → Comma-separated list of valid categories
- `{examples}` → Formatted few-shot examples (auto-generated)

---

## Experiment Results

Each experiment run creates its own folder:

```
results/run_20251203_151718_baseline_v1/
├── run_config.json     # Configuration: models, datasets, techniques, limit
├── results.json        # Full results with per-class metrics for each experiment
├── summary.csv         # CSV with one row per experiment
└── run_summary.json    # High-level stats: duration, avg accuracy, best/worst
```

### Metrics Computed

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Macro-averaged precision across all classes
- **Recall**: Macro-averaged recall across all classes
- **F1-Score**: Macro-averaged F1 (harmonic mean of precision & recall)
- **Parse Failure Rate**: Percentage of responses that couldn't be parsed to a category

### Viewing Results

```bash
# List all runs
ls results/

# View run summary
cat results/run_20251203_151718_baseline_v1/run_summary.json

# View CSV summary
cat results/run_20251203_151718_baseline_v1/summary.csv

# Analyze with Python
uv run python -c "
import pandas as pd
df = pd.read_csv('results/run_20251203_151718_baseline_v1/summary.csv')
print(df.groupby('technique')['accuracy'].mean().sort_values(ascending=False))
"
```

---

## Module Documentation

### `experiments.py` - Main Orchestrator

The central entry point that coordinates all other modules.

**Key Class: `ExperimentPipeline`**

```python
pipeline = ExperimentPipeline()

# Run custom experiment
pipeline.run_custom_experiment(
    models=['gemma3:12b', 'phi4:14b'],
    datasets=['ag_news', 'bbc_news'],
    techniques=['zero_shot', 'few_shot'],
    limit=100,
    run_name='my_experiment'
)

# Run full experiment (all combinations)
pipeline.run_full_experiment(limit=50, run_name='full_test')

# Run single experiment
result = pipeline.run_single_experiment('ag_news', 'gemma3:12b', 'zero_shot', limit=10)
```

**Key Methods:**
- `run_custom_experiment()` - Flexible experiment with specified combinations
- `run_full_experiment()` - All models × datasets × techniques
- `run_single_experiment()` - One specific combination
- `_start_run()` - Creates timestamped run folder
- `_save_run_config()` - Saves configuration JSON
- `_save_run_summary()` - Saves final summary with duration and stats

---

### `dataset_loader.py` - Dataset Management

Downloads, preprocesses, and caches datasets.

**Key Class: `DatasetLoader`**

```python
loader = DatasetLoader(data_dir="data")

# Load dataset (auto-downloads if missing)
test_df = loader.load_dataset('ag_news', split='test')
# Returns DataFrame with columns: text, label, label_text

# Get category names
categories = loader.get_categories('ag_news')
# Returns: ['Business', 'Sci/Tech', 'Sports', 'World']

# Get few-shot examples (one per category)
examples = loader.get_few_shot_examples('ag_news', n=3)
# Returns: [('Article text...', 'Sports'), ...]
```

**Supported Datasets:**
- `ag_news` - Downloaded from GitHub (CharCnn_Keras mirror)
- `bbc_news` - Downloaded from GitHub/GCS mirrors (with fallback)
- `20newsgroups` - Via scikit-learn's `fetch_20newsgroups()`

---

### `prompt_loader.py` - Prompt Template System

Loads and formats prompt templates from text files.

**Key Class: `PromptLoader`**

```python
loader = PromptLoader(prompts_dir="prompts")

# Get available techniques
techniques = loader.get_available_techniques()
# Returns: ['zero_shot', 'few_shot', 'constrained', 'chain_of_thought', 'self_consistency']

# Load raw template
template = loader.load_template('zero_shot')

# Format prompt with data
prompt = loader.format_prompt(
    technique='few_shot',
    text='Lakers win championship...',
    categories=['Sports', 'Business', 'World', 'Sci/Tech'],
    examples=[('Fed raises rates...', 'Business'), ...]
)
```

---

### `model_inference.py` - Ollama API Wrapper

Handles model interaction, prediction, and response parsing.

**Key Class: `ModelInference`**

```python
inference = ModelInference()

# Check available models
models = inference.get_available_models()
# Returns: ['gemma3:12b', 'phi4:14b', ...]

# Check if specific model is available
available = inference.is_model_available('gemma3:12b')

# Get prediction
result = inference.predict(
    model_name='gemma3:12b',
    prompt='Classify this article...',
    categories=['Sports', 'Business', 'World', 'Sci/Tech'],
    technique='zero_shot'
)
# Returns: {'prediction': 'Sports', 'raw_response': 'Sports', 'technique': 'zero_shot', ...}

# Self-consistency (5 samples, majority vote)
prediction = inference.predict_with_self_consistency(
    model_name='gemma3:12b',
    prompt='...',
    n_samples=5,
    temperature=0.7,
    categories=['Sports', 'Business', ...]
)
```

**Response Parsing Strategies:**
1. Direct match (case-insensitive)
2. Category appears anywhere in response
3. Extract first word/line
4. Fuzzy matching for variations (e.g., "sci/tech" vs "science/technology")

---

### `evaluation.py` - Metrics Computation

Computes classification metrics.

**Key Class: `ClassificationMetrics`**

```python
metrics_calc = ClassificationMetrics()

# Compute all metrics
metrics = metrics_calc.compute_metrics(
    y_true=['Sports', 'Business', 'Sports', ...],
    y_pred=['Sports', 'World', 'Sports', ...],
    categories=['Sports', 'Business', 'World', 'Sci/Tech']
)
# Returns: {
#   'accuracy': 0.85,
#   'precision': 0.82,
#   'recall': 0.80,
#   'f1_score': 0.81,
#   'per_class': {'Sports': {'precision': 0.9, 'recall': 0.85, ...}, ...}
# }

# Print formatted results
metrics_calc.print_metrics(metrics, 'ag_news', 'gemma3:12b', 'zero_shot')
```

---

## Adding New Components

### Add a New Dataset

1. Add config to `dataset_loader.py`:
   ```python
   self.dataset_configs['my_dataset'] = {
       'categories': ['cat1', 'cat2', 'cat3'],
       'train_file': 'my_dataset_train.csv',
       'test_file': 'my_dataset_test.csv'
   }
   ```

2. Add download method:
   ```python
   def _process_my_dataset(self):
       # Download and save to self.data_dir
       # Must create CSV with columns: text, label, label_text
   ```

3. Add to `_download_and_preprocess()`:
   ```python
   elif dataset_name == 'my_dataset':
       self._process_my_dataset()
   ```

4. Add to `experiments.py`:
   ```python
   self.datasets = ['ag_news', 'bbc_news', '20newsgroups', 'my_dataset']
   ```

### Add a New Prompting Technique

1. Create `prompts/my_technique.txt`:
   ```
   Your prompt template here...
   Article: {text}
   Categories: {categories}
   {examples}
   Answer:
   ```

2. Register in `prompt_loader.py`:
   ```python
   self.prompt_templates['my_technique'] = 'my_technique.txt'
   ```

3. Add to `experiments.py`:
   ```python
   self.techniques = [..., 'my_technique']
   ```

4. (Optional) Add special handling in `model_inference.py` if needed.

### Add a New Model

Just install it via Ollama:
```bash
ollama pull new-model:7b
```

The pipeline automatically detects available models. To add it to the default list:
```python
# In experiments.py
self.models = [..., 'new-model:7b']
```

---

## Troubleshooting

### Model Not Available
```bash
ollama list              # See installed models
ollama pull gemma3:4b    # Install a model
```

### Dataset Download Fails
```bash
# Re-run dataset download
uv run python dataset_loader.py

# Check data folder
ls -la data/
```

### High Parse Failure Rate (>10%)
- Check raw responses in `results.json`
- Make prompts more explicit about output format
- Add parsing rules to `model_inference.py::parse_category()`

### Out of Memory
- Use smaller models (gemma3:4b, llama3.2:1b)
- Use `--limit` to test on fewer samples
- Text is auto-truncated to 1000 characters

---

## Research Context

This pipeline implements the experimental methodology from:

> "Systematic Evaluation of Prompt Engineering Techniques for News Classification with Open-Source Large Language Models"

The goal is to systematically compare prompting strategies across multiple LLMs and datasets to identify best practices for news classification without fine-tuning.

---

## License

Research/educational use. See individual dataset licenses for data usage terms.
