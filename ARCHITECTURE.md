# Architecture & Technical Reference

This document provides a deep technical overview of the news classification pipeline for developers and LLM assistants working on feature enhancements.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI ENTRY POINT                                 │
│                            experiments.py --args                             │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ExperimentPipeline                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │DatasetLoader│  │PromptLoader │  │ModelInference│ │ClassificationMetrics│ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                     │            │
└─────────┼────────────────┼────────────────┼─────────────────────┼────────────┘
          │                │                │                     │
          ▼                ▼                ▼                     ▼
    ┌──────────┐    ┌───────────┐    ┌───────────┐         ┌───────────┐
    │  data/   │    │  prompts/ │    │  Ollama   │         │ Metrics   │
    │  *.csv   │    │  *.txt    │    │  API      │         │ Engine    │
    └──────────┘    └───────────┘    └───────────┘         └───────────┘
```

---

## Data Flow

### 1. Experiment Initialization

```python
# User runs:
python experiments.py --models gemma3:12b --datasets ag_news --techniques zero_shot --limit 10

# Flow:
main()
  → parse_list('gemma3:12b', all_models, 'models')  # Validate models
  → parse_list('ag_news', all_datasets, 'datasets')  # Validate datasets
  → parse_list('zero_shot', all_techniques, 'techniques')  # Validate techniques
  → pipeline.run_custom_experiment(models, datasets, techniques, limit, run_name)
```

### 2. Run Folder Creation

```python
pipeline._start_run(run_name='my_run')
  → timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  → folder_name = f"run_{timestamp}_{run_name}"  # e.g., run_20251203_151718_my_run
  → self.run_dir = Path('results') / folder_name
  → self.run_dir.mkdir(exist_ok=True)
```

### 3. Single Experiment Flow

```python
run_single_experiment(dataset_name, model_name, technique, limit)
  │
  ├── Check model availability
  │     → model_inference.is_model_available(model_name)
  │
  ├── Load test data
  │     → data_loader.load_dataset(dataset_name, split='test')
  │     → Returns: DataFrame[text, label, label_text]
  │
  ├── Get categories
  │     → data_loader.get_categories(dataset_name)
  │     → Returns: ['Sports', 'Business', 'Sci/Tech', 'World']
  │
  ├── Get few-shot examples (if technique != 'zero_shot')
  │     → data_loader.get_few_shot_examples(dataset_name, n=3)
  │     → Returns: [('Article text...', 'Sports'), ...]
  │
  ├── For each row in test_df:
  │     │
  │     ├── Truncate text (max 1000 chars)
  │     │
  │     ├── Format prompt
  │     │     → prompt_loader.format_prompt(technique, text, categories, examples)
  │     │     → Returns: "Classify this article: {text}..."
  │     │
  │     └── Get prediction
  │           → model_inference.predict(model_name, prompt, categories, technique)
  │           → Returns: {'prediction': 'Sports', 'raw_response': '...', ...}
  │
  ├── Compute metrics
  │     → metrics_calc.compute_metrics(y_true, y_pred, categories)
  │     → Returns: {accuracy, precision, recall, f1_score, per_class}
  │
  └── Return results dict
```

### 4. Result Saving

```python
_save_results(all_results)
  │
  ├── Save results.json (full details)
  │     → json.dump(results, f)
  │
  └── Save summary.csv
        → pd.DataFrame(summary_data).to_csv(csv_file)

_save_run_summary(results)
  │
  └── Save run_summary.json
        → {run_name, start_time, end_time, duration, avg_accuracy, best_result, worst_result}
```

---

## Module Details

### `dataset_loader.py`

**Purpose**: Download, preprocess, and cache datasets in consistent CSV format.

**Key Data Structures**:
```python
self.dataset_configs = {
    'ag_news': {
        'categories': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'train_file': 'ag_news_train.csv',
        'test_file': 'ag_news_test.csv'
    },
    'bbc_news': {...},
    '20newsgroups': {...}
}
```

**CSV Schema** (all datasets):
| Column | Type | Description |
|--------|------|-------------|
| `text` | str | Article content |
| `label` | int | Numeric label (0-indexed) |
| `label_text` | str | Human-readable category name |

**Download Sources**:
- AG News: `https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/`
- BBC News: Multiple fallback URLs (GitHub mirrors, Google Cloud Storage)
- 20 Newsgroups: scikit-learn `fetch_20newsgroups()`

**Key Methods**:
| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `load_dataset(name, split)` | dataset name, 'train'/'test' | DataFrame | Load cached data or download |
| `get_categories(name)` | dataset name | List[str] | Sorted unique categories |
| `get_few_shot_examples(name, n)` | dataset name, count | List[Tuple] | Sample (text, label) pairs |

---

### `prompt_loader.py`

**Purpose**: Load and format prompt templates from text files.

**Template Mapping**:
```python
self.prompt_templates = {
    'zero_shot': 'zero_shot.txt',
    'few_shot': 'few_shot.txt',
    'constrained': 'constrained.txt',
    'chain_of_thought': 'chain_of_thought.txt',
    'self_consistency': 'self_consistency.txt'
}
```

**Placeholder System**:
| Placeholder | Replacement | Example |
|-------------|-------------|---------|
| `{text}` | Article content | "Lakers win NBA championship..." |
| `{categories}` | Comma-separated list | "Sports, Business, Sci/Tech, World" |
| `{examples}` | Formatted few-shot examples | `- "Article..." → Sports` |

**Example Formatting**:
```python
def _format_examples(self, examples):
    # Input: [('Article text', 'Sports'), ...]
    # Output: '- "Article text" → Sports\n- "Other text" → Business'
    formatted = []
    for text, label in examples:
        formatted.append(f'- "{text}" → {label}')
    return '\n'.join(formatted)
```

---

### `model_inference.py`

**Purpose**: Wrap Ollama API for inference and parse model responses.

**Ollama API Interaction**:
```python
def generate(self, model_name, prompt, temperature=0.0, max_tokens=100):
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            'temperature': temperature,
            'num_predict': max_tokens,
        }
    )
    return response['response'].strip()
```

**Response Parsing Strategies** (in order):
1. **Direct match**: Response exactly matches category (case-insensitive)
2. **Substring match**: Category name appears in response
3. **First word/line**: Extract first token, match against categories
4. **Fuzzy matching**: Handle variations like "sci/tech" ↔ "science/technology"

```python
category_variations = {
    'sci/tech': ['scien', 'tech', 'sci'],
    'business': ['busi', 'econ', 'financ'],
    'entertainment': ['entertain', 'ent'],
    'politics': ['politic', 'polit']
}
```

**Self-Consistency Implementation**:
```python
def predict_with_self_consistency(self, model_name, prompt, n_samples=5, temperature=0.7, categories=None):
    predictions = []
    for i in range(n_samples):  # Generate 5 predictions
        response = self.generate(model_name, prompt, temperature=temperature)
        parsed = self.parse_category(response, categories)
        if parsed:
            predictions.append(parsed)
    
    # Majority vote
    vote_counts = Counter(predictions)
    majority_category, _ = vote_counts.most_common(1)[0]
    return majority_category
```

---

### `evaluation.py`

**Purpose**: Compute classification metrics.

**Metrics Computed**:
| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | correct / total | Overall correct predictions |
| Precision | TP / (TP + FP) | Per-class, then macro-averaged |
| Recall | TP / (TP + FN) | Per-class, then macro-averaged |
| F1-Score | 2 × (P × R) / (P + R) | Per-class, then macro-averaged |

**Return Structure**:
```python
{
    'accuracy': 0.85,
    'precision': 0.82,
    'recall': 0.80,
    'f1_score': 0.81,
    'per_class': {
        'Sports': {'precision': 0.9, 'recall': 0.85, 'f1_score': 0.87, 'support': 1900},
        'Business': {'precision': 0.8, 'recall': 0.78, 'f1_score': 0.79, 'support': 1900},
        ...
    }
}
```

---

### `experiments.py`

**Purpose**: Main orchestrator and CLI entry point.

**Class: ExperimentPipeline**

**Instance Variables**:
```python
self.data_loader = DatasetLoader()
self.prompt_loader = PromptLoader()
self.model_inference = ModelInference()
self.metrics_calc = ClassificationMetrics()

self.base_results_dir = Path("results")
self.run_dir = None  # Set when run starts
self.run_start_time = None

# Defaults (can be overridden by CLI)
self.models = ['gemma3:270m', 'gemma3:12b', 'phi4:14b', ...]
self.datasets = ['ag_news', 'bbc_news', '20newsgroups']
self.techniques = ['zero_shot', 'few_shot', 'constrained', 'chain_of_thought', 'self_consistency']
```

**CLI Argument Parsing**:
```python
parser.add_argument('--models', type=str)      # "gemma3:12b,phi4:14b" or "all"
parser.add_argument('--datasets', type=str)    # "ag_news,bbc_news" or "all"
parser.add_argument('--techniques', type=str)  # "zero_shot,few_shot" or "all"
parser.add_argument('--limit', type=int)       # e.g., 50
parser.add_argument('--run-name', type=str)    # e.g., "baseline_v1"
parser.add_argument('--full', action='store_true')  # Run all combinations
```

**List Parsing Logic**:
```python
def parse_list(value, all_values, name):
    if value is None:
        return None
    if value.lower() == 'all':
        return all_values
    items = [x.strip() for x in value.split(',')]
    # Validate and filter
    invalid = [x for x in items if x not in all_values]
    if invalid:
        print(f"Warning: Unknown {name}: {invalid}")
    return [x for x in items if x in all_values]
```

---

## Result File Formats

### `run_config.json`
```json
{
  "run_name": "baseline_v1",
  "start_time": "2025-12-03T15:17:18.123456",
  "datasets": ["ag_news", "bbc_news"],
  "models": ["gemma3:12b", "phi4:14b"],
  "techniques": ["zero_shot", "few_shot"],
  "limit": 100,
  "total_experiments": 8
}
```

### `results.json`
```json
[
  {
    "dataset": "ag_news",
    "model": "gemma3:12b",
    "technique": "zero_shot",
    "timestamp": "2025-12-03T15:20:30.123456",
    "n_samples": 100,
    "metrics": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.80,
      "f1_score": 0.81,
      "parse_failure_rate": 0.02
    },
    "per_class_metrics": {
      "Sports": {"precision": 0.9, "recall": 0.85, "f1_score": 0.87, "support": 25},
      ...
    }
  },
  ...
]
```

### `summary.csv`
| dataset | model | technique | accuracy | precision | recall | f1_score | parse_failure_rate | n_samples | timestamp |
|---------|-------|-----------|----------|-----------|--------|----------|-------------------|-----------|-----------|
| ag_news | gemma3:12b | zero_shot | 0.85 | 0.82 | 0.80 | 0.81 | 0.02 | 100 | 2025-12-03T15:20:30 |

### `run_summary.json`
```json
{
  "run_name": "baseline_v1",
  "start_time": "2025-12-03T15:17:18.123456",
  "end_time": "2025-12-03T16:45:30.654321",
  "duration_seconds": 5292.53,
  "duration_formatted": "1:28:12.530865",
  "total_experiments": 8,
  "avg_accuracy": 0.78,
  "avg_f1_score": 0.75,
  "best_result": {"dataset": "ag_news", "model": "phi4:14b", "technique": "few_shot", ...},
  "worst_result": {"dataset": "20newsgroups", "model": "gemma3:270m", "technique": "zero_shot", ...}
}
```

---

## Extension Points

### Adding a New Prompting Technique

**Files to modify**:
1. `prompts/my_technique.txt` - Create template
2. `prompt_loader.py` - Register template
3. `experiments.py` - Add to techniques list
4. `model_inference.py` - (Optional) Add special handling

**Example: Adding a "structured" technique**:
```python
# 1. prompts/structured.txt
"""
Task: Classify the following news article.
Categories: {categories}

Article:
{text}

Instructions: Output a JSON object with "category" and "confidence" fields.
Response:
"""

# 2. prompt_loader.py
self.prompt_templates['structured'] = 'structured.txt'

# 3. experiments.py
self.techniques = [..., 'structured']

# 4. model_inference.py (optional special handling)
if technique == 'structured':
    # Parse JSON response
    try:
        result = json.loads(raw_response)
        prediction = result.get('category')
    except:
        prediction = self.parse_category(raw_response, categories)
```

### Adding a New Dataset

**Files to modify**:
1. `dataset_loader.py` - Add config, download method
2. `experiments.py` - Add to datasets list

**Required CSV format**:
```
text,label,label_text
"Article content here...",0,CategoryName
"Another article...",1,OtherCategory
```

### Adding Custom Metrics

**File to modify**: `evaluation.py`

```python
def compute_metrics(self, y_true, y_pred, categories):
    # ... existing code ...
    
    # Add custom metric
    metrics['custom_metric'] = self._compute_custom_metric(y_true, y_pred)
    
    return metrics

def _compute_custom_metric(self, y_true, y_pred):
    # Your computation here
    return value
```

---

## Error Handling

### Model Unavailable
```python
if not self.model_inference.is_model_available(model_name):
    print(f"⚠️  Model {model_name} not available. Skipping.")
    return None
```

### Dataset Download Failure
```python
urls_to_try = [url1, url2, url3]
for url in urls_to_try:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        continue
raise Exception("Dataset unavailable from all sources")
```

### Parse Failure Tracking
```python
if result['prediction'] == 'unknown':
    parse_failures += 1

metrics['parse_failure_rate'] = parse_failures / len(test_df)
```

### Continuous Saving (Fault Tolerance)
```python
for dataset_name in run_datasets:
    for model_name in run_models:
        for technique in run_techniques:
            result = self.run_single_experiment(...)
            if result:
                all_results.append(result)
                self._save_results(all_results)  # Save after each experiment
```

---

## Performance Considerations

### Text Truncation
```python
if len(text) > 1000:
    text = text[:1000] + "..."
```

### Model Configuration
```python
ollama.generate(
    model=model_name,
    prompt=prompt,
    options={
        'temperature': 0.0,  # Deterministic (except self-consistency)
        'num_predict': 100,  # Max tokens
    }
)
```

### Self-Consistency Cost
- 5× API calls per prediction
- Higher temperature (0.7) for diversity
- Use sparingly on large test sets

---

## Testing

### Module Tests
```bash
uv run python dataset_loader.py   # Test dataset download
uv run python prompt_loader.py    # Test prompt formatting
uv run python model_inference.py  # Test Ollama connection
uv run python evaluation.py       # Test metrics computation
uv run python check_status.py     # Full system check
```

### Quick Experiment Test
```bash
uv run python experiments.py --models gemma3:4b --datasets ag_news --techniques zero_shot --limit 5
```
