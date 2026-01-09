# Quick Start Guide

Get running in 5 minutes.

---

## Prerequisites

1. **Python 3.8+**
2. **uv package manager**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** with at least one model:
   ```bash
   # Install from https://ollama.ai
   ollama pull gemma3:4b
   ```

---

## Setup (1 minute)

```bash
# Automated (recommended)
./setup.sh

# Or manual
uv pip install -r requirements.txt
uv run python dataset_loader.py
```

---

## Verify (10 seconds)

```bash
uv run python check_status.py
```

Expected output:
```
✓ Dependencies installed
✓ Datasets downloaded
✓ Prompts ready
✓ Ollama models available: gemma3:4b, ...
```

---

## Run Experiments

### Quick Test (30 seconds)

```bash
uv run python experiments.py \
  --models gemma3:4b \
  --datasets ag_news \
  --techniques zero_shot \
  --limit 10 \
  --run-name quick_test
```

### Single Model, All Datasets & Techniques

```bash
uv run python experiments.py \
  --models gemma3:12b \
  --datasets all \
  --techniques all \
  --run-name gemma_full
```

### Compare Multiple Models

```bash
uv run python experiments.py \
  --models gemma3:12b,phi4:14b,qwen3:8b \
  --datasets ag_news \
  --techniques zero_shot,few_shot \
  --run-name model_comparison
```

### Full Pipeline (All Combinations)

```bash
# WARNING: Takes hours depending on hardware
uv run python experiments.py --full --run-name baseline_v1

# Or quick test version
uv run python experiments.py --full --limit 50 --run-name quick_full
```

---

## View Results

```bash
# List runs
ls results/

# View summary
cat results/run_*_quick_test/run_summary.json

# CSV overview
cat results/run_*_quick_test/summary.csv

# Detailed analysis
uv run python -c "
import pandas as pd
import glob, os

# Find latest run
runs = sorted(glob.glob('results/run_*'))
latest = runs[-1]
print(f'Analyzing: {latest}')

df = pd.read_csv(f'{latest}/summary.csv')
print('\nTop 5 by Accuracy:')
print(df.nlargest(5, 'accuracy')[['model', 'technique', 'dataset', 'accuracy']])
print('\nBy Technique:')
print(df.groupby('technique')['accuracy'].mean().sort_values(ascending=False))
"
```

---

## Customize Prompts

Edit any file in `prompts/`:

```bash
# Edit zero-shot prompt
nano prompts/zero_shot.txt

# Test immediately
uv run python experiments.py \
  --models gemma3:4b \
  --datasets ag_news \
  --techniques zero_shot \
  --limit 10
```

### Placeholders

- `{text}` → Article content
- `{categories}` → Valid category names
- `{examples}` → Few-shot examples

---

## Common Issues

### "Model not available"
```bash
ollama list              # See installed models
ollama pull gemma3:4b    # Install a model
```

### "Dataset not found"
```bash
uv run python dataset_loader.py
```

### High parse failure rate
- Edit prompts to be more explicit
- Add "Answer with ONLY ONE WORD" instructions

---

## Next Steps

- 📖 **README.md** - Full documentation
- 🏗️ **ARCHITECTURE.md** - Technical deep-dive
- 🧪 Create custom prompts in `prompts/`
- 📊 Analyze results in `results/`
