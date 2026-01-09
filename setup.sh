#!/bin/bash
# Quick setup script for news classification experiments

echo "=================================================="
echo "News Classification Experiments - Setup Script"
echo "=================================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed!"
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✓ uv package manager found"
echo ""

# Install dependencies using uv pip
echo "Installing Python dependencies with uv..."
uv pip install -r requirements.txt

echo ""
echo "Downloading and preprocessing datasets..."
echo "This may take a few minutes..."
uv run python dataset_loader.py

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "✓ Dependencies installed"
echo "✓ Datasets downloaded and preprocessed"
echo ""
echo "Next steps:"
echo ""
echo "1. Install Ollama models (if not already installed):"
echo "   ollama pull gemma3:4b"
echo "   ollama pull gemma3:12b"
echo "   ollama pull phi4:14b"
echo "   ollama pull qwen3:8b"
echo ""
echo "2. Check system status:"
echo "   uv run python check_status.py"
echo ""
echo "3. Run a test experiment:"
echo "   uv run python experiments.py --dataset ag_news --model gemma3:4b --technique zero_shot --limit 10"
echo ""
echo "4. Check available models:"
echo "   ollama list"
echo ""
echo "5. Run full pipeline:"
echo "   uv run python experiments.py --full"
echo ""
echo "=================================================="
