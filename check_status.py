"""
System status checker - verifies that all components are ready for experiments.
"""

import sys
from pathlib import Path
from dataset_loader import DatasetLoader
from model_inference import ModelInference


def check_dependencies():
    """Check if required Python packages are installed."""
    print("Checking Python dependencies...")
    required = ['pandas', 'sklearn', 'requests', 'tqdm', 'ollama', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def check_datasets():
    """Check if datasets are downloaded and ready."""
    print("\nChecking datasets...")
    loader = DatasetLoader()
    datasets = ['ag_news', 'bbc_news', '20newsgroups']
    
    all_ready = True
    for dataset in datasets:
        try:
            train_df = loader.load_dataset(dataset, split='train')
            test_df = loader.load_dataset(dataset, split='test')
            print(f"  ✓ {dataset}: {len(train_df)} train, {len(test_df)} test samples")
        except Exception as e:
            print(f"  ✗ {dataset} - NOT READY ({str(e)[:50]}...)")
            all_ready = False
    
    if not all_ready:
        print("\n⚠️  Some datasets are missing.")
        print("Download with: python dataset_loader.py")
        return False
    return True


def check_prompts():
    """Check if prompt templates exist."""
    print("\nChecking prompt templates...")
    prompts_dir = Path('prompts')
    templates = ['zero_shot.txt', 'few_shot.txt', 'constrained.txt', 
                'chain_of_thought.txt', 'self_consistency.txt']
    
    all_exist = True
    for template in templates:
        template_path = prompts_dir / template
        if template_path.exists():
            print(f"  ✓ {template}")
        else:
            print(f"  ✗ {template} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some prompt templates are missing.")
        return False
    return True


def check_models():
    """Check available Ollama models."""
    print("\nChecking Ollama models...")
    inference = ModelInference()
    available = inference.get_available_models()
    
    recommended = [
        'gemma3:4b', 'gemma3:12b', 'phi4:14b', 'qwen3:8b',
        'mistral-small3.2:24b', 'deepseek-r1:8b', 'olmo2:7b'
    ]
    
    if not available:
        print("  ✗ No models found!")
        print("\n⚠️  No Ollama models installed.")
        print("Install with: ollama pull <model_name>")
        print("Example: ollama pull gemma3:4b")
        return False
    
    print(f"\n  Available models ({len(available)}):")
    for model in available:
        is_recommended = any(model.startswith(rec) for rec in recommended)
        marker = "✓" if is_recommended else " "
        print(f"    {marker} {model}")
    
    print(f"\n  Recommended models for this project:")
    for rec in recommended:
        has_it = any(model.startswith(rec) for model in available)
        marker = "✓" if has_it else "✗"
        print(f"    {marker} {rec}")
    
    return True


def check_directory_structure():
    """Check if required directories exist."""
    print("\nChecking directory structure...")
    dirs = ['data', 'prompts', 'results']
    
    all_exist = True
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            dir_path.mkdir(exist_ok=True)
            print(f"    Created {dir_name}/")
    
    return True


def main():
    """Run all system checks."""
    print("="*70)
    print("SYSTEM STATUS CHECK")
    print("="*70)
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Prompts", check_prompts),
        ("Datasets", check_datasets),
        ("Models", check_models),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results[name] = False
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ READY" if passed else "✗ NOT READY"
        print(f"  {status:15} - {name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All systems ready! You can start running experiments.")
        print("\nQuick start:")
        print("  python experiments.py --dataset ag_news --model gemma3:4b --technique zero_shot --limit 10")
    else:
        print("\n⚠️  System not fully ready. Please address the issues above.")
        print("\nQuick fix:")
        print("  1. pip install -r requirements.txt")
        print("  2. python dataset_loader.py")
        print("  3. ollama pull gemma3:4b")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
