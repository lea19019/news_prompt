"""
Main experiment pipeline for news classification with prompt engineering.
Runs comprehensive evaluation across models, datasets, and prompting techniques.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from dataset_loader import DatasetLoader
from prompt_loader import PromptLoader
from model_inference import ModelInference
from evaluation import ClassificationMetrics


class ExperimentPipeline:
    """Main pipeline for running classification experiments."""
    
    def __init__(self, results_dir="results", run_name=None):
        self.data_loader = DatasetLoader()
        self.prompt_loader = PromptLoader()
        self.model_inference = ModelInference()
        self.metrics_calc = ClassificationMetrics()
        
        self.base_results_dir = Path(results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Create run-specific directory
        self.run_name = run_name
        self.run_dir = None
        self.run_start_time = None
        
        # Model configurations from project description
        self.models = [
            'gemma3:270m',
            'gemma3:12b',
            'phi4:14b',
            'qwen3:8b',
            'qwen3:32b',
            'gpt-oss:20b',
            'llama3.2:1b',
            'mistral:7b',
            'r1-1776:70b'
        ]
        
        # Dataset configurations
        self.datasets = ['ag_news', 'bbc_news', '20newsgroups']
        
        # Prompting techniques
        self.techniques = [
            'zero_shot',
            'few_shot_3',
            'few_shot_5',
            'chain_of_thought',
            'zero_shot_enriched',
            'few_shot_3_enriched',
            'few_shot_5_enriched',
            'chain_of_thought_enriched'
        ]
    
    def _stratified_sample(self, df, n, random_state=42):
        """
        Perform stratified sampling to get n samples with proportional category representation.
        
        Args:
            df: DataFrame with 'label_text' column
            n: Number of samples to return
            random_state: Random seed for reproducibility
        
        Returns:
            DataFrame with n stratified samples
        """
        from sklearn.model_selection import train_test_split
        
        # If n >= len(df), return the full dataframe
        if n >= len(df):
            return df
        
        # Calculate the fraction to sample
        frac = n / len(df)
        
        try:
            # Use train_test_split for stratified sampling
            # We split and keep only the "test" portion of size n
            _, sampled_df = train_test_split(
                df, 
                test_size=frac, 
                stratify=df['label_text'],
                random_state=random_state
            )
            return sampled_df.reset_index(drop=True)
        except ValueError:
            # Fallback if stratification fails (e.g., too few samples in a class)
            # Use simple random sampling instead
            print("⚠️  Stratified sampling failed, using random sampling")
            return df.sample(n=n, random_state=random_state).reset_index(drop=True)
    
    def _start_run(self, run_name=None):
        """Initialize a new experiment run with its own folder."""
        self.run_start_time = datetime.now()
        timestamp = self.run_start_time.strftime('%Y%m%d_%H%M%S')
        
        if run_name:
            folder_name = f"run_{timestamp}_{run_name}"
        else:
            folder_name = f"run_{timestamp}"
        
        self.run_dir = self.base_results_dir / folder_name
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Created run folder: {self.run_dir}")
        return self.run_dir
    
    def _get_results_dir(self):
        """Get the current results directory (run folder or base folder)."""
        if self.run_dir:
            return self.run_dir
        return self.base_results_dir
    
    def run_single_experiment(self, dataset_name, model_name, technique, limit=None):
        """
        Run a single experiment: one dataset, one model, one technique.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the Ollama model
            technique: Prompting technique to use
            limit: Limit number of test samples (for quick testing)
        
        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*70}")
        print(f"Running: {dataset_name} | {model_name} | {technique}")
        print('='*70)
        
        # Check if model is available
        if not self.model_inference.is_model_available(model_name):
            print(f"⚠️  Model {model_name} not available. Skipping.")
            return None
        
        # Load test data
        try:
            test_df = self.data_loader.load_dataset(dataset_name, split='test')
            categories = self.data_loader.get_categories(dataset_name)
            
            # Limit for testing with stratified sampling
            if limit:
                test_df = self._stratified_sample(test_df, limit, random_state=42)
            
            print(f"Test samples: {len(test_df)}")
            print(f"Categories: {categories}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        # Determine number of examples needed based on technique
        n_examples = 0
        if 'few_shot_3' in technique:
            n_examples = 3
        elif 'few_shot_5' in technique:
            n_examples = 5
        elif technique == 'few_shot':
            n_examples = 3
        
        # Get few-shot examples if needed
        examples = None
        if n_examples > 0:
            examples = self.data_loader.get_few_shot_examples(dataset_name, n=n_examples)
        
        # Run predictions
        predictions = []
        raw_responses = []
        parse_failures = 0
        
        print(f"Generating predictions...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            text = row['text']
            
            # Handle NaN/missing text values
            if pd.isna(text):
                text = ""
            else:
                text = str(text)
            
            # Truncate very long texts to avoid context length issues
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            # Format prompt
            prompt = self.prompt_loader.format_prompt(
                technique=technique,
                text=text,
                categories=categories,
                examples=examples,
                dataset_name=dataset_name
            )
            
            # Get prediction
            result = self.model_inference.predict(
                model_name=model_name,
                prompt=prompt,
                categories=categories,
                technique=technique
            )
            
            predictions.append(result['prediction'])
            raw_responses.append(result['raw_response'])
            
            if result['prediction'] == 'unknown':
                parse_failures += 1
        
        # Compute metrics
        y_true = test_df['label_text'].tolist()
        y_pred = predictions
        
        metrics = self.metrics_calc.compute_metrics(y_true, y_pred, categories)
        
        # Add metadata
        metrics['parse_failure_rate'] = parse_failures / len(test_df)
        
        # Print results
        self.metrics_calc.print_metrics(metrics, dataset_name, model_name, technique)
        print(f"Parse failures: {parse_failures}/{len(test_df)} ({metrics['parse_failure_rate']:.2%})")
        
        # Prepare results dictionary
        results = {
            'dataset': dataset_name,
            'model': model_name,
            'technique': technique,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(test_df),
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'parse_failure_rate': metrics['parse_failure_rate']
            },
            'per_class_metrics': metrics['per_class']
        }
        
        return results
    
    def run_full_experiment(self, limit=None, save_results=True, run_name=None):
        """
        Run the full experimental pipeline across all combinations.
        
        Args:
            limit: Limit number of test samples per experiment (for quick testing)
            save_results: Whether to save results to file
            run_name: Optional name for this experiment run
        """
        return self.run_custom_experiment(
            models=self.models,
            datasets=self.datasets,
            techniques=self.techniques,
            limit=limit,
            save_results=save_results,
            run_name=run_name
        )
    
    def run_custom_experiment(self, models=None, datasets=None, techniques=None,
                               limit=None, save_results=True, run_name=None):
        """
        Run experiments with custom combinations of models, datasets, and techniques.
        
        Args:
            models: List of model names to test (defaults to all available)
            datasets: List of dataset names to test (defaults to all)
            techniques: List of technique names to test (defaults to all)
            limit: Limit number of test samples per experiment
            save_results: Whether to save results to file
            run_name: Optional name for this experiment run
        
        Returns:
            List of all experiment results
        """
        # Use defaults if not specified
        models = models or self.models
        datasets = datasets or self.datasets
        techniques = techniques or self.techniques
        
        all_results = []
        
        # Start a new run (creates dedicated folder)
        self._start_run(run_name=run_name)
        
        # Update instance variables for this run (used in config saving)
        run_models = models
        run_datasets = datasets
        run_techniques = techniques
        
        print(f"\n{'#'*70}")
        print(f"STARTING CUSTOM EXPERIMENT PIPELINE")
        print(f"{'#'*70}")
        print(f"Run folder: {self.run_dir}")
        print(f"Datasets: {run_datasets}")
        print(f"Models: {run_models}")
        print(f"Techniques: {run_techniques}")
        print(f"Total combinations: {len(run_datasets) * len(run_models) * len(run_techniques)}")
        
        if limit:
            print(f"⚠️  TESTING MODE: Limited to {limit} samples per experiment")
        
        print(f"{'#'*70}\n")
        
        # Save run configuration
        self._save_run_config(limit, models=run_models, datasets=run_datasets, techniques=run_techniques)
        
        # Iterate through all combinations
        for dataset_name in run_datasets:
            for model_name in run_models:
                for technique in run_techniques:
                    result = self.run_single_experiment(
                        dataset_name, model_name, technique, limit=limit
                    )
                    
                    if result:
                        all_results.append(result)
                        
                        # Save intermediate results
                        if save_results:
                            self._save_results(all_results)
        
        # Generate and save summary
        self._print_summary(all_results)
        self._save_run_summary(all_results)
        
        return all_results
    
    def _save_run_config(self, limit=None, models=None, datasets=None, techniques=None):
        """Save the configuration for this run."""
        run_models = models or self.models
        run_datasets = datasets or self.datasets
        run_techniques = techniques or self.techniques
        
        config = {
            'run_name': self.run_name,
            'start_time': self.run_start_time.isoformat(),
            'datasets': run_datasets,
            'models': run_models,
            'techniques': run_techniques,
            'limit': limit,
            'total_experiments': len(run_datasets) * len(run_models) * len(run_techniques)
        }
        
        config_file = self._get_results_dir() / 'run_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _save_run_summary(self, results):
        """Save a final summary for the run."""
        if not results:
            return
        
        run_end_time = datetime.now()
        duration = run_end_time - self.run_start_time
        
        summary = {
            'run_name': self.run_name,
            'start_time': self.run_start_time.isoformat(),
            'end_time': run_end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration),
            'total_experiments': len(results),
            'avg_accuracy': sum(r['metrics']['accuracy'] for r in results) / len(results),
            'avg_f1_score': sum(r['metrics']['f1_score'] for r in results) / len(results),
            'best_result': max(results, key=lambda x: x['metrics']['accuracy']),
            'worst_result': min(results, key=lambda x: x['metrics']['accuracy'])
        }
        
        summary_file = self._get_results_dir() / 'run_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 Run summary saved to: {summary_file}")
    
    def _save_results(self, results):
        """Save results to JSON and CSV files."""
        results_dir = self._get_results_dir()
        
        # Save detailed JSON (always overwrite with latest)
        json_file = results_dir / 'results.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for result in results:
            row = {
                'dataset': result['dataset'],
                'model': result['model'],
                'technique': result['technique'],
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1_score': result['metrics']['f1_score'],
                'parse_failure_rate': result['metrics']['parse_failure_rate'],
                'n_samples': result['n_samples'],
                'timestamp': result['timestamp']
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        csv_file = results_dir / 'summary.csv'
        summary_df.to_csv(csv_file, index=False)
        
        print(f"\n💾 Results saved to: {results_dir}/")
        print(f"   - results.json ({len(results)} experiments)")
        print(f"   - summary.csv")
    
    def _print_summary(self, results):
        """Print a summary of all results."""
        if not results:
            print("\n⚠️  No results to summarize.")
            return
        
        print(f"\n\n{'#'*70}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'#'*70}\n")
        
        df = pd.DataFrame([
            {
                'dataset': r['dataset'],
                'model': r['model'],
                'technique': r['technique'],
                'accuracy': r['metrics']['accuracy'],
                'f1_score': r['metrics']['f1_score']
            }
            for r in results
        ])
        
        print("Top 10 Results by Accuracy:")
        print(df.nlargest(10, 'accuracy')[['dataset', 'model', 'technique', 'accuracy', 'f1_score']])
        
        print("\n\nAverage Performance by Technique:")
        print(df.groupby('technique')[['accuracy', 'f1_score']].mean().sort_values('accuracy', ascending=False))
        
        print("\n\nAverage Performance by Model:")
        print(df.groupby('model')[['accuracy', 'f1_score']].mean().sort_values('accuracy', ascending=False))
        
        print("\n\nAverage Performance by Dataset:")
        print(df.groupby('dataset')[['accuracy', 'f1_score']].mean().sort_values('accuracy', ascending=False))
        
        print(f"\n{'#'*70}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run news classification experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single model on all datasets and prompts
  python experiments.py --models gemma3:12b --datasets all --techniques all

  # Test multiple models on one dataset
  python experiments.py --models gemma3:12b,phi4:14b --datasets ag_news --techniques all

  # Compare techniques on specific models and datasets  
  python experiments.py --models gemma3:12b --datasets ag_news,bbc_news --techniques zero_shot,few_shot

  # Full pipeline (all combinations)
  python experiments.py --full

  # Quick test with sample limit
  python experiments.py --models gemma3:12b --datasets all --techniques all --limit 10 --run-name quick_test

  # Legacy single experiment syntax (still works)
  python experiments.py --dataset ag_news --model gemma3:12b --technique zero_shot
        """
    )
    
    # New flexible arguments (plural, comma-separated)
    parser.add_argument('--models', type=str, 
                        help='Comma-separated list of models, or "all" for all available models')
    parser.add_argument('--datasets', type=str,
                        help='Comma-separated list of datasets, or "all" for all datasets')
    parser.add_argument('--techniques', type=str,
                        help='Comma-separated list of techniques, or "all" for all techniques')
    
    # Legacy single-value arguments (for backward compatibility)
    parser.add_argument('--dataset', type=str, help='Single dataset (legacy, use --datasets)')
    parser.add_argument('--model', type=str, help='Single model (legacy, use --models)')
    parser.add_argument('--technique', type=str, help='Single technique (legacy, use --techniques)')
    
    parser.add_argument('--limit', type=int, help='Limit number of test samples (for testing)')
    parser.add_argument('--full', action='store_true', help='Run full experiment pipeline (all combinations)')
    parser.add_argument('--run-name', type=str, help='Name for this experiment run (used in folder name)')
    
    args = parser.parse_args()
    
    pipeline = ExperimentPipeline()
    
    # Helper to parse comma-separated or "all"
    def parse_list(value, all_values, name):
        if value is None:
            return None
        if value.lower() == 'all':
            return all_values
        items = [x.strip() for x in value.split(',')]
        # Validate items
        invalid = [x for x in items if x not in all_values]
        if invalid:
            print(f"⚠️  Warning: Unknown {name}: {invalid}")
            print(f"   Available: {all_values}")
        return [x for x in items if x in all_values]
    
    # Get all available options
    all_models = pipeline.model_inference.get_available_models()
    all_datasets = pipeline.datasets
    all_techniques = pipeline.techniques
    
    if args.full:
        # Run full pipeline (all combinations)
        pipeline.run_full_experiment(limit=args.limit, run_name=args.run_name)
    
    elif args.models or args.datasets or args.techniques:
        # New flexible mode: parse lists
        models = parse_list(args.models, all_models, 'models') or all_models
        datasets = parse_list(args.datasets, all_datasets, 'datasets') or all_datasets
        techniques = parse_list(args.techniques, all_techniques, 'techniques') or all_techniques
        
        # Run custom experiment with specified combinations
        pipeline.run_custom_experiment(
            models=models,
            datasets=datasets,
            techniques=techniques,
            limit=args.limit,
            run_name=args.run_name
        )
    
    elif args.dataset and args.model and args.technique:
        # Legacy single experiment mode
        run_name = args.run_name or f"{args.dataset}_{args.model.replace(':', '-')}_{args.technique}"
        pipeline._start_run(run_name=run_name)
        result = pipeline.run_single_experiment(
            args.dataset, args.model, args.technique, limit=args.limit
        )
        if result:
            pipeline._save_results([result])
            pipeline._save_run_summary([result])
    
    else:
        # Interactive mode - show available options
        print("News Classification Experiment Pipeline")
        print("="*70)
        print("\nAvailable models (installed in Ollama):")
        for model in all_models:
            print(f"  - {model}")
        
        print("\nAvailable datasets:")
        for dataset in all_datasets:
            print(f"  - {dataset}")
        
        print("\nAvailable techniques:")
        for technique in all_techniques:
            print(f"  - {technique}")
        
        print("\n" + "="*70)
        print("Usage examples:")
        print("\n  # Flexible experiment configuration:")
        print("  python experiments.py --models gemma3:12b --datasets all --techniques all")
        print("  python experiments.py --models gemma3:12b,phi4:14b --datasets ag_news --techniques zero_shot,few_shot")
        print("  python experiments.py --models all --datasets bbc_news --techniques chain_of_thought --limit 20")
        print("\n  # Full pipeline (all combinations):")
        print("  python experiments.py --full")
        print("  python experiments.py --full --limit 50 --run-name quick_test")
        print("\n  # Legacy single experiment:")
        print("  python experiments.py --dataset ag_news --model gemma3:12b --technique zero_shot")
        print("\nResults are saved in: results/run_YYYYMMDD_HHMMSS[_name]/")


if __name__ == "__main__":
    main()