"""
Results Analysis and Visualization Script for News Classification Experiments.

Generates tables, charts, and insights from experiment results.

Usage:
    python analyze_results.py <run_folder>
    python analyze_results.py results/run_20251204_083400_baseline_vs_enriched_v2
    
    # Save figures to files instead of displaying
    python analyze_results.py <run_folder> --save
    
    # Generate HTML report
    python analyze_results.py <run_folder> --html
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


def load_results(run_folder: str) -> tuple[dict, dict, list]:
    """Load all JSON files from a run folder."""
    run_path = Path(run_folder)
    
    with open(run_path / 'run_config.json') as f:
        config = json.load(f)
    
    with open(run_path / 'run_summary.json') as f:
        summary = json.load(f)
    
    with open(run_path / 'results.json') as f:
        results = json.load(f)
    
    return config, summary, results


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert results list to a flat DataFrame."""
    rows = []
    for r in results:
        row = {
            'dataset': r['dataset'],
            'model': r['model'],
            'technique': r['technique'],
            'accuracy': r['metrics']['accuracy'],
            'precision': r['metrics']['precision'],
            'recall': r['metrics']['recall'],
            'f1_score': r['metrics']['f1_score'],
            'parse_failure_rate': r['metrics']['parse_failure_rate'],
            'n_samples': r['n_samples'],
            'timestamp': r['timestamp']
        }
        # Add flag for enriched vs baseline
        row['is_enriched'] = '_enriched' in r['technique']
        row['base_technique'] = r['technique'].replace('_enriched', '')
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_run_summary(config: dict, summary: dict):
    """Print overview of the experiment run."""
    print_header("EXPERIMENT RUN SUMMARY")
    
    duration = timedelta(seconds=summary['duration_seconds'])
    
    print(f"📅 Start Time:     {summary['start_time']}")
    print(f"📅 End Time:       {summary['end_time']}")
    print(f"⏱️  Duration:       {duration}")
    print(f"🧪 Total Experiments: {summary['total_experiments']}")
    print(f"📊 Sample Limit:   {config.get('limit', 'None')}")
    print()
    print(f"📈 Average Accuracy: {summary['avg_accuracy']:.2%}")
    print(f"📈 Average F1 Score: {summary['avg_f1_score']:.2%}")
    print()
    print(f"🏆 Best Result:")
    best = summary['best_result']
    print(f"   Dataset: {best['dataset']}, Model: {best['model']}, Technique: {best['technique']}")
    print(f"   Accuracy: {best['metrics']['accuracy']:.2%}, F1: {best['metrics']['f1_score']:.2%}")
    print()
    print(f"Models tested: {', '.join(config['models'])}")
    print(f"Datasets tested: {', '.join(config['datasets'])}")
    print(f"Techniques tested: {', '.join(config['techniques'])}")


def print_overall_rankings(df: pd.DataFrame):
    """Print overall performance rankings."""
    print_header("OVERALL PERFORMANCE RANKINGS")
    
    # Top 10 experiments by accuracy
    print("🏆 Top 10 Experiments by Accuracy:")
    top10 = df.nlargest(10, 'accuracy')[['dataset', 'model', 'technique', 'accuracy', 'f1_score', 'parse_failure_rate']]
    top10_display = top10.copy()
    top10_display['accuracy'] = top10_display['accuracy'].apply(lambda x: f"{x:.2%}")
    top10_display['f1_score'] = top10_display['f1_score'].apply(lambda x: f"{x:.2%}")
    top10_display['parse_failure_rate'] = top10_display['parse_failure_rate'].apply(lambda x: f"{x:.1%}")
    print(top10_display.to_string(index=False))
    
    print()
    
    # Bottom 5 experiments
    print("📉 Bottom 5 Experiments by Accuracy:")
    bottom5 = df.nsmallest(5, 'accuracy')[['dataset', 'model', 'technique', 'accuracy', 'f1_score', 'parse_failure_rate']]
    bottom5_display = bottom5.copy()
    bottom5_display['accuracy'] = bottom5_display['accuracy'].apply(lambda x: f"{x:.2%}")
    bottom5_display['f1_score'] = bottom5_display['f1_score'].apply(lambda x: f"{x:.2%}")
    bottom5_display['parse_failure_rate'] = bottom5_display['parse_failure_rate'].apply(lambda x: f"{x:.1%}")
    print(bottom5_display.to_string(index=False))


def print_technique_comparison(df: pd.DataFrame):
    """Print comparison between prompting techniques."""
    print_header("TECHNIQUE COMPARISON")
    
    # Overall technique performance
    technique_stats = df.groupby('technique').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std'],
        'parse_failure_rate': 'mean'
    }).round(4)
    
    technique_stats.columns = ['acc_mean', 'acc_std', 'acc_min', 'acc_max', 'f1_mean', 'f1_std', 'parse_fail']
    technique_stats = technique_stats.sort_values('acc_mean', ascending=False)
    
    print("Average Performance by Technique:")
    print("-" * 90)
    print(f"{'Technique':<30} {'Accuracy':>12} {'± Std':>8} {'F1 Score':>12} {'Parse Fail':>12}")
    print("-" * 90)
    for tech, row in technique_stats.iterrows():
        print(f"{tech:<30} {row['acc_mean']:>11.2%} {row['acc_std']:>7.2%} {row['f1_mean']:>11.2%} {row['parse_fail']:>11.1%}")
    print("-" * 90)


def print_baseline_vs_enriched(df: pd.DataFrame):
    """Print detailed comparison of baseline vs enriched prompts."""
    print_header("BASELINE vs ENRICHED COMPARISON")
    
    # Group by base technique and enrichment status
    comparison = df.groupby(['base_technique', 'is_enriched']).agg({
        'accuracy': 'mean',
        'f1_score': 'mean',
        'parse_failure_rate': 'mean'
    }).round(4)
    
    # Pivot for easier comparison
    baseline_df = df[~df['is_enriched']].groupby('base_technique')[['accuracy', 'f1_score']].mean()
    enriched_df = df[df['is_enriched']].groupby('base_technique')[['accuracy', 'f1_score']].mean()
    
    print("Accuracy Comparison (Baseline → Enriched):")
    print("-" * 70)
    print(f"{'Technique':<20} {'Baseline':>12} {'Enriched':>12} {'Δ Change':>12} {'% Improve':>12}")
    print("-" * 70)
    
    for tech in baseline_df.index:
        if tech in enriched_df.index:
            baseline_acc = baseline_df.loc[tech, 'accuracy']
            enriched_acc = enriched_df.loc[tech, 'accuracy']
            delta = enriched_acc - baseline_acc
            pct_change = (delta / baseline_acc) * 100 if baseline_acc > 0 else 0
            
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"{tech:<20} {baseline_acc:>11.2%} {enriched_acc:>11.2%} {arrow} {delta:>+10.2%} {pct_change:>+11.1f}%")
    print("-" * 70)
    
    # Overall baseline vs enriched
    overall_baseline = df[~df['is_enriched']]['accuracy'].mean()
    overall_enriched = df[df['is_enriched']]['accuracy'].mean()
    overall_delta = overall_enriched - overall_baseline
    
    print()
    print(f"📊 Overall Baseline Average: {overall_baseline:.2%}")
    print(f"📊 Overall Enriched Average: {overall_enriched:.2%}")
    print(f"📊 Overall Improvement:      {overall_delta:+.2%} ({(overall_delta/overall_baseline)*100:+.1f}%)")
    
    # Per-dataset breakdown
    print("\n\nPer-Dataset Baseline vs Enriched:")
    print("-" * 60)
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        ds_baseline = ds_df[~ds_df['is_enriched']]['accuracy'].mean()
        ds_enriched = ds_df[ds_df['is_enriched']]['accuracy'].mean()
        ds_delta = ds_enriched - ds_baseline
        arrow = "↑" if ds_delta > 0 else "↓" if ds_delta < 0 else "="
        print(f"{dataset:<20} Baseline: {ds_baseline:.2%} → Enriched: {ds_enriched:.2%} ({arrow} {ds_delta:+.2%})")
    print("-" * 60)


def print_model_comparison(df: pd.DataFrame):
    """Print comparison between models."""
    print_header("MODEL COMPARISON")
    
    model_stats = df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1_score': 'mean',
        'parse_failure_rate': 'mean'
    }).round(4)
    
    model_stats.columns = ['acc_mean', 'acc_std', 'acc_min', 'acc_max', 'f1_mean', 'parse_fail']
    model_stats = model_stats.sort_values('acc_mean', ascending=False)
    
    print("Average Performance by Model:")
    print("-" * 95)
    print(f"{'Model':<20} {'Accuracy':>12} {'± Std':>8} {'Min':>10} {'Max':>10} {'F1':>10} {'Parse Fail':>12}")
    print("-" * 95)
    for model, row in model_stats.iterrows():
        print(f"{model:<20} {row['acc_mean']:>11.2%} {row['acc_std']:>7.2%} {row['acc_min']:>9.2%} {row['acc_max']:>9.2%} {row['f1_mean']:>9.2%} {row['parse_fail']:>11.1%}")
    print("-" * 95)


def print_dataset_comparison(df: pd.DataFrame):
    """Print comparison between datasets."""
    print_header("DATASET COMPARISON")
    
    dataset_stats = df.groupby('dataset').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1_score': 'mean',
        'parse_failure_rate': 'mean'
    }).round(4)
    
    dataset_stats.columns = ['acc_mean', 'acc_std', 'acc_min', 'acc_max', 'f1_mean', 'parse_fail']
    dataset_stats = dataset_stats.sort_values('acc_mean', ascending=False)
    
    print("Average Performance by Dataset:")
    print("-" * 95)
    print(f"{'Dataset':<20} {'Accuracy':>12} {'± Std':>8} {'Min':>10} {'Max':>10} {'F1':>10} {'Parse Fail':>12}")
    print("-" * 95)
    for dataset, row in dataset_stats.iterrows():
        print(f"{dataset:<20} {row['acc_mean']:>11.2%} {row['acc_std']:>7.2%} {row['acc_min']:>9.2%} {row['acc_max']:>9.2%} {row['f1_mean']:>9.2%} {row['parse_fail']:>11.1%}")
    print("-" * 95)


def print_pivot_tables(df: pd.DataFrame):
    """Print pivot tables for detailed analysis."""
    print_header("DETAILED PIVOT TABLES")
    
    # Accuracy by Model × Technique
    print("📊 Accuracy by Model × Technique:")
    pivot1 = df.pivot_table(values='accuracy', index='model', columns='technique', aggfunc='mean')
    pivot1_pct = pivot1.map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    print(pivot1_pct.to_string())
    
    print("\n")
    
    # Accuracy by Dataset × Technique
    print("📊 Accuracy by Dataset × Technique:")
    pivot2 = df.pivot_table(values='accuracy', index='dataset', columns='technique', aggfunc='mean')
    pivot2_pct = pivot2.map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    print(pivot2_pct.to_string())
    
    print("\n")
    
    # Accuracy by Dataset × Model
    print("📊 Accuracy by Dataset × Model:")
    pivot3 = df.pivot_table(values='accuracy', index='dataset', columns='model', aggfunc='mean')
    pivot3_pct = pivot3.map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    print(pivot3_pct.to_string())


def create_visualizations(df: pd.DataFrame, run_folder: str, save: bool = False):
    """Create visualization plots."""
    print_header("GENERATING VISUALIZATIONS")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    run_path = Path(run_folder)
    figures_dir = run_path / 'figures'
    if save:
        figures_dir.mkdir(exist_ok=True)
    
    # Figure 1: Overall accuracy by technique (bar chart)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    technique_acc = df.groupby('technique')['accuracy'].mean().sort_values(ascending=True)
    colors = ['#2ecc71' if '_enriched' in t else '#3498db' for t in technique_acc.index]
    bars = ax1.barh(technique_acc.index, technique_acc.values, color=colors)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Average Accuracy by Prompting Technique', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, technique_acc.values):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}', 
                va='center', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Baseline'),
                       Patch(facecolor='#2ecc71', label='Enriched')]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    if save:
        fig1.savefig(figures_dir / 'accuracy_by_technique.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: accuracy_by_technique.png")
    
    # Figure 2: Heatmap - Model × Technique
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    pivot_heatmap = df.pivot_table(values='accuracy', index='model', columns='technique', aggfunc='mean')
    sns.heatmap(pivot_heatmap, annot=True, fmt='.1%', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, ax=ax2,
                annot_kws={'fontsize': 9})
    ax2.set_title('Accuracy Heatmap: Model × Technique', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save:
        fig2.savefig(figures_dir / 'heatmap_model_technique.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: heatmap_model_technique.png")
    
    # Figure 3: Heatmap - Dataset × Technique
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    pivot_heatmap2 = df.pivot_table(values='accuracy', index='dataset', columns='technique', aggfunc='mean')
    sns.heatmap(pivot_heatmap2, annot=True, fmt='.1%', cmap='RdYlGn',
                center=0.5, vmin=0, vmax=1, ax=ax3,
                annot_kws={'fontsize': 10})
    ax3.set_title('Accuracy Heatmap: Dataset × Technique', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save:
        fig3.savefig(figures_dir / 'heatmap_dataset_technique.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: heatmap_dataset_technique.png")
    
    # Figure 4: Baseline vs Enriched comparison (grouped bar)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    baseline_enriched = df.groupby(['base_technique', 'is_enriched'])['accuracy'].mean().unstack()
    baseline_enriched.columns = ['Baseline', 'Enriched']
    baseline_enriched.plot(kind='bar', ax=ax4, color=['#3498db', '#2ecc71'], width=0.7)
    ax4.set_xlabel('Technique', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Baseline vs Enriched Prompts by Technique', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.legend(title='Prompt Type')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=8,
                      padding=2, 
                      labels=[f'{v*100:.1f}%' for v in container.datavalues])
    
    plt.tight_layout()
    if save:
        fig4.savefig(figures_dir / 'baseline_vs_enriched.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: baseline_vs_enriched.png")
    
    # Figure 5: Model performance comparison (box plot)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    model_order = df.groupby('model')['accuracy'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='model', y='accuracy', order=model_order, ax=ax5, palette='viridis')
    ax5.set_xlabel('Model', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_title('Accuracy Distribution by Model', fontsize=14, fontweight='bold')
    ax5.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save:
        fig5.savefig(figures_dir / 'model_boxplot.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: model_boxplot.png")
    
    # Figure 6: Dataset difficulty comparison
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    dataset_order = df.groupby('dataset')['accuracy'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='dataset', y='accuracy', order=dataset_order, ax=ax6, palette='Set2')
    ax6.set_xlabel('Dataset', fontsize=12)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_title('Accuracy Distribution by Dataset', fontsize=14, fontweight='bold')
    ax6.set_ylim(0, 1)
    plt.tight_layout()
    if save:
        fig6.savefig(figures_dir / 'dataset_boxplot.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: dataset_boxplot.png")
    
    # Figure 7: Parse failure rate by model
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    parse_fails = df.groupby('model')['parse_failure_rate'].mean().sort_values(ascending=True)
    colors = ['#e74c3c' if v > 0.1 else '#f39c12' if v > 0.05 else '#27ae60' for v in parse_fails.values]
    bars = ax7.barh(parse_fails.index, parse_fails.values * 100, color=colors)
    ax7.set_xlabel('Parse Failure Rate (%)', fontsize=12)
    ax7.set_title('Average Parse Failure Rate by Model', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, parse_fails.values):
        ax7.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1%}',
                va='center', fontsize=10)
    plt.tight_layout()
    if save:
        fig7.savefig(figures_dir / 'parse_failure_rate.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: parse_failure_rate.png")
    
    # Figure 8: Accuracy vs F1 Score scatter
    fig8, ax8 = plt.subplots(figsize=(10, 8))
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        ax8.scatter(ds_df['accuracy'], ds_df['f1_score'], 
                   label=dataset, alpha=0.7, s=80)
    ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect correlation')
    ax8.set_xlabel('Accuracy', fontsize=12)
    ax8.set_ylabel('F1 Score', fontsize=12)
    ax8.set_title('Accuracy vs F1 Score by Dataset', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    plt.tight_layout()
    if save:
        fig8.savefig(figures_dir / 'accuracy_vs_f1.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: accuracy_vs_f1.png")
    
    # Figure 9: Enriched improvement by dataset (delta chart)
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    improvements = []
    for dataset in df['dataset'].unique():
        for tech in df['base_technique'].unique():
            baseline = df[(df['dataset'] == dataset) & (df['base_technique'] == tech) & (~df['is_enriched'])]['accuracy'].mean()
            enriched = df[(df['dataset'] == dataset) & (df['base_technique'] == tech) & (df['is_enriched'])]['accuracy'].mean()
            if pd.notna(baseline) and pd.notna(enriched):
                improvements.append({
                    'dataset': dataset,
                    'technique': tech,
                    'improvement': (enriched - baseline) * 100
                })
    
    imp_df = pd.DataFrame(improvements)
    if len(imp_df) > 0:
        pivot_imp = imp_df.pivot(index='technique', columns='dataset', values='improvement')
        pivot_imp.plot(kind='bar', ax=ax9, width=0.8)
        ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax9.set_xlabel('Technique', fontsize=12)
        ax9.set_ylabel('Accuracy Improvement (percentage points)', fontsize=12)
        ax9.set_title('Enriched vs Baseline: Accuracy Improvement by Dataset', fontsize=14, fontweight='bold')
        ax9.legend(title='Dataset')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save:
        fig9.savefig(figures_dir / 'enriched_improvement.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: enriched_improvement.png")
    
    if save:
        print(f"\n📁 All figures saved to: {figures_dir}/")
    else:
        plt.show()
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9


def generate_html_report(df: pd.DataFrame, config: dict, summary: dict, run_folder: str):
    """Generate an HTML report with embedded visualizations."""
    print_header("GENERATING HTML REPORT")
    
    run_path = Path(run_folder)
    
    # First, save figures
    create_visualizations(df, run_folder, save=True)
    
    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Results Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 8px;
            text-align: center;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        tr:hover {{
            background: #f1f1f1;
        }}
        .figure-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }}
        .figure-container img {{
            max-width: 100%;
            height: auto;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
    </style>
</head>
<body>
    <h1>📊 News Classification Experiment Results</h1>
    
    <div class="summary-box">
        <h2>Run Summary</h2>
        <div class="metric">
            <div class="metric-value">{summary['total_experiments']}</div>
            <div class="metric-label">Total Experiments</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['avg_accuracy']:.1%}</div>
            <div class="metric-label">Avg Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['avg_f1_score']:.1%}</div>
            <div class="metric-label">Avg F1 Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['duration_formatted'].split('.')[0]}</div>
            <div class="metric-label">Duration</div>
        </div>
        <p><strong>Models:</strong> {', '.join(config['models'])}</p>
        <p><strong>Datasets:</strong> {', '.join(config['datasets'])}</p>
        <p><strong>Techniques:</strong> {', '.join(config['techniques'])}</p>
    </div>
    
    <h2>🏆 Top 10 Results</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Dataset</th>
            <th>Model</th>
            <th>Technique</th>
            <th>Accuracy</th>
            <th>F1 Score</th>
        </tr>
"""
    
    top10 = df.nlargest(10, 'accuracy')
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        html_content += f"""        <tr>
            <td>{i}</td>
            <td>{row['dataset']}</td>
            <td>{row['model']}</td>
            <td>{row['technique']}</td>
            <td>{row['accuracy']:.1%}</td>
            <td>{row['f1_score']:.1%}</td>
        </tr>
"""
    
    html_content += """    </table>
    
    <h2>📈 Baseline vs Enriched Comparison</h2>
"""
    
    # Add baseline vs enriched table
    baseline_enriched_data = []
    for tech in df['base_technique'].unique():
        baseline = df[(df['base_technique'] == tech) & (~df['is_enriched'])]['accuracy'].mean()
        enriched = df[(df['base_technique'] == tech) & (df['is_enriched'])]['accuracy'].mean()
        if pd.notna(baseline) and pd.notna(enriched):
            delta = enriched - baseline
            baseline_enriched_data.append((tech, baseline, enriched, delta))
    
    html_content += """    <table>
        <tr>
            <th>Technique</th>
            <th>Baseline</th>
            <th>Enriched</th>
            <th>Improvement</th>
        </tr>
"""
    for tech, baseline, enriched, delta in baseline_enriched_data:
        delta_class = 'positive' if delta > 0 else 'negative'
        arrow = '↑' if delta > 0 else '↓'
        html_content += f"""        <tr>
            <td>{tech}</td>
            <td>{baseline:.1%}</td>
            <td>{enriched:.1%}</td>
            <td class="{delta_class}">{arrow} {delta:+.1%}</td>
        </tr>
"""
    html_content += """    </table>
    
    <h2>📊 Visualizations</h2>
    
    <div class="grid">
        <div class="figure-container">
            <h3>Accuracy by Technique</h3>
            <img src="figures/accuracy_by_technique.png" alt="Accuracy by Technique">
        </div>
        <div class="figure-container">
            <h3>Baseline vs Enriched</h3>
            <img src="figures/baseline_vs_enriched.png" alt="Baseline vs Enriched">
        </div>
    </div>
    
    <div class="figure-container">
        <h3>Model × Technique Heatmap</h3>
        <img src="figures/heatmap_model_technique.png" alt="Heatmap Model Technique">
    </div>
    
    <div class="figure-container">
        <h3>Dataset × Technique Heatmap</h3>
        <img src="figures/heatmap_dataset_technique.png" alt="Heatmap Dataset Technique">
    </div>
    
    <div class="grid">
        <div class="figure-container">
            <h3>Model Performance Distribution</h3>
            <img src="figures/model_boxplot.png" alt="Model Boxplot">
        </div>
        <div class="figure-container">
            <h3>Dataset Difficulty</h3>
            <img src="figures/dataset_boxplot.png" alt="Dataset Boxplot">
        </div>
    </div>
    
    <div class="grid">
        <div class="figure-container">
            <h3>Accuracy vs F1 Score</h3>
            <img src="figures/accuracy_vs_f1.png" alt="Accuracy vs F1">
        </div>
        <div class="figure-container">
            <h3>Parse Failure Rate</h3>
            <img src="figures/parse_failure_rate.png" alt="Parse Failure Rate">
        </div>
    </div>
    
    <div class="figure-container">
        <h3>Enriched Improvement by Dataset</h3>
        <img src="figures/enriched_improvement.png" alt="Enriched Improvement">
    </div>
    
    <footer style="text-align: center; color: #7f8c8d; margin-top: 40px; padding: 20px;">
        Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    </footer>
</body>
</html>
"""
    
    report_path = run_path / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize news classification experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py results/run_20251204_083400_baseline_vs_enriched_v2
  python analyze_results.py results/run_20251204_083400_baseline_vs_enriched_v2 --save
  python analyze_results.py results/run_20251204_083400_baseline_vs_enriched_v2 --html
        """
    )
    parser.add_argument('run_folder', type=str, help='Path to the run folder containing results')
    parser.add_argument('--save', action='store_true', help='Save figures to files instead of displaying')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from: {args.run_folder}")
    config, summary, results = load_results(args.run_folder)
    df = results_to_dataframe(results)
    
    # Print tables
    print_run_summary(config, summary)
    print_overall_rankings(df)
    print_technique_comparison(df)
    print_baseline_vs_enriched(df)
    print_model_comparison(df)
    print_dataset_comparison(df)
    print_pivot_tables(df)
    
    # Generate visualizations
    if not args.no_plots:
        if args.html:
            generate_html_report(df, config, summary, args.run_folder)
        else:
            create_visualizations(df, args.run_folder, save=args.save)
    
    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
