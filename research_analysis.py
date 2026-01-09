"""
Research-Oriented Analysis and Visualization for News Classification Experiments.

Generates publication-ready tables, comprehensive metric analysis, and research-focused
visualizations. Outputs LaTeX tables, plain text tables for reports, and statistical summaries.

Usage:
    python research_analysis.py <run_folder>
    python research_analysis.py results/run_20251204_083400_baseline_vs_enriched_v2
    
    # Generate all outputs (tables, figures, reports)
    python research_analysis.py <run_folder> --all
    
    # Generate only tables (txt, csv, latex)
    python research_analysis.py <run_folder> --tables
    
    # Generate only figures
    python research_analysis.py <run_folder> --figures
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_results(run_folder: str) -> Tuple[dict, dict, list]:
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
    """Convert results list to a comprehensive DataFrame with all metrics."""
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
        # Add enrichment flags
        row['is_enriched'] = '_enriched' in r['technique']
        row['base_technique'] = r['technique'].replace('_enriched', '')
        row['prompt_type'] = 'Enriched' if row['is_enriched'] else 'Baseline'
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_per_class_dataframe(results: list) -> pd.DataFrame:
    """Extract per-class metrics into a DataFrame for detailed analysis."""
    rows = []
    for r in results:
        for class_name, metrics in r['per_class_metrics'].items():
            row = {
                'dataset': r['dataset'],
                'model': r['model'],
                'technique': r['technique'],
                'class': class_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'support': metrics['support']
            }
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# TABLE GENERATION - PUBLICATION READY
# =============================================================================

def format_metric(value: float, as_percent: bool = True, decimals: int = 2) -> str:
    """Format a metric value for display."""
    if pd.isna(value):
        return "-"
    if as_percent:
        return f"{value * 100:.{decimals}f}"
    return f"{value:.{decimals}f}"


def format_metric_with_std(mean: float, std: float, as_percent: bool = True) -> str:
    """Format mean ± std for publication."""
    if pd.isna(mean):
        return "-"
    if as_percent:
        return f"{mean * 100:.2f} ± {std * 100:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def generate_main_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the main results table with all metrics.
    Rows: Model × Technique combinations
    Columns: Accuracy, Precision, Recall, F1, Parse Failure Rate
    """
    # Group by model and technique, aggregate across datasets
    agg_df = df.groupby(['model', 'technique']).agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'parse_failure_rate': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    
    return agg_df


def generate_technique_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technique comparison table aggregated across all models and datasets.
    For comparing prompting strategies.
    """
    agg = df.groupby('technique').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'parse_failure_rate': ['mean', 'std']
    }).round(4)
    
    agg.columns = ['_'.join(col) for col in agg.columns]
    agg = agg.reset_index()
    agg = agg.sort_values('f1_score_mean', ascending=False)
    
    return agg


def generate_model_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate model comparison table aggregated across all techniques and datasets.
    """
    agg = df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'parse_failure_rate': ['mean', 'std']
    }).round(4)
    
    agg.columns = ['_'.join(col) for col in agg.columns]
    agg = agg.reset_index()
    agg = agg.sort_values('f1_score_mean', ascending=False)
    
    return agg


def generate_dataset_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate dataset difficulty comparison table.
    """
    agg = df.groupby('dataset').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'parse_failure_rate': ['mean', 'std']
    }).round(4)
    
    agg.columns = ['_'.join(col) for col in agg.columns]
    agg = agg.reset_index()
    agg = agg.sort_values('f1_score_mean', ascending=False)
    
    return agg


def generate_baseline_vs_enriched_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed baseline vs enriched comparison with all metrics.
    """
    rows = []
    
    for tech in df['base_technique'].unique():
        baseline = df[(df['base_technique'] == tech) & (~df['is_enriched'])]
        enriched = df[(df['base_technique'] == tech) & (df['is_enriched'])]
        
        if len(baseline) > 0 and len(enriched) > 0:
            row = {
                'technique': tech,
                # Accuracy
                'acc_baseline': baseline['accuracy'].mean(),
                'acc_enriched': enriched['accuracy'].mean(),
                'acc_delta': enriched['accuracy'].mean() - baseline['accuracy'].mean(),
                # Precision
                'prec_baseline': baseline['precision'].mean(),
                'prec_enriched': enriched['precision'].mean(),
                'prec_delta': enriched['precision'].mean() - baseline['precision'].mean(),
                # Recall
                'rec_baseline': baseline['recall'].mean(),
                'rec_enriched': enriched['recall'].mean(),
                'rec_delta': enriched['recall'].mean() - baseline['recall'].mean(),
                # F1
                'f1_baseline': baseline['f1_score'].mean(),
                'f1_enriched': enriched['f1_score'].mean(),
                'f1_delta': enriched['f1_score'].mean() - baseline['f1_score'].mean(),
                # Parse Failure
                'pf_baseline': baseline['parse_failure_rate'].mean(),
                'pf_enriched': enriched['parse_failure_rate'].mean(),
                'pf_delta': enriched['parse_failure_rate'].mean() - baseline['parse_failure_rate'].mean(),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_full_results_matrix(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate pivot tables for each metric: Dataset × Model × Technique
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'parse_failure_rate']
    tables = {}
    
    for metric in metrics:
        # Create pivot: rows = (dataset, model), columns = technique
        pivot = df.pivot_table(
            values=metric,
            index=['dataset', 'model'],
            columns='technique',
            aggfunc='mean'
        ).round(4)
        tables[metric] = pivot
    
    return tables


# =============================================================================
# TEXT/LATEX TABLE EXPORT
# =============================================================================

def df_to_text_table(df: pd.DataFrame, title: str = "", float_format: str = "%.4f") -> str:
    """Convert DataFrame to formatted text table."""
    lines = []
    if title:
        lines.append(f"\n{'=' * 80}")
        lines.append(f" {title}")
        lines.append(f"{'=' * 80}\n")
    
    # Use pandas to_string for clean formatting
    lines.append(df.to_string(index=True, float_format=lambda x: float_format % x if isinstance(x, float) else str(x)))
    lines.append("")
    
    return "\n".join(lines)


def df_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert DataFrame to LaTeX table."""
    latex = df.to_latex(
        index=True,
        float_format="%.3f",
        caption=caption,
        label=label,
        bold_rows=True,
        column_format='l' + 'c' * len(df.columns)
    )
    return latex


def generate_publication_tables(df: pd.DataFrame, output_dir: Path):
    """Generate all publication-ready tables in multiple formats."""
    
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    all_text = []
    all_latex = []
    
    # 1. Main Results Table
    main_results = generate_main_results_table(df)
    
    # Format for display
    display_df = pd.DataFrame({
        'Model': main_results['model'],
        'Technique': main_results['technique'],
        'Accuracy': main_results.apply(lambda r: f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}", axis=1),
        'Precision': main_results.apply(lambda r: f"{r['precision_mean']*100:.2f} ± {r['precision_std']*100:.2f}", axis=1),
        'Recall': main_results.apply(lambda r: f"{r['recall_mean']*100:.2f} ± {r['recall_std']*100:.2f}", axis=1),
        'F1 Score': main_results.apply(lambda r: f"{r['f1_score_mean']*100:.2f} ± {r['f1_score_std']*100:.2f}", axis=1),
        'Parse Fail %': main_results.apply(lambda r: f"{r['parse_failure_rate_mean']*100:.1f} ± {r['parse_failure_rate_std']*100:.1f}", axis=1),
    })
    
    all_text.append(df_to_text_table(display_df.set_index(['Model', 'Technique']), 
                                      "TABLE 1: Complete Results by Model and Technique (Mean ± Std %)"))
    main_results.to_csv(tables_dir / 'table1_main_results.csv', index=False)
    
    # 2. Technique Comparison Table
    tech_comp = generate_technique_comparison_table(df)
    tech_display = pd.DataFrame({
        'Technique': tech_comp['technique'],
        'Accuracy': tech_comp.apply(lambda r: f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}", axis=1),
        'Precision': tech_comp.apply(lambda r: f"{r['precision_mean']*100:.2f} ± {r['precision_std']*100:.2f}", axis=1),
        'Recall': tech_comp.apply(lambda r: f"{r['recall_mean']*100:.2f} ± {r['recall_std']*100:.2f}", axis=1),
        'F1 Score': tech_comp.apply(lambda r: f"{r['f1_score_mean']*100:.2f} ± {r['f1_score_std']*100:.2f}", axis=1),
        'Parse Fail %': tech_comp.apply(lambda r: f"{r['parse_failure_rate_mean']*100:.1f} ± {r['parse_failure_rate_std']*100:.1f}", axis=1),
    })
    
    all_text.append(df_to_text_table(tech_display.set_index('Technique'),
                                      "TABLE 2: Prompting Technique Comparison (Mean ± Std %)"))
    tech_comp.to_csv(tables_dir / 'table2_technique_comparison.csv', index=False)
    
    # 3. Model Comparison Table
    model_comp = generate_model_comparison_table(df)
    model_display = pd.DataFrame({
        'Model': model_comp['model'],
        'Accuracy': model_comp.apply(lambda r: f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}", axis=1),
        'Precision': model_comp.apply(lambda r: f"{r['precision_mean']*100:.2f} ± {r['precision_std']*100:.2f}", axis=1),
        'Recall': model_comp.apply(lambda r: f"{r['recall_mean']*100:.2f} ± {r['recall_std']*100:.2f}", axis=1),
        'F1 Score': model_comp.apply(lambda r: f"{r['f1_score_mean']*100:.2f} ± {r['f1_score_std']*100:.2f}", axis=1),
        'Parse Fail %': model_comp.apply(lambda r: f"{r['parse_failure_rate_mean']*100:.1f} ± {r['parse_failure_rate_std']*100:.1f}", axis=1),
    })
    
    all_text.append(df_to_text_table(model_display.set_index('Model'),
                                      "TABLE 3: Model Performance Comparison (Mean ± Std %)"))
    model_comp.to_csv(tables_dir / 'table3_model_comparison.csv', index=False)
    
    # 4. Dataset Comparison Table
    ds_comp = generate_dataset_comparison_table(df)
    ds_display = pd.DataFrame({
        'Dataset': ds_comp['dataset'],
        'Accuracy': ds_comp.apply(lambda r: f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}", axis=1),
        'Precision': ds_comp.apply(lambda r: f"{r['precision_mean']*100:.2f} ± {r['precision_std']*100:.2f}", axis=1),
        'Recall': ds_comp.apply(lambda r: f"{r['recall_mean']*100:.2f} ± {r['recall_std']*100:.2f}", axis=1),
        'F1 Score': ds_comp.apply(lambda r: f"{r['f1_score_mean']*100:.2f} ± {r['f1_score_std']*100:.2f}", axis=1),
        'Parse Fail %': ds_comp.apply(lambda r: f"{r['parse_failure_rate_mean']*100:.1f} ± {r['parse_failure_rate_std']*100:.1f}", axis=1),
    })
    
    all_text.append(df_to_text_table(ds_display.set_index('Dataset'),
                                      "TABLE 4: Dataset Difficulty Comparison (Mean ± Std %)"))
    ds_comp.to_csv(tables_dir / 'table4_dataset_comparison.csv', index=False)
    
    # 5. Baseline vs Enriched Comparison
    be_comp = generate_baseline_vs_enriched_table(df)
    if len(be_comp) > 0:
        be_display = pd.DataFrame({
            'Technique': be_comp['technique'],
            'Acc Base': be_comp['acc_baseline'].apply(lambda x: f"{x*100:.2f}"),
            'Acc Enr': be_comp['acc_enriched'].apply(lambda x: f"{x*100:.2f}"),
            'Acc Δ': be_comp['acc_delta'].apply(lambda x: f"{x*100:+.2f}"),
            'F1 Base': be_comp['f1_baseline'].apply(lambda x: f"{x*100:.2f}"),
            'F1 Enr': be_comp['f1_enriched'].apply(lambda x: f"{x*100:.2f}"),
            'F1 Δ': be_comp['f1_delta'].apply(lambda x: f"{x*100:+.2f}"),
            'PF Base': be_comp['pf_baseline'].apply(lambda x: f"{x*100:.1f}"),
            'PF Enr': be_comp['pf_enriched'].apply(lambda x: f"{x*100:.1f}"),
            'PF Δ': be_comp['pf_delta'].apply(lambda x: f"{x*100:+.1f}"),
        })
        
        all_text.append(df_to_text_table(be_display.set_index('Technique'),
                                          "TABLE 5: Baseline vs Enriched Prompts Comparison (%)"))
        be_comp.to_csv(tables_dir / 'table5_baseline_vs_enriched.csv', index=False)
    
    # 6. Full Results Matrix by Metric
    matrices = generate_full_results_matrix(df)
    for metric_name, pivot in matrices.items():
        formatted = pivot.map(lambda x: f"{x*100:.2f}" if pd.notna(x) else "-")
        all_text.append(df_to_text_table(formatted, 
                                          f"TABLE: {metric_name.upper()} by Dataset × Model × Technique (%)"))
        pivot.to_csv(tables_dir / f'matrix_{metric_name}.csv')
    
    # 7. Per-Dataset Detailed Results
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        ds_pivot = ds_df.pivot_table(
            values=['accuracy', 'precision', 'recall', 'f1_score', 'parse_failure_rate'],
            index='model',
            columns='technique',
            aggfunc='mean'
        ).round(4)
        
        # Flatten and format
        detail_rows = []
        for model in ds_df['model'].unique():
            for tech in ds_df['technique'].unique():
                row_data = ds_df[(ds_df['model'] == model) & (ds_df['technique'] == tech)]
                if len(row_data) > 0:
                    r = row_data.iloc[0]
                    detail_rows.append({
                        'Model': model,
                        'Technique': tech,
                        'Acc': f"{r['accuracy']*100:.2f}",
                        'Prec': f"{r['precision']*100:.2f}",
                        'Rec': f"{r['recall']*100:.2f}",
                        'F1': f"{r['f1_score']*100:.2f}",
                        'PF%': f"{r['parse_failure_rate']*100:.1f}",
                    })
        
        detail_df = pd.DataFrame(detail_rows)
        if len(detail_df) > 0:
            all_text.append(df_to_text_table(detail_df.set_index(['Model', 'Technique']),
                                              f"TABLE: Detailed Results for {dataset} (%)"))
    
    # Write all tables to text file
    with open(tables_dir / 'all_tables.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" NEWS CLASSIFICATION EXPERIMENT RESULTS\n")
        f.write(" Publication-Ready Tables\n")
        f.write("=" * 80 + "\n\n")
        f.write("Note: All metric values are percentages (%).\n")
        f.write("Mean ± Std indicates average across conditions with standard deviation.\n")
        f.write("Δ indicates change from baseline to enriched (positive = improvement).\n\n")
        f.write("\n".join(all_text))
    
    print(f"  ✓ Generated tables in: {tables_dir}/")
    print(f"    - all_tables.txt (copy-paste ready)")
    print(f"    - CSV files for each table")
    
    return tables_dir


# =============================================================================
# RESEARCH-FOCUSED VISUALIZATIONS
# =============================================================================

def create_research_figures(df: pd.DataFrame, output_dir: Path):
    """Create publication-quality research figures."""
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'serif',
    })
    
    # Color schemes for consistency
    metric_colors = {
        'accuracy': '#3498db',
        'precision': '#e74c3c', 
        'recall': '#2ecc71',
        'f1_score': '#9b59b6',
        'parse_failure_rate': '#f39c12'
    }
    
    # ==========================================================================
    # FIGURE 1: Multi-Metric Comparison by Technique (Research Summary)
    # ==========================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig1.suptitle('Comprehensive Metric Comparison by Prompting Technique', fontweight='bold', y=1.02)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'parse_failure_rate']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Parse Failure Rate']
    
    tech_order = df.groupby('technique')['f1_score'].mean().sort_values(ascending=False).index.tolist()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes.flatten()[idx]
        
        data = df.groupby('technique')[metric].agg(['mean', 'std']).reindex(tech_order)
        
        colors = ['#2ecc71' if '_enriched' in t else '#3498db' for t in data.index]
        bars = ax.bar(range(len(data)), data['mean'], yerr=data['std'], 
                     capsize=3, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([t.replace('_', '\n') for t in data.index], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'{label} (%)')
        ax.set_title(label)
        ax.set_ylim(0, 1.1 if metric != 'parse_failure_rate' else max(data['mean'] + data['std']) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars, data['mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val*100:.1f}', ha='center', va='bottom', fontsize=7)
    
    # Add legend in last subplot
    axes.flatten()[5].axis('off')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Baseline'),
                       Patch(facecolor='#2ecc71', label='Enriched')]
    axes.flatten()[5].legend(handles=legend_elements, loc='center', fontsize=12)
    axes.flatten()[5].text(0.5, 0.3, 'Error bars show\nstandard deviation', 
                           ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    fig1.savefig(figures_dir / 'fig1_technique_all_metrics.png', dpi=300, bbox_inches='tight')
    fig1.savefig(figures_dir / 'fig1_technique_all_metrics.pdf', bbox_inches='tight')
    plt.close(fig1)
    
    # ==========================================================================
    # FIGURE 2: Model Performance Radar/Spider Chart
    # ==========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    metrics_radar = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels_radar = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    models = df['model'].unique()
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    colors_models = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        values = [df[df['model'] == model][m].mean() for m in metrics_radar]
        values += values[:1]  # Complete the loop
        ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_models[i])
        ax2.fill(angles, values, alpha=0.15, color=colors_models[i])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels_radar, fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_title('Model Performance Profile\n(All Metrics)', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    fig2.savefig(figures_dir / 'fig2_model_radar.png', dpi=300, bbox_inches='tight')
    fig2.savefig(figures_dir / 'fig2_model_radar.pdf', bbox_inches='tight')
    plt.close(fig2)
    
    # ==========================================================================
    # FIGURE 3: Precision-Recall Trade-off Analysis
    # ==========================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Precision-Recall Trade-off by Dataset', fontweight='bold')
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        ax = axes3[idx]
        ds_df = df[df['dataset'] == dataset]
        
        for model in ds_df['model'].unique():
            model_df = ds_df[ds_df['model'] == model]
            # Color by enriched vs baseline
            colors = ['#2ecc71' if e else '#3498db' for e in model_df['is_enriched']]
            markers = ['o' if not e else '^' for e in model_df['is_enriched']]
            
            for _, row in model_df.iterrows():
                marker = '^' if row['is_enriched'] else 'o'
                ax.scatter(row['recall'], row['precision'], 
                          c='#2ecc71' if row['is_enriched'] else '#3498db',
                          marker=marker, s=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(dataset)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal reference
        
        # Add F1 iso-lines
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            precision_line = np.linspace(0.01, 1, 100)
            recall_line = (f1 * precision_line) / (2 * precision_line - f1)
            mask = (recall_line > 0) & (recall_line <= 1)
            ax.plot(recall_line[mask], precision_line[mask], '--', color='gray', alpha=0.3, linewidth=0.5)
            # Label F1 curve
            if np.any(mask):
                mid_idx = len(recall_line[mask]) // 2
                ax.text(recall_line[mask][mid_idx], precision_line[mask][mid_idx], 
                       f'F1={f1}', fontsize=7, color='gray')
    
    # Add legend
    legend_elements = [plt.scatter([], [], c='#3498db', marker='o', s=100, label='Baseline'),
                       plt.scatter([], [], c='#2ecc71', marker='^', s=100, label='Enriched')]
    fig3.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig3.savefig(figures_dir / 'fig3_precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    fig3.savefig(figures_dir / 'fig3_precision_recall_tradeoff.pdf', bbox_inches='tight')
    plt.close(fig3)
    
    # ==========================================================================
    # FIGURE 4: Parse Failure Analysis
    # ==========================================================================
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('Parse Failure Analysis', fontweight='bold')
    
    # 4a: Parse failure by model and technique
    ax4a = axes4[0]
    pf_pivot = df.pivot_table(values='parse_failure_rate', index='model', columns='technique', aggfunc='mean')
    sns.heatmap(pf_pivot * 100, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4a,
               cbar_kws={'label': 'Parse Failure Rate (%)'})
    ax4a.set_title('Parse Failure Rate by Model × Technique')
    ax4a.set_xticklabels(ax4a.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # 4b: Parse failure vs F1 score (correlation)
    ax4b = axes4[1]
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        ax4b.scatter(ds_df['parse_failure_rate'] * 100, ds_df['f1_score'] * 100,
                    label=dataset, alpha=0.6, s=80)
    
    ax4b.set_xlabel('Parse Failure Rate (%)')
    ax4b.set_ylabel('F1 Score (%)')
    ax4b.set_title('Impact of Parse Failures on F1 Score')
    ax4b.legend()
    
    # Add correlation line
    z = np.polyfit(df['parse_failure_rate'], df['f1_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, df['parse_failure_rate'].max(), 100)
    ax4b.plot(x_line * 100, p(x_line) * 100, 'r--', alpha=0.5, label='Trend')
    
    # Calculate and show correlation
    corr = df['parse_failure_rate'].corr(df['f1_score'])
    ax4b.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax4b.transAxes, 
             ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig4.savefig(figures_dir / 'fig4_parse_failure_analysis.png', dpi=300, bbox_inches='tight')
    fig4.savefig(figures_dir / 'fig4_parse_failure_analysis.pdf', bbox_inches='tight')
    plt.close(fig4)
    
    # ==========================================================================
    # FIGURE 5: Baseline vs Enriched Delta Analysis
    # ==========================================================================
    be_comp = generate_baseline_vs_enriched_table(df)
    
    if len(be_comp) > 0:
        fig5, axes5 = plt.subplots(1, 4, figsize=(14, 5))
        fig5.suptitle('Effect of Enriched Category Descriptions (Δ from Baseline)', fontweight='bold')
        
        metrics_delta = [('acc_delta', 'Accuracy'), ('prec_delta', 'Precision'), 
                        ('rec_delta', 'Recall'), ('f1_delta', 'F1 Score')]
        
        for idx, (col, label) in enumerate(metrics_delta):
            ax = axes5[idx]
            values = be_comp[col] * 100
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
            
            bars = ax.barh(be_comp['technique'], values, color=colors, edgecolor='black', linewidth=0.5)
            ax.axvline(x=0, color='black', linewidth=1)
            ax.set_xlabel(f'Δ {label} (percentage points)')
            ax.set_title(label)
            
            # Add value labels
            for bar, val in zip(bars, values):
                x_pos = val + 0.5 if val >= 0 else val - 0.5
                ha = 'left' if val >= 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}',
                       ha=ha, va='center', fontsize=9)
        
        plt.tight_layout()
        fig5.savefig(figures_dir / 'fig5_enriched_delta_analysis.png', dpi=300, bbox_inches='tight')
        fig5.savefig(figures_dir / 'fig5_enriched_delta_analysis.pdf', bbox_inches='tight')
        plt.close(fig5)
    
    # ==========================================================================
    # FIGURE 6: Comprehensive Heatmaps for All Metrics
    # ==========================================================================
    fig6, axes6 = plt.subplots(2, 3, figsize=(16, 10))
    fig6.suptitle('Performance Heatmaps by Dataset × Technique', fontweight='bold', y=1.02)
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes6.flatten()[idx]
        
        pivot = df.pivot_table(values=metric, index='dataset', columns='technique', aggfunc='mean')
        
        # Choose colormap based on metric
        if metric == 'parse_failure_rate':
            cmap = 'YlOrRd'  # Red is bad for parse failures
            fmt = '.1f'
            pivot_display = pivot * 100
        else:
            cmap = 'RdYlGn'  # Green is good for other metrics
            fmt = '.1f'
            pivot_display = pivot * 100
        
        sns.heatmap(pivot_display, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                   vmin=0 if metric == 'parse_failure_rate' else None,
                   vmax=100 if metric != 'parse_failure_rate' else None,
                   cbar_kws={'label': f'{label} (%)'})
        ax.set_title(label)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Empty last subplot - add summary statistics
    ax_summary = axes6.flatten()[5]
    ax_summary.axis('off')
    
    summary_text = "Summary Statistics\n" + "=" * 30 + "\n\n"
    for metric, label in zip(metrics[:4], metric_labels[:4]):
        mean_val = df[metric].mean() * 100
        std_val = df[metric].std() * 100
        summary_text += f"{label}:\n  {mean_val:.2f} ± {std_val:.2f}%\n\n"
    
    summary_text += f"Parse Failure Rate:\n  {df['parse_failure_rate'].mean()*100:.2f} ± {df['parse_failure_rate'].std()*100:.2f}%"
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig6.savefig(figures_dir / 'fig6_all_metrics_heatmaps.png', dpi=300, bbox_inches='tight')
    fig6.savefig(figures_dir / 'fig6_all_metrics_heatmaps.pdf', bbox_inches='tight')
    plt.close(fig6)
    
    # ==========================================================================
    # FIGURE 7: Per-Dataset Performance Summary
    # ==========================================================================
    fig7, axes7 = plt.subplots(1, 3, figsize=(15, 5))
    fig7.suptitle('Per-Dataset Model Performance (F1 Score)', fontweight='bold')
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        ax = axes7[idx]
        ds_df = df[df['dataset'] == dataset]
        
        pivot = ds_df.pivot_table(values='f1_score', index='model', columns='technique', aggfunc='mean')
        pivot_display = pivot * 100
        
        sns.heatmap(pivot_display, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                   vmin=0, vmax=100, cbar_kws={'label': 'F1 Score (%)'})
        ax.set_title(f'{dataset}\n(n_classes={len(ds_df["technique"].unique())})')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    
    plt.tight_layout()
    fig7.savefig(figures_dir / 'fig7_per_dataset_f1_heatmaps.png', dpi=300, bbox_inches='tight')
    fig7.savefig(figures_dir / 'fig7_per_dataset_f1_heatmaps.pdf', bbox_inches='tight')
    plt.close(fig7)
    
    # ==========================================================================
    # FIGURE 8: Box Plots for Metric Distributions
    # ==========================================================================
    fig8, axes8 = plt.subplots(2, 2, figsize=(12, 10))
    fig8.suptitle('Metric Distributions by Prompt Type', fontweight='bold')
    
    metrics_box = [('accuracy', 'Accuracy'), ('f1_score', 'F1 Score'),
                   ('precision', 'Precision'), ('recall', 'Recall')]
    
    for idx, (metric, label) in enumerate(metrics_box):
        ax = axes8.flatten()[idx]
        
        # Create combined category for cleaner visualization
        df_plot = df.copy()
        df_plot['group'] = df_plot['model'] + '\n' + df_plot['prompt_type']
        
        order = []
        for model in df['model'].unique():
            order.append(f"{model}\nBaseline")
            order.append(f"{model}\nEnriched")
        
        colors = ['#3498db', '#2ecc71'] * len(df['model'].unique())
        
        box = sns.boxplot(data=df_plot, x='group', y=metric, order=order,
                         palette=colors, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(f'{label} (%)')
        ax.set_title(label)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        # Convert y-axis to percentage
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    
    plt.tight_layout()
    fig8.savefig(figures_dir / 'fig8_metric_distributions.png', dpi=300, bbox_inches='tight')
    fig8.savefig(figures_dir / 'fig8_metric_distributions.pdf', bbox_inches='tight')
    plt.close(fig8)
    
    print(f"  ✓ Generated figures in: {figures_dir}/")
    print(f"    - PNG files (for preview)")
    print(f"    - PDF files (for publication)")
    
    return figures_dir


# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================

def generate_statistical_summary(df: pd.DataFrame, output_dir: Path):
    """Generate statistical summary report."""
    
    report_path = output_dir / 'statistical_summary.txt'
    
    lines = []
    lines.append("=" * 80)
    lines.append(" STATISTICAL SUMMARY REPORT")
    lines.append(" News Classification Experiment Analysis")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall Statistics
    lines.append("1. OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"   Total experiments: {len(df)}")
    lines.append(f"   Models tested: {df['model'].nunique()} ({', '.join(df['model'].unique())})")
    lines.append(f"   Datasets tested: {df['dataset'].nunique()} ({', '.join(df['dataset'].unique())})")
    lines.append(f"   Techniques tested: {df['technique'].nunique()}")
    lines.append("")
    
    # Metric summaries
    lines.append("2. METRIC DISTRIBUTIONS")
    lines.append("-" * 40)
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'parse_failure_rate']:
        lines.append(f"\n   {metric.upper().replace('_', ' ')}:")
        lines.append(f"      Mean:   {df[metric].mean()*100:.2f}%")
        lines.append(f"      Std:    {df[metric].std()*100:.2f}%")
        lines.append(f"      Median: {df[metric].median()*100:.2f}%")
        lines.append(f"      Min:    {df[metric].min()*100:.2f}%")
        lines.append(f"      Max:    {df[metric].max()*100:.2f}%")
        lines.append(f"      IQR:    {(df[metric].quantile(0.75) - df[metric].quantile(0.25))*100:.2f}%")
    
    # Best/Worst configurations
    lines.append("\n3. BEST AND WORST CONFIGURATIONS")
    lines.append("-" * 40)
    
    best_f1 = df.loc[df['f1_score'].idxmax()]
    worst_f1 = df.loc[df['f1_score'].idxmin()]
    
    lines.append("\n   Best F1 Score:")
    lines.append(f"      Dataset: {best_f1['dataset']}")
    lines.append(f"      Model: {best_f1['model']}")
    lines.append(f"      Technique: {best_f1['technique']}")
    lines.append(f"      Metrics: Acc={best_f1['accuracy']*100:.2f}%, P={best_f1['precision']*100:.2f}%, R={best_f1['recall']*100:.2f}%, F1={best_f1['f1_score']*100:.2f}%")
    
    lines.append("\n   Worst F1 Score:")
    lines.append(f"      Dataset: {worst_f1['dataset']}")
    lines.append(f"      Model: {worst_f1['model']}")
    lines.append(f"      Technique: {worst_f1['technique']}")
    lines.append(f"      Metrics: Acc={worst_f1['accuracy']*100:.2f}%, P={worst_f1['precision']*100:.2f}%, R={worst_f1['recall']*100:.2f}%, F1={worst_f1['f1_score']*100:.2f}%")
    
    # Baseline vs Enriched statistical comparison
    lines.append("\n4. BASELINE VS ENRICHED ANALYSIS")
    lines.append("-" * 40)
    
    baseline = df[~df['is_enriched']]
    enriched = df[df['is_enriched']]
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'parse_failure_rate']:
        base_mean = baseline[metric].mean()
        enr_mean = enriched[metric].mean()
        delta = enr_mean - base_mean
        pct_change = (delta / base_mean * 100) if base_mean != 0 else 0
        
        lines.append(f"\n   {metric.upper().replace('_', ' ')}:")
        lines.append(f"      Baseline:  {base_mean*100:.2f}% ± {baseline[metric].std()*100:.2f}%")
        lines.append(f"      Enriched:  {enr_mean*100:.2f}% ± {enriched[metric].std()*100:.2f}%")
        lines.append(f"      Δ Change:  {delta*100:+.2f} percentage points ({pct_change:+.1f}%)")
    
    # Per-dataset analysis
    lines.append("\n5. PER-DATASET ANALYSIS")
    lines.append("-" * 40)
    
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        lines.append(f"\n   {dataset.upper()}:")
        lines.append(f"      F1 Score: {ds_df['f1_score'].mean()*100:.2f}% ± {ds_df['f1_score'].std()*100:.2f}%")
        lines.append(f"      Accuracy: {ds_df['accuracy'].mean()*100:.2f}% ± {ds_df['accuracy'].std()*100:.2f}%")
        lines.append(f"      Parse Fail: {ds_df['parse_failure_rate'].mean()*100:.1f}%")
        
        # Best technique for this dataset
        best_tech = ds_df.groupby('technique')['f1_score'].mean().idxmax()
        best_f1 = ds_df.groupby('technique')['f1_score'].mean().max()
        lines.append(f"      Best technique: {best_tech} (F1={best_f1*100:.2f}%)")
    
    # Model ranking
    lines.append("\n6. MODEL RANKING (by mean F1 Score)")
    lines.append("-" * 40)
    
    model_ranking = df.groupby('model')['f1_score'].mean().sort_values(ascending=False)
    for rank, (model, f1) in enumerate(model_ranking.items(), 1):
        lines.append(f"   {rank}. {model}: {f1*100:.2f}%")
    
    # Technique ranking
    lines.append("\n7. TECHNIQUE RANKING (by mean F1 Score)")
    lines.append("-" * 40)
    
    tech_ranking = df.groupby('technique')['f1_score'].mean().sort_values(ascending=False)
    for rank, (tech, f1) in enumerate(tech_ranking.items(), 1):
        lines.append(f"   {rank}. {tech}: {f1*100:.2f}%")
    
    # Correlation analysis
    lines.append("\n8. CORRELATION ANALYSIS")
    lines.append("-" * 40)
    
    metrics_corr = ['accuracy', 'precision', 'recall', 'f1_score', 'parse_failure_rate']
    corr_matrix = df[metrics_corr].corr()
    
    lines.append("\n   Pearson Correlation Matrix:")
    lines.append("   " + corr_matrix.to_string().replace('\n', '\n   '))
    
    lines.append("\n   Key Correlations:")
    lines.append(f"      Parse Failure ↔ F1 Score: r = {corr_matrix.loc['parse_failure_rate', 'f1_score']:.3f}")
    lines.append(f"      Precision ↔ Recall: r = {corr_matrix.loc['precision', 'recall']:.3f}")
    lines.append(f"      Accuracy ↔ F1 Score: r = {corr_matrix.loc['accuracy', 'f1_score']:.3f}")
    
    lines.append("\n" + "=" * 80)
    lines.append(" END OF REPORT")
    lines.append("=" * 80)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Generated statistical summary: {report_path}")
    
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Research-oriented analysis for news classification experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_analysis.py results/run_20251204_083400_baseline_vs_enriched_v2 --all
  python research_analysis.py results/run_20251204_083400_baseline_vs_enriched_v2 --tables
  python research_analysis.py results/run_20251204_083400_baseline_vs_enriched_v2 --figures
        """
    )
    parser.add_argument('run_folder', type=str, help='Path to the run folder containing results')
    parser.add_argument('--all', action='store_true', help='Generate all outputs (tables, figures, reports)')
    parser.add_argument('--tables', action='store_true', help='Generate publication-ready tables')
    parser.add_argument('--figures', action='store_true', help='Generate research figures')
    parser.add_argument('--stats', action='store_true', help='Generate statistical summary')
    
    args = parser.parse_args()
    
    # Default to --all if no specific option selected
    if not (args.tables or args.figures or args.stats):
        args.all = True
    
    # Load data
    print(f"\n{'='*60}")
    print(f" RESEARCH ANALYSIS: {Path(args.run_folder).name}")
    print(f"{'='*60}\n")
    
    print("Loading results...")
    config, summary, results = load_results(args.run_folder)
    df = results_to_dataframe(results)
    
    output_dir = Path(args.run_folder)
    
    print(f"  Experiments: {len(df)}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Techniques: {df['technique'].nunique()}")
    print()
    
    if args.all or args.tables:
        print("Generating publication-ready tables...")
        generate_publication_tables(df, output_dir)
        print()
    
    if args.all or args.figures:
        print("Generating research figures...")
        create_research_figures(df, output_dir)
        print()
    
    if args.all or args.stats:
        print("Generating statistical summary...")
        generate_statistical_summary(df, output_dir)
        print()
    
    print(f"{'='*60}")
    print(f" ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - tables/     : CSV and TXT tables for reports")
    print(f"  - figures/    : PNG and PDF visualizations")
    print(f"  - statistical_summary.txt : Detailed statistics")


if __name__ == "__main__":
    main()
