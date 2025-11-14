#!/usr/bin/env python3
"""
XtalNet CCSG Results Plotting Script

This script creates visualization plots comparing baseline XtalNet and RAG-XtalNet
performance based on the analysis results from analyze_ccsg_results.py.

Usage:
    python scripts/plot_ccsg_results.py \
        --input analysis_results.csv \
        --output_dir plots/ \
        --plot_types match_curve rmse_dist boxplot

    python scripts/plot_ccsg_results.py \
        --input analysis_results.json \
        --output_dir comparison_plots/ \
        --split test
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_results(input_path: Path) -> pd.DataFrame:
    """Load analysis results from CSV or JSON file"""
    if input_path.suffix == '.csv':
        return pd.read_csv(input_path)
    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

def get_experiment_colors_and_markers() -> Dict[str, Tuple[str, str]]:
    """Define colors and markers for different experiment types"""
    config = {
        'baseline': ('#2E86AB', 'o'),      # Blue, circle
        'rag': ('#A23B72', 's'),           # Purple, square
        'rag_align': ('#F18F01', '^'),     # Orange, triangle
        'retrieval': ('#C73E1D', 'D'),     # Red, diamond
        'default': ('#808080', 'x')        # Gray, cross
    }
    return config

def extract_experiment_type(exp_name: str) -> str:
    """Extract experiment type from experiment name"""
    exp_name_lower = exp_name.lower()
    if 'baseline' in exp_name_lower or 'vanilla' in exp_name_lower:
        return 'baseline'
    elif 'rag_align' in exp_name_lower or 'align' in exp_name_lower:
        return 'rag_align'
    elif 'rag' in exp_name_lower or 'retrieval' in exp_name_lower:
        return 'rag'
    else:
        return 'default'

def plot_match_at_k_curves(df: pd.DataFrame, output_path: Path, split: str = 'test'):
    """Plot Top-k Match curves for different experiments"""
    # Filter data for specified split
    split_df = df[df['split'] == split].copy()

    if split_df.empty:
        print(f"No data found for split: {split}")
        return

    # Extract match@k data
    k_values = [1, 3, 5, 10]
    match_cols = [f'match_{k}' for k in k_values]

    # Check if we have match data
    available_cols = [col for col in match_cols if col in split_df.columns]
    if not available_cols:
        print("No match@k data found in the input file")
        return

    k_values = [int(col.split('_')[1]) for col in available_cols]

    plt.figure(figsize=(10, 6))

    colors_markers = get_experiment_colors_and_markers()

    for _, row in split_df.iterrows():
        exp_name = row['exp_name']
        exp_type = extract_experiment_type(exp_name)
        color, marker = colors_markers.get(exp_type, colors_markers['default'])

        # Extract match values
        match_values = [row.get(f'match_{k}', np.nan) for k in k_values]

        # Filter out NaN values
        valid_k = []
        valid_matches = []
        for k, match in zip(k_values, match_values):
            if not np.isnan(match):
                valid_k.append(k)
                valid_matches.append(match * 100)  # Convert to percentage

        if valid_k:
            label = f"{exp_name}"
            if row.get('use_retrieval', False):
                label += f" (RAG)"
            plt.plot(valid_k, valid_matches,
                    marker=marker, markersize=8, linewidth=2,
                    color=color, label=label)

    plt.xlabel('k', fontsize=12)
    plt.ylabel('Match@k (%)', fontsize=12)
    plt.title(f'Top-k Match Accuracy Comparison ({split.upper()} set)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(k_values)
    plt.tight_layout()

    plt.savefig(output_path / f'match_at_k_curves_{split}.png', bbox_inches='tight')
    plt.savefig(output_path / f'match_at_k_curves_{split}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Match@k curves saved to {output_path / f'match_at_k_curves_{split}.png'}")

def plot_rmse_distribution(df: pd.DataFrame, output_path: Path, split: str = 'test'):
    """Plot RMSE distribution (histogram and cumulative)"""
    split_df = df[df['split'] == split].copy()

    if split_df.empty:
        print(f"No data found for split: {split}")
        return

    if 'rmse_mean' not in split_df.columns:
        print("No RMSE data found in the input file")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors_markers = get_experiment_colors_and_markers()

    # Histogram
    for _, row in split_df.iterrows():
        exp_name = row['exp_name']
        exp_type = extract_experiment_type(exp_name)
        color, _ = colors_markers.get(exp_type, colors_markers['default'])

        rmse = row['rmse_mean']

        label = f"{exp_name}"
        if row.get('use_retrieval', False):
            label += f" (RAG)"

        ax1.bar(label, rmse, color=color, alpha=0.7, width=0.6)

    ax1.set_ylabel('RMSE (mean)', fontsize=12)
    ax1.set_title(f'RMSE Comparison ({split.upper()} set)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add median if available
    if 'rmse_median' in split_df.columns:
        for i, (_, row) in enumerate(split_df.iterrows()):
            exp_name = row['exp_name']
            median_rmse = row['rmse_median']
            ax1.plot([i-0.3, i+0.3], [median_rmse, median_rmse],
                    'k--', linewidth=2, label='Median' if i == 0 else "")

    # Cumulative distribution style (sorted by performance)
    sorted_df = split_df.sort_values('rmse_mean')
    positions = range(1, len(sorted_df) + 1)

    ax2.plot(positions, sorted_df['rmse_mean'], 'o-', linewidth=2, markersize=8)
    ax2.set_xlabel('Rank (by RMSE)', fontsize=12)
    ax2.set_ylabel('RMSE (mean)', fontsize=12)
    ax2.set_title(f'RMSE Ranking ({split.upper()} set)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add experiment names as labels
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax2.annotate(row['exp_name'][:15] + '...' if len(row['exp_name']) > 15 else row['exp_name'],
                    (i+1, row['rmse_mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path / f'rmse_distribution_{split}.png', bbox_inches='tight')
    plt.savefig(output_path / f'rmse_distribution_{split}.pdf', bbox_inches='tight')
    plt.close()

    print(f"RMSE distribution plots saved to {output_path / f'rmse_distribution_{split}.png'}")

def plot_atom_bucket_boxplot(df: pd.DataFrame, output_path: Path, split: str = 'test'):
    """Plot atom count bucket boxplots"""
    split_df = df[df['split'] == split].copy()

    if split_df.empty:
        print(f"No data found for split: {split}")
        return

    # Find bucket columns
    bucket_cols = [col for col in split_df.columns if col.startswith('bucket_') and col.endswith('_mean')]

    if not bucket_cols:
        print("No atom bucket data found in the input file")
        return

    # Extract bucket ranges
    bucket_ranges = []
    for col in bucket_cols:
        try:
            bucket_num = col.split('_')[1]
            bucket_ranges.append(bucket_num)
        except:
            continue

    if not bucket_ranges:
        print("Could not parse bucket ranges from column names")
        return

    # Prepare data for plotting
    experiments = []
    bucket_data = {bucket: [] for bucket in bucket_ranges}

    for _, row in split_df.iterrows():
        exp_name = row['exp_name']
        if row.get('use_retrieval', False):
            exp_name += " (RAG)"
        experiments.append(exp_name)

        for bucket in bucket_ranges:
            col_name = f'bucket_{bucket}_mean'
            if col_name in row and not pd.isna(row[col_name]):
                bucket_data[bucket].append(row[col_name])
            else:
                bucket_data[bucket].append(np.nan)

    # Create boxplot
    plt.figure(figsize=(12, 8))

    colors_markers = get_experiment_colors_and_markers()

    # Group experiments by type for coloring
    exp_types = [extract_experiment_type(exp.split(' (')[0]) for exp in experiments]
    colors = [colors_markers.get(exp_type, colors_markers['default'])[0] for exp_type in exp_types]

    # Prepare data for boxplot
    plot_data = []
    labels = []
    for bucket in bucket_ranges:
        values = [v for v in bucket_data[bucket] if not np.isnan(v)]
        if values:
            plot_data.append(values)
            labels.append(f'Bucket {bucket}')

    if not plot_data:
        print("No valid bucket data for plotting")
        return

    bp = plt.boxplot(plot_data, labels=labels, patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    plt.xlabel('Atom Count Bucket', fontsize=12)
    plt.ylabel('RMSE (mean)', fontsize=12)
    plt.title(f'RMSE by Atom Count Bucket ({split.upper()} set)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)

    # Add legend for experiment types
    legend_elements = []
    for exp_type in set(exp_types):
        if exp_type in colors_markers:
            color, marker = colors_markers[exp_type]
            legend_elements.append(mpatches.Patch(color=color, label=exp_type.title()))

    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / f'atom_bucket_boxplot_{split}.png', bbox_inches='tight')
    plt.savefig(output_path / f'atom_bucket_boxplot_{split}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Atom bucket boxplot saved to {output_path / f'atom_bucket_boxplot_{split}.png'}")

def create_summary_table(df: pd.DataFrame, output_path: Path, split: str = 'test'):
    """Create a summary table image"""
    split_df = df[df['split'] == split].copy()

    if split_df.empty:
        print(f"No data found for split: {split}")
        return

    # Select key columns for summary
    key_cols = ['exp_name', 'num_samples', 'match_1', 'match_5', 'rmse_mean', 'rmse_median']
    available_cols = [col for col in key_cols if col in split_df.columns]

    if len(available_cols) < 2:
        print("Not enough data for summary table")
        return

    summary_df = split_df[available_cols].copy()

    # Format for display
    if 'match_1' in summary_df.columns:
        summary_df['match_1'] = summary_df['match_1'] * 100
    if 'match_5' in summary_df.columns:
        summary_df['match_5'] = summary_df['match_5'] * 100

    # Rename columns for better display
    column_renames = {
        'exp_name': 'Experiment',
        'num_samples': 'Samples',
        'match_1': 'Match@1 (%)',
        'match_5': 'Match@5 (%)',
        'rmse_mean': 'RMSE (mean)',
        'rmse_median': 'RMSE (median)'
    }

    for old_name, new_name in column_renames.items():
        if old_name in summary_df.columns:
            summary_df = summary_df.rename(columns={old_name: new_name})

    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=summary_df.round(4).values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_fontsize(12)
            cell.set_facecolor('#E6E6FA')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#F8F8F8')

    plt.title(f'CCSG Results Summary ({split.upper()} set)', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path / f'summary_table_{split}.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_path / f'summary_table_{split}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Summary table saved to {output_path / f'summary_table_{split}.png'}")

def main():
    parser = argparse.ArgumentParser(
        description="Create visualization plots for CCSG experiment results"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input analysis file (CSV or JSON from analyze_ccsg_results.py)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for plots"
    )
    parser.add_argument(
        "--plot_types",
        nargs="+",
        choices=["match_curve", "rmse_dist", "boxplot", "summary"],
        default=["match_curve", "rmse_dist", "boxplot", "summary"],
        help="Types of plots to generate (default: all)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "val"],
        help="Data splits to plot (default: test val)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="both",
        help="Output format for plots (default: both)"
    )

    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_path}")
    df = load_results(input_path)
    print(f"Loaded {len(df)} results")

    # Generate plots
    for split in args.splits:
        print(f"\nGenerating plots for {split.upper()} split...")

        if "match_curve" in args.plot_types:
            plot_match_at_k_curves(df, output_dir, split)

        if "rmse_dist" in args.plot_types:
            plot_rmse_distribution(df, output_dir, split)

        if "boxplot" in args.plot_types:
            plot_atom_bucket_boxplot(df, output_dir, split)

        if "summary" in args.plot_types:
            create_summary_table(df, output_dir, split)

    print(f"\nAll plots saved to {output_dir}")
    print("Generated plots:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()