#!/usr/bin/env python3
"""
XtalNet Case Study Visualization Script

This script creates detailed visualizations for individual crystal structure predictions,
comparing ground truth vs predicted structures with their simulated PXRD patterns.

Usage:
    python scripts/visualize_case_study.py \
        --results_file results.json \
        --sample_ids 0 1 2 \
        --output_dir case_studies/

    python scripts/visualize_case_study.py \
        --results_file results.json \
        --rmse_threshold 0.1 \
        --top_n 5 \
        --output_dir case_studies/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_results(results_file: Path) -> Dict[str, Any]:
    """Load results from compute_ccsg_metrics.py output"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return {}

def find_sample_data(results: Dict[str, Any], sample_id: int) -> Optional[Dict[str, Any]]:
    """Find specific sample data in results"""
    # Look for sample data in various possible locations
    if 'samples' in results:
        samples = results['samples']
        if isinstance(samples, list) and sample_id < len(samples):
            return samples[sample_id]
        elif isinstance(samples, dict) and str(sample_id) in samples:
            return samples[str(sample_id)]

    # Try to find in nested structure
    for key, value in results.items():
        if isinstance(value, dict) and 'samples' in value:
            samples = value['samples']
            if isinstance(samples, list) and sample_id < len(samples):
                return samples[sample_id]
            elif isinstance(samples, dict) and str(sample_id) in samples:
                return samples[str(sample_id)]

    return None

def simulate_pxrd_pattern(structure_data: Dict[str, Any],
                         two_theta_range: Tuple[float, float] = (5, 50),
                         num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate PXRD pattern from crystal structure data.
    This is a simplified simulation for visualization purposes.
    """
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], num_points)

    # Generate simulated peaks based on structure
    # In reality, this would use proper crystallographic calculations
    # Here we create a simplified simulation for demonstration

    if 'peaks' in structure_data:
        # Use provided peak data if available
        peaks = structure_data['peaks']
        pattern = np.zeros_like(two_theta)

        for peak in peaks:
            peak_pos = peak.get('position', 20)
            peak_intensity = peak.get('intensity', 1.0)
            peak_width = peak.get('width', 0.5)

            # Add Gaussian peak
            peak_contribution = peak_intensity * np.exp(
                -0.5 * ((two_theta - peak_pos) / peak_width) ** 2
            )
            pattern += peak_contribution
    else:
        # Generate synthetic pattern based on lattice parameters
        if 'lattice' in structure_data:
            lattice = structure_data['lattice']
            # Generate synthetic peaks based on lattice
            pattern = np.zeros_like(two_theta)

            # Add some synthetic peaks based on lattice parameters
            base_peaks = [10, 15, 20, 25, 30, 35, 40, 45]
            for i, base_pos in enumerate(base_peaks):
                # Modulate position based on lattice parameters
                if len(lattice) >= 3:  # a, b, c
                    scale_factor = (lattice[0] + lattice[1] + lattice[2]) / 30.0
                    peak_pos = base_pos * scale_factor
                else:
                    peak_pos = base_pos

                # Add some variation
                peak_pos += np.random.normal(0, 0.1)
                peak_intensity = np.random.uniform(0.3, 1.0)
                peak_width = np.random.uniform(0.3, 0.8)

                peak_contribution = peak_intensity * np.exp(
                    -0.5 * ((two_theta - peak_pos) / peak_width) ** 2
                )
                pattern += peak_contribution
        else:
            # Generate completely synthetic pattern
            pattern = np.zeros_like(two_theta)
            num_peaks = np.random.randint(5, 15)

            for _ in range(num_peaks):
                peak_pos = np.random.uniform(5, 50)
                peak_intensity = np.random.uniform(0.2, 1.0)
                peak_width = np.random.uniform(0.2, 1.0)

                peak_contribution = peak_intensity * np.exp(
                    -0.5 * ((two_theta - peak_pos) / peak_width) ** 2
                )
                pattern += peak_contribution

    # Add some noise
    noise = np.random.normal(0, 0.01, len(two_theta))
    pattern = np.maximum(pattern + noise, 0)

    return two_theta, pattern

def plot_structure_comparison(sample_data: Dict[str, Any],
                            sample_id: int,
                            output_path: Path,
                            show_pxrd: bool = True):
    """Create comparison plot for ground truth vs predicted structure"""

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extract ground truth and prediction data
    gt_data = sample_data.get('ground_truth', {})
    pred_data = sample_data.get('prediction', {})

    # Get metrics
    rmse = sample_data.get('rmse', 'N/A')
    match_score = sample_data.get('match_score', 'N/A')

    # Title
    fig.suptitle(f'Crystal Structure Prediction - Sample {sample_id}\n' +
                 f'RMSE: {rmse:.4f} | Match Score: {match_score:.3f}',
                 fontsize=16, fontweight='bold')

    if show_pxrd:
        # PXRD patterns
        ax1 = fig.add_subplot(gs[0, :])

        # Generate PXRD patterns
        two_theta, gt_pxrd = simulate_pxrd_pattern(gt_data)
        _, pred_pxrd = simulate_pxrd_pattern(pred_data)

        ax1.plot(two_theta, gt_pxrd, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(two_theta, pred_pxrd, 'r--', linewidth=2, label='Prediction', alpha=0.8)

        ax1.set_xlabel('2θ (degrees)', fontsize=12)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.set_title('PXRD Pattern Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(5, 50)

    # Structure info
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')

    # Ground truth structure info
    gt_info = []
    if 'formula' in gt_data:
        gt_info.append(f"Formula: {gt_data['formula']}")
    if 'num_atoms' in gt_data:
        gt_info.append(f"Atoms: {gt_data['num_atoms']}")
    if 'lattice' in gt_data:
        lattice = gt_data['lattice']
        if len(lattice) >= 6:
            gt_info.append(f"a={lattice[0]:.3f}Å, b={lattice[1]:.3f}Å, c={lattice[2]:.3f}Å")
            gt_info.append(f"α={lattice[3]:.1f}°, β={lattice[4]:.1f}°, γ={lattice[5]:.1f}°")

    gt_text = "Ground Truth Structure:\n" + "\n".join(gt_info)
    ax2.text(0.05, 0.95, gt_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Predicted structure info
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    pred_info = []
    if 'formula' in pred_data:
        pred_info.append(f"Formula: {pred_data['formula']}")
    if 'num_atoms' in pred_data:
        pred_info.append(f"Atoms: {pred_data['num_atoms']}")
    if 'lattice' in pred_data:
        lattice = pred_data['lattice']
        if len(lattice) >= 6:
            pred_info.append(f"a={lattice[0]:.3f}Å, b={lattice[1]:.3f}Å, c={lattice[2]:.3f}Å")
            pred_info.append(f"α={lattice[3]:.1f}°, β={lattice[4]:.1f}°, γ={lattice[5]:.1f}°")

    pred_text = "Predicted Structure:\n" + "\n".join(pred_info)
    ax3.text(0.05, 0.95, pred_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Additional metrics and analysis
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Collect additional metrics
    metrics_text = "Additional Metrics:\n"

    if 'lattice_rmse' in sample_data:
        metrics_text += f"Lattice RMSE: {sample_data['lattice_rmse']:.4f}\n"

    if 'atom_rmse' in sample_data:
        metrics_text += f"Atomic RMSE: {sample_data['atom_rmse']:.4f}\n"

    if 'retrieval_info' in sample_data:
        retrieval_info = sample_data['retrieval_info']
        if 'top_k_match' in retrieval_info:
            metrics_text += f"Retrieval Top-k Match: {retrieval_info['top_k_match']:.3f}\n"
        if 'template_similarity' in retrieval_info:
            metrics_text += f"Template Similarity: {retrieval_info['template_similarity']:.3f}\n"

    if 'generation_info' in sample_data:
        gen_info = sample_data['generation_info']
        if 'num_steps' in gen_info:
            metrics_text += f"Generation Steps: {gen_info['num_steps']}\n"
        if 'convergence_time' in gen_info:
            metrics_text += f"Convergence Time: {gen_info['convergence_time']:.2f}s\n"

    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path / f'case_study_{sample_id:04d}.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_path / f'case_study_{sample_id:04d}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Case study {sample_id} saved to {output_path / f'case_study_{sample_id:04d}.png'}")

def export_cif_files(sample_data: Dict[str, Any],
                    sample_id: int,
                    output_path: Path):
    """Export ground truth and predicted structures as CIF files"""

    gt_data = sample_data.get('ground_truth', {})
    pred_data = sample_data.get('prediction', {})

    # Export ground truth CIF
    if 'cif_content' in gt_data:
        with open(output_path / f'ground_truth_{sample_id:04d}.cif', 'w') as f:
            f.write(gt_data['cif_content'])
    else:
        # Generate simplified CIF if not provided
        cif_content = generate_simple_cif(gt_data, f'ground_truth_{sample_id:04d}')
        with open(output_path / f'ground_truth_{sample_id:04d}.cif', 'w') as f:
            f.write(cif_content)

    # Export predicted CIF
    if 'cif_content' in pred_data:
        with open(output_path / f'prediction_{sample_id:04d}.cif', 'w') as f:
            f.write(pred_data['cif_content'])
    else:
        # Generate simplified CIF if not provided
        cif_content = generate_simple_cif(pred_data, f'prediction_{sample_id:04d}')
        with open(output_path / f'prediction_{sample_id:04d}.cif', 'w') as f:
            f.write(cif_content)

    print(f"CIF files exported for sample {sample_id}")

def generate_simple_cif(structure_data: Dict[str, Any], name: str) -> str:
    """Generate a simplified CIF file from structure data"""

    cif_lines = [
        f"data_{name}",
        "_audit_creation_method 'Generated by XtalNet'",
        "_chemical_name_mineral 'Unknown'",
    ]

    if 'formula' in structure_data:
        cif_lines.append(f"_chemical_formula_sum '{structure_data['formula']}'")

    if 'lattice' in structure_data:
        lattice = structure_data['lattice']
        if len(lattice) >= 6:
            cif_lines.extend([
                f"_cell_length_a {lattice[0]:.6f}",
                f"_cell_length_b {lattice[1]:.6f}",
                f"_cell_length_c {lattice[2]:.6f}",
                f"_cell_angle_alpha {lattice[3]:.3f}",
                f"_cell_angle_beta {lattice[4]:.3f}",
                f"_cell_angle_gamma {lattice[5]:.3f}",
            ])

    # Add atomic positions if available
    if 'atoms' in structure_data:
        atoms = structure_data['atoms']
        cif_lines.extend([
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z"
        ])

        for i, atom in enumerate(atoms):
            if isinstance(atom, dict):
                element = atom.get('element', 'C')
                x, y, z = atom.get('position', [0, 0, 0])
                cif_lines.append(f"{i+1} {element} {x:.6f} {y:.6f} {z:.6f}")

    cif_lines.append("")  # Empty line at end

    return "\n".join(cif_lines)

def find_samples_by_rmse(results: Dict[str, Any],
                        rmse_threshold: float,
                        top_n: Optional[int] = None) -> List[Tuple[int, Dict[str, Any]]]:
    """Find samples with RMSE below threshold or top N best samples"""

    samples = []

    # Look for samples in various locations
    sample_lists = []

    if 'samples' in results:
        sample_lists.append(results['samples'])

    # Check nested structures
    for key, value in results.items():
        if isinstance(value, dict) and 'samples' in value:
            sample_lists.append(value['samples'])

    # Collect all samples with RMSE
    for sample_list in sample_lists:
        if isinstance(sample_list, list):
            for i, sample in enumerate(sample_list):
                if isinstance(sample, dict) and 'rmse' in sample:
                    samples.append((i, sample))
        elif isinstance(sample_list, dict):
            for sample_id_str, sample in sample_list.items():
                if isinstance(sample, dict) and 'rmse' in sample:
                    try:
                        sample_id = int(sample_id_str)
                        samples.append((sample_id, sample))
                    except ValueError:
                        continue

    # Filter by RMSE threshold
    if rmse_threshold > 0:
        filtered_samples = [(sid, s) for sid, s in samples if s['rmse'] <= rmse_threshold]
        if filtered_samples:
            samples = filtered_samples

    # Sort by RMSE
    samples.sort(key=lambda x: x[1]['rmse'])

    # Take top N if specified
    if top_n is not None:
        samples = samples[:top_n]

    return samples

def create_summary_report(sample_results: List[Tuple[int, Dict[str, Any]]],
                         output_path: Path):
    """Create a summary report of all visualized samples"""

    report_lines = [
        "# Crystal Structure Prediction Case Study Report",
        "",
        f"Number of samples analyzed: {len(sample_results)}",
        "",
        "## Sample Summary",
        "",
        "| Sample ID | RMSE | Match Score | Formula (GT) | Formula (Pred) |",
        "|-----------|------|-------------|--------------|----------------|",
    ]

    for sample_id, sample_data in sample_results:
        rmse = sample_data.get('rmse', 'N/A')
        match_score = sample_data.get('match_score', 'N/A')

        gt_formula = sample_data.get('ground_truth', {}).get('formula', 'N/A')
        pred_formula = sample_data.get('prediction', {}).get('formula', 'N/A')

        report_lines.append(
            f"| {sample_id:04d} | {rmse:.4f} | {match_score:.3f} | {gt_formula} | {pred_formula} |"
        )

    report_lines.extend([
        "",
        "## Files Generated",
        "",
    ])

    # List generated files
    for file_path in sorted(output_path.glob("*")):
        if file_path.is_file():
            report_lines.append(f"- {file_path.name}")

    report_content = "\n".join(report_lines)

    with open(output_path / "case_study_report.md", 'w') as f:
        f.write(report_content)

    print(f"Summary report saved to {output_path / 'case_study_report.md'}")

def main():
    parser = argparse.ArgumentParser(
        description="Create detailed case study visualizations for crystal structure predictions"
    )
    parser.add_argument(
        "--results_file",
        required=True,
        help="Results file from compute_ccsg_metrics.py"
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        help="Specific sample IDs to visualize"
    )
    parser.add_argument(
        "--rmse_threshold",
        type=float,
        default=0.0,
        help="Select samples with RMSE below this threshold (default: 0.0)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        help="Select top N samples by RMSE (default: all matching criteria)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for visualizations and CIF files"
    )
    parser.add_argument(
        "--export_cif",
        action="store_true",
        help="Export CIF files for structures"
    )
    parser.add_argument(
        "--no_pxrd",
        action="store_true",
        help="Skip PXRD pattern plots"
    )

    args = parser.parse_args()

    # Load results
    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)

    if not results_file.exists():
        print(f"Error: Results file {results_file} does not exist!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_file}")
    results = load_results(results_file)

    if not results:
        print("Error: Could not load results from file!")
        return

    # Determine which samples to visualize
    sample_results = []

    if args.sample_ids:
        # Use specific sample IDs
        for sample_id in args.sample_ids:
            sample_data = find_sample_data(results, sample_id)
            if sample_data:
                sample_results.append((sample_id, sample_data))
            else:
                print(f"Warning: Sample {sample_id} not found in results")
    else:
        # Find samples by RMSE threshold or top N
        sample_results = find_samples_by_rmse(results, args.rmse_threshold, args.top_n)

    if not sample_results:
        print("No samples found matching the criteria!")
        return

    print(f"Found {len(sample_results)} samples to visualize")

    # Create visualizations
    for sample_id, sample_data in sample_results:
        print(f"\nProcessing sample {sample_id}...")

        # Create comparison plot
        plot_structure_comparison(
            sample_data, sample_id, output_dir,
            show_pxrd=not args.no_pxrd
        )

        # Export CIF files if requested
        if args.export_cif:
            export_cif_files(sample_data, sample_id, output_dir)

    # Create summary report
    create_summary_report(sample_results, output_dir)

    print(f"\nCase study visualizations complete!")
    print(f"Output directory: {output_dir}")
    print(f"Generated {len(sample_results)} case studies")

if __name__ == "__main__":
    main()