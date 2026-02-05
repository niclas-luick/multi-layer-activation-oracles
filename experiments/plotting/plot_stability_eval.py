"""
Plot stability evaluation results from existing JSON files.

Allows comparing multiple noise scales on the same plot.

Usage:
    python plot_stability_eval.py
    
Configure NOISE_SCALES list below to select which experiments to include.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Model configuration (must match the JSON files you want to load)
MODEL_NAME = "Qwen/Qwen3-8B"
VERBALIZER_LORA = "nluick/MLAO-Qwen3-8B-3L-3N"
DATASET_NAME = "language_identification"

# Select which noise scales to plot (must have corresponding JSON files)
NOISE_SCALES = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]

# Input/output paths
INPUT_DIR = "plots/stability"
OUTPUT_DIR = "plots/stability"

# ============================================================================
# Plotting Functions
# ============================================================================


def compute_accuracy_coverage_at_threshold(
    results: list[dict],
    threshold: float,
) -> tuple[float, float, int]:
    """
    Compute accuracy and coverage for samples with agreement >= threshold.
    
    Returns: (accuracy, coverage, n_samples)
    """
    filtered = [r for r in results if r["agreement_rate"] >= threshold]
    n_filtered = len(filtered)
    n_total = len(results)

    if n_filtered == 0:
        return 0.0, 0.0, 0

    accuracy = sum(r["is_correct"] for r in filtered) / n_filtered
    coverage = n_filtered / n_total

    return accuracy, coverage, n_filtered


def load_results_json(json_path: str) -> dict | None:
    """Load results from JSON file, return None if not found."""
    if not os.path.exists(json_path):
        print(f"WARNING: JSON not found: {json_path}")
        return None
    
    with open(json_path, "r") as f:
        return json.load(f)


def get_json_path(model_name: str, lora_name: str, dataset_name: str, noise_scale: float) -> str:
    """Construct the expected JSON path for a given configuration."""
    model_name_str = model_name.split("/")[-1]
    lora_name_str = lora_name.split("/")[-1]
    return f"{INPUT_DIR}/stability_{model_name_str}_{lora_name_str}_{dataset_name}_noise{noise_scale}.json"


def plot_multi_noise_comparison(
    results_by_noise: dict[float, dict],
    output_path: str,
    title: str = "Stability Analysis: Accuracy & Coverage vs. Threshold",
):
    """
    Plot accuracy and coverage curves for multiple noise scales on the same plot.
    
    Args:
        results_by_noise: Dict mapping noise_scale -> loaded JSON data
        output_path: Where to save the plot
        title: Plot title
    """
    thresholds = np.linspace(0.5, 1.0, 11)  # Use linspace to avoid floating point issues
    
    # Color palette for different noise scales
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_by_noise)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Threshold
    for i, (noise_scale, data) in enumerate(sorted(results_by_noise.items())):
        results = data["results"]
        baseline_acc = data["summary"]["baseline_accuracy"]
        
        accuracies = []
        for thresh in thresholds:
            acc, _, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            accuracies.append(acc)
        
        ax1.plot(
            thresholds, accuracies, 
            "o-", color=colors[i], linewidth=2, markersize=6,
            label=f"noise={noise_scale} (baseline={baseline_acc:.3f})"
        )
    
    ax1.set_xlabel("Agreement Threshold", fontsize=12)
    ax1.set_ylabel("Selective Accuracy", fontsize=12)
    ax1.set_ylim(0.6, 0.9)
    ax1.set_xlim(0.45, 1.05)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy vs. Agreement Threshold", fontsize=12)
    
    # Plot 2: Coverage vs Threshold
    for i, (noise_scale, data) in enumerate(sorted(results_by_noise.items())):
        results = data["results"]
        
        coverages = []
        for thresh in thresholds:
            _, cov, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            coverages.append(cov)
        
        ax2.plot(
            thresholds, coverages,
            "s--", color=colors[i], linewidth=2, markersize=6,
            label=f"noise={noise_scale}"
        )
    
    ax2.set_xlabel("Agreement Threshold", fontsize=12)
    ax2.set_ylabel("Coverage (fraction of data)", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0.45, 1.05)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Coverage vs. Agreement Threshold", fontsize=12)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_accuracy_coverage_tradeoff(
    results_by_noise: dict[float, dict],
    output_path: str,
    title: str = "Accuracy-Coverage Tradeoff",
):
    """
    Plot accuracy vs coverage (parametric in threshold) for multiple noise scales.
    
    This shows the tradeoff: higher threshold = higher accuracy but lower coverage.
    """
    thresholds = np.linspace(0.5, 1.0, 11)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_by_noise)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (noise_scale, data) in enumerate(sorted(results_by_noise.items())):
        results = data["results"]
        
        accuracies = []
        coverages = []
        for thresh in thresholds:
            acc, cov, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            accuracies.append(acc)
            coverages.append(cov)
        
        ax.plot(
            coverages, accuracies,
            "o-", color=colors[i], linewidth=2, markersize=6,
            label=f"noise={noise_scale}"
        )
        
        # # Mark the threshold=0.5 point (highest coverage)
        # ax.annotate(
        #     "t=0.5", (coverages[0], accuracies[0]),
        #     textcoords="offset points", xytext=(5, 5), fontsize=8, color=colors[i]
        # )
        # # Mark the threshold=1.0 point (highest accuracy)
        # ax.annotate(
        #     "t=1.0", (coverages[-1], accuracies[-1]),
        #     textcoords="offset points", xytext=(5, -10), fontsize=8, color=colors[i]
        # )
    
    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Accuracy", fontsize=12)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.6, 0.9)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def print_summary_table(results_by_noise: dict[float, dict]):
    """Print a summary table of key metrics for each noise scale."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Noise Scale':<12} {'Baseline Acc':<14} {'Mean Agreement':<16} {'N Examples':<12}")
    print("-" * 80)
    
    for noise_scale, data in sorted(results_by_noise.items()):
        summary = data["summary"]
        print(f"{noise_scale:<12} {summary['baseline_accuracy']:<14.3f} "
              f"{summary['mean_agreement_rate']:<16.3f} {summary['n_examples']:<12}")
    
    print("=" * 80 + "\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"{'=' * 60}")
    print(f"Stability Evaluation Plotting")
    print(f"Model: {MODEL_NAME}")
    print(f"Verbalizer: {VERBALIZER_LORA}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Noise scales: {NOISE_SCALES}")
    print(f"{'=' * 60}")
    
    # Load all results
    results_by_noise: dict[float, dict] = {}
    
    for noise_scale in NOISE_SCALES:
        json_path = get_json_path(MODEL_NAME, VERBALIZER_LORA, DATASET_NAME, noise_scale)
        data = load_results_json(json_path)
        if data is not None:
            results_by_noise[noise_scale] = data
            print(f"Loaded: noise={noise_scale} ({len(data['results'])} examples)")
    
    if not results_by_noise:
        print("ERROR: No results found! Check your configuration and JSON paths.")
        exit(1)
    
    # Print summary
    print_summary_table(results_by_noise)
    
    # Generate output filename
    model_name_str = MODEL_NAME.split("/")[-1]
    lora_name_str = VERBALIZER_LORA.split("/")[-1]
    noise_str = "_".join(str(n) for n in sorted(results_by_noise.keys()))
    output_base = f"{OUTPUT_DIR}/stability_comparison_{model_name_str}_{lora_name_str}_{DATASET_NAME}"
    
    # Plot 1: Accuracy & Coverage vs Threshold (side by side)
    plot_multi_noise_comparison(
        results_by_noise=results_by_noise,
        output_path=f"{output_base}_curves.png",
        title=f"Stability Analysis: {DATASET_NAME}\n{lora_name_str}",
    )
    
    # Plot 2: Accuracy-Coverage Tradeoff
    plot_accuracy_coverage_tradeoff(
        results_by_noise=results_by_noise,
        output_path=f"{output_base}_tradeoff.png",
        title=f"Accuracy-Coverage Tradeoff: {DATASET_NAME}\n{lora_name_str}",
    )
    
    print("\nDone!")
