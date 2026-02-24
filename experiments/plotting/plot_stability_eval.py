"""
Plot stability evaluation results from existing JSON files.

Supports noise, temperature, threshold, and prompt modes. Configure the
EXPERIMENTS list below to select which JSON files to load and overlay on the
same plot.

Usage:
    python plot_stability_eval.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Model configuration (must match the JSON files you want to load)
MODEL_NAME = "Qwen/Qwen3-8B"
VERBALIZER_LORA = "nluick/MLAO-Qwen3-8B-3L-3N"
DATASET_NAME = "language_identification"

# Each entry: (mode, param_value)
#   mode="noise"       -> param_value is noise_scale (e.g. 0.003)
#   mode="temperature"  -> param_value is temperature (e.g. 1.0)
#   mode="threshold"    -> param_value is ignored (use 0), single deterministic pass
#   mode="prompt"       -> param_value is a string flags combo: "qp", "q", "p", or "none"
EXPERIMENTS: list[tuple[str, float | str]] = [
    ("noise", 0.001),
    ("noise", 0.003),
    ("noise", 0.005),
    ("noise", 0.01),
    ("noise", 0.02),
    #("noise", 0.05),
    ("temperature", 0.3),
    ("temperature", 0.5),
    ("temperature", 0.7),
    ("temperature", 1.0),
    ("temperature", 1.5),
    ("temperature", 2.0),
    ("threshold", 0),
    ("prompt", "qp"),
    #("prompt", "q"),
    #("prompt", "p"),
]

# Input/output paths
INPUT_DIR = "plots/stability"
OUTPUT_DIR = "plots/stability"

# ============================================================================
# Helpers
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
        print(f"  WARNING: not found: {json_path}")
        return None

    with open(json_path, "r") as f:
        return json.load(f)


def get_json_path(model_name: str, lora_name: str, dataset_name: str, mode: str, param: float | str) -> str:
    """Construct the expected JSON path for a given configuration."""
    model_name_str = model_name.split("/")[-1]
    lora_name_str = lora_name.split("/")[-1]
    if mode == "noise":
        param_str = f"noise{param}"
    elif mode == "temperature":
        param_str = f"temp{param}"
    elif mode == "prompt":
        param_str = f"promptvar_{param}" if param else "promptvar_none"
    else:
        param_str = "logitconf"
    return f"{INPUT_DIR}/stability_{model_name_str}_{lora_name_str}_{dataset_name}_{param_str}.json"


def make_label(mode: str, param: float | str) -> str:
    """Human-readable label for a (mode, param) pair."""
    if mode == "noise":
        return f"noise={param}"
    if mode == "temperature":
        return f"T={param}"
    if mode == "prompt":
        flag_labels = {"qp": "question+prefix", "q": "question", "p": "prefix", "none": "baseline"}
        return f"prompt ({flag_labels.get(param, param)})"
    return "logit conf."


def assign_colors(
    entries: list[tuple[str, str, float]],
) -> list[tuple[float, float, float, float]]:
    """
    Assign colors to entries based on mode.
    
    Blues for noise, oranges/reds for temperature, green for threshold.
    Intensity varies within each family (lighter = smaller param, darker = larger param).
    
    Args:
        entries: list of (label, mode, param) tuples
    
    Returns:
        List of RGBA color tuples, one per entry.
    """
    noise_entries = [(i, p) for i, (_, m, p) in enumerate(entries) if m == "noise"]
    temp_entries = [(i, p) for i, (_, m, p) in enumerate(entries) if m == "temperature"]
    threshold_entries = [(i, p) for i, (_, m, p) in enumerate(entries) if m == "threshold"]
    prompt_entries = [(i, p) for i, (_, m, p) in enumerate(entries) if m == "prompt"]

    colors: list[tuple[float, float, float, float] | None] = [None] * len(entries)

    # Blues for noise (range 0.35–0.9 to avoid too-light / too-dark extremes)
    if noise_entries:
        n = len(noise_entries)
        blue_values = plt.cm.Blues(np.linspace(0.35, 0.9, n))
        for j, (idx, _) in enumerate(noise_entries):
            colors[idx] = tuple(blue_values[j])

    # Oranges/reds for temperature
    if temp_entries:
        n = len(temp_entries)
        red_values = plt.cm.OrRd(np.linspace(0.35, 0.9, n))
        for j, (idx, _) in enumerate(temp_entries):
            colors[idx] = tuple(red_values[j])

    # Green for threshold (single entry, distinct from both families)
    for idx, _ in threshold_entries:
        colors[idx] = (0.15, 0.65, 0.15, 1.0)  # Forest green

    # Purples for prompt paraphrase
    if prompt_entries:
        n = len(prompt_entries)
        purple_values = plt.cm.Purples(np.linspace(0.45, 0.9, n))
        for j, (idx, _) in enumerate(prompt_entries):
            colors[idx] = tuple(purple_values[j])

    return colors


def assign_markers(mode: str) -> str:
    """Return marker style based on mode."""
    if mode == "noise":
        return "o"
    if mode == "temperature":
        return "s"
    if mode == "prompt":
        return "^"  # Triangle for prompt
    return "D"  # Diamond for threshold


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_curves(
    entries: list[tuple[str, str, float, dict]],
    output_path: str,
    title: str = "Stability Analysis: Accuracy & Coverage vs. Threshold",
):
    """
    Plot accuracy and coverage curves for multiple experiments on the same plot.
    
    Args:
        entries: list of (label, mode, param, data) tuples
    """
    thresholds = np.linspace(0.5, 1.0, 11)

    color_inputs = [(label, mode, param) for label, mode, param, _ in entries]
    colors = assign_colors(color_inputs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs Threshold
    for i, (label, mode, param, data) in enumerate(entries):
        results = data["results"]
        baseline_acc = data["summary"]["baseline_accuracy"]
        marker = assign_markers(mode)

        accuracies = []
        for thresh in thresholds:
            acc, _, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            accuracies.append(acc)

        ax1.plot(
            thresholds, accuracies,
            f"{marker}-", color=colors[i], linewidth=2, markersize=6,
            label=f"{label} (baseline={baseline_acc:.3f})"
        )

    ax1.set_xlabel("Agreement Threshold", fontsize=12)
    ax1.set_ylabel("Selective Accuracy", fontsize=12)
    ax1.set_xlim(0.45, 1.05)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy vs. Agreement Threshold", fontsize=12)

    # Plot 2: Coverage vs Threshold
    for i, (label, mode, param, data) in enumerate(entries):
        results = data["results"]
        marker = assign_markers(mode)

        coverages = []
        for thresh in thresholds:
            _, cov, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            coverages.append(cov)

        ax2.plot(
            thresholds, coverages,
            f"{marker}--", color=colors[i], linewidth=2, markersize=6,
            label=label,
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


def plot_tradeoff(
    entries: list[tuple[str, str, float, dict]],
    output_path: str,
    title: str = "Accuracy-Coverage Tradeoff",
):
    """
    Plot accuracy vs coverage (parametric in threshold) for multiple experiments.
    """
    thresholds = np.linspace(0.5, 1.0, 11)

    color_inputs = [(label, mode, param) for label, mode, param, _ in entries]
    colors = assign_colors(color_inputs)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, mode, param, data) in enumerate(entries):
        results = data["results"]
        marker = assign_markers(mode)

        accuracies = []
        coverages = []
        for thresh in thresholds:
            acc, cov, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            accuracies.append(acc)
            coverages.append(cov)

        ax.plot(
            coverages, accuracies,
            f"{marker}-", color=colors[i], linewidth=2, markersize=6,
            label=label,
        )

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Accuracy", fontsize=12)
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0.6, 0.9)

    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def print_summary_table(entries: list[tuple[str, str, float, dict]]):
    """Print a summary table of key metrics for each experiment."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Experiment':<24} {'Mode':<14} {'Baseline Acc':<14} {'Mean Agreement':<16} {'N Examples':<12}")
    print("-" * 80)

    for label, mode, param, data in entries:
        summary = data["summary"]
        print(f"{label:<24} {mode:<14} {summary['baseline_accuracy']:<14.3f} "
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
    print(f"Experiments: {EXPERIMENTS}")
    print(f"{'=' * 60}")

    # Load all results as (label, mode, param, data) tuples
    entries: list[tuple[str, str, float, dict]] = []

    for mode, param in EXPERIMENTS:
        json_path = get_json_path(MODEL_NAME, VERBALIZER_LORA, DATASET_NAME, mode, param)
        data = load_results_json(json_path)
        if data is not None:
            label = make_label(mode, param)
            entries.append((label, mode, param, data))
            print(f"  Loaded: {label} ({len(data['results'])} examples)")

    if not entries:
        print("ERROR: No results found! Check your configuration and JSON paths.")
        exit(1)

    # Print summary
    print_summary_table(entries)

    # Check which modes are present
    modes_present = set(m for _, m, _, _ in entries)
    mode_descriptions = {
        "noise": "noise (blues)",
        "temperature": "temperature (oranges/reds)",
        "threshold": "threshold (green)",
        "prompt": "prompt (purples)",
    }
    present_str = ", ".join(mode_descriptions[m] for m in modes_present if m in mode_descriptions)
    print(f"Plotting: {present_str}")

    # Generate output filename that encodes which modes are included
    model_name_str = MODEL_NAME.split("/")[-1]
    lora_name_str = VERBALIZER_LORA.split("/")[-1]
    modes_suffix = "_".join(sorted(modes_present))
    output_base = f"{OUTPUT_DIR}/stability_comparison_{model_name_str}_{lora_name_str}_{DATASET_NAME}_{modes_suffix}"

    # Plot 1: Accuracy & Coverage vs Threshold (side by side)
    plot_curves(
        entries=entries,
        output_path=f"{output_base}_curves.png",
        title=f"Stability Analysis: {DATASET_NAME}\n{lora_name_str}",
    )

    # Plot 2: Accuracy-Coverage Tradeoff
    plot_tradeoff(
        entries=entries,
        output_path=f"{output_base}_tradeoff.png",
        title=f"Accuracy-Coverage Tradeoff: {DATASET_NAME}\n{lora_name_str}",
    )

    print("\nDone!")
