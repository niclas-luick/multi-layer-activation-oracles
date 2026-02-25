"""
Plot PersonaQA open-ended stability evaluation results.

Loads per-question-type JSON files produced by personaqa_stability_eval.py and
plots accuracy-coverage curves (one line per question type).

Usage:
    python plot_personaqa_stability_eval.py
    python plot_personaqa_stability_eval.py --question-types country favorite_food
    python plot_personaqa_stability_eval.py --mode noise --noise-scale 0.005
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_stability_eval import (
    compute_accuracy_coverage_at_threshold,
    print_summary_table,
)

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
VERBALIZER_LORA = "nluick/MLAO-Qwen3-8B-3L-3N"

PROMPT_TYPES = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

INPUT_DIR = "plots/stability/data"
OUTPUT_DIR = "plots/stability/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Helpers
# ============================================================================


def get_personaqa_json_path(
    model_name: str,
    verbalizer_lora: str,
    question_type: str,
    n_samples: int,
    mode: str = "prompt",
    noise_scale: float = 0.0,
) -> str:
    """Construct the expected JSON path for a personaqa stability result."""
    model_str = model_name.split("/")[-1]
    verbalizer_str = verbalizer_lora.split("/")[-1]
    if mode == "prompt":
        mode_str = "prompt"
    elif mode == "noise":
        mode_str = f"noise{noise_scale}"
    else:  # combined
        mode_str = f"combined{noise_scale}"
    return (
        f"{INPUT_DIR}/personaqa_stability_{model_str}_{verbalizer_str}"
        f"_{question_type}_{mode_str}_n{n_samples}.json"
    )


def load_results_json(json_path: str) -> dict | None:
    """Load results from JSON file, return None if not found."""
    if not os.path.exists(json_path):
        print(f"  WARNING: not found: {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


# ============================================================================
# Plotting
# ============================================================================


def _compute_all_curves(
    entries: list[tuple[str, dict]],
    thresholds: np.ndarray,
) -> list[tuple[list[float], list[float]]]:
    """Compute (accuracies, coverages) curves for each entry."""
    curves = []
    for _, data in entries:
        results = data["results"]
        accs, covs = [], []
        for thresh in thresholds:
            acc, cov, _ = compute_accuracy_coverage_at_threshold(results, thresh)
            accs.append(acc)
            covs.append(cov)
        curves.append((accs, covs))
    return curves


def _mean_curve(curves: list[tuple[list[float], list[float]]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean accuracy and coverage across all curves."""
    all_accs = np.array([accs for accs, _ in curves])
    all_covs = np.array([covs for _, covs in curves])
    return all_accs.mean(axis=0), all_covs.mean(axis=0)


def plot_curves(
    entries: list[tuple[str, dict]],
    output_path: str,
    title: str = "PersonaQA Stability: Accuracy & Coverage vs. Threshold",
):
    """Plot accuracy and coverage curves for multiple question types plus average."""
    thresholds = np.linspace(0.0, 1.0, 21)
    cmap = plt.cm.tab10
    colors = [cmap(i / max(len(entries) - 1, 1)) for i in range(len(entries))]

    curves = _compute_all_curves(entries, thresholds)
    mean_accs, mean_covs = _mean_curve(curves)
    mean_baseline = np.mean([d["summary"]["baseline_accuracy"] for _, d in entries])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (label, data) in enumerate(entries):
        baseline_acc = data["summary"]["baseline_accuracy"]
        accs, covs = curves[i]

        ax1.plot(
            thresholds, accs,
            "o-", color=colors[i], linewidth=1.5, markersize=4, alpha=0.6,
            label=f"{label} (baseline={baseline_acc:.3f})",
        )
        ax2.plot(
            thresholds, covs,
            "o--", color=colors[i], linewidth=1.5, markersize=4, alpha=0.6,
            label=label,
        )

    # Average line
    ax1.plot(
        thresholds, mean_accs,
        "s-", color="black", linewidth=2.5, markersize=5,
        label=f"AVERAGE (baseline={mean_baseline:.3f})",
    )
    ax2.plot(
        thresholds, mean_covs,
        "s--", color="black", linewidth=2.5, markersize=5,
        label="AVERAGE",
    )

    ax1.set_xlabel("Agreement Threshold", fontsize=12)
    ax1.set_ylabel("Selective Accuracy", fontsize=12)
    ax1.set_xlim(-0.05, 1.05)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy vs. Agreement Threshold", fontsize=12)

    ax2.set_xlabel("Agreement Threshold", fontsize=12)
    ax2.set_ylabel("Coverage (fraction of data)", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Coverage vs. Agreement Threshold", fontsize=12)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_tradeoff(
    entries: list[tuple[str, dict]],
    output_path: str,
    title: str = "PersonaQA Stability: Accuracy-Coverage Tradeoff",
):
    """Plot accuracy vs coverage (parametric in threshold) for multiple question types plus average."""
    thresholds = np.linspace(0.0, 1.0, 21)
    cmap = plt.cm.tab10
    colors = [cmap(i / max(len(entries) - 1, 1)) for i in range(len(entries))]

    curves = _compute_all_curves(entries, thresholds)
    mean_accs, mean_covs = _mean_curve(curves)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, _) in enumerate(entries):
        accs, covs = curves[i]
        ax.plot(
            covs, accs,
            "o-", color=colors[i], linewidth=1.5, markersize=4, alpha=0.6,
            label=label,
        )

    # Average line
    ax.plot(
        mean_covs, mean_accs,
        "s-", color="black", linewidth=2.5, markersize=5,
        label="AVERAGE",
    )

    ax.set_xlabel("Coverage", fontsize=12)
    ax.set_ylabel("Selective Accuracy", fontsize=12)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_average_only(
    entries: list[tuple[str, dict]],
    output_path: str,
    title: str = "PersonaQA Stability: Average across question types",
):
    """Plot only the average accuracy and coverage curves (no per-type lines)."""
    thresholds = np.linspace(0.0, 1.0, 21)

    curves = _compute_all_curves(entries, thresholds)
    mean_accs, mean_covs = _mean_curve(curves)
    mean_baseline = np.mean([d["summary"]["baseline_accuracy"] for _, d in entries])
    mean_agreement = np.mean([d["summary"]["mean_agreement_rate"] for _, d in entries])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy vs threshold
    ax1.plot(
        thresholds, mean_accs,
        "s-", color="black", linewidth=2.5, markersize=6,
        label=f"Average (baseline={mean_baseline:.3f})",
    )
    ax1.axhline(y=mean_baseline, color="gray", linestyle=":", linewidth=1, label="Baseline (no filtering)")
    ax1.set_xlabel("Agreement Threshold", fontsize=12)
    ax1.set_ylabel("Selective Accuracy", fontsize=12)
    ax1.set_xlim(-0.05, 1.05)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Accuracy vs. Agreement Threshold", fontsize=12)

    # Coverage vs threshold
    ax2.plot(
        thresholds, mean_covs,
        "s--", color="black", linewidth=2.5, markersize=6,
        label=f"Average (mean agreement={mean_agreement:.3f})",
    )
    ax2.set_xlabel("Agreement Threshold", fontsize=12)
    ax2.set_ylabel("Coverage (fraction of data)", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Coverage vs. Agreement Threshold", fontsize=12)

    # Accuracy-coverage tradeoff
    ax3.plot(
        mean_covs, mean_accs,
        "s-", color="black", linewidth=2.5, markersize=6,
        label="Average",
    )
    ax3.set_xlabel("Coverage", fontsize=12)
    ax3.set_ylabel("Selective Accuracy", fontsize=12)
    ax3.set_xlim(0.0, 1.05)
    ax3.set_ylim(0.0, 1.05)
    ax3.legend(loc="lower left", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Accuracy-Coverage Tradeoff", fontsize=12)

    fig.suptitle(f"{title}\n(averaged over {len(entries)} question types)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PersonaQA stability evaluation results")
    parser.add_argument(
        "--question-types", nargs="+", default=None,
        help="Question types to plot (default: all 6)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="N samples used in eval (default: 50)",
    )
    parser.add_argument(
        "--mode", choices=["prompt", "noise", "combined"], default="prompt",
        help="Stability mode (default: prompt)",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.005,
        help="Noise scale used in eval (default: 0.005)",
    )
    args = parser.parse_args()

    question_types = args.question_types or PROMPT_TYPES

    print(f"{'=' * 60}")
    print(f"PersonaQA Stability Evaluation Plotting")
    print(f"Model: {MODEL_NAME}")
    print(f"Verbalizer: {VERBALIZER_LORA}")
    print(f"Mode: {args.mode}")
    if args.mode in ("noise", "combined"):
        print(f"Noise scale: {args.noise_scale}")
    print(f"N samples: {args.n_samples}")
    print(f"Question types: {question_types}")
    print(f"{'=' * 60}")

    # Load all results as (label, data) tuples
    entries: list[tuple[str, dict]] = []
    for pt in question_types:
        json_path = get_personaqa_json_path(
            MODEL_NAME, VERBALIZER_LORA, pt,
            args.n_samples, args.mode, args.noise_scale,
        )
        data = load_results_json(json_path)
        if data is not None:
            entries.append((pt, data))
            s = data["summary"]
            print(f"  Loaded: {pt} (acc={s['baseline_accuracy']:.3f}, "
                  f"agreement={s['mean_agreement_rate']:.3f}, "
                  f"n={s['n_examples']})")

    if not entries:
        print("ERROR: No results found! Check paths and run personaqa_stability_eval.py first.")
        exit(1)

    # Print summary table (reuse from plot_stability_eval)
    table_entries = [(pt, "prompt", 0, data) for pt, data in entries]
    print_summary_table(table_entries)

    # Build output filename
    model_str = MODEL_NAME.split("/")[-1]
    lora_str = VERBALIZER_LORA.split("/")[-1]
    if args.mode == "prompt":
        mode_str = "prompt"
    elif args.mode == "noise":
        mode_str = f"noise{args.noise_scale}"
    else:
        mode_str = f"combined{args.noise_scale}"
    output_base = (
        f"{OUTPUT_DIR}/personaqa_stability_{model_str}_{lora_str}"
        f"_{mode_str}_n{args.n_samples}"
    )

    # Plot curves
    title_suffix = f"{args.mode} | n={args.n_samples}"
    if args.mode in ("noise", "combined"):
        title_suffix += f" | noise={args.noise_scale}"
    plot_curves(
        entries=entries,
        output_path=f"{output_base}_curves.png",
        title=f"PersonaQA Stability: {title_suffix}\n{lora_str}",
    )

    # Plot tradeoff
    plot_tradeoff(
        entries=entries,
        output_path=f"{output_base}_tradeoff.png",
        title=f"PersonaQA Accuracy-Coverage Tradeoff: {title_suffix}\n{lora_str}",
    )

    # Plot average only (separate clean plot)
    if len(entries) > 1:
        plot_average_only(
            entries=entries,
            output_path=f"{output_base}_average.png",
            title=f"PersonaQA Stability Average: {title_suffix}\n{lora_str}",
        )

    print("\nDone!")
