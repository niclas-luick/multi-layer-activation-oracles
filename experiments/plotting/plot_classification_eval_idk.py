"""
Plot results from classification_eval_idk.py (3-way Yes/No/IDK evaluation).

Designed to work with one or a few IDK-trained models. Reads pre-computed
metrics from the JSON (overall_accuracy, selective_accuracy, coverage, idk_rate).
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
EXPERIMENTS_DIR = "experiments/classification_idk"
PLOT_DIR = "plots/classification_idk"
os.makedirs(PLOT_DIR, exist_ok=True)

# Dataset split (must match classification.py / classification_eval_idk.py)
IID_DATASETS = [
    "geometry_of_truth", "relations", "sst2", "md_gender",
    "snli", "ner", "tense",
]
OOD_DATASETS = [
    "ag_news", "language_identification", "singular_plural",
    "engels_headline_istrump", "engels_headline_isobama",
    "engels_headline_ischina", "engels_hist_fig_ismale",
]

# Display name for the single model (or first one found)
MODEL_DISPLAY_NAMES = {
    "MLAO-Qwen3-4B-3L-1N": "MLAO Qwen3-4B-3L-1N",
    "MLAO-Qwen3-4B-3L-3N": "MLAO Qwen3-4B-3L-3N",
    "MLAO-Qwen3-4B-6L-1N": "MLAO Qwen3-4B-6L-1N",
    "MLAO-Qwen3-4B-6L-3N": "MLAO Qwen3-4B-6L-3N",
    "MLAO-Qwen3-4B-6L-6N": "MLAO Qwen3-4B-6L-6N",
    "MLAO-Qwen3-8B-3L-1N": "MLAO Qwen3-8B-3L-1N",
    "MLAO-Qwen3-8B-3L-3N": "MLAO Qwen3-8B-3L-3N",
}


def get_model_display_name(lora_path: str) -> str:
    """Get short display name from LoRA path."""
    if not lora_path:
        return "base_model"
    short = lora_path.split("/")[-1]
    return MODEL_DISPLAY_NAMES.get(short, short.replace("_", " "))


def load_idk_results():
    """Load all classification_idk result JSONs; return list of (display_name, data)."""
    pattern = os.path.join(EXPERIMENTS_DIR, "**", "classification_idk_results_*.json")
    files = glob.glob(pattern)
    results = []
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        meta = data.get("meta", {})
        if meta.get("eval_type") != "idk_3way":
            continue
        lora_path = meta.get("investigator_lora_path") or ""
        display_name = get_model_display_name(lora_path)
        results.append((display_name, data))
    return results


def plot_iid_ood_summary(data_list: list[tuple[str, dict]]) -> None:
    """Single model: IID vs OOD bar chart with overall acc, selective acc, coverage, IDK rate."""
    if not data_list:
        return
    # Use first model when only one
    display_name, data = data_list[0]
    metrics = data.get("metrics", {})

    iid = metrics.get("iid_aggregate", {})
    ood = metrics.get("ood_aggregate", {})
    if not iid and not ood:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns_setup(ax)

    x = np.array([0, 1])
    width = 0.2

    metric_keys = ["overall_accuracy", "selective_accuracy", "coverage", "idk_rate"]
    labels = ["Overall Acc", "Selective Acc", "Coverage", "IDK Rate"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

    for i, (key, label) in enumerate(zip(metric_keys, labels)):
        vals = [
            iid.get(key, 0) * 100 if iid else 0,
            ood.get(key, 0) * 100 if ood else 0,
        ]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[i], edgecolor="black", linewidth=0.8)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(["IID", "OOD"])
    ax.set_ylabel("(%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"IDK Evaluation: {display_name}\nIID vs OOD")
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "idk_iid_ood_summary.png"), dpi=300)
    plt.close()
    print(f"  Saved {PLOT_DIR}/idk_iid_ood_summary.png")


def plot_per_dataset_breakdown(data_list: list[tuple[str, dict]], split_name: str, datasets: list[str]) -> None:
    """Per-dataset bars for one model: overall acc, selective acc, coverage, IDK rate."""
    if not data_list:
        return
    display_name, data = data_list[0]
    metrics = data.get("metrics", {})
    per_ds = metrics.get("per_dataset", {})

    # Restrict to requested datasets and order
    ds_order = [d for d in datasets if d in per_ds]
    if not ds_order:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(ds_order) * 1.2), 6))
    sns_setup(ax)

    x = np.arange(len(ds_order))
    width = 0.2

    metric_keys = ["overall_accuracy", "selective_accuracy", "coverage", "idk_rate"]
    labels = ["Overall Acc", "Selective Acc", "Coverage", "IDK Rate"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

    for i, (key, label) in enumerate(zip(metric_keys, labels)):
        vals = [per_ds[d].get(key, 0) * 100 for d in ds_order]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[i], edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=25, ha="right")
    ax.set_ylabel("(%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"IDK Evaluation: {display_name} — {split_name} datasets")
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    fname = f"idk_breakdown_{split_name.lower()}.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=300)
    plt.close()
    print(f"  Saved {PLOT_DIR}/{fname}")


def plot_overall_single_model(data_list: list[tuple[str, dict]]) -> None:
    """One horizontal bar chart: overall metrics for the single model."""
    if not data_list:
        return
    display_name, data = data_list[0]
    overall = data.get("metrics", {}).get("overall", {})
    if not overall:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    sns_setup(ax)

    keys = ["overall_accuracy", "selective_accuracy", "coverage", "idk_rate"]
    labels = ["Overall Accuracy", "Selective Accuracy", "Coverage", "IDK Rate"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    vals = [overall.get(k, 0) * 100 for k in keys]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("(%)")
    ax.set_title(f"IDK Evaluation: {display_name} (all datasets)")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "idk_overall_single.png"), dpi=300)
    plt.close()
    print(f"  Saved {PLOT_DIR}/idk_overall_single.png")


def sns_setup(ax):
    """Light grid styling."""
    ax.set_facecolor("#fafafa")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)


def print_summary(data_list: list[tuple[str, dict]]) -> None:
    """Print metrics summary to console."""
    if not data_list:
        print("No IDK results found.")
        return
    for display_name, data in data_list:
        metrics = data.get("metrics", {})
        overall = metrics.get("overall", {})
        print(f"\n{display_name}")
        print(f"  Overall accuracy:   {overall.get('overall_accuracy', 0) * 100:.1f}%")
        print(f"  Selective accuracy: {overall.get('selective_accuracy', 0) * 100:.1f}%")
        print(f"  Coverage:          {overall.get('coverage', 0) * 100:.1f}%")
        print(f"  IDK rate:          {overall.get('idk_rate', 0) * 100:.1f}%")
        iid = metrics.get("iid_aggregate", {})
        ood = metrics.get("ood_aggregate", {})
        if iid:
            print(f"  IID — Acc: {iid.get('overall_accuracy', 0)*100:.1f}%, IDK: {iid.get('idk_rate', 0)*100:.1f}%")
        if ood:
            print(f"  OOD — Acc: {ood.get('overall_accuracy', 0)*100:.1f}%, IDK: {ood.get('idk_rate', 0)*100:.1f}%")


if __name__ == "__main__":
    print(f"Scanning {EXPERIMENTS_DIR} for IDK results...")
    data_list = load_idk_results()
    if not data_list:
        print("No classification_idk result files found. Run classification_eval_idk.py first.")
    else:
        print(f"Found {len(data_list)} result set(s): {[d[0] for d in data_list]}")
        print_summary(data_list)
        print("\nGenerating plots...")
        plot_overall_single_model(data_list)
        plot_iid_ood_summary(data_list)
        plot_per_dataset_breakdown(data_list, "IID", IID_DATASETS)
        plot_per_dataset_breakdown(data_list, "OOD", OOD_DATASETS)
        print("\nDone.")
