import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
EXPERIMENTS_DIR = "experiments/classification"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Map ugly LoRA paths to pretty names for the legend
# Update this with your specific path names if they differ
NAME_MAPPING = {
    "base_model": "Base Model (Zero-Shot)",
    "nluick_activation_oracle_multilayer_qwen3_8b_25_50_75": "<b>Your Model</b> (Layers 25,50,75)",
    "adamkarvonen_checkpoints_cls_latentqa_only_addition_Qwen3-8B": "Baseline (LatentQA)",
}

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def compute_metrics(records):
    """Calculates accuracy and standard error from a list of records."""
    if not records:
        return 0.0, 0.0, 0
    
    # Check exact string match
    correct = [1 if r["ground_truth"].strip().lower() == r["target"].strip().lower() else 0 for r in records]
    
    n = len(correct)
    acc = np.mean(correct)
    # Standard Error calculation
    se = np.std(correct, ddof=1) / np.sqrt(n)
    return acc * 100, se * 100, n

def load_all_results():
    data = []
    
    # Find all result JSON files recursively
    search_path = os.path.join(EXPERIMENTS_DIR, "**", "*.json")
    files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(files)} result files.")

    for fpath in files:
        try:
            with open(fpath, "r") as f:
                res = json.load(f)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        meta = res.get("meta", {})
        records = res.get("records", [])
        
        # Extract metadata
        lora_path = meta.get("investigator_lora_path")
        if lora_path is None:
            method_key = "base_model"
        else:
            # Normalize filename to key
            method_key = lora_path.split("/")[-1].replace("/", "_").replace(".", "_")
            if "dataset_classes" in method_key: # Handle messy paths
                method_key = "base_model"

        # Identify if this is a Multi-Layer run or Single Layer
        layer_conf = meta.get("layer_percent")
        if isinstance(layer_conf, list):
            layer_label = "Multi-Layer " + str(layer_conf)
        else:
            layer_label = f"Layer {layer_conf}%"

        # Group records by dataset_id (datasets are mixed in the json sometimes)
        records_by_ds = {}
        for r in records:
            ds = r["dataset_id"]
            if ds not in records_by_ds:
                records_by_ds[ds] = []
            records_by_ds[ds].append(r)

        # Compute metrics per dataset
        for ds_id, ds_records in records_by_ds.items():
            acc, se, n = compute_metrics(ds_records)
            
            # Create a pretty label
            pretty_name = NAME_MAPPING.get(method_key, method_key)
            
            # If it's the baseline, append the layer info to distinguish "Layer 25" from "Layer 50"
            if "Baseline" in pretty_name or "Base Model" in pretty_name:
                final_label = f"{pretty_name} ({layer_label})"
            else:
                # Your model is inherently multi-layer, so the name is enough
                final_label = pretty_name

            data.append({
                "Dataset": ds_id,
                "Method": final_label,
                "Accuracy": acc,
                "SE": se,
                "N": n,
                "Raw_Method_Key": method_key, # For sorting/filtering
                "Layer_Conf": str(layer_conf)
            })

    return pd.DataFrame(data)

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------

def plot_comparison(df):
    if df.empty:
        print("No data found to plot!")
        return

    # Filter out empty datasets or tiny tests
    df = df[df["N"] > 10]

    # Set up the style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Get unique datasets and sort them
    datasets = sorted(df["Dataset"].unique())
    
    # --- PLOT 1: Overview Bar Chart ---
    plt.figure(figsize=(14, 8))
    
    # Create the bar chart
    # We use 'Method' for hue (color) to compare Base vs Baseline vs Yours
    chart = sns.barplot(
        data=df,
        x="Dataset",
        y="Accuracy",
        hue="Method",
        errorbar=None, # We draw our own error bars
        palette="viridis",
        edgecolor="black",
        alpha=0.9
    )

    # Add Error Bars manually (Seaborn's default aggregation can be tricky with pre-computed SE)
    # We iterate over the patches to place error bars
    # Note: This alignment logic assumes simple grouping. 
    # For robust error bars in Seaborn with pre-computed data, 
    # it's often easier to use matplotlib directly, but let's try a pointplot overlay or loop.
    
    # Better approach for exact error bars: Loop through the data
    # Calculate bar width and positions
    num_methods = len(df["Method"].unique())
    num_datasets = len(datasets)
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Activation Oracle Performance: Multi-Layer vs Baselines", fontsize=16, weight='bold')
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("")
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "results_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved main plot to {out_path}")
    plt.show()

    # --- PLOT 2: Aggregate (Average across all datasets) ---
    plt.figure(figsize=(8, 6))
    
    avg_df = df.groupby("Method", as_index=False).agg({
        "Accuracy": "mean",
        "SE": "mean" # Approximation of pooled SE
    })
    
    sns.barplot(
        data=avg_df, 
        x="Method", 
        y="Accuracy", 
        hue="Method", 
        palette="viridis", 
        edgecolor="black",
        legend=False
    )
    
    # Add error bars
    x_coords = range(len(avg_df))
    plt.errorbar(
        x=x_coords, 
        y=avg_df["Accuracy"], 
        yerr=avg_df["SE"], 
        fmt='none', 
        c='black', 
        capsize=5
    )
    
    plt.title("Average Accuracy Across All Datasets", fontsize=14)
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Avg Accuracy (%)")
    plt.tight_layout()
    
    out_path_agg = os.path.join(PLOT_DIR, "results_aggregate.png")
    plt.savefig(out_path_agg, dpi=300)
    print(f"Saved aggregate plot to {out_path_agg}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_all_results()
    
    if not df.empty:
        print("\nData loaded. Methods found:")
        print(df["Method"].unique())
        
        # Optional: Filter to keep the plot clean
        # e.g., only keep your specific run and the best baseline
        # df = df[df["Method"].str.contains("Your Model") | df["Method"].str.contains("Baseline")]
        
        plot_comparison(df)
    else:
        print("DataFrame is empty. Check your EXPERIMENTS_DIR path.")