import json
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
EXPERIMENTS_DIR = "experiments/classification_MLAO_1xlatentqa_1xpast_lens_1xcls"
PLOT_DIR = "plots/MLAO/training_data_1xlatentqa_1xpast_lens_1xcls"
os.makedirs(PLOT_DIR, exist_ok=True)

# IGNORE LIST
IGNORE_FOLDERS = []
IGNORE_MODELS = ["base_model", "None"]

# DATASET DEFINITIONS
OOD_DATASETS = [
    "ag_news", "language_identification", "singular_plural",
    "engels_headline_istrump", "engels_headline_isobama",
    "engels_headline_ischina", "engels_hist_fig_ismale"
]

IID_DATASETS = [
    "geometry_of_truth", "relations", "sst2", "md_gender",
    "snli", "ner", "tense"
]

# # --- NAME MAPPING (Defines Order) ---
# NAME_MAPPING = {
#     # 4B Models
#     "checkpoints_latentqa_cls_past_lens_Qwen3-4B": "AO Qwen-4B",
#     "activation-oracle-multilayer-qwen3-8b-25-50-75": "MLAO Qwen-4B-3L",
#     "activation-oracle-multilayer-qwen3-4b-6L-3xlayer-loop": "MLAO Qwen-4B-6L",
    
#     # 8B Models
#     "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "AO Qwen-8B",
#     "activation-oracle-multilayer-qwen3-14b-25-50-75": "MLAO Qwen-8B-3L",
# }

# --- NAME MAPPING (Defines Order) ---
NAME_MAPPING = {
    # 4B Models
    "checkpoints_latentqa_cls_past_lens_Qwen3-4B": "AO Qwen-4B",
    "activation-oracle-multilayer-qwen3-4b-3L": "MLAO Qwen-4B-3L",
    "activation-oracle-multilayer-qwen3-4b-6L": "MLAO Qwen-4B-6L",
    
    # 8B Models
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "AO Qwen-8B",
    "activation-oracle-multilayer-qwen3-8b-8L": "MLAO Qwen-8B-3L",
}

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------

def compute_metrics(records):
    if not records: return 0.0, 0.0, 0
    correct = []
    for r in records:
        gt = str(r.get("ground_truth", "")).strip().lower().replace(".", "")
        pred = str(r.get("target", "")).strip().lower().replace(".", "")
        if gt == pred: correct.append(1)
        elif gt in pred and len(pred) < 20: correct.append(1)
        else: correct.append(0)

    n = len(correct)
    if n == 0: return 0.0, 0.0, 0
    acc = np.mean(correct) * 100
    se = (np.std(correct, ddof=1) / np.sqrt(n)) * 100
    return acc, se, n

def get_clean_model_name(lora_path, filename):
    if lora_path is None or "base_model" in filename or lora_path == "base_model":
        return "base_model"
    
    short_key = lora_path.split("/")[-1]
    
    if short_key in NAME_MAPPING: return NAME_MAPPING[short_key]
    
    for k, v in NAME_MAPPING.items():
        if k in short_key: return v
            
    return short_key.replace("_", " ")

def load_all_results():
    print(f"🔍 Scanning {EXPERIMENTS_DIR}...")
    files = glob.glob(os.path.join(EXPERIMENTS_DIR, "**", "*.json"), recursive=True)
    deduped_data = {}
    
    for fpath in files:
        if any(ignore in fpath for ignore in IGNORE_FOLDERS): continue

        try:
            with open(fpath, "r") as f: res = json.load(f)
        except Exception: continue

        meta = res.get("meta", {})
        records = res.get("records", [])
        raw_lora_path = meta.get("investigator_lora_path", "base_model")
        
        # Filter base models
        if raw_lora_path in IGNORE_MODELS or raw_lora_path is None: continue
        if "base_model" in str(raw_lora_path): continue

        model_label = get_clean_model_name(raw_lora_path, os.path.basename(fpath))
        if model_label == "base_model": continue

        records_by_ds = {}
        for r in records:
            ds = r["dataset_id"]
            if ds not in records_by_ds: records_by_ds[ds] = []
            records_by_ds[ds].append(r)

        for ds_id, ds_recs in records_by_ds.items():
            if ds_id not in IID_DATASETS and ds_id not in OOD_DATASETS: continue
            acc, se, n = compute_metrics(ds_recs)
            split_type = "IID" if ds_id in IID_DATASETS else "OOD"
            
            key = (model_label, ds_id)
            if key not in deduped_data or n > deduped_data[key]["N"]:
                deduped_data[key] = {
                    "Raw_Path": raw_lora_path,
                    "Model": model_label,
                    "Dataset": ds_id,
                    "Split": split_type,
                    "Accuracy": acc,
                    "SE": se,
                    "N": n
                }

    print(f"✅ Loaded {len(deduped_data)} unique pairs (Labels Cleaned).")
    return pd.DataFrame(list(deduped_data.values()))

# ---------------------------------------------------------
# 3. OUTPUT & PLOTTING
# ---------------------------------------------------------

def print_raw_results_text(df):
    if df.empty: return
    defined_order = [v for v in NAME_MAPPING.values() if v in df["Model"].unique()]
    other_models = [m for m in df["Model"].unique() if m not in defined_order]
    sorted_models = defined_order + other_models

    print("\n" + "="*60 + "\n📊 RAW EVALUATION RESULTS (Clean Labels)\n" + "="*60)

    for clean_name in sorted_models:
        model_df = df[df["Model"] == clean_name].sort_values("Dataset")
        raw_path = model_df.iloc[0]["Raw_Path"]
        
        print(f"\nModel: {clean_name}")
        print(f"Path:  {raw_path}")
        
        for _, row in model_df.iterrows():
            print(f"  {row['Dataset']}: {row['Accuracy']:.2f}% (n={row['N']})")
            
        def get_pool_stats(split_name):
            split_rows = model_df[model_df["Split"] == split_name]
            if split_rows.empty: return None
            total_n = split_rows["N"].sum()
            weighted_acc = np.sum(split_rows["Accuracy"] * split_rows["N"]) / total_n
            p = weighted_acc / 100.0
            pooled_se = np.sqrt(p * (1 - p) / total_n) * 100
            return weighted_acc, pooled_se, total_n

        iid_stats = get_pool_stats("IID")
        if iid_stats: print(f"  IID Accuracy: {iid_stats[0]:.2f}% ± {iid_stats[1]:.2f}% (n={iid_stats[2]})")
        ood_stats = get_pool_stats("OOD")
        if ood_stats: print(f"  OOD Accuracy: {ood_stats[0]:.2f}% ± {ood_stats[1]:.2f}% (n={ood_stats[2]})")

def get_color_mapping(df):
    """Creates a consistent color mapping based on the NAME_MAPPING order."""
    unique_models = [v for v in NAME_MAPPING.values() if v in df["Model"].unique()]
    remaining = [m for m in df["Model"].unique() if m not in unique_models]
    all_models = unique_models + remaining
    
    # Use 'Spectral' to match your category breakdown preference
    palette = sns.color_palette("Spectral", n_colors=len(all_models))
    return dict(zip(all_models, palette))

def plot_aggregated_bar(df, split_name, filename, color_map):
    subset = df[df["Split"] == split_name]
    if subset.empty: return
    
    # Pre-aggregate means and propagated SE
    agg_df = subset.groupby("Model").agg({
        "Accuracy": "mean", "SE": lambda x: np.sqrt(np.sum(x**2)) / len(x)
    }).reset_index()

    # Enforce order
    plot_order = [m for m in color_map.keys() if m in agg_df["Model"].values]

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. Main Bar Plot
    ax = sns.barplot(
        data=agg_df, x="Model", y="Accuracy", hue="Model", 
        order=plot_order, palette=color_map, # Uses same colors as breakdown
        edgecolor="black", alpha=0.9, legend=False
    )
    
    # 2. FIX: Explicitly draw Error Bars (since we plotted pre-aggregated means)
    # Re-sort agg_df to match plot_order so error bars align with bars
    agg_df = agg_df.set_index("Model").reindex(plot_order).reset_index()
    
    plt.errorbar(
        x=range(len(agg_df)), 
        y=agg_df["Accuracy"], 
        yerr=agg_df["SE"], 
        fmt='none',       # No marker for the mean (bars show that)
        c='black',        # Color of the error bar line
        capsize=5,        # Width of the caps
        elinewidth=2      # Thickness of the line
    )

    for container in ax.containers:
        if hasattr(container, "patches"): ax.bar_label(container, fmt='%.1f', padding=5, fontsize=11)
    
    plt.title(f"{split_name} Performance", fontsize=18, weight="bold")
    plt.ylabel("Average Accuracy (%)")
    plt.xlabel("")
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha="right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

def plot_category_breakdown(df, color_map):
    if df.empty: return
    plt.figure(figsize=(20, 10))
    sns.set_theme(style="whitegrid", context="talk")
    iid_ds = sorted([d for d in df["Dataset"].unique() if d in IID_DATASETS])
    ood_ds = sorted([d for d in df["Dataset"].unique() if d in OOD_DATASETS])
    ds_order = iid_ds + ood_ds
    
    hue_order = list(color_map.keys())

    ax = sns.barplot(
        data=df, x="Dataset", y="Accuracy", hue="Model", 
        order=ds_order, hue_order=hue_order, 
        palette=color_map, edgecolor="black", alpha=0.9
    )
    plt.title("Detailed Performance Breakdown", fontsize=22, weight="bold", pad=20)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("")
    plt.ylim(0, 110)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc="upper center", ncol=3, frameon=False)
    
    if len(iid_ds) > 0:
        plt.axvline(x=len(iid_ds) - 0.5, color="black", linestyle="--", alpha=0.5, linewidth=2)
        plt.text(len(iid_ds)/2 - 0.5, 105, "IID (Seen)", ha="center", weight="bold", fontsize=14)
        plt.text(len(iid_ds) + len(ood_ds)/2 - 0.5, 105, "OOD (Unseen)", ha="center", weight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_Category_Breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_all_results()
    if not df.empty:
        print_raw_results_text(df)
        
        # Create global color map for consistency
        color_map = get_color_mapping(df)
        
        print("\nGenerating Plots...")
        plot_aggregated_bar(df, "OOD", "1_OOD_Performance.png", color_map)
        plot_aggregated_bar(df, "IID", "2_IID_Performance.png", color_map)
        plot_category_breakdown(df, color_map)
        print("\n✅ Done!")
    else:
        print("❌ No data found.")