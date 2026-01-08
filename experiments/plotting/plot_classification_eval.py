import json
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.patches as mpatches
import colorsys

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
EXPERIMENTS_DIR = "experiments/classification/"
PLOT_DIR = "plots/final/"
os.makedirs(PLOT_DIR, exist_ok=True)

# COLOR ADJUSTMENT
# Lower this value to make color differences within a group smaller
COLOR_SPREAD = 0.05

# SPACING CONFIGURATION
BAR_WIDTH = 0.6
SUBGROUP_GAP = 0.3  # Reduced slightly to keep groups cohesive
DATASET_GAP = 1.2   # Space between different datasets

# TITLES
IID_PLOT_TITLE = "Classification Performance, IID to Training"
OOD_PLOT_TITLE = "Classification Performance, Out-Of-Distribution (OOD)"

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

# --- NAME MAPPING (Defines Order) ---
NAME_MAPPING = {
    # 4B Models
    "checkpoints_latentqa_cls_past_lens_Qwen3-4B": "AO Qwen3-4B",
    "MLAO-Qwen3-4B-3L-1N": "MLAO Qwen3-4B-3L-1N",
    "MLAO-Qwen3-4B-3L-3N": "MLAO Qwen3-4B-3L-3N",
    "MLAO-Qwen3-4B-6L-1N": "MLAO Qwen3-4B-6L-1N",
    "MLAO-Qwen3-4B-6L-3N": "MLAO Qwen3-4B-6L-3N",
    "MLAO-Qwen3-4B-6L-6N": "MLAO Qwen3-4B-6L-6N",

    # 8B Models
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "AO Qwen3-8B",
    "MLAO-Qwen3-8B-3L-1N": "MLAO Qwen3-8B-3L-1N",
    "MLAO-Qwen3-8B-3L-3N": "MLAO Qwen3-8B-3L-3N",
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
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

def parse_model_info(name):
    suffixes = ["-1N", "-3N", "-6N"]
    for s in suffixes:
        if name.endswith(s):
            return name.replace(s, "").strip(), s
    return name, ""

def get_subtle_color(base_color, idx, total):
    try:
        c = mc.cnames[base_color] if base_color in mc.cnames else base_color
    except:
        c = base_color
    r, g, b = mc.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    if total <= 1:
        return (r, g, b)
    
    spread = COLOR_SPREAD
    step = spread / max(1, total - 1)
    
    new_l = l - (spread / 2) + (idx * step)
    new_l = max(0.2, min(0.9, new_l))
    
    return colorsys.hls_to_rgb(h, new_l, s)

def get_group_colors(df):
    unique_models = [v for v in NAME_MAPPING.values() if v in df["Model"].unique()]
    remaining = [m for m in df["Model"].unique() if m not in unique_models]
    all_models = unique_models + remaining
    
    groups = {}
    for m in all_models:
        grp, _ = parse_model_info(m)
        if grp not in groups: groups[grp] = []
        groups[grp].append(m)
        
    group_names = list(groups.keys())
    
    # Use Spectral Palette (SAME as Plot 1 & 2)
    base_palette = sns.color_palette("Spectral", n_colors=len(group_names))
    
    model_colors = {}
    for i, grp in enumerate(group_names):
        base = base_palette[i]
        members = groups[grp]
        for j, member in enumerate(members):
            model_colors[member] = get_subtle_color(base, j, len(members))
            
    return model_colors

# ---------------------------------------------------------
# 4. OUTPUT & PLOTTING
# ---------------------------------------------------------

def print_raw_results_text(df):
    if df.empty: return
    defined_order = [v for v in NAME_MAPPING.values() if v in df["Model"].unique()]
    other = [m for m in df["Model"].unique() if m not in defined_order]
    sorted_models = defined_order + other

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
            w_acc = np.sum(split_rows["Accuracy"] * split_rows["N"]) / total_n
            p = w_acc / 100.0
            se = np.sqrt(p * (1 - p) / total_n) * 100
            return w_acc, se, total_n

        iid = get_pool_stats("IID")
        if iid: print(f"  IID Accuracy: {iid[0]:.2f}% ± {iid[1]:.2f}% (n={iid[2]})")
        ood = get_pool_stats("OOD")
        if ood: print(f"  OOD Accuracy: {ood[0]:.2f}% ± {ood[1]:.2f}% (n={ood[2]})")

def plot_aggregated_bar(df, split_name, title, filename, color_map):
    subset = df[df["Split"] == split_name]
    if subset.empty: return
    
    agg_df = subset.groupby("Model").agg({
        "Accuracy": "mean", "SE": lambda x: np.sqrt(np.sum(x**2)) / len(x)
    }).reset_index()

    plot_order = [m for m in color_map.keys() if m in agg_df["Model"].values]
    
    plt.figure(figsize=(15, 9)) 
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    ax = plt.gca()
    ax.xaxis.grid(False) 
    
    x_pos = 0.0
    bar_width = 0.8
    group_gap = 0.8
    
    xticks_locs = []
    xticks_labels = []
    group_intervals = {} 
    current_group = None
    
    for model in plot_order:
        row = agg_df[agg_df["Model"] == model].iloc[0]
        group, suffix = parse_model_info(model)
        
        if current_group is not None and group != current_group:
            x_pos += group_gap
        current_group = group
        
        center_x = x_pos
        if group not in group_intervals:
            group_intervals[group] = [center_x, center_x]
        else:
            group_intervals[group][1] = center_x
            
        color = color_map[model]
        
        plt.bar(x_pos, row["Accuracy"], width=bar_width, color=color, 
                edgecolor="black", alpha=0.9, linewidth=1.0)
        
        plt.errorbar(x_pos, row["Accuracy"], yerr=row["SE"], 
                     fmt='none', c='black', capsize=6, elinewidth=2)
        
        plt.text(x_pos, row["Accuracy"] + 1.5, f"{row['Accuracy']:.1f}", 
                 ha='center', va='bottom', fontsize=20, color="#333333")

        xticks_locs.append(x_pos)
        xticks_labels.append(suffix)
        x_pos += bar_width

    ax.set_xticks(xticks_locs)
    ax.set_xticklabels(xticks_labels, fontsize=18)
    
    trans = ax.get_xaxis_transform()
    for grp, (min_x, max_x) in group_intervals.items():
        center = (min_x + max_x) / 2
        plt.text(center, -0.08, grp, transform=trans, 
                 ha='center', va='top', fontsize=18, color="#333333")

    plt.title(title, fontsize=28, pad=20)
    plt.ylabel("Average Accuracy (%)", fontsize=24, labelpad=20)
    plt.xlabel("")
    plt.ylim(0, 110)
    plt.yticks(fontsize=22)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

def plot_single_breakdown(df, target_datasets, title, filename, color_map):
    """
    Plots using same colors as aggregated plots, with manual spacing logic.
    """
    subset = df[df["Dataset"].isin(target_datasets)].copy()
    if subset.empty: return

    ds_order = [d for d in target_datasets if d in subset["Dataset"].unique()]
    hue_order = list(color_map.keys())

    # Dynamic Width
    plot_width = max(20, len(ds_order) * 3)
    plt.figure(figsize=(plot_width, 14)) 
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.3)
    ax = plt.gca()
    ax.xaxis.grid(False)
    
    # ---------------- Manual Plotting ----------------
    
    current_x = 0
    xticks_locs = []
    
    # Track legends: use color_map strictly to ensure identical colors
    legend_handles = []
    seen_models_for_legend = set()

    for ds in ds_order:
        ds_data = subset[subset["Dataset"] == ds]
        
        bars_x_positions = []
        current_group = None
        
        # Iterate in specific Hue Order to ensure color consistency
        for model in hue_order:
            row = ds_data[ds_data["Model"] == model]
            if row.empty: continue
            
            acc = row.iloc[0]["Accuracy"]
            se = row.iloc[0]["SE"]
            
            group, _ = parse_model_info(model)
            
            if current_group is not None and group != current_group:
                current_x += SUBGROUP_GAP
            
            current_group = group
            
            color = color_map[model]
            
            # Plot Bar
            plt.bar(current_x, acc, width=BAR_WIDTH, color=color, 
                    edgecolor="black", linewidth=1.0, alpha=0.9) # alpha/edge match other plots
            
            if model not in seen_models_for_legend:
                patch = mpatches.Patch(color=color, label=model)
                legend_handles.append(patch)
                seen_models_for_legend.add(model)
            
            if se > 0:
                plt.errorbar(current_x, acc, yerr=se, 
                             fmt='none', c='black', capsize=3, elinewidth=1.5)
            
            bars_x_positions.append(current_x)
            current_x += BAR_WIDTH 
            
        if bars_x_positions:
            center = (min(bars_x_positions) + max(bars_x_positions)) / 2
            xticks_locs.append(center)
        
        current_x += DATASET_GAP

    # ---------------- Styling ----------------

    plt.title(title, fontsize=36, pad=25)
    plt.ylabel("Accuracy (%)", fontsize=34, labelpad=20)
    plt.xlabel("")
    plt.ylim(0, 115)
    
    ax.set_xticks(xticks_locs)
    ax.set_xticklabels(ds_order, rotation=20, ha="right", fontsize=24)
    plt.yticks(fontsize=28)
    
    # Sort handles to match original NAME_MAPPING order
    sorted_handles = [h for m in hue_order for h in legend_handles if h.get_label() == m]
    
    cols = min(3, len(sorted_handles)) 
    plt.legend(
        handles=sorted_handles,
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.2),
        ncol=cols, 
        frameon=False,
        fontsize=22
    )
    
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    df = load_all_results()
    if not df.empty:
        print_raw_results_text(df)
        
        color_map = get_group_colors(df)
        
        print("\nGenerating Plots...")
        
        plot_aggregated_bar(df, "OOD", OOD_PLOT_TITLE, "1_OOD_Performance.png", color_map)
        plot_aggregated_bar(df, "IID", IID_PLOT_TITLE, "2_IID_Performance.png", color_map)
        
        plot_single_breakdown(df, IID_DATASETS, "Classification Task Breakdown (IID)", "3_IID_Breakdown.png", color_map)
        plot_single_breakdown(df, OOD_DATASETS, "Classification Task Breakdown (OOD)", "4_OOD_Breakdown.png", color_map)
        
        print("\n✅ Done!")