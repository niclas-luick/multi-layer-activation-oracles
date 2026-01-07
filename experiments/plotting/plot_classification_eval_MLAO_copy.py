import json
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
EXPERIMENT_DIRS = {
    "1N": "experiments/classification_MLAO_1xlatentqa_1xpast_lens_1xcls",
    "3N": "experiments/classification_MLAO_1xlatentqa_1xpast_lens_3xcls",
    "6N": "experiments/classification_MLAO_1xlatentqa_1xpast_lens_6xcls",
    # If your base models are in a separate folder, add it here. 
    # Based on your script, they might be mixed in the folders above.
    "BASE": "experiments/classification_final" 
}

PLOT_DIR = "plots/MLAO/combine_data_runs_fixed_v2"
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

# --- NAME MAPPING (CRITICAL UPDATE) ---
# This matches the "investigator_lora_path" found in your training script configuration.
NAME_MAPPING = {
    # --- 4B Models ---
    # AO / Base
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B": "AO Qwen-4B",
    "checkpoints_latentqa_cls_past_lens_Qwen3-4B": "AO Qwen-4B",
    
    # 3L Variants
    "nluick/activation-oracle-multilayer-qwen3-4b-3L": "MLAO Qwen-4B-3L-1N",
    "activation-oracle-multilayer-qwen3-4b-3L": "MLAO Qwen-4B-3L-1N", # Fallback
    
    "nluick/activation-oracle-multilayer-qwen3-8b-25-50-75": "MLAO Qwen-4B-3L-3N", # Note: User mentioned HF name typo in script (8b->4b)
    "activation-oracle-multilayer-qwen3-8b-25-50-75": "MLAO Qwen-4B-3L-3N",

    # 6L Variants
    "nluick/activation-oracle-multilayer-qwen3-4b-6L": "MLAO Qwen-4B-6L-1N",
    "activation-oracle-multilayer-qwen3-4b-6L": "MLAO Qwen-4B-6L-1N",

    "nluick/activation-oracle-multilayer-qwen3-4b-6L-3xlayer-loop": "MLAO Qwen-4B-6L-3N",
    "activation-oracle-multilayer-qwen3-4b-6L-3xlayer-loop": "MLAO Qwen-4B-6L-3N",

    "nluick/activation-oracle-multilayer-qwen3-8b-15-30-45-60-75-90": "MLAO Qwen-4B-6L-6N",
    "activation-oracle-multilayer-qwen3-8b-15-30-45-60-75-90": "MLAO Qwen-4B-6L-6N",


    # --- 8B Models ---
    # AO / Base
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "AO Qwen-8B",
    "checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B": "AO Qwen-8B",

    # 3L Variants
    "nluick/activation-oracle-multilayer-qwen3-8b-3L": "MLAO Qwen-8B-3L-1N", # HF Name typo 14b->8b logic handled here?
    "activation-oracle-multilayer-qwen3-8b-3L": "MLAO Qwen-8B-3L-1N",

    "nluick/activation-oracle-multilayer-qwen3-14b-25-50-75": "MLAO Qwen-8B-3L-3N",
    "activation-oracle-multilayer-qwen3-14b-25-50-75": "MLAO Qwen-8B-3L-3N",
}

# ---------------------------------------------------------
# 2. DATA LOADING & PROCESSING
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

def parse_model_info(raw_path, filename):
    """
    Returns: (Display Name, Family, Variant)
    """
    matched_name = None
    
    # 1. Try exact match on raw_path (Handling potential trailing slashes or user typos)
    if raw_path in NAME_MAPPING:
        matched_name = NAME_MAPPING[raw_path]
    
    # 2. Try substring match if exact failed
    if not matched_name and raw_path:
        for k, v in NAME_MAPPING.items():
            if k in raw_path: 
                matched_name = v
                break
                
    # 3. Fallback: Check filename
    if not matched_name:
        for k, v in NAME_MAPPING.items():
            if k in filename:
                matched_name = v
                break

    if not matched_name: return None, None, None

    # Parse Family and Variant
    # Example: "MLAO Qwen-4B-3L-1N" -> Family: "MLAO Qwen-4B-3L", Variant: "1N"
    if matched_name.endswith("-1N"):
        return matched_name, matched_name[:-3], "1N"
    elif matched_name.endswith("-3N"):
        return matched_name, matched_name[:-3], "3N"
    elif matched_name.endswith("-6N"):
        return matched_name, matched_name[:-3], "6N"
    else:
        # Assumed to be "AO ..." or Reference
        return matched_name, matched_name, "Ref"

def load_all_results():
    deduped_data = {}
    
    for source_label, dir_path in EXPERIMENT_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"⚠️ Warning: Path not found: {dir_path}")
            continue
            
        print(f"🔍 Scanning {dir_path}...")
        files = glob.glob(os.path.join(dir_path, "**", "*.json"), recursive=True)
        
        for fpath in files:
            try:
                with open(fpath, "r") as f: res = json.load(f)
            except Exception: continue

            meta = res.get("meta", {})
            raw_path = meta.get("investigator_lora_path", "")
            filename = os.path.basename(fpath)
            
            # Skip explicit base_model runs if they are just raw pre-training checks
            if raw_path == "base_model" or raw_path is None: continue

            # IDENTIFY
            display_name, family, variant = parse_model_info(raw_path, filename)
            if display_name is None: continue

            # PROCESS RECORDS
            records = res.get("records", [])
            records_by_ds = {}
            for r in records:
                ds = r["dataset_id"]
                if ds not in records_by_ds: records_by_ds[ds] = []
                records_by_ds[ds].append(r)

            for ds_id, ds_recs in records_by_ds.items():
                if ds_id not in IID_DATASETS + OOD_DATASETS: continue
                
                acc, se, n = compute_metrics(ds_recs)
                split = "IID" if ds_id in IID_DATASETS else "OOD"
                
                # Store (Deduplicate: prefer higher N)
                key = (display_name, ds_id)
                if key not in deduped_data or n > deduped_data[key]["N"]:
                    deduped_data[key] = {
                        "Model": display_name,
                        "Family": family,
                        "Variant": variant,
                        "Dataset": ds_id,
                        "Split": split,
                        "Accuracy": acc,
                        "SE": se,
                        "N": n
                    }

    print(f"✅ Loaded {len(deduped_data)} unique pairs.")
    return pd.DataFrame(list(deduped_data.values()))

# ---------------------------------------------------------
# 3. PLOTTING
# ---------------------------------------------------------

def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def get_hierarchical_palette(df):
    """
    Assigns colors based on Family, shaded by Variant.
    """
    families = sorted(df["Family"].unique())
    # Use Spectral for high contrast between families
    base_palette = sns.color_palette("Spectral", n_colors=len(families))
    fam_map = dict(zip(families, base_palette))
    
    full_palette = {}
    for _, row in df.iterrows():
        mod = row["Model"]
        fam = row["Family"]
        var = row["Variant"]
        
        base_c = fam_map.get(fam, (0.5, 0.5, 0.5))
        
        if var == "1N":
            full_palette[mod] = adjust_lightness(base_c, 1.4) # Lighter
        elif var == "3N":
            full_palette[mod] = base_c # Base
        elif var == "6N":
            full_palette[mod] = adjust_lightness(base_c, 0.6) # Darker
        else:
            full_palette[mod] = base_c # Reference (AO)
            
    return full_palette

def plot_grouped_side_by_side(df, split_name, filename, palette):
    subset = df[df["Split"] == split_name]
    if subset.empty: return

    # Aggregate
    agg = subset.groupby(["Family", "Variant", "Model"])["Accuracy"].mean().reset_index()
    
    # Sort Families (AO first)
    families = sorted(agg["Family"].unique(), key=lambda x: ("AO" not in x, x))
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # Settings for grouping
    bar_width = 0.22
    variants_order = ["1N", "3N", "6N"]
    
    for i, fam in enumerate(families):
        fam_data = agg[agg["Family"] == fam]
        
        # Determine valid bars for this family
        valid_vars = []
        if "Ref" in fam_data["Variant"].values:
            valid_vars.append("Ref")
        else:
            for v in variants_order:
                if v in fam_data["Variant"].values: valid_vars.append(v)
        
        # Calculate X positions
        n_bars = len(valid_vars)
        total_w = n_bars * bar_width
        start_x = i - (total_w / 2) + (bar_width / 2)
        
        for j, var in enumerate(valid_vars):
            row = fam_data[fam_data["Variant"] == var].iloc[0]
            val = row["Accuracy"]
            mod = row["Model"]
            color = palette.get(mod, "grey")
            
            x_pos = start_x + (j * bar_width)
            
            # Plot Bar
            ax.bar(x_pos, val, bar_width, color=color, edgecolor="black")
            
            # Label (1N, 3N)
            lbl = var if var != "Ref" else "AO"
            txt_col = "white" if var == "6N" else "black"
            
            # Label inside top
            ax.text(x_pos, val - 3, lbl, ha='center', va='top', fontsize=10, 
                    fontweight='bold', color=txt_col)
            
            # Value above
            ax.text(x_pos, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color="black")

    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"{split_name} Performance", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

def plot_breakdown(df, palette):
    if df.empty: return
    
    # Sort for grouping
    rank = {"Ref":0, "1N":1, "3N":2, "6N":3}
    df = df.copy()
    df["Rank"] = df["Variant"].map(rank)
    df = df.sort_values(["Family", "Rank"])
    
    hue_order = df["Model"].unique()
    ds_order = sorted([d for d in df["Dataset"].unique() if d in IID_DATASETS]) + \
               sorted([d for d in df["Dataset"].unique() if d in OOD_DATASETS])

    plt.figure(figsize=(24, 10))
    sns.set_theme(style="whitegrid", context="talk")
    
    sns.barplot(
        data=df, x="Dataset", y="Accuracy", hue="Model",
        order=ds_order, hue_order=hue_order, palette=palette,
        edgecolor="black", alpha=0.9
    )
    
    plt.ylim(0, 115)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.title("Detailed Performance Breakdown", fontsize=20, weight="bold")
    plt.legend(bbox_to_anchor=(0.5, -0.35), loc="upper center", ncol=5, frameon=False, fontsize=11)
    
    # Divider
    n_iid = len([d for d in ds_order if d in IID_DATASETS])
    if n_iid > 0:
        plt.axvline(x=n_iid - 0.5, color="black", linestyle="--", alpha=0.5)
        plt.text(n_iid/2, 112, "IID", ha="center", weight="bold")
        plt.text(n_iid + (len(ds_order)-n_iid)/2, 112, "OOD", ha="center", weight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "3_Category_Breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    df = load_all_results()
    
    if not df.empty:
        # Debug print
        print("\nModels Found:")
        print(df[["Model", "Family", "Variant"]].drop_duplicates().to_string(index=False))
        
        palette = get_hierarchical_palette(df)
        
        print("\nGenerating Plots...")
        plot_grouped_side_by_side(df, "OOD", "1_OOD_Grouped.png", palette)
        plot_grouped_side_by_side(df, "IID", "2_IID_Grouped.png", palette)
        plot_breakdown(df, palette)
        print("✅ Done!")
    else:
        print("❌ No data found.")