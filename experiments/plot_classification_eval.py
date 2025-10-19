# %%

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BitsAndBytesConfig

# External project imports (assumed available in your env)
from nl_probes.utils.eval import parse_answer, run_evaluation

# -----------------------------
# Configuration - tune here
# -----------------------------

# When False: skip expensive eval and try to load from RESULTS_FILENAME
RUN_FRESH_EVAL = True

# Model and eval config
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
HOOK_LAYER = 1
DTYPE = torch.bfloat16
BATCH_SIZE = 32
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}

model_name_str = MODEL_NAME.split("/")[-1].replace(".", "_").replace(" ", "_")
RESULTS_FILENAME = f"0919_{model_name_str}_classification_results.json"
EXPERIMENTS_DIR = ""
DATA_DIR = "classification_eval"
RESULTS_FILENAME = f"{DATA_DIR}/{RESULTS_FILENAME}"


device = torch.device("cuda")
dtype = torch.bfloat16
print(f"Using device={device}, dtype={dtype}")

model_kwargs = {}

if MODEL_NAME == "meta-llama/Llama-3.3-70B-Instruct":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
    )
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=dtype,
    # )
    model_kwargs = {"quantization_config": bnb_config}

# Dataset selection
MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "relations": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "sst2": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "md_gender": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "snli": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ag_news": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "tense": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "language_identification": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
}

# Groupings for plotting
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]
OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
]

# Layer percent settings used by loaders
LAYER_PERCENTS = [25, 50, 75]

KEY_FOR_NONE = "original"


@dataclass(frozen=True)
class Method:
    label: str
    lora_path: str


LORA_DIR = ""

METHODS: list[Method] = [
    # Method(label="Original", lora_path=KEY_FOR_NONE),
    # Method(
    #     label="Past / Future Lens -> 1 Classification Epoch",
    #     lora_path=f"{LORA_DIR}adamkarvonen/checkpoints_act_pretrain_cls_latentqa_fixed_posttrain_Llama-3_3-70B-Instruct",
    # ),
    # Method(
    #     label="1 Classification Epoch",
    #     lora_path=f"{LORA_DIR}adamkarvonen/checkpoints_classification_only_Llama-3_3-70B-Instruct",
    # ),
    Method(
        label="Past / Future Lens -> 1 Classification Epoch Step 5000",
        lora_path=f"{LORA_DIR}model_lora/step_5000",
    ),
    # Method(
    #     label="SAE Pretrain Mix -> 1 Classification Epoch",
    #     lora_path=f"{LORA_DIR}checkpoints_sae_pretrain_1_token_-3_-5_classification_posttrain/final",
    # ),
    # Method(
    #     label="SAE and Past Lens Pretrain Mix -> 1 Classification Epoch",
    #     lora_path=f"{LORA_DIR}checkpoints_all_pretrain_1_token_-3_-5_sae_explanation_posttrain/final",
    # ),
    # Method(
    #     label="2 Classification Epochs",
    #     lora_path=f"{LORA_DIR}checkpoints_classification_only_1_token_-3_-5_2_epochs/final",
    # ),
]

LABEL_BY_LORA_PATH: dict[str, str] = {m.lora_path: m.label for m in METHODS}

# Optional external baselines keyed by dataset id; set to None to skip plotting.
LINEAR_PROBE_BASELINE: dict[str, Any] | None = {
    "label": "Linear Probe Baseline",
    "values": {
        "ag_news": 0.8338,
        "geometry_of_truth": 0.8760,
        "language_identification": 0.8862,
        "md_gender": 0.9382,
        "singular_plural": 0.9618,
        "sst2": 0.8227,
        "tense": 0.9484,
    },
}
# -----------------------------
# Lightweight helpers
# -----------------------------


def canonical_dataset_id(name: str) -> str:
    """Strip 'classification_' prefix if present so keys match your IID/OOD lists."""
    if name.startswith("classification_"):
        return name[len("classification_") :]
    return name


def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """Return p, se, lower, upper for a binomial proportion with normal approx CI."""
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def score_predictions(cleaned_responses: list[str], target_responses: list[str]) -> dict[str, Any]:
    """Compute correctness stats given cleaned model outputs and cleaned targets."""
    assert len(cleaned_responses) == len(target_responses)
    n = len(cleaned_responses)
    is_correct_list = [cr == tr for cr, tr in zip(cleaned_responses, target_responses)]
    correct = sum(is_correct_list)
    p, se, lower, upper = proportion_confidence(correct, n)
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }


# Orchestrate load-or-run
results_by_ds: dict[str, dict[str, Any]] = {}
meta_loaded: dict[str, Any] | None = None

if Path(RESULTS_FILENAME).exists():
    print(f"Loading existing results from {RESULTS_FILENAME}")
    with open(RESULTS_FILENAME, "r") as f:
        results_by_ds = json.load(f)
else:
    raise ValueError(f"plotting only, {RESULTS_FILENAME}")

# %%
# Plotting utilities - robust to missing entries


def _score_and_err(result: dict[str, Any] | None) -> tuple[float, float, float]:
    """Return (p, lower_err, upper_err). If missing, return (nan,0,0)."""
    if not result:
        return float("nan"), 0.0, 0.0
    p = result.get("p")
    if p is None:
        n = result.get("n", 0) or 0
        c = result.get("correct", 0) or 0
        p = (c / n) if n else float("nan")
    lower = result.get("ci_lower")
    upper = result.get("ci_upper")
    if lower is not None and upper is not None and isinstance(p, (int, float)):
        return float(p), max(0.0, float(p) - float(lower)), max(0.0, float(upper) - float(p))
    se = result.get("se")
    if se is not None and isinstance(p, (int, float)):
        return float(p), float(se), float(se)
    return float(p) if isinstance(p, (int, float)) else float("nan"), 0.0, 0.0


def plot_group(
    group_name: str,
    datasets: list[str],
    results: dict[str, dict[str, Any]],
    label_by_key: dict[str, str],
    *,
    baseline: float | None = 0.5,
    extra_series: list[dict[str, Any]] | None = None,
) -> None:
    present = [ds for ds in datasets if ds in results]
    missing = [ds for ds in datasets if ds not in results]
    if missing:
        print(f"[plot {group_name}] Skipping missing datasets: {missing}")

    if not present:
        print(f"[plot {group_name}] Nothing to plot.")
        return

    x = np.arange(len(present))
    plt.figure(figsize=(10, 6))

    # For consistent legend order, iterate methods in METHODS order but include only those present
    for m in METHODS:
        y = []
        yerr_low = []
        yerr_high = []
        any_present = False
        for ds in present:
            r = results[ds].get(m.lora_path)
            p, el, eh = _score_and_err(r)
            if not np.isnan(p):
                any_present = True
            y.append(p)
            yerr_low.append(el)
            yerr_high.append(eh)
        if any_present:
            # Build legend label with per-plot average over non-NaN values
            base_label = label_by_key.get(m.lora_path, m.lora_path)
            if len(base_label) > 22:
                base_label = base_label.replace(" -> ", "->\n ")
            valid = [v for v in y if not np.isnan(v)]
            label = f"{base_label} (avg={np.mean(valid):.3f})" if valid else base_label

            plt.errorbar(
                x,
                y,
                yerr=[yerr_low, yerr_high],
                label=label,
                marker="o",
                linestyle="--",
                linewidth=2,
                markersize=6,
                alpha=0.9,
                capsize=4,
            )

    if extra_series:
        for series in extra_series:
            values = series.get("values") or series.get("data") or {}
            if not isinstance(values, dict):
                continue
            y = []
            any_present = False
            for ds in present:
                val = values.get(ds)
                if val is None:
                    y.append(np.nan)
                else:
                    any_present = True
                    y.append(float(val))
            if not any_present:
                continue
            label = series.get("label", "Extra")
            valid = [v for v in y if not np.isnan(v)]
            display_label = f"{label} (avg={np.mean(valid):.3f})" if valid else label
            plt.plot(
                x,
                y,
                marker="s",
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                alpha=0.8,
                label=display_label,
            )

    if baseline is not None:
        plt.axhline(y=baseline, linestyle=":", linewidth=1.5, alpha=0.7, label=f"Baseline {baseline:.2f}")

    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title(group_name)
    plt.xticks(x, present, rotation=45, ha="right")
    # plt.legend(loc="best", fontsize=10)
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc="center", ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim([0.45, 1.0])
    plt.tight_layout()
    plt.show()


# Plot IID and OOD

plot_group(
    "IID (In-Distribution) Dataset Performance, Single Token",
    IID_DATASETS,
    results_by_ds,
    LABEL_BY_LORA_PATH,
    baseline=0.5,
    extra_series=[LINEAR_PROBE_BASELINE] if LINEAR_PROBE_BASELINE else None,
)
plot_group(
    "OOD (Out-of-Distribution) Dataset Performance, Single Token",
    OOD_DATASETS,
    results_by_ds,
    LABEL_BY_LORA_PATH,
    baseline=0.5,
    extra_series=[LINEAR_PROBE_BASELINE] if LINEAR_PROBE_BASELINE else None,
)
# %%
# Inspect a single dataset's per-example correctness, if you have raw predictions in-memory.
# This cell is optional - it shows how you could re-run scoring on a dataset interactively.


def analyze_results_debug(
    eval_data: list[Any],
    raw_api_responses: list[str],
) -> dict[str, Any]:
    """
    Convenience to debug a single method on a single dataset.
    Prints first few mismatches and returns stats.
    """
    cleaned = [parse_answer(x) for x in raw_api_responses]
    targets = [parse_answer(dp.target_output) for dp in eval_data]

    stats = score_predictions(cleaned, targets)

    # Print a handful of mismatches for quick inspection
    mismatches = [(i, c, t) for i, (c, t, ok) in enumerate(zip(cleaned, targets, stats["is_correct_list"])) if not ok][
        :10
    ]
    if mismatches:
        print("\nFirst few mismatches (index, response, target):")
        for i, c, t in mismatches:
            print(f"{i:4d}: {c!r} vs {t!r}")

    print(
        f"\ncorrect={stats['correct']}, n={stats['n']}, p={stats['p']:.4f} "
        f"95%CI=[{stats['ci_lower']:.4f},{stats['ci_upper']:.4f}]"
    )
    return stats


# Example usage (uncomment to run live on one dataset and one method):
# ds = "sst2"
# method = METHODS[0]  # Original
# fresh_raw = run_evaluation(
#     eval_data=all_eval_data[ds],
#     model=model,
#     tokenizer=tokenizer,
#     submodule=submodule,
#     device=device,
#     dtype=dtype,
#     global_step=-1,
#     lora_path=method.lora_path,
#     eval_batch_size=BATCH_SIZE,
#     steering_coefficient=STEERING_COEFFICIENT,
#     generation_kwargs=GENERATION_KWARGS,
# )
# analyze_results_debug(all_eval_data[ds], [r.api_response for r in fresh_raw])

# %%
