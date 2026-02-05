# %%
"""
Perturbation Stability Experiment

This script evaluates how stable model predictions are under small perturbations
to the activation/steering vectors. The hypothesis is that robustly encoded concepts
should be stable under noise, while uncertain predictions will be unstable.

Outputs:
- Accuracy vs. threshold plot (with coverage curve)
- Raw results JSON
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from dataclasses import dataclass, asdict
from typing import Any
import torch
from peft import LoraConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import parse_answer
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.dataset_utils import (
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
    TrainingDataPoint,
)
from nl_probes.base_experiment import sanitize_lora_name

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StabilityConfig:
    n_samples: int = 10  # Number of noisy forward passes
    noise_scale: float = 0.05  # Fraction of activation norm for noise std


# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"
VERBALIZER_LORA = "nluick/MLAO-Qwen3-8B-3L-3N"
LAYER_PERCENTS = [25, 50, 75]  # Multi-layer config for 3L model

# Evaluation settings
INJECTION_LAYER = 1
DTYPE = torch.bfloat16
BATCH_SIZE = 1  # Process one at a time for stability sampling
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}

# Dataset settings
DATASET_NAME = "language_identification"
NUM_TEST_EXAMPLES = 100

# Output settings
OUTPUT_DIR = "plots/stability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda")

# ============================================================================
# Stability Evaluation Functions
# ============================================================================


def add_noise_to_vectors(
    vectors: list[torch.Tensor],
    noise_scale: float,
) -> list[torch.Tensor]:
    """Add Gaussian noise to steering vectors, scaled by their norm."""
    noisy_vectors = []
    for vec in vectors:
        # Compute noise magnitude relative to vector norm
        vec_norm = vec.norm(dim=-1, keepdim=True).mean()
        noise_std = noise_scale * vec_norm
        noise = torch.randn_like(vec) * noise_std
        noisy_vectors.append(vec + noise)
    return noisy_vectors


@torch.no_grad()
def run_single_inference(
    model,
    tokenizer,
    submodule,
    batch_data,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    noisy_vectors: list[torch.Tensor] | None = None,
) -> str:
    """Run a single forward pass, optionally with noisy steering vectors."""
    vectors = noisy_vectors if noisy_vectors is not None else batch_data.steering_vectors
    positions = batch_data.positions

    hook_fn = get_hf_activation_steering_hook(
        vectors=vectors,
        positions=positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": batch_data.input_ids,
        "attention_mask": batch_data.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    generated_tokens = output_ids[:, batch_data.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return decoded_output[0]  # Single example


def evaluate_with_stability(
    model,
    tokenizer,
    submodule,
    eval_data: list[TrainingDataPoint],
    stability_config: StabilityConfig,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    """
    Run stability evaluation on all examples.
    
    For each example:
    1. Run N noisy forward passes
    2. Compute agreement rate
    3. Store predictions and metrics
    """
    results = []

    for i, datapoint in enumerate(tqdm(eval_data, desc="Stability evaluation")):
        # Prepare single example
        dp = get_prompt_tokens_only(datapoint)
        batch = materialize_missing_steering_vectors([dp], tokenizer, model)
        batch_data = construct_batch(batch, tokenizer, device)

        # Get base steering vectors
        base_vectors = batch_data.steering_vectors

        # Run N noisy forward passes
        predictions = []
        for _ in range(stability_config.n_samples):
            noisy_vectors = add_noise_to_vectors(base_vectors, stability_config.noise_scale)
            pred = run_single_inference(
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                batch_data=batch_data,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
                device=device,
                dtype=dtype,
                noisy_vectors=noisy_vectors,
            )
            predictions.append(parse_answer(pred))

        # Compute stability metrics
        yes_count = sum(1 for p in predictions if p == "yes")
        no_count = sum(1 for p in predictions if p == "no")
        other_count = stability_config.n_samples - yes_count - no_count

        # Majority vote
        if yes_count >= no_count and yes_count >= other_count:
            majority_vote = "yes"
            majority_count = yes_count
        elif no_count >= yes_count and no_count >= other_count:
            majority_vote = "no"
            majority_count = no_count
        else:
            majority_vote = "other"
            majority_count = other_count

        agreement_rate = majority_count / stability_config.n_samples

        # Ground truth
        ground_truth = parse_answer(datapoint.target_output)
        is_correct = majority_vote == ground_truth

        result = {
            "index": i,
            "ground_truth": ground_truth,
            "majority_vote": majority_vote,
            "is_correct": is_correct,
            "agreement_rate": agreement_rate,
            "yes_count": yes_count,
            "no_count": no_count,
            "other_count": other_count,
            "predictions": predictions,
        }
        results.append(result)

    return results


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


def plot_accuracy_coverage_vs_threshold(
    results: list[dict],
    output_path: str,
    title: str = "Accuracy & Coverage vs. Agreement Threshold",
):
    """Plot accuracy and coverage as a function of agreement threshold."""
    thresholds = np.arange(0.5, 1.01, 0.05)

    accuracies = []
    coverages = []
    n_samples_list = []

    for thresh in thresholds:
        acc, cov, n = compute_accuracy_coverage_at_threshold(results, thresh)
        accuracies.append(acc)
        coverages.append(cov)
        n_samples_list.append(n)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot accuracy
    color1 = "tab:blue"
    ax1.set_xlabel("Agreement Threshold", fontsize=12)
    ax1.set_ylabel("Accuracy", color=color1, fontsize=12)
    line1 = ax1.plot(thresholds, accuracies, "o-", color=color1, linewidth=2, markersize=8, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)

    # Plot coverage on secondary y-axis
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Coverage (fraction of data)", color=color2, fontsize=12)
    line2 = ax2.plot(thresholds, coverages, "s--", color=color2, linewidth=2, markersize=8, label="Coverage")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    # Add baseline accuracy (no filtering)
    baseline_acc = sum(r["is_correct"] for r in results) / len(results)
    ax1.axhline(y=baseline_acc, color="gray", linestyle=":", linewidth=2, label=f"Baseline Acc: {baseline_acc:.3f}")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    # Add sample counts as text annotations
    for i, (thresh, acc, n) in enumerate(zip(thresholds, accuracies, n_samples_list)):
        if i % 2 == 0:  # Annotate every other point to avoid clutter
            ax1.annotate(
                f"n={n}",
                (thresh, acc),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    ax1.set_title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"{'=' * 60}")
    print(f"Stability Evaluation Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Verbalizer: {VERBALIZER_LORA}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"{'=' * 60}")

    stability_config = StabilityConfig(n_samples=10, noise_scale=0.003)

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)
    submodule = get_hf_submodule(model, INJECTION_LAYER)

    # Add dummy adapter for PEFT compatibility
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Load verbalizer LoRA
    print(f"Loading verbalizer LoRA: {VERBALIZER_LORA}")
    sanitized_name = sanitize_lora_name(VERBALIZER_LORA)
    model.load_adapter(VERBALIZER_LORA, adapter_name=sanitized_name, is_trainable=False, low_cpu_mem_usage=True)
    model.set_adapter(sanitized_name)

    # Load dataset
    print(f"\nLoading dataset: {DATASET_NAME} (n={NUM_TEST_EXAMPLES})")
    classification_config = ClassificationDatasetConfig(
        classification_dataset_name=DATASET_NAME,
        max_end_offset=-3,
        min_end_offset=-3,
        max_window_size=1,
        min_window_size=1,
    )
    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=classification_config,
        num_train=0,
        num_test=NUM_TEST_EXAMPLES,
        splits=["test"],
        model_name=MODEL_NAME,
        layer_percents=LAYER_PERCENTS,
        save_acts=True,
        batch_size=16,  # For activation collection
    )
    dataset_loader = ClassificationDatasetLoader(
        dataset_config=dataset_config,
        model=model,
    )
    eval_data = dataset_loader.load_dataset("test")
    print(f"Loaded {len(eval_data)} test examples")

    # Run stability evaluation
    print(f"\nRunning stability evaluation (n_samples={stability_config.n_samples}, noise_scale={stability_config.noise_scale})")
    results = evaluate_with_stability(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        eval_data=eval_data,
        stability_config=stability_config,
        steering_coefficient=STEERING_COEFFICIENT,
        generation_kwargs=GENERATION_KWARGS,
        device=device,
        dtype=DTYPE,
    )

    # Compute summary statistics
    baseline_accuracy = sum(r["is_correct"] for r in results) / len(results)
    mean_agreement = np.mean([r["agreement_rate"] for r in results])
    print(f"\nBaseline accuracy (no filtering): {baseline_accuracy:.3f}")
    print(f"Mean agreement rate: {mean_agreement:.3f}")

    # Save results JSON
    model_name_str = MODEL_NAME.split("/")[-1]
    lora_name_str = VERBALIZER_LORA.split("/")[-1]
    output_base = f"{OUTPUT_DIR}/stability_{model_name_str}_{lora_name_str}_{DATASET_NAME}_noise{stability_config.noise_scale}"

    results_json = {
        "config": {
            "model_name": MODEL_NAME,
            "verbalizer_lora": VERBALIZER_LORA,
            "layer_percents": LAYER_PERCENTS,
            "dataset_name": DATASET_NAME,
            "n_test_examples": NUM_TEST_EXAMPLES,
            "stability_config": asdict(stability_config),
            "steering_coefficient": STEERING_COEFFICIENT,
        },
        "summary": {
            "baseline_accuracy": baseline_accuracy,
            "mean_agreement_rate": mean_agreement,
            "n_examples": len(results),
        },
        "results": results,
    }

    json_path = f"{output_base}.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results to {json_path}")

    # Plot accuracy vs threshold
    plot_path = f"{output_base}.png"
    plot_accuracy_coverage_vs_threshold(
        results=results,
        output_path=plot_path,
        title=f"Stability Analysis: {DATASET_NAME}\n{lora_name_str} (noise_scale={stability_config.noise_scale})",
    )

    print(f"\nDone!")
