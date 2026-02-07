# %%
"""
Perturbation Stability Experiment

Two modes for measuring prediction stability:
  - noise: Add Gaussian noise to activation/steering vectors
  - temperature: Use temperature sampling (do_sample=True, temperature=1.0)

Both modes run N forward passes, compute agreement rate via majority vote,
and output the same JSON format for downstream plotting.

Usage:
    python stability_eval.py --mode noise              # Activation noise (default)
    python stability_eval.py --mode temperature         # Temperature sampling
    python stability_eval.py --mode noise --force-rerun # Force re-run

Outputs:
- Raw results JSON (plots are generated separately via plotting/plot_stability_eval.py)
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from dataclasses import dataclass, asdict
from typing import Any
import torch
from peft import LoraConfig
from tqdm import tqdm
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
    mode: str = "noise"  # "noise" or "temperature"
    n_samples: int = 10  # Number of forward passes
    noise_scale: float = 0.05  # (noise mode) Fraction of activation norm for noise std
    temperature: float = 1.0  # (temperature mode) Sampling temperature


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
    1. Run N forward passes (with noise or temperature sampling)
    2. Compute agreement rate via majority vote
    3. Store predictions and metrics
    """
    is_temperature_mode = stability_config.mode == "temperature"
    
    # Build generation kwargs for this mode
    if is_temperature_mode:
        gen_kwargs = {
            **generation_kwargs,
            "do_sample": True,
            "temperature": stability_config.temperature,
        }
        desc = f"Temperature eval (T={stability_config.temperature})"
    else:
        gen_kwargs = generation_kwargs
        desc = f"Noise eval (scale={stability_config.noise_scale})"
    
    results = []

    for i, datapoint in enumerate(tqdm(eval_data, desc=desc)):
        # Prepare single example
        dp = get_prompt_tokens_only(datapoint)
        batch = materialize_missing_steering_vectors([dp], tokenizer, model)
        batch_data = construct_batch(batch, tokenizer, device)

        base_vectors = batch_data.steering_vectors

        # Run N forward passes
        predictions = []
        for _ in range(stability_config.n_samples):
            if is_temperature_mode:
                # Temperature mode: use clean vectors, stochastic decoding
                pred = run_single_inference(
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    batch_data=batch_data,
                    steering_coefficient=steering_coefficient,
                    generation_kwargs=gen_kwargs,
                    device=device,
                    dtype=dtype,
                )
            else:
                # Noise mode: perturb vectors, deterministic decoding
                noisy_vectors = add_noise_to_vectors(base_vectors, stability_config.noise_scale)
                pred = run_single_inference(
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    batch_data=batch_data,
                    steering_coefficient=steering_coefficient,
                    generation_kwargs=gen_kwargs,
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


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stability evaluation experiment")
    parser.add_argument("--mode", choices=["noise", "temperature"], default="noise",
                        help="Stability mode: 'noise' perturbs activations, 'temperature' uses stochastic decoding")
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0.003],
                        help="Noise scale(s) for noise mode (default: 0.003)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                        help="Temperature(s) for temperature mode (default: 1.0)")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run even if JSON exists")
    args = parser.parse_args()

    # Build list of (mode, param_value) pairs to evaluate
    if args.mode == "noise":
        param_values = args.noise_scales
    else:
        param_values = args.temperatures

    print(f"{'=' * 60}")
    print(f"Stability Evaluation Experiment")
    print(f"Mode: {args.mode}")
    print(f"Param values: {param_values}")
    print(f"Model: {MODEL_NAME}")
    print(f"Verbalizer: {VERBALIZER_LORA}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"{'=' * 60}")

    model_name_str = MODEL_NAME.split("/")[-1]
    lora_name_str = VERBALIZER_LORA.split("/")[-1]

    # Determine which param values actually need evaluation
    configs_to_run: list[tuple[StabilityConfig, str]] = []  # (config, json_path)
    for param in param_values:
        if args.mode == "noise":
            cfg = StabilityConfig(mode="noise", n_samples=10, noise_scale=param)
            param_str = f"noise{param}"
        else:
            cfg = StabilityConfig(mode="temperature", n_samples=10, temperature=param)
            param_str = f"temp{param}"

        json_path = f"{OUTPUT_DIR}/stability_{model_name_str}_{lora_name_str}_{DATASET_NAME}_{param_str}.json"

        if os.path.exists(json_path) and not args.force_rerun:
            with open(json_path, "r") as f:
                results_json = json.load(f)
            s = results_json["summary"]
            print(f"\n[SKIP] {param_str}: already exists ({s['n_examples']} examples, "
                  f"acc={s['baseline_accuracy']:.3f}, agreement={s['mean_agreement_rate']:.3f})")
        else:
            configs_to_run.append((cfg, json_path))
            reason = "force-rerun" if os.path.exists(json_path) else "no existing results"
            print(f"\n[QUEUE] {param_str}: will run ({reason})")

    if not configs_to_run:
        print("\nAll param values already have results. Use --force-rerun to re-run.")
        print("Done! Use experiments/plotting/plot_stability_eval.py to generate plots.")
        exit(0)

    # Load model, tokenizer, and dataset once (shared across all runs)
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

    # Run stability evaluation for each param value
    for run_idx, (stability_config, json_path) in enumerate(configs_to_run):
        print(f"\n{'=' * 60}")
        print(f"Run {run_idx + 1}/{len(configs_to_run)}: {stability_config.mode} mode, "
              f"{'noise_scale' if stability_config.mode == 'noise' else 'temperature'}="
              f"{stability_config.noise_scale if stability_config.mode == 'noise' else stability_config.temperature}")
        print(f"Output: {json_path}")
        print(f"{'=' * 60}")

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

        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved results to {json_path}")

    print(f"\nAll done! Use experiments/plotting/plot_stability_eval.py to generate plots.")
