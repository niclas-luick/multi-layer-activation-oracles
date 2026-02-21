"""
Generate confidence labels for classification training data.

For each classification training datapoint, runs the oracle N times with
perturbation (temperature sampling or activation noise) and computes:
  confidence = fraction of oracle responses matching ground truth label.

Saves a JSON sidecar file alongside each .pt file:
  <pt_stem>_confidence.json

Usage:
    python generate_confidence_labels.py --mode temperature --temperature 1.0
    python generate_confidence_labels.py --mode noise --noise-scale 0.05
    python generate_confidence_labels.py --force-rerun
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.eval import parse_answer
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
    construct_batch,
)
from nl_probes.utils.confidence_utils import get_confidence_json_path
from nl_probes.base_experiment import sanitize_lora_name

# Reuse perturbation functions from stability_eval
from experiments.stability_eval import (
    StabilityConfig,
    add_noise_to_vectors,
)
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook


# ============================================================================
# Configuration (same defaults as stability_eval.py)
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
VERBALIZER_LORA = "nluick/MLAO-Qwen3-8B-3L-3N"
INJECTION_LAYER = 1
DTYPE = torch.bfloat16
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}
DATASET_FOLDER = "sft_training_data"


# ============================================================================
# Core logic
# ============================================================================


def find_classification_train_pt_files(folder: str) -> list[Path]:
    """Find all classification training .pt files in the dataset folder."""
    return sorted(Path(folder).glob("classification_*_train_*.pt"))


@torch.no_grad()
def run_batch_inference(
    model,
    tokenizer,
    submodule,
    batch_data,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    noisy_vectors: list[torch.Tensor] | None = None,
) -> list[str]:
    """Run a single forward pass on a batch, returning one prediction per example."""
    vectors = noisy_vectors if noisy_vectors is not None else batch_data.steering_vectors

    hook_fn = get_hf_activation_steering_hook(
        vectors=vectors,
        positions=batch_data.positions,
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

    generated_tokens = output_ids[:, batch_data.input_ids.shape[1]:]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def compute_confidence_for_dataset(
    datapoints: list[TrainingDataPoint],
    model,
    tokenizer,
    submodule,
    stability_config: StabilityConfig,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 64,
) -> list[dict]:
    """
    For each datapoint, run oracle N times and compute confidence =
    fraction of responses matching ground truth.

    Processes datapoints in batches for efficiency. Each batch runs
    n_samples forward passes (one per sample round).
    """
    mode = stability_config.mode

    if mode == "temperature":
        gen_kwargs = {
            **generation_kwargs,
            "do_sample": True,
            "temperature": stability_config.temperature,
        }
        desc = f"Confidence (temp={stability_config.temperature}, batch={batch_size})"
    else:
        gen_kwargs = generation_kwargs
        desc = f"Confidence (noise={stability_config.noise_scale}, batch={batch_size})"

    # Separate scorable (yes/no) from IDK datapoints
    scorable_indices = []
    idk_indices = set()
    for i, dp in enumerate(datapoints):
        gt = parse_answer(dp.target_output)
        if gt in ("yes", "no"):
            scorable_indices.append(i)
        else:
            idk_indices.add(i)

    # Storage for per-datapoint predictions across sample rounds
    all_predictions: dict[int, list[str]] = {i: [] for i in scorable_indices}

    # Process scorable datapoints in batches
    n_batches = (len(scorable_indices) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(scorable_indices), batch_size),
                            total=n_batches, desc=desc):
        batch_indices = scorable_indices[batch_start:batch_start + batch_size]
        batch_dps = [get_prompt_tokens_only(datapoints[i]) for i in batch_indices]
        batch_dps = materialize_missing_steering_vectors(batch_dps, tokenizer, model)
        batch_data = construct_batch(batch_dps, tokenizer, device)

        base_vectors = batch_data.steering_vectors

        # Run n_samples forward passes on this batch
        for _ in range(stability_config.n_samples):
            if mode == "noise":
                noisy_vectors = add_noise_to_vectors(base_vectors, stability_config.noise_scale)
            else:
                noisy_vectors = None

            preds = run_batch_inference(
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

            for idx, pred in zip(batch_indices, preds):
                all_predictions[idx].append(parse_answer(pred))

    # Build results list in original order
    results = []
    for i, dp in enumerate(datapoints):
        ground_truth = parse_answer(dp.target_output)

        if i in idk_indices:
            results.append({
                "index": i,
                "ground_truth": ground_truth,
                "target_output": dp.target_output,
                "confidence": None,
                "match_count": None,
                "n_samples": stability_config.n_samples,
                "yes_count": None,
                "no_count": None,
                "other_count": None,
                "predictions": None,
                "skipped": True,
            })
        else:
            predictions = all_predictions[i]
            match_count = sum(1 for p in predictions if p == ground_truth)
            confidence = match_count / stability_config.n_samples
            yes_count = sum(1 for p in predictions if p == "yes")
            no_count = sum(1 for p in predictions if p == "no")
            other_count = stability_config.n_samples - yes_count - no_count

            results.append({
                "index": i,
                "ground_truth": ground_truth,
                "target_output": dp.target_output,
                "confidence": confidence,
                "match_count": match_count,
                "n_samples": stability_config.n_samples,
                "yes_count": yes_count,
                "no_count": no_count,
                "other_count": other_count,
                "predictions": predictions,
                "skipped": False,
            })

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate confidence labels for classification training data"
    )
    parser.add_argument(
        "--mode", choices=["noise", "temperature"], default="temperature",
        help="Perturbation mode (default: temperature)",
    )
    parser.add_argument("--n-samples", type=int, default=10, help="Oracle passes per example")
    parser.add_argument("--noise-scale", type=float, default=0.05, help="Noise scale (noise mode)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (temperature mode)")
    parser.add_argument("--dataset-folder", type=str, default=DATASET_FOLDER)
    parser.add_argument("--force-rerun", action="store_true", help="Re-run even if JSON exists")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--max-datapoints", type=int, default=None, help="Limit datapoints per file (for quick testing)")
    args = parser.parse_args()

    # Build stability config
    if args.mode == "noise":
        stability_config = StabilityConfig(
            mode="noise", n_samples=args.n_samples, noise_scale=args.noise_scale,
        )
    else:
        stability_config = StabilityConfig(
            mode="temperature", n_samples=args.n_samples, temperature=args.temperature,
        )

    # Build JSON suffix from config: e.g. "confidence_temp1.0_n10" or "confidence_noise0.05_n10"
    model_short = MODEL_NAME.split("/")[-1]
    if args.mode == "noise":
        json_suffix = f"confidence_{model_short}_noise{args.noise_scale}_n{args.n_samples}"
    else:
        json_suffix = f"confidence_{model_short}_temp{args.temperature}_n{args.n_samples}"

    # Find classification training .pt files
    pt_files = find_classification_train_pt_files(args.dataset_folder)
    print(f"{'=' * 60}")
    print(f"Confidence Label Generation")
    print(f"Mode: {args.mode}")
    print(f"N samples: {args.n_samples}")
    if args.mode == "noise":
        print(f"Noise scale: {args.noise_scale}")
    else:
        print(f"Temperature: {args.temperature}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset folder: {args.dataset_folder}")
    print(f"JSON suffix: {json_suffix}")
    print(f"Found {len(pt_files)} classification training .pt files")
    print(f"{'=' * 60}")

    # Filter to files that need processing
    files_to_process: list[tuple[Path, Path]] = []  # (pt_path, json_path)
    for pt_path in pt_files:
        json_path = get_confidence_json_path(pt_path, suffix=json_suffix)
        if json_path.exists() and not args.force_rerun:
            print(f"  [SKIP] {pt_path.name}")
        else:
            files_to_process.append((pt_path, json_path))
            reason = "force-rerun" if json_path.exists() else "no existing JSON"
            print(f"  [QUEUE] {pt_path.name} ({reason})")

    if not files_to_process:
        print("\nAll files already have confidence JSONs. Use --force-rerun to re-run.")
        exit(0)

    # Load model once (shared across all .pt files)
    device = torch.device("cuda")
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)
    submodule = get_hf_submodule(model, INJECTION_LAYER)

    # Wrap as PeftModel for compatibility with materialize_missing_steering_vectors
    model = get_peft_model(model, LoraConfig())

    # Load verbalizer LoRA
    print(f"Loading verbalizer LoRA: {VERBALIZER_LORA}")
    sanitized_name = sanitize_lora_name(VERBALIZER_LORA)
    model.load_adapter(
        VERBALIZER_LORA, adapter_name=sanitized_name,
        is_trainable=False, low_cpu_mem_usage=True,
    )
    model.set_adapter(sanitized_name)

    # Process each .pt file
    for pt_path, json_path in files_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing: {pt_path.name}")
        print(f"Output: {json_path.name}")
        print(f"{'=' * 60}")

        saved_object = torch.load(pt_path, weights_only=False)
        data_dicts = saved_object["data"]
        datapoints = [TrainingDataPoint(**d) for d in data_dicts]
        if args.max_datapoints is not None:
            datapoints = datapoints[:args.max_datapoints]
        print(f"Loaded {len(datapoints)} datapoints")

        results = compute_confidence_for_dataset(
            datapoints=datapoints,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            stability_config=stability_config,
            steering_coefficient=STEERING_COEFFICIENT,
            generation_kwargs=GENERATION_KWARGS,
            device=device,
            dtype=DTYPE,
            batch_size=args.batch_size,
        )

        # Summary stats (exclude skipped IDK datapoints)
        scored = [r for r in results if not r.get("skipped", False)]
        skipped = len(results) - len(scored)
        if scored:
            confidences = [r["confidence"] for r in scored]
            mean_conf = sum(confidences) / len(confidences)
            correct_count = sum(1 for c in confidences if c > 0.5)
            print(f"Scored: {len(scored)} datapoints  |  Skipped (IDK): {skipped}")
            print(f"Mean confidence: {mean_conf:.3f}")
            print(f"Majority-correct: {correct_count}/{len(scored)} ({correct_count/len(scored)*100:.1f}%)")
        else:
            print(f"All {skipped} datapoints skipped (IDK)")

        # Save JSON
        output = {
            "pt_filename": pt_path.name,
            "stability_config": {
                "mode": stability_config.mode,
                "n_samples": stability_config.n_samples,
                "noise_scale": stability_config.noise_scale,
                "temperature": stability_config.temperature,
            },
            "model_name": MODEL_NAME,
            "verbalizer_lora": VERBALIZER_LORA,
            "steering_coefficient": STEERING_COEFFICIENT,
            "results": results,
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved confidence JSON to {json_path}")

    print(f"\nDone! Confidence JSONs saved alongside .pt files in {args.dataset_folder}/")
