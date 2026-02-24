# %%
"""
Perturbation Stability Experiment

Four modes for measuring prediction stability:
  - noise: Add Gaussian noise to activation/steering vectors (N passes)
  - temperature: Use temperature sampling (N passes)
  - threshold: Single deterministic pass, use softmax confidence as stability score
  - prompt: Re-ask with paraphrased prompts (N passes with different wording)

All modes output the same JSON format (agreement_rate field) for downstream plotting.

Usage:
    python stability_eval.py --mode noise              # Activation noise (default)
    python stability_eval.py --mode temperature         # Temperature sampling
    python stability_eval.py --mode threshold           # Logit confidence
    python stability_eval.py --mode prompt              # Prompt paraphrasing (both question + prefix)
    python stability_eval.py --mode prompt --no-vary-prefix   # Question paraphrasing only
    python stability_eval.py --mode prompt --no-vary-question # Prefix paraphrasing only
    python stability_eval.py --mode noise --force-rerun # Force re-run

Outputs:
- Raw results JSON (plots are generated separately via plotting/plot_stability_eval.py)
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import random
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
    mode: str = "noise"  # "noise", "temperature", "threshold", or "prompt"
    n_samples: int = 10  # Number of forward passes (ignored for threshold mode)
    noise_scale: float = 0.05  # (noise mode) Fraction of activation norm for noise std
    temperature: float = 1.0  # (temperature mode) Sampling temperature
    vary_question: bool = True  # (prompt mode) Paraphrase the question
    vary_prefix: bool = True  # (prompt mode) Paraphrase the instruction prefix


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
OUTPUT_DIR = "plots/stability/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda")

# Prompt paraphrase constants
ORIGINAL_PREFIX = "Answer with 'Yes' or 'No' only. "

INSTRUCTION_PARAPHRASES = [
    "Answer with 'Yes' or 'No' only. ",
    "Respond with just 'Yes' or 'No'. ",
    "Reply only with 'Yes' or 'No'. ",
    "Your answer should be 'Yes' or 'No' only. ",
    "Give a 'Yes' or 'No' answer. ",
    "Please respond with either 'Yes' or 'No'. ",
    "Simply say 'Yes' or 'No'. ",
    "Provide your answer as 'Yes' or 'No'. ",
    "State 'Yes' or 'No'. ",
    "Only answer 'Yes' or 'No'. ",
]

PARAPHRASES_JSON_PATH = "datasets/classification_datasets/paraphrases/question.json"


def load_question_paraphrases(dataset_name: str) -> list[str]:
    """Load question paraphrase templates from question.json for a given dataset."""
    with open(PARAPHRASES_JSON_PATH) as f:
        all_paraphrases = json.load(f)
    templates = all_paraphrases[dataset_name]
    if isinstance(templates, dict):
        # Flatten label-keyed dicts (e.g., sst2 has {"positive": [...], "negative": [...]})
        flat = []
        for v in templates.values():
            flat.extend(v)
        return flat
    return templates


def extract_fill_value(question_text: str, templates: list[str]) -> str | None:
    """Extract the fill value from a question by matching against known templates.

    E.g., question_text="Is this text written in English?" with
    template="Is this text written in {}?" → returns "English".
    """
    for template in templates:
        if "{}" not in template:
            continue
        # Convert "Is this text written in {}?" → regex "Is this text written in (.+?)\\?"
        pattern = re.escape(template).replace(r"\{\}", "(.+?)")
        pattern = "^" + pattern + "$"
        match = re.match(pattern, question_text)
        if match:
            return match.group(1)
    return None


def swap_classification_prompt(
    datapoint: "TrainingDataPoint",
    tokenizer,
    new_prefix: str,
    new_question: str,
) -> "TrainingDataPoint":
    """Replace the instruction prefix + question in a datapoint's input_ids.

    Decodes input_ids → replaces the classification prompt → re-encodes.
    Steering vectors and positions are preserved (they precede the swapped region).
    """
    decoded = tokenizer.decode(datapoint.input_ids, skip_special_tokens=False)

    # Find the original classification prompt: "Answer with 'Yes' or 'No' only. # <question>"
    prefix_idx = decoded.index(ORIGINAL_PREFIX)
    end_marker = "<|im_end|>"
    end_idx = decoded.index(end_marker, prefix_idx)

    new_cls_prompt = f"{new_prefix}# {new_question}"
    new_decoded = decoded[:prefix_idx] + new_cls_prompt + decoded[end_idx:]

    new_ids = tokenizer.encode(new_decoded, add_special_tokens=False)

    new_dp = datapoint.model_copy()
    new_dp.input_ids = new_ids
    return new_dp


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


@torch.no_grad()
def run_inference_with_confidence(
    model,
    tokenizer,
    submodule,
    batch_data,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, float]:
    """Single deterministic forward pass returning (decoded_answer, confidence).
    
    Confidence is the softmax probability of the first generated token.
    """
    vectors = batch_data.steering_vectors
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

    gen_kwargs = {
        **generation_kwargs,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    with add_hook(submodule, hook_fn):
        outputs = model.generate(**tokenized_input, **gen_kwargs)

    # scores[0] = logits at first generated token, shape (batch, vocab_size)
    first_token_logits = outputs.scores[0][0]  # (vocab_size,)
    probs = torch.softmax(first_token_logits, dim=-1)

    # Confidence = probability assigned to the token the model actually generated
    generated_token_id = outputs.sequences[0, batch_data.input_ids.shape[1]]
    confidence = probs[generated_token_id].item()

    generated_tokens = outputs.sequences[:, batch_data.input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return decoded, confidence


def _compute_majority_vote(predictions: list[str], n_samples: int) -> dict:
    """Compute majority vote statistics from a list of parsed predictions."""
    yes_count = sum(1 for p in predictions if p == "yes")
    no_count = sum(1 for p in predictions if p == "no")
    other_count = n_samples - yes_count - no_count

    if yes_count >= no_count and yes_count >= other_count:
        majority_vote = "yes"
        majority_count = yes_count
    elif no_count >= yes_count and no_count >= other_count:
        majority_vote = "no"
        majority_count = no_count
    else:
        majority_vote = "other"
        majority_count = other_count

    return {
        "majority_vote": majority_vote,
        "agreement_rate": majority_count / n_samples,
        "yes_count": yes_count,
        "no_count": no_count,
        "other_count": other_count,
    }


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
    question_paraphrases: list[str] | None = None,
) -> list[dict]:
    """
    Run stability evaluation on all examples.

    Modes:
    - noise/temperature: Run N forward passes, compute agreement via majority vote
    - threshold: Single deterministic pass, use softmax confidence as agreement_rate
    - prompt: Re-ask with paraphrased prompts, compute agreement via majority vote
    """
    mode = stability_config.mode

    # Build generation kwargs for this mode
    if mode == "temperature":
        gen_kwargs = {
            **generation_kwargs,
            "do_sample": True,
            "temperature": stability_config.temperature,
        }
        desc = f"Temperature eval (T={stability_config.temperature})"
    elif mode == "threshold":
        gen_kwargs = generation_kwargs  # deterministic
        desc = "Threshold eval (logit confidence)"
    elif mode == "prompt":
        gen_kwargs = generation_kwargs  # deterministic
        vary_q = stability_config.vary_question
        vary_p = stability_config.vary_prefix
        desc = f"Prompt eval (vary_question={vary_q}, vary_prefix={vary_p})"
    else:
        gen_kwargs = generation_kwargs
        desc = f"Noise eval (scale={stability_config.noise_scale})"

    results = []

    for i, datapoint in enumerate(tqdm(eval_data, desc=desc)):
        # Prepare single example
        dp = get_prompt_tokens_only(datapoint)
        batch = materialize_missing_steering_vectors([dp], tokenizer, model)
        dp_with_vectors = batch[0]

        ground_truth = parse_answer(datapoint.target_output)

        if mode == "threshold":
            batch_data = construct_batch(batch, tokenizer, device)
            # Single forward pass with confidence extraction
            pred_text, confidence = run_inference_with_confidence(
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                batch_data=batch_data,
                steering_coefficient=steering_coefficient,
                generation_kwargs=gen_kwargs,
                device=device,
                dtype=dtype,
            )
            predicted = parse_answer(pred_text)

            result = {
                "index": i,
                "ground_truth": ground_truth,
                "majority_vote": predicted,
                "is_correct": predicted == ground_truth,
                "agreement_rate": confidence,  # Softmax confidence as stability score
                "yes_count": 1 if predicted == "yes" else 0,
                "no_count": 1 if predicted == "no" else 0,
                "other_count": 1 if predicted not in ("yes", "no") else 0,
                "predictions": [predicted],
                "confidence": confidence,
            }
        elif mode == "prompt":
            # Prompt paraphrase mode: run with N different prompt wordings
            vary_q = stability_config.vary_question
            vary_p = stability_config.vary_prefix
            n_samples = stability_config.n_samples

            # Extract the original question text and fill value
            decoded = tokenizer.decode(dp_with_vectors.input_ids, skip_special_tokens=False)
            prefix_idx = decoded.index(ORIGINAL_PREFIX)
            end_idx = decoded.index("<|im_end|>", prefix_idx)
            original_cls = decoded[prefix_idx + len(ORIGINAL_PREFIX):end_idx]
            # Strip "# " prefix from question
            original_question = original_cls[2:] if original_cls.startswith("# ") else original_cls

            fill_value = None
            if vary_q and question_paraphrases is not None:
                fill_value = extract_fill_value(original_question, question_paraphrases)

            # Build all (prefix, question) pairs and sample n_samples unique ones
            prefixes = INSTRUCTION_PARAPHRASES if vary_p else [ORIGINAL_PREFIX]
            if vary_q and question_paraphrases is not None:
                questions = []
                for t in question_paraphrases:
                    if "{}" in t and fill_value is not None:
                        questions.append(t.format(fill_value))
                    else:
                        questions.append(t)
            else:
                questions = [original_question]

            all_pairs = [(p, q) for p in prefixes for q in questions]
            if n_samples <= len(all_pairs):
                sampled_pairs = random.sample(all_pairs, n_samples)
            else:
                sampled_pairs = all_pairs[:]
                while len(sampled_pairs) < n_samples:
                    sampled_pairs.append(random.choice(all_pairs))

            predictions = []
            prompt_variants = []
            for new_prefix, new_question in sampled_pairs:

                if new_prefix != ORIGINAL_PREFIX or new_question != original_question:
                    swapped_dp = swap_classification_prompt(
                        dp_with_vectors, tokenizer, new_prefix, new_question,
                    )
                    variant_batch = construct_batch([swapped_dp], tokenizer, device)
                else:
                    variant_batch = construct_batch([dp_with_vectors], tokenizer, device)

                pred = run_single_inference(
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    batch_data=variant_batch,
                    steering_coefficient=steering_coefficient,
                    generation_kwargs=gen_kwargs,
                    device=device,
                    dtype=dtype,
                )
                parsed = parse_answer(pred)
                predictions.append(parsed)
                prompt_variants.append(f"{new_prefix}# {new_question}")

            stats = _compute_majority_vote(predictions, n_samples)
            result = {
                "index": i,
                "ground_truth": ground_truth,
                "majority_vote": stats["majority_vote"],
                "is_correct": stats["majority_vote"] == ground_truth,
                "agreement_rate": stats["agreement_rate"],
                "yes_count": stats["yes_count"],
                "no_count": stats["no_count"],
                "other_count": stats["other_count"],
                "predictions": predictions,
                "prompt_variants": prompt_variants,
            }
        else:
            # Sampling modes (noise/temperature): N forward passes
            batch_data = construct_batch(batch, tokenizer, device)
            base_vectors = batch_data.steering_vectors
            predictions = []
            for _ in range(stability_config.n_samples):
                if mode == "temperature":
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

            stats = _compute_majority_vote(predictions, stability_config.n_samples)
            result = {
                "index": i,
                "ground_truth": ground_truth,
                "majority_vote": stats["majority_vote"],
                "is_correct": stats["majority_vote"] == ground_truth,
                "agreement_rate": stats["agreement_rate"],
                "yes_count": stats["yes_count"],
                "no_count": stats["no_count"],
                "other_count": stats["other_count"],
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
    parser.add_argument("--mode", choices=["noise", "temperature", "threshold", "prompt"], default="noise",
                        help="Stability mode: 'noise' perturbs activations, 'temperature' uses stochastic decoding, "
                             "'threshold' uses logit confidence, 'prompt' uses paraphrased prompts")
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0.003],
                        help="Noise scale(s) for noise mode (default: 0.003)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                        help="Temperature(s) for temperature mode (default: 1.0)")
    parser.add_argument("--vary-question", action=argparse.BooleanOptionalAction, default=True,
                        help="(prompt mode) Paraphrase the question (default: True)")
    parser.add_argument("--vary-prefix", action=argparse.BooleanOptionalAction, default=True,
                        help="(prompt mode) Paraphrase the instruction prefix (default: True)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of forward passes per example (default: 10)")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run even if JSON exists")
    args = parser.parse_args()

    # Build list of param values to evaluate
    if args.mode == "noise":
        param_values = args.noise_scales
    elif args.mode == "temperature":
        param_values = args.temperatures
    else:
        param_values = [0]  # Threshold/prompt mode: single run, no sweep parameter

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
            cfg = StabilityConfig(mode="noise", n_samples=args.n_samples, noise_scale=param)
            param_str = f"noise{param}_n{cfg.n_samples}"
        elif args.mode == "temperature":
            cfg = StabilityConfig(mode="temperature", n_samples=args.n_samples, temperature=param)
            param_str = f"temp{param}_n{cfg.n_samples}"
        elif args.mode == "prompt":
            cfg = StabilityConfig(
                mode="prompt", n_samples=args.n_samples,
                vary_question=args.vary_question, vary_prefix=args.vary_prefix,
            )
            flags = ("q" if args.vary_question else "") + ("p" if args.vary_prefix else "")
            param_str = f"promptvar_{flags}_n{cfg.n_samples}" if flags else f"promptvar_none_n{cfg.n_samples}"
        else:
            cfg = StabilityConfig(mode="threshold")
            param_str = "logitconf"

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

    # Load question paraphrases for prompt mode
    question_paraphrases = None
    if args.mode == "prompt":
        question_paraphrases = load_question_paraphrases(DATASET_NAME)
        print(f"Loaded {len(question_paraphrases)} question paraphrases for '{DATASET_NAME}'")

    # Run stability evaluation for each param value
    for run_idx, (stability_config, json_path) in enumerate(configs_to_run):
        print(f"\n{'=' * 60}")
        if stability_config.mode == "prompt":
            print(f"Run {run_idx + 1}/{len(configs_to_run)}: prompt mode, "
                  f"vary_question={stability_config.vary_question}, vary_prefix={stability_config.vary_prefix}")
        else:
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
            question_paraphrases=question_paraphrases,
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
