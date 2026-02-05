# %%
"""
Classification evaluation with IDK (I don't know) support.

This script evaluates models trained with IDK mixing on standard classification datasets.
The key difference from classification_eval.py:
- Uses 3-way prompt: "Answer with 'Yes', 'No', or 'I don't know'."
- Tracks when model says IDK to compute selective accuracy and coverage
- Ground truth is still Yes/No (no IDK test samples), but model can abstain
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
from dataclasses import dataclass
from typing import Any, Union
import gc
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
    IID_DATASETS,
    OOD_DATASETS,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation
from nl_probes.base_experiment import sanitize_lora_name

# -----------------------------
# Configuration
# -----------------------------

CONFIG_3L = [25, 50, 75]
CONFIG_6L = [15, 30, 45, 60, 75, 90]

# Models trained with IDK mixing
MODEL_CONFIGS = {
    # "Qwen/Qwen3-4B": [
    #     ["nluick/MLAO-Qwen3-4B-3L-1N", CONFIG_3L],
    #     ["nluick/MLAO-Qwen3-4B-3L-3N", CONFIG_3L],
    #     ["nluick/MLAO-Qwen3-4B-6L-1N", CONFIG_6L],
    #     ["nluick/MLAO-Qwen3-4B-6L-3N", CONFIG_6L],
    #     ["nluick/MLAO-Qwen3-4B-6L-6N", CONFIG_6L],
    # ],
    "Qwen/Qwen3-8B": [
        ["nluick/MLAO-Qwen3-8B-3L-1N-IDK", CONFIG_3L],
#        ["nluick/MLAO-Qwen3-8B-3L-3N", CONFIG_3L],
    ],
}

INJECTION_LAYER = 1
DTYPE = torch.bfloat16
BASE_BATCH_SIZE = 256
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}

# 3-way prompt for IDK evaluation
PREFIX = "Answer with 'Yes', 'No', or 'I don't know'. "

SINGLE_TOKEN_MODE = True
mode_str = "single_token" if SINGLE_TOKEN_MODE else "multi_token"

EXPERIMENTS_DIR = "experiments"
DATA_DIR = "classification_idk"

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(f"{EXPERIMENTS_DIR}/{DATA_DIR}", exist_ok=True)

device = torch.device("cuda")
dtype = torch.bfloat16
print(f"Using device={device}, dtype={dtype}")

# Dataset selection - same as classification_eval.py
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
    "engels_headline_istrump": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_isobama": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_ischina": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_hist_fig_ismale": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_news_class_politics": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isjournalist": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isathlete": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_ispolitician": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_issinger": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isresearcher": {"num_train": 0, "num_test": 250, "splits": ["test"]},
}


def canonical_dataset_id(name: str) -> str:
    """Strip 'classification_' prefix if present."""
    if name.startswith("classification_"):
        return name[len("classification_"):]
    return name


def is_iid_dataset(ds_id: str) -> bool:
    """Check if dataset is in-distribution."""
    return ds_id in IID_DATASETS


def parse_3way_response(response: str) -> str:
    """
    Parse model response into one of: 'Yes', 'No', 'IDK', 'invalid'.
    
    Returns:
        'Yes', 'No', 'IDK', or 'invalid' if cannot parse
    """
    response_lower = response.lower().strip()
    
    # Check for IDK first (more specific patterns)
    idk_patterns = [
        "i don't know",
        "i dont know", 
        "don't know",
        "dont know",
        "idk",
        "unknown",
        "uncertain",
        "not sure",
        "cannot determine",
        "can't determine",
    ]
    for pattern in idk_patterns:
        if pattern in response_lower:
            return "IDK"
    
    # Check for Yes/No
    if response_lower.startswith("yes") or response_lower == "y":
        return "Yes"
    if response_lower.startswith("no") or response_lower == "n":
        return "No"
    
    # Try to find yes/no anywhere in short responses
    if len(response_lower) < 20:
        if "yes" in response_lower:
            return "Yes"
        if "no" in response_lower:
            return "No"
    
    return "invalid"


def get_model_kwargs(model_name: str) -> dict:
    """Return model kwargs based on model name."""
    if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        return {"quantization_config": bnb_config}
    return {}


def get_batch_size(model_name: str) -> int:
    """Return batch size based on model name."""
    if model_name == "Qwen/Qwen3-32B":
        return BASE_BATCH_SIZE // 4
    return BASE_BATCH_SIZE


def load_datasets_for_layer_percent(
    model_name: str, layer_percent: Union[int, list[int]], model_kwargs: dict, model=None
) -> dict[str, list[Any]]:
    """Load all classification datasets for a specific model and layer percent."""
    batch_size = get_batch_size(model_name)

    if isinstance(layer_percent, list):
        current_layers = layer_percent
    else:
        current_layers = [layer_percent]

    classification_dataset_loaders: list[ClassificationDatasetLoader] = []
    for dataset_name, dcfg in CLASSIFICATION_DATASETS.items():
        if "language_identification" in dataset_name:
            ds_batch_size = batch_size // 8
        else:
            ds_batch_size = batch_size

        if SINGLE_TOKEN_MODE:
            classification_config = ClassificationDatasetConfig(
                classification_dataset_name=dataset_name,
                max_end_offset=-3,
                min_end_offset=-3,
                max_window_size=1,
                min_window_size=1,
                use_3way_prompt=True,  # So model sees "Yes", "No", or "I don't know" as options
            )
        else:
            classification_config = ClassificationDatasetConfig(
                classification_dataset_name=dataset_name,
                max_end_offset=-1,
                min_end_offset=-1,
                max_window_size=50,
                min_window_size=50,
                use_3way_prompt=True,
            )
        dataset_config = DatasetLoaderConfig(
            custom_dataset_params=classification_config,
            num_train=dcfg["num_train"],
            num_test=dcfg["num_test"],
            splits=dcfg["splits"],
            model_name=model_name,
            layer_percents=current_layers,
            save_acts=True,
            batch_size=ds_batch_size,
        )
        classification_dataset_loaders.append(
            ClassificationDatasetLoader(dataset_config=dataset_config, model_kwargs=model_kwargs, model=model)
        )

    # Pull test sets for evaluation
    all_eval_data: dict[str, list[Any]] = {}
    for loader in classification_dataset_loaders:
        if "test" in loader.dataset_config.splits:
            ds_id = canonical_dataset_id(loader.dataset_config.dataset_name)
            all_eval_data[ds_id] = loader.load_dataset("test")

    return all_eval_data


def compute_metrics(records: list[dict]) -> dict:
    """
    Compute metrics from evaluation records.
    
    Returns dict with:
        - overall_accuracy: accuracy on all samples
        - selective_accuracy: accuracy only on non-IDK predictions
        - coverage: fraction of samples where model gave Yes/No (not IDK)
        - idk_rate: fraction of samples where model said IDK
        - per_dataset: breakdown by dataset
        - iid_vs_ood: breakdown by IID/OOD
    """
    metrics = {
        "overall": {},
        "per_dataset": {},
        "iid_aggregate": {},
        "ood_aggregate": {},
    }
    
    # Overall metrics
    total = len(records)
    if total == 0:
        return metrics
    
    correct = sum(1 for r in records if r["predicted"] == r["target"])
    idk_count = sum(1 for r in records if r["predicted"] == "IDK")
    invalid_count = sum(1 for r in records if r["predicted"] == "invalid")
    answered = total - idk_count - invalid_count
    
    answered_correct = sum(
        1 for r in records 
        if r["predicted"] == r["target"] and r["predicted"] not in ("IDK", "invalid")
    )
    
    metrics["overall"] = {
        "total": total,
        "overall_accuracy": correct / total if total > 0 else 0,
        "selective_accuracy": answered_correct / answered if answered > 0 else 0,
        "coverage": answered / total if total > 0 else 0,
        "idk_rate": idk_count / total if total > 0 else 0,
        "invalid_rate": invalid_count / total if total > 0 else 0,
        "answered": answered,
        "idk_count": idk_count,
        "invalid_count": invalid_count,
    }
    
    # Per-dataset metrics
    datasets = set(r["dataset_id"] for r in records)
    for ds_id in datasets:
        ds_records = [r for r in records if r["dataset_id"] == ds_id]
        ds_total = len(ds_records)
        ds_correct = sum(1 for r in ds_records if r["predicted"] == r["target"])
        ds_idk = sum(1 for r in ds_records if r["predicted"] == "IDK")
        ds_invalid = sum(1 for r in ds_records if r["predicted"] == "invalid")
        ds_answered = ds_total - ds_idk - ds_invalid
        ds_answered_correct = sum(
            1 for r in ds_records 
            if r["predicted"] == r["target"] and r["predicted"] not in ("IDK", "invalid")
        )
        
        metrics["per_dataset"][ds_id] = {
            "total": ds_total,
            "overall_accuracy": ds_correct / ds_total if ds_total > 0 else 0,
            "selective_accuracy": ds_answered_correct / ds_answered if ds_answered > 0 else 0,
            "coverage": ds_answered / ds_total if ds_total > 0 else 0,
            "idk_rate": ds_idk / ds_total if ds_total > 0 else 0,
            "invalid_rate": ds_invalid / ds_total if ds_total > 0 else 0,
            "is_iid": is_iid_dataset(ds_id),
        }
    
    # IID vs OOD aggregate
    for split_name, is_iid in [("iid_aggregate", True), ("ood_aggregate", False)]:
        split_records = [r for r in records if is_iid_dataset(r["dataset_id"]) == is_iid]
        if not split_records:
            continue
            
        split_total = len(split_records)
        split_correct = sum(1 for r in split_records if r["predicted"] == r["target"])
        split_idk = sum(1 for r in split_records if r["predicted"] == "IDK")
        split_invalid = sum(1 for r in split_records if r["predicted"] == "invalid")
        split_answered = split_total - split_idk - split_invalid
        split_answered_correct = sum(
            1 for r in split_records 
            if r["predicted"] == r["target"] and r["predicted"] not in ("IDK", "invalid")
        )
        
        metrics[split_name] = {
            "total": split_total,
            "overall_accuracy": split_correct / split_total if split_total > 0 else 0,
            "selective_accuracy": split_answered_correct / split_answered if split_answered > 0 else 0,
            "coverage": split_answered / split_total if split_total > 0 else 0,
            "idk_rate": split_idk / split_total if split_total > 0 else 0,
            "invalid_rate": split_invalid / split_total if split_total > 0 else 0,
        }
    
    return metrics


def run_eval_for_datasets(
    model,
    tokenizer,
    submodule,
    model_name: str,
    layer_percent: Union[int, list[int]],
    lora_path: str | None,
    eval_data_by_ds: dict[str, list[Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Run evaluation and return results with IDK tracking."""
    
    sanitized_lora_name = None
    if lora_path is not None:
        sanitized_lora_name = sanitize_lora_name(lora_path)
        if sanitized_lora_name not in model.peft_config:
            print(f"Loading LoRA: {lora_path}")
            model.load_adapter(
                lora_path,
                adapter_name=sanitized_lora_name,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )
        model.set_adapter(sanitized_lora_name)

    records = []

    for ds_id, eval_data in eval_data_by_ds.items():
        print(f"  Evaluating {ds_id} ({len(eval_data)} samples)...")
        
        raw_results = run_evaluation(
            eval_data=eval_data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=-1,
            lora_path=lora_path,
            eval_batch_size=batch_size,
            steering_coefficient=STEERING_COEFFICIENT,
            generation_kwargs=GENERATION_KWARGS,
        )

        for response, target in zip(raw_results, eval_data, strict=True):
            raw_response = response.api_response
            predicted = parse_3way_response(raw_response)
            
            record = {
                "dataset_id": ds_id,
                "raw_response": raw_response,
                "predicted": predicted,
                "target": target.target_output,
                "is_correct": predicted == target.target_output,
                "is_idk": predicted == "IDK",
                "is_invalid": predicted == "invalid",
            }
            records.append(record)

    # Compute metrics
    metrics = compute_metrics(records)
    
    results = {
        "meta": {
            "model_name": model_name,
            "dtype": str(DTYPE),
            "layer_percent": layer_percent,
            "injection_layer": INJECTION_LAYER,
            "investigator_lora_path": lora_path,
            "steering_coefficient": STEERING_COEFFICIENT,
            "eval_batch_size": batch_size,
            "generation_kwargs": GENERATION_KWARGS,
            "single_token_mode": SINGLE_TOKEN_MODE,
            "prefix": PREFIX,
            "eval_type": "idk_3way",
        },
        "metrics": metrics,
        "records": records,
    }

    if sanitized_lora_name is not None and sanitized_lora_name in model.peft_config:
        model.delete_adapter(sanitized_lora_name)

    return results


def print_sample_eval_data(
    eval_data_by_ds: dict[str, list[Any]], 
    tokenizer, 
    num_samples: int = 2
) -> None:
    """Print sample evaluation data for verification."""
    print("\n" + "=" * 70)
    print("SAMPLE EVALUATION DATA (for verification)")
    print("=" * 70)
    
    for ds_id, eval_data in list(eval_data_by_ds.items())[:3]:  # First 3 datasets
        print(f"\n--- Dataset: {ds_id} ({len(eval_data)} samples) ---")
        
        for i, dp in enumerate(eval_data[:num_samples]):
            print(f"\n  Sample {i+1}:")
            
            # Decode oracle/context prompt
            if dp.context_input_ids is not None:
                oracle_prompt = tokenizer.decode(dp.context_input_ids, skip_special_tokens=True)
                if len(oracle_prompt) > 200:
                    oracle_prompt = oracle_prompt[:100] + " ... " + oracle_prompt[-80:]
                print(f"    Oracle Prompt:  {oracle_prompt}")
            
            # Decode full input (includes question)
            full_input = tokenizer.decode(dp.input_ids, skip_special_tokens=True)
            
            # Extract question part (after oracle prompt)
            if dp.context_input_ids is not None:
                oracle_text = tokenizer.decode(dp.context_input_ids, skip_special_tokens=True)
                question_part = full_input[len(oracle_text):].strip() if oracle_text in full_input else full_input
            else:
                question_part = full_input
            
            if len(question_part) > 150:
                question_part = question_part[:150] + "..."
            print(f"    Question:       {question_part}")
            print(f"    Target Answer:  {dp.target_output}")
    
    print("\n" + "=" * 70 + "\n")


def print_prediction_examples(
    records: list[dict],
    eval_data_by_ds: dict[str, list[Any]],
    tokenizer,
    num_per_category: int = 2,
) -> None:
    """Print example predictions, showing correct, incorrect, and IDK cases."""
    print("\n" + "=" * 70)
    print("PREDICTION EXAMPLES")
    print("=" * 70)
    
    # Group by prediction type
    correct = [r for r in records if r["is_correct"] and not r["is_idk"]]
    incorrect = [r for r in records if not r["is_correct"] and not r["is_idk"] and not r["is_invalid"]]
    idk_cases = [r for r in records if r["is_idk"]]
    invalid_cases = [r for r in records if r["is_invalid"]]
    
    def print_examples(category_name: str, examples: list[dict], n: int):
        if not examples:
            print(f"\n  {category_name}: (none)")
            return
        print(f"\n  {category_name} ({len(examples)} total):")
        for i, ex in enumerate(examples[:n]):
            print(f"    [{i+1}] Dataset: {ex['dataset_id']}")
            print(f"        Target: {ex['target']}, Predicted: {ex['predicted']}")
            raw = ex['raw_response']
            if len(raw) > 80:
                raw = raw[:80] + "..."
            print(f"        Raw response: \"{raw}\"")
    
    print_examples("CORRECT predictions", correct, num_per_category)
    print_examples("INCORRECT predictions", incorrect, num_per_category)
    print_examples("IDK responses", idk_cases, num_per_category)
    if invalid_cases:
        print_examples("INVALID responses", invalid_cases, num_per_category)
    
    print("\n" + "=" * 70 + "\n")


def print_metrics_summary(metrics: dict, lora_name: str) -> None:
    """Print a summary of the metrics."""
    print(f"\n{'=' * 60}")
    print(f"Results for {lora_name}")
    print(f"{'=' * 60}")
    
    overall = metrics.get("overall", {})
    print(f"\nOverall ({overall.get('total', 0)} samples):")
    print(f"  Overall Accuracy:   {overall.get('overall_accuracy', 0):.1%}")
    print(f"  Selective Accuracy: {overall.get('selective_accuracy', 0):.1%}")
    print(f"  Coverage:           {overall.get('coverage', 0):.1%}")
    print(f"  IDK Rate:           {overall.get('idk_rate', 0):.1%}")
    
    iid = metrics.get("iid_aggregate", {})
    if iid:
        print(f"\nIID Datasets ({iid.get('total', 0)} samples):")
        print(f"  Overall Accuracy:   {iid.get('overall_accuracy', 0):.1%}")
        print(f"  Selective Accuracy: {iid.get('selective_accuracy', 0):.1%}")
        print(f"  Coverage:           {iid.get('coverage', 0):.1%}")
        print(f"  IDK Rate:           {iid.get('idk_rate', 0):.1%}")
    
    ood = metrics.get("ood_aggregate", {})
    if ood:
        print(f"\nOOD Datasets ({ood.get('total', 0)} samples):")
        print(f"  Overall Accuracy:   {ood.get('overall_accuracy', 0):.1%}")
        print(f"  Selective Accuracy: {ood.get('selective_accuracy', 0):.1%}")
        print(f"  Coverage:           {ood.get('coverage', 0):.1%}")
        print(f"  IDK Rate:           {ood.get('idk_rate', 0):.1%}")
    
    print(f"\nPer-dataset IDK rates:")
    per_ds = metrics.get("per_dataset", {})
    for ds_id, ds_metrics in sorted(per_ds.items(), key=lambda x: -x[1].get("idk_rate", 0)):
        iid_marker = "[IID]" if ds_metrics.get("is_iid") else "[OOD]"
        print(f"  {ds_id:30s} {iid_marker}: IDK={ds_metrics.get('idk_rate', 0):.1%}, "
              f"Acc={ds_metrics.get('overall_accuracy', 0):.1%}, "
              f"SelAcc={ds_metrics.get('selective_accuracy', 0):.1%}")


# %%
# Main loop

if __name__ == "__main__":
    for model_name in MODEL_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 60}")

        investigator_lora_paths = MODEL_CONFIGS[model_name]
        model_kwargs = get_model_kwargs(model_name)
        batch_size = get_batch_size(model_name)

        model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")

        # Load model and tokenizer
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name, dtype, **model_kwargs)
        submodule = get_hf_submodule(model, INJECTION_LAYER)

        dummy_config = LoraConfig()
        model.add_adapter(dummy_config, adapter_name="default")

        for lora_params in investigator_lora_paths:
            lora = lora_params[0]
            print(f"\nEvaluating LoRA: {lora}")
            
            if lora is None:
                active_lora_path = None
                lora_name = "base_model"
            else:
                active_lora_path = lora
                lora_name = lora.split("/")[-1].replace("/", "_").replace(".", "_")

            layer_percent = lora_params[1]

            # Create run_dir
            if isinstance(layer_percent, list):
                lp_str = "_".join(map(str, layer_percent))
            else:
                lp_str = str(layer_percent)
            run_dir = f"{EXPERIMENTS_DIR}/{DATA_DIR}/classification_idk_{model_name_str}_{mode_str}_{lp_str}/"
            os.makedirs(run_dir, exist_ok=True)

            # Load datasets
            all_eval_data = load_datasets_for_layer_percent(model_name, layer_percent, model_kwargs, model=model)
            print(f"Loaded datasets: {list(all_eval_data.keys())}")
            
            # Print sample data for verification
            print_sample_eval_data(all_eval_data, tokenizer, num_samples=2)

            # Run evaluation
            results = run_eval_for_datasets(
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                model_name=model_name,
                layer_percent=layer_percent,
                lora_path=active_lora_path,
                eval_data_by_ds=all_eval_data,
                batch_size=batch_size,
            )

            # Print prediction examples
            print_prediction_examples(results["records"], all_eval_data, tokenizer, num_per_category=3)
            
            # Print summary
            print_metrics_summary(results["metrics"], lora_name)

            # Save results
            output_json = f"{run_dir}classification_idk_results_lora_{lora_name}.json"
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {output_json}")

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
