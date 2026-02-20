import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import random
from datetime import timedelta

# All necessary imports are now included above
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from transformers.optimization import get_linear_schedule_with_warmup
import torch.distributed as dist
import wandb

import nl_probes.dataset_classes.classification as classification
from nl_probes.utils.steering_hooks import (
    add_hook,
    get_hf_activation_steering_hook,
)
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.eval import run_evaluation, score_eval_responses
from nl_probes.utils.confidence_utils import (
    get_confidence_json_path,
    load_confidence_map,
    apply_confidence_labels_to_dataset,
)


def push_lora_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    private: bool,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download, upload_file

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name,
                filename="config.json",
                cache_dir=None,
                force_download=False,
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SAE Introspection

This is a LoRA (Low-Rank Adaptation) adapter trained for SAE (Sparse Autoencoder) introspection tasks.

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: SAE Feature Introspection

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using the lightweight SAE introspection training script to help the model understand and explain SAE features through activation steering.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch.steering_vectors
    batch_positions = training_batch.positions

    # # --- SANITY CHECK START (Run this once to verify, then comment out) ---
    # if random.random() < 0.01: # Only print 1% of the time to avoid spam
    #     print("\n" + "="*50)
    #     print("🔍 TRAINING DATA SANITY CHECK")
        
    #     # 1. Decode the text to see the structure
    #     # We look at the first item in the batch
    #     input_tokens = training_batch.input_ids[0]
    #     text = tokenizer.decode(input_tokens, skip_special_tokens=False)
    #     print(f"📄 PROMPT (First 300 chars):\n{text[:300]}...")
        
    #     # 2. Check the Steering Positions
    #     # Where are we injecting vectors?
    #     positions = batch_positions[0] # List of ints
    #     print(f"\n📍 INJECTION INDICES: {positions}")
        
    #     # 3. verify Alignment: Are we hitting the '?' tokens?
    #     # The token at each position SHOULD be your special token (e.g. ' ?')
    #     target_tokens = [tokenizer.decode(input_tokens[p]) for p in positions]
    #     print(f"🎯 TARGET TOKENS AT INDICES: {target_tokens}")
    #     if not all('?' in t for t in target_tokens):
    #         print("❌ WARNING: Vectors are NOT hitting '?' tokens! Check alignment!")
        
    #     # 4. Check Vector Shapes
    #     # We expect [Num_Positions, Hidden_Dim]
    #     # In your multi-layer setup, Num_Positions should equal (Num_Layers * Window_Size)
    #     vectors = batch_steering_vectors[0]
    #     print(f"📐 VECTOR SHAPE: {vectors.shape}")
        
    #     expected_len = len(positions)
    #     if vectors.shape[0] != expected_len:
    #         print(f"❌ MISMATCH: {expected_len} positions but {vectors.shape[0]} vectors!")
    #     else:
    #         print("✅ Shape and Count Match.")
            
    #     print("="*50 + "\n")
    # # --- SANITY CHECK END ---

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch.labels).loss

    return loss


def eval_all_datasets(
    cfg: SelfInterpTrainingConfig,
    eval_datasets: dict[str, list[TrainingDataPoint]],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> None:
    model.eval()
    eval_results = {}
    for ds in eval_datasets:
        eval_responses = run_evaluation(
            eval_data=eval_datasets[ds],
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=global_step,
            lora_path=None,
            eval_batch_size=cfg.eval_batch_size,
            steering_coefficient=cfg.steering_coefficient,
            generation_kwargs=cfg.generation_kwargs,
        )
        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])
        eval_results[f"eval_format_correct/{ds}"] = percent_format_correct
        eval_results[f"eval_ans_correct/{ds}"] = percent_ans_correct
        print(f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")

    wandb.log(
        eval_results,
        step=global_step,
    )
    wandb.summary.update(eval_results)
    model.train()

    # Have occasionally seen OOMs on first training step after eval, so clear cache here
    torch.cuda.empty_cache()
    gc.collect()


def oom_preflight_check(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    long_prompts = [longest_prompt] * cfg.train_batch_size
    long_prompts = materialize_missing_steering_vectors(long_prompts, tokenizer, model)
    largest_possible_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        loss = train_features_batch(cfg, largest_possible_batch, model, submodule, device, dtype)
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    print("OOM preflight check complete")


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    eval_datasets: dict[str, list[TrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    verbose: bool = False,
):
    # Distributed settings (always on; launch with torchrun, even on 1 GPU)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Ensure loads happen on this GPU only (important for quantized models)
    model_kwargs = {
        **model_kwargs,
        "device_map": {"": f"cuda:{local_rank}"},
    }

    set_seed(cfg.seed)
    model = load_model(cfg.model_name, dtype, **model_kwargs)

    model.enable_input_require_grads()

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.use_lora and cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if vlm_targets and target_modules == "all-linear":
            print(f"VLM detected ({cfg.model_name}): excluding vision tower from LoRA")
            target_modules = vlm_targets

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        load_lora_path = Path(cfg.load_lora_path)
        assert load_lora_path.exists()
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)

    model.print_trainable_parameters()

    # Wrap with DDP for training, but keep the PEFT model reference for hooks/eval
    torch.cuda.set_device(local_rank)
    train_model_module: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )

    train_model_module.train()

    oom_preflight_check(cfg, training_data, model, submodule, tokenizer, device, dtype)

    set_seed(cfg.seed)

    optimizer = torch.optim.AdamW(train_model_module.parameters(), lr=cfg.lr)

    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data):
        print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
        training_data = training_data[:effective_steps]

    # Token accounting (approx): count tokens after the DDP trim and before sharding.
    # This slightly overestimates actual training tokens because we later trim per-rank
    # to align with gradient_accumulation_steps.
    if rank == 0:
        tokens_per_epoch_est = sum(len(dp.input_ids) for dp in training_data)
        total_training_tokens_est = tokens_per_epoch_est * cfg.num_epochs
        num_examples_pre_shard = len(training_data)

    # Shard dataset per rank (simple strided split)
    training_data = training_data[rank::world_size]

    num_batches_per_epoch = len(training_data) // cfg.train_batch_size
    batches_per_epoch = (num_batches_per_epoch // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
    trimmed_examples = batches_per_epoch * cfg.train_batch_size
    if trimmed_examples != len(training_data) and rank == 0:
        print(
            f"Trimming per-rank training_data from {len(training_data)} to {trimmed_examples} "
            "to align with gradient_accumulation_steps"
        )
    training_data = training_data[:trimmed_examples]

    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert steps_per_epoch > 0, "No optimizer steps will be run; check dataset/batch/accumulation sizes"
    total_training_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0

    # Init Weights & Biases only on rank 0
    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/total_tokens_est"] = total_training_tokens_est
        wandb.summary["train/num_examples_pre_shard"] = num_examples_pre_shard

    for epoch in range(cfg.num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()
        for step_idx, start in enumerate(
            tqdm(
                range(0, len(training_data), cfg.train_batch_size),
                desc=f"Training epoch {epoch + 1}",
                disable=rank != 0,
            )
        ):
            t_batch_list: list[TrainingDataPoint] = training_data[start : start + cfg.train_batch_size]

            # Compute missing steering vectors using the PEFT model (not DDP wrapper)
            t_batch_list = materialize_missing_steering_vectors(t_batch_list, tokenizer, model)

            t_batch = construct_batch(t_batch_list, tokenizer, device)

            # Forward/backward on the DDP-wrapped module if enabled
            loss = train_features_batch(cfg, t_batch, train_model_module, submodule, device, dtype)
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0

            if is_update_step:
                clip_grad_norm_(train_model_module.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0:
                    wandb.log(
                        {
                            "train/loss": accumulated_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )
                    if verbose:
                        print(f"Step {global_step} loss: {accumulated_loss}")

                # -------------------------------- evaluation --------------------------------
                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                    if rank == 0:
                        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
                    dist.barrier()

                if global_step % cfg.save_steps == 0 and global_step > 0:
                    if rank == 0:
                        model.save_pretrained(f"{cfg.save_dir}/step_{global_step}")
                        if cfg.hf_push_to_hub and cfg.hf_repo_id:
                            print("Pushing LoRA adapter to Hugging Face Hub...")
                            push_lora_to_hf(
                                model=model,
                                tokenizer=tokenizer,
                                repo_id=cfg.hf_repo_id + f"-step-{global_step}",
                                private=cfg.hf_private_repo,
                                commit_message=(f"SAE introspection LoRA - {cfg.wandb_run_name} - step {global_step}"),
                            )
                            print("Pushed LoRA adapter to Hugging Face Hub.")
                    dist.barrier()

                global_step += 1
                accumulated_loss = 0.0

    print("Training complete.")

    # Save final model
    if rank == 0:
        print("Saving final model...")
        model.save_pretrained(f"{cfg.save_dir}/final")

        # Final evaluation
        print("Running final evaluation...")
        eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)
        wandb.finish()

        # Push to Hugging Face if configured
        if cfg.hf_push_to_hub and cfg.hf_repo_id:
            print("Pushing LoRA adapter to Hugging Face Hub...")
            push_lora_to_hf(
                model=model,
                tokenizer=tokenizer,
                repo_id=cfg.hf_repo_id,
                commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - final model",
                private=cfg.hf_private_repo,
            )
    dist.barrier()


def length_grouped_reorder(
    data: list[TrainingDataPoint],
    batch_size: int,
    window_mult: int,
) -> list[TrainingDataPoint]:
    lengths = [len(d.input_ids) for d in data]

    indices = list(range(len(data)))
    megabatch_size = window_mult * batch_size

    # Slice into mega-batches
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    # Sort within each mega-batch by length desc
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    new_order = [i for mb in megabatches for i in mb]
    return [data[i] for i in new_order]


def _maybe_apply_confidence(
    train_data: list[TrainingDataPoint],
    dataset_loader: ActDatasetLoader,
    split: str,
    tokenizer: PreTrainedTokenizer | None,
) -> list[TrainingDataPoint]:
    """
    If a confidence JSON sidecar exists for this dataset's .pt file,
    apply confidence labels to the training data. Otherwise, return unchanged.
    """
    assert tokenizer is not None, "tokenizer required for confidence labeling"

    pt_filename = dataset_loader.get_dataset_filename(split)
    pt_path = Path(dataset_loader.dataset_config.dataset_folder) / pt_filename
    json_path = get_confidence_json_path(pt_path)

    if not json_path.exists():
        print(f"  [CONFIDENCE] No JSON found for {pt_filename}, using original labels")
        return train_data

    print(f"  [CONFIDENCE] Loading from {json_path.name}")
    confidence_map = load_confidence_map(json_path)
    # confidence_map may be smaller than train_data (IDK entries are excluded)
    assert len(confidence_map) <= len(train_data), (
        f"Confidence map size {len(confidence_map)} > dataset size {len(train_data)}"
    )

    relabeled = apply_confidence_labels_to_dataset(train_data, confidence_map, tokenizer)

    if confidence_map:
        confidences = list(confidence_map.values())
        mean_conf = sum(confidences) / len(confidences)
        skipped = len(train_data) - len(confidence_map)
        print(f"  [CONFIDENCE] Applied to {len(confidence_map)} datapoints "
              f"(skipped {skipped} IDK): mean={mean_conf:.2f}, "
              f"min={min(confidences):.2f}, max={max(confidences):.2f}")

    return relabeled


def build_datasets(
    cfg: SelfInterpTrainingConfig,
    dataset_loaders: list[ActDatasetLoader],
    max_len_percentile: float | None = 0.999,
    window_mult: int | None = 20,
    apply_confidence_labels: bool = False,
    tokenizer: PreTrainedTokenizer | None = None,
) -> tuple[list[TrainingDataPoint], dict[str, list[TrainingDataPoint]]]:
    set_seed(cfg.seed)
    all_training_data: list[TrainingDataPoint] = []
    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for dataset_loader in dataset_loaders:
        if "train" in dataset_loader.dataset_config.splits:
            train_data = dataset_loader.load_dataset("train")

            # Apply confidence labels only to classification datasets
            if (
                apply_confidence_labels
                and dataset_loader.dataset_config.dataset_name.startswith("classification_")
            ):
                train_data = _maybe_apply_confidence(train_data, dataset_loader, "train", tokenizer)

            all_training_data.extend(train_data)
        if "test" in dataset_loader.dataset_config.splits:
            all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

    p = max_len_percentile
    if p is not None:
        if p >= 1.0 or p <= 0.0:
            raise ValueError("max_len_percentile must be less than 1.0 and greater than 0.0")

        lengths = sorted(len(td.input_ids) for td in all_training_data)
        median_length = lengths[len(lengths) // 2]
        print(f"Max length: {lengths[-1]}, Min length: {lengths[0]}, Median length: {median_length}")
        # Inclusive quantile index
        idx = int((len(lengths) - 1) * p)
        threshold = lengths[idx]

        before = len(all_training_data)
        all_training_data = [td for td in all_training_data if len(td.input_ids) <= threshold]
        removed = before - len(all_training_data)
        print(f"Percentile trim: kept <= {threshold} tokens (p={p:.6f}). Removed {removed}/{before} examples.")

    set_seed(cfg.seed)
    random.shuffle(all_training_data)

    if window_mult is not None:
        all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size, window_mult)

    return all_training_data, all_eval_data


# Helper to cut repetition when building DatasetLoaderConfig
def mk_cfg(
    custom_params,
    *,
    num_train: int,
    num_test: int,
    splits: list[str],
    model_name: str,
    layer_percents: list[int],
    save_acts: bool,
    batch_size: int,
) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        custom_dataset_params=custom_params,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
    )


def build_loader_groups(
    *,
    model_name: str,
    layer_percents: list[int],
    act_collection_batch_size: int,
    save_acts: bool,
    classification_datasets: dict[str, dict[str, Any]],
    model_kwargs: dict[str, Any],
    # Classification-specific options
    position_resample_repeats: int = 1,
    enable_idk_mixing: bool = False,
    idk_ratio: float = 0.33,
) -> dict[str, list[ActDatasetLoader]]:
    DEBUG = False
    num_datapoints = 100_000

    # DEBUG = True

    if DEBUG:
        print("DEBUG mode: using small datasets")
        num_datapoints = 100

    # PastLens: build both single-token and multi-token variants
    past_lens_single = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=1,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    past_lens_multi = PastLensDatasetLoader(
        dataset_config=mk_cfg(
            PastLensDatasetConfig(
                max_k_activations=50,
                max_k_tokens=50,
            ),
            num_train=num_datapoints,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    latent_qa_loader = LatentQADatasetLoader(
        dataset_config=mk_cfg(
            custom_params=LatentQADatasetConfig(),
            num_train=100_000,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=False,
            batch_size=train_batch_size,
        )
    )

    # Classification: build both single-token and multi-token variants for each dataset
    # position_resample_repeats: 1 for -1N models, 3 for -3N models, 6 for -6N models
    # enable_idk_mixing: set to True to include "I don't know" category
    #   NOTE: When enable_idk_mixing=True, uses a SINGLE combined dataset from all IID sources
    #         with Yes/No/IDK samples mixed. The per-dataset loop is skipped.
    # These are now passed as function parameters.
    
    classification_loaders: list[ActDatasetLoader] = []
    
    if enable_idk_mixing:
        # IDK mixing mode: create ONE combined dataset from all IID datasets
        # Uses "iid" keyword to load only IID datasets (OOD held out for eval)
        total_train = sum(meta["num_train"] for meta in classification_datasets.values() if "train" in meta["splits"])
        total_test = sum(meta["num_test"] for meta in classification_datasets.values() if "test" in meta["splits"])
        
        single_params_idk = ClassificationDatasetConfig(
            classification_dataset_name="iid",  # Uses all IID datasets
            max_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=2,
            position_resample_repeats=position_resample_repeats,
            enable_idk_mixing=True,
            idk_ratio=idk_ratio,
            use_3way_prompt=True,  # IDK requires 3-way prompt
        )
        multi_params_idk = ClassificationDatasetConfig(
            classification_dataset_name="iid",  # Uses all IID datasets
            max_window_size=50,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
            position_resample_repeats=position_resample_repeats,
            enable_idk_mixing=True,
            idk_ratio=idk_ratio,
            use_3way_prompt=True,  # IDK requires 3-way prompt
        )
        
        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    single_params_idk,
                    num_train=total_train,
                    num_test=total_test,
                    splits=["train", "test"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=train_batch_size,
                ),
                model_kwargs=model_kwargs,
            )
        )
        classification_loaders.append(
            ClassificationDatasetLoader(
                dataset_config=mk_cfg(
                    multi_params_idk,
                    num_train=total_train,
                    num_test=total_test,
                    splits=["train", "test"],
                    model_name=model_name,
                    layer_percents=layer_percents,
                    save_acts=save_acts,
                    batch_size=train_batch_size,
                ),
                model_kwargs=model_kwargs,
            )
        )
    else:
        # Standard mode: per-dataset loaders
        for ds_name, meta in classification_datasets.items():
            single_params = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_window_size=1,
                min_end_offset=-1,
                max_end_offset=-5,
                num_qa_per_sample=2,
                position_resample_repeats=position_resample_repeats,
            )
            multi_params = ClassificationDatasetConfig(
                classification_dataset_name=ds_name,
                max_window_size=50,
                min_end_offset=-1,
                max_end_offset=-5,
                num_qa_per_sample=1,
                position_resample_repeats=position_resample_repeats,
            )

            # language identification has very long sequence lengths
            if "batch_size" in meta:
                bs = meta["batch_size"]
            else:
                bs = train_batch_size

            classification_loaders.append(
                ClassificationDatasetLoader(
                    dataset_config=mk_cfg(
                        single_params,
                        num_train=meta["num_train"],
                        num_test=meta["num_test"],
                        splits=meta["splits"],
                        model_name=model_name,
                        layer_percents=layer_percents,
                        save_acts=save_acts,
                        batch_size=bs,
                    ),
                    model_kwargs=model_kwargs,
                )
            )

            classification_loaders.append(
                ClassificationDatasetLoader(
                    dataset_config=mk_cfg(
                        multi_params,
                        num_train=meta["num_train"],
                        num_test=meta["num_test"],
                        splits=meta["splits"],
                        model_name=model_name,
                        layer_percents=layer_percents,
                        save_acts=save_acts,
                        batch_size=train_batch_size,
                    ),
                    model_kwargs=model_kwargs,
                )
            )

    return {
        "past_lens_loaders": [past_lens_single, past_lens_multi],
        "latentqa_loaders": [latent_qa_loader],
        "classification_loaders": classification_loaders,
        # Config info for validation and banner
        "classification_config": {
            "position_resample_repeats": position_resample_repeats,
            "enable_idk_mixing": enable_idk_mixing,
            "idk_ratio": idk_ratio,
        },
    }


def print_classification_config_banner(cls_config: dict[str, Any]) -> None:
    """Print a prominent banner showing classification training config."""
    enable_idk = cls_config.get("enable_idk_mixing", False)
    idk_ratio = cls_config.get("idk_ratio", 0.33)
    repeats = cls_config.get("position_resample_repeats", 1)
    
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║              CLASSIFICATION TRAINING CONFIGURATION               ║
╠══════════════════════════════════════════════════════════════════╣
║  IDK Mixing:              {idk_status:<40} ║
║  Position Resample:       {repeats}x                                          ║
║  Prompt Style:            {prompt_style:<40} ║"""
    
    if enable_idk:
        yes_no_pct = (1 - idk_ratio) * 100
        idk_pct = idk_ratio * 100
        banner += f"""
║  Expected Distribution:   ~{yes_no_pct/2:.0f}% Yes, ~{yes_no_pct/2:.0f}% No, ~{idk_pct:.0f}% IDK            ║"""
    else:
        banner += """
║  Expected Distribution:   ~50% Yes, ~50% No                      ║"""
    
    banner += """
╚══════════════════════════════════════════════════════════════════╝"""
    
    idk_status = "ENABLED" if enable_idk else "DISABLED (binary Yes/No only)"
    prompt_style = "3-way (Yes/No/IDK)" if enable_idk else "Binary (Yes/No only)"
    
    print(banner.format(
        idk_status=idk_status,
        repeats=repeats,
        prompt_style=prompt_style,
    ))


def print_sample_training_data(training_data: list[TrainingDataPoint], tokenizer, num_samples: int = 3) -> None:
    """Print a few sample training examples for visual verification."""
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING DATA (for verification)")
    print("=" * 60)
    
    # Count target distributions
    target_counts: dict[str, int] = {}
    for dp in training_data:
        target = dp.target_output
        target_counts[target] = target_counts.get(target, 0) + 1
    
    total = len(training_data)
    print(f"\nTarget Distribution ({total} total samples):")
    for target, count in sorted(target_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {target}: {count} ({pct:.1f}%)")
    
    # Print samples
    print(f"\nFirst {num_samples} samples:")
    for i, dp in enumerate(training_data[:num_samples]):
        prompt_text = tokenizer.decode(dp.input_ids, skip_special_tokens=False)
        # Truncate for display
        if len(prompt_text) > 300:
            prompt_text = prompt_text[:150] + " ... " + prompt_text[-100:]
        print(f"\n--- Sample {i+1} ---")
        print(f"Target: {dp.target_output}")
        print(f"Prompt: {prompt_text}")
    
    print("\n" + "=" * 60 + "\n")


def validate_idk_training_data(
    training_data: list[TrainingDataPoint], 
    enable_idk_mixing: bool,
    tokenizer=None,
) -> None:
    """Validate that training data matches expected IDK configuration.
    
    Performs two checks:
    1. IDK sample count matches enable_idk_mixing setting
    2. Prompt format matches expected style (binary vs 3-way)
    """
    idk_count = sum(1 for dp in training_data if dp.target_output == "I don't know")
    total = len(training_data)
    
    # Check 1: IDK sample count
    if enable_idk_mixing:
        if idk_count == 0:
            raise ValueError(
                "CRITICAL: enable_idk_mixing=True but NO 'I don't know' samples found in training data!\n"
                "This means the model will NOT learn to say IDK. Check your dataset configuration."
            )
        idk_pct = idk_count / total * 100
        print(f"✓ IDK validation passed: {idk_count}/{total} ({idk_pct:.1f}%) IDK samples found")
    else:
        if idk_count > 0:
            print(f"⚠ WARNING: enable_idk_mixing=False but found {idk_count} IDK samples. "
                  "This may indicate a configuration mismatch.")
    
    # Check 2: Prompt format (sample a few examples)
    if tokenizer is not None and len(training_data) > 0:
        sample_size = min(10, len(training_data))
        has_binary_prompt = 0
        has_3way_prompt = 0
        
        for dp in training_data[:sample_size]:
            prompt_text = tokenizer.decode(dp.input_ids, skip_special_tokens=True)
            if "Answer with 'Yes' or 'No' only" in prompt_text:
                has_binary_prompt += 1
            elif "Answer with 'Yes', 'No', or 'I don't know'" in prompt_text:
                has_3way_prompt += 1
        
        if enable_idk_mixing:
            if has_binary_prompt > 0 and has_3way_prompt == 0:
                raise ValueError(
                    "CRITICAL: enable_idk_mixing=True but prompts use BINARY format!\n"
                    f"Found {has_binary_prompt} binary prompts, {has_3way_prompt} 3-way prompts.\n"
                    "This means the model won't learn when to say IDK. Check use_3way_prompt setting."
                )
        else:
            if has_3way_prompt > 0:
                print(f"⚠ WARNING: enable_idk_mixing=False but found 3-way prompts. "
                      "This may indicate a configuration mismatch.")


def _ensure_datasets_exist(dataset_loaders: list[ActDatasetLoader]) -> None:
    """Materialize datasets on disk using a single process (rank 0).

    Each loader's `load_dataset` will create and save if missing; otherwise it
    simply loads. This avoids race conditions when multiple ranks start up.
    """

    # TODO: Switch to multiprocessing for speed

    old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    # Make only GPU 0 visible for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        for dl in dataset_loaders:
            for split in dl.dataset_config.splits:
                _ = dl.load_dataset(split)
    finally:
        # Revert to original state
        if old_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices


if __name__ == "__main__":
    # for gemma: export TORCHDYNAMO_DISABLE=1
    # Always initialize DDP (launch with torchrun, even for 1 GPU)
    # time delta of two hours because currently it can take 1 hour to build all datasets
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    main_train_size = 6000
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "relations": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "sst2": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "md_gender": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "snli": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "ag_news": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
        "ner": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "tense": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["train", "test"],
        },
        "language_identification": {
            "num_train": main_train_size,
            "num_test": main_test_size,
            "splits": ["test"],
            # language identification has very long sequence lengths
            "batch_size": 4,
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    hook_layer = 1
    # model_name = "Qwen/Qwen3-32B"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # model_name = "google/gemma-2-9b-it"
    # model_name = "Qwen/Qwen3-8B"

    models = [
        # "Qwen/Qwen3-14B",
        # "google/gemma-2-27b-it",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "google/gemma-3-4b-it",
        # "google/gemma-3-12b-it",
        # "google/gemma-3-27b-it",
        "Qwen/Qwen3-8B",
    ]

    for model_name in models:
        # ═══════════════════════════════════════════════════════════════════
        # CLASSIFICATION CONFIG - REVIEW CAREFULLY BEFORE TRAINING!
        # ═══════════════════════════════════════════════════════════════════
        position_resample_repeats = 1  # 1 for -1N, 3 for -3N, 6 for -6N
        enable_idk_mixing = True       # True = train with IDK samples (~1/3 yes, 1/3 no, 1/3 idk)
        idk_ratio = 0.33               # Only used if enable_idk_mixing=True
        apply_confidence_labels = False # True = relabel classification data with confidence scores

        # Layer config
        layer_percents = [25, 50, 75]   # 3L config; use [15, 30, 45, 60, 75, 90] for 6L
        
        # ═══════════════════════════════════════════════════════════════════
        # HF repo naming: MLAO-{Model}-{Layers}L-{Repeats}N[-IDK]
        # ═══════════════════════════════════════════════════════════════════
        num_layers = len(layer_percents)
        model_short_name = model_name.split("/")[-1].replace(".", "-")
        hf_repo_name = f"MLAO-{model_short_name}-{num_layers}L-{position_resample_repeats}N"
        if enable_idk_mixing:
            hf_repo_name += "-IDK-fixed"
        print(f"hf_repo_name: {hf_repo_name}")
        model_name_str = model_name.split("/")[-1].replace(".", "_").replace(" ", "_")

        #train_batch_size = 16
        train_batch_size = 8
        gradient_checkpointing = True
        model_kwargs = {}

        if model_name == "Qwen/Qwen3-32B" or model_name == "meta-llama/Llama-3.3-70B-Instruct":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=dtype,
            )
            model_kwargs = {"quantization_config": bnb_config}

        # if model_name == "meta-llama/Llama-3.3-70B-Instruct":
        # train_batch_size = train_batch_size * 4  # increase gpu utilization on 4x GPUs
        # cuts training time by ~50%

        print("Global train batch size:", train_batch_size)
        assert train_batch_size % world_size == 0, (
            f"Global batch size {train_batch_size} must be divisible by world_size {world_size}"
        )
        train_batch_size = train_batch_size // world_size
        print(f"Per-rank train batch size: {train_batch_size}, world size: {world_size}")

        save_acts = False

        gradient_accumulation_steps = 2

        # Build loader groups (single + multi variants)
        loader_groups = build_loader_groups(
            model_name=model_name,
            layer_percents=layer_percents,
            act_collection_batch_size=train_batch_size,
            save_acts=save_acts,
            classification_datasets=classification_datasets,
            model_kwargs=model_kwargs,
            position_resample_repeats=position_resample_repeats,
            enable_idk_mixing=enable_idk_mixing,
            idk_ratio=idk_ratio,
        )

        classification_dataset_loaders = loader_groups["classification_loaders"]
        past_lens_loaders = loader_groups["past_lens_loaders"]
        latentqa_loaders = loader_groups["latentqa_loaders"]
        classification_config = loader_groups["classification_config"]
        
        # Print config banner (rank 0 only to avoid spam)
        if local_rank == 0:
            print_classification_config_banner(classification_config)

        iterations = [
            # Default dataset mixture
            # Set load_lora_path to checkpoint path to continue training
            {
                "load_lora_path": None,
                "dataset_loaders": latentqa_loaders + classification_dataset_loaders + past_lens_loaders,
                "wandb_suffix": f"_latentqa_cls_past_lens_{model_name_str}",
            },
            # {
            #     "load_lora_path": None,
            #     "dataset_loaders": latentqa_loaders,
            #     "wandb_suffix": f"_latentqa_only_{model_name_str}",
            # },
        ]

        for hyperparam_override in iterations:
            loop_dataset_loaders = hyperparam_override.pop("dataset_loaders")
            if hyperparam_override["load_lora_path"] is not None:
                assert os.path.exists(hyperparam_override["load_lora_path"]), f"{hyperparam_override['load_lora_path']}"

            cfg = SelfInterpTrainingConfig(
                model_name=model_name,
                hook_onto_layer=hook_layer,
                hf_repo_name=hf_repo_name,
                # wandb_suffix=wandb_suffix,
                layer_percents=layer_percents,
                train_batch_size=train_batch_size,
                activation_collection_batch_size=train_batch_size * 4,
                eval_batch_size=train_batch_size * 8,
                eval_steps=10_000,
                eval_on_start=True,
                gradient_checkpointing=gradient_checkpointing,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **hyperparam_override,
            )

            cfg.finalize(dataset_loaders=loop_dataset_loaders)

            print(f"save dir: {cfg.save_dir}")

            tokenizer = load_tokenizer(cfg.model_name)

            # Ensure only rank 0 performs any on-disk dataset creation
            if local_rank == 0:
                _ensure_datasets_exist(loop_dataset_loaders)
            dist.barrier()

            all_training_data, all_eval_data = build_datasets(
                cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult,
                apply_confidence_labels=apply_confidence_labels, tokenizer=tokenizer,
            )

            # for debugging
            # all_training_data = all_training_data[:100]
            # eval_keys = list(all_eval_data.keys())
            # assert len(eval_keys) == 1
            # eval_key = eval_keys[0]
            # all_eval_data = {eval_key: all_training_data[:]}

            print(f"training data length: {len(all_training_data)}, eval data length: {len(all_eval_data)}")
            
            # Validate IDK training data and print samples (rank 0 only)
            if local_rank == 0:
                validate_idk_training_data(
                    all_training_data, 
                    enable_idk_mixing=classification_config["enable_idk_mixing"],
                    tokenizer=tokenizer,
                )
                # print_sample_training_data(all_training_data, tokenizer, num_samples=3)

            print(asdict(cfg))

            train_model(
                cfg=cfg,
                training_data=all_training_data,
                eval_datasets=all_eval_data,
                tokenizer=tokenizer,
                dtype=dtype,
                device=device,
                model_kwargs=model_kwargs,
                verbose=True,
            )

    # Clean up DDP
    dist.destroy_process_group()
