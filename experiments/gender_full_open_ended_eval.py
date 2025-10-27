# %%

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import random
import itertools
from tqdm import tqdm

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# nl_probes imports
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation


# ========================================
# CONFIGURATION - edit here
# ========================================

# Model and dtype
MODEL_NAME = "google/gemma-2-9b-it"
DTYPE = torch.bfloat16
model_name_str = MODEL_NAME.split("/")[-1].replace(".", "_")

VERBOSE = False
# VERBOSE = True

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUFFIX = "_50_mix"
SUFFIX = ""

PREFIX = "Answer with 'Male' or 'Female' only. "

if MODEL_NAME == "google/gemma-2-9b-it":
    INVESTIGATOR_LORA_PATHS = [
        # "adamkarvonen/checkpoints_cls_latentqa_only_addition_gemma-2-9b-it",
        "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
        # "adamkarvonen/checkpoints_cls_only_addition_gemma-2-9b-it",
        # "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it",
    ]
    ACTIVE_LORA_PATH_TEMPLATE: Optional[str] = "bcywinski/gemma-2-9b-it-user-{word}"
else:
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

# Layers for activation collection and injection
LAYER_PERCENTS = [25, 50, 75]  # Layers to collect activations from
ACT_LAYERS = [layer_percent_to_layer(MODEL_NAME, lp) for lp in LAYER_PERCENTS]
ACTIVE_LAYER = ACT_LAYERS[1]

INJECTION_LAYER: int = 1  # where to inject steering vectors during evaluation
ACT_BATCH_SIZE: int = 128

# Evaluation params
STEERING_COEFFICIENT: float = 1.0
EVAL_BATCH_SIZE: int = 1024
GENERATION_KWARGS = {"do_sample": False, "temperature": 1.0, "max_new_tokens": 40}

# Chat template params
ADD_GENERATION_PROMPT = True
ENABLE_THINKING = False

ADD_RESPONSE_TO_CONTEXT_PROMPT = False

PROMPT_TYPE = "all_direct"
# PROMPT_TYPE = "all_standard"

EXPERIMENTS_DIR: str = "experiments/gender_eval_results"
OUTPUT_JSON_DIR: str = f"{EXPERIMENTS_DIR}/{model_name_str}_open_ended"
OUTPUT_JSON_DIR: str = f"{EXPERIMENTS_DIR}/{model_name_str}_open_ended_{PROMPT_TYPE}{SUFFIX}"

os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
# Optional: save results to disk as JSON
OUTPUT_JSON_TEMPLATE: Optional[str] = f"{OUTPUT_JSON_DIR}/" + "gender_results_open_{lora}.json"

# ========================================
# PROMPT TYPES AND QUESTIONS
# ========================================

WORD_NAMES = [
    "male",
    "female",
]

if PROMPT_TYPE == "all_direct":
    context_prompt_filename = "datasets/gender/gender_direct_test.txt"

    with open(context_prompt_filename, 'r') as f:
        CONTEXT_PROMPTS = [line.strip() for line in f]
elif PROMPT_TYPE == "all_standard":
    context_prompt_filename = "datasets/gender/gender_standard_test.txt"

    with open(context_prompt_filename, 'r') as f:
        CONTEXT_PROMPTS = [line.strip() for line in f]
else:
    raise ValueError(f"Unsupported PROMPT_TYPE: {PROMPT_TYPE}")

VERBALIZER_PROMPTS: list[str] = [
    "What gender is the user?",
    "What gender is most likely being referred to?",
]


# Control output size during dev
MAX_WORDS: Optional[int] = 20  # set to an int to cap, or None for all
if MAX_WORDS is not None:
    CONTEXT_PROMPTS = CONTEXT_PROMPTS[:MAX_WORDS]

# ========================================
# HELPERS
# ========================================


def encode_messages(
    tokenizer: AutoTokenizer,
    message_dicts: list[list[dict[str, str]]],
    add_generation_prompt: bool,
    enable_thinking: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = []
    for source in message_dicts:
        rendered = tokenizer.apply_chat_template(
            source,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        messages.append(rendered)
    inputs_BL = tokenizer(messages, return_tensors="pt", add_special_tokens=False, padding=True).to(device)
    return inputs_BL


def download_hf_folder(repo_id: str, folder_prefix: str, local_dir: str) -> None:
    """
    Download a specific folder from a Hugging Face repo.
    Example:
        download_hf_folder("adamkarvonen/loras", "model_lora_Qwen_Qwen3-8B_evil_claude37/", "model_lora")
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{folder_prefix}*",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {folder_prefix} from {repo_id} into {local_dir}")


def collect_activations_without_lora(
    model: AutoModelForCausalLM,
    submodules: dict,
    inputs_BL: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    model.disable_adapters()
    orig = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )
    model.enable_adapters()
    return orig


def collect_activations_lora_only(
    model: AutoModelForCausalLM,
    submodules: dict,
    inputs_BL: dict[str, torch.Tensor],
) -> dict[int, torch.Tensor]:
    model.enable_adapters()
    lora = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )

    return lora


def collect_activations_lora_and_orig(
    model: AutoModelForCausalLM,
    submodules: dict,
    inputs_BL: dict[str, torch.Tensor],
    act_layers: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    model.enable_adapters()
    lora = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )

    model.disable_adapters()
    orig = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )
    model.enable_adapters()

    diff = {}
    for layer in act_layers:
        diff[layer] = lora[layer] - orig[layer]
        # Quick sanity print
        print(
            f"[collect] layer {layer} - lora sum {lora[layer].sum().item():.2f} - orig sum {orig[layer].sum().item():.2f}"
        )
    return lora, orig, diff


def create_training_data_from_activations(
    acts_BLD_by_layer_dict: dict[int, torch.Tensor],
    context_input_ids: list[int],
    investigator_prompt: str,
    act_layer: int,
    prompt_layer: int,
    tokenizer: AutoTokenizer,
    batch_idx: int = 0,
    left_pad: int = 0,
    base_meta: dict[str, Any] | None = None,
) -> list[TrainingDataPoint]:
    training_data: list[TrainingDataPoint] = []

    # Token-level probes
    for i in range(len(context_input_ids)):
        context_positions_rel = [i]
        context_positions_abs = [left_pad + i]
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
        acts_BD = acts_BLD[context_positions_abs]  # [1, D]
        meta = {"dp_kind": "token", "token_index": i}
        if base_meta is not None:
            meta.update(base_meta)
        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions_rel),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions_rel,
            ds_label="N/A",
            meta_info=meta,
        )
        training_data.append(dp)

    # Full-sequence probes - last 10 tokens, repeat 10 times for stability
    for _ in range(10):
        start_rel = len(context_input_ids) - 10
        context_positions_rel = list(range(start_rel, len(context_input_ids)))
        context_positions_abs = [left_pad + p for p in context_positions_rel]
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
        acts_BD = acts_BLD[context_positions_abs]  # [L, D]
        meta = {"dp_kind": "full_last10"}
        if base_meta is not None:
            meta.update(base_meta)
        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions_rel),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions_rel,
            ds_label="N/A",
            meta_info=meta,
        )
        training_data.append(dp)

    # Full-sequence probes - all tokens, repeat 10 times for stability
    for _ in range(10):
        context_positions_rel = list(range(len(context_input_ids)))
        context_positions_abs = [left_pad + p for p in context_positions_rel]
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]  # [L, D]
        acts_BD = acts_BLD[context_positions_abs]  # [L, D]
        meta = {"dp_kind": "full_all"}
        if base_meta is not None:
            meta.update(base_meta)
        dp = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions_rel),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions_rel,
            ds_label="N/A",
            meta_info=meta,
        )
        training_data.append(dp)

    return training_data


# ========================================
# MAIN
# ========================================
# %%


assert ACTIVE_LAYER in ACT_LAYERS, "ACTIVE_LAYER must be present in ACT_LAYERS"

# Load tokenizer and model
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = load_tokenizer(MODEL_NAME)

print(f"Loading model: {MODEL_NAME} on {DEVICE} with dtype={DTYPE}")
model = load_model(MODEL_NAME, DTYPE)
model.eval()

# Add dummy adapter so peft_config exists
dummy_config = LoraConfig()
model.add_adapter(dummy_config, adapter_name="default")

# %%

# Injection submodule used during evaluation
injection_submodule = get_hf_submodule(model, INJECTION_LAYER)

total_iterations = len(INVESTIGATOR_LORA_PATHS) * len(WORD_NAMES) * len(CONTEXT_PROMPTS) * len(VERBALIZER_PROMPTS)

pbar = tqdm(total=total_iterations, desc="Overall Progress")

for INVESTIGATOR_LORA_PATH in INVESTIGATOR_LORA_PATHS:
    # Load ACTIVE_LORA_PATH adapter if specified
    if INVESTIGATOR_LORA_PATH not in model.peft_config:
        print(f"Loading ACTIVE LoRA: {INVESTIGATOR_LORA_PATH}")
        model.load_adapter(
            INVESTIGATOR_LORA_PATH,
            adapter_name=INVESTIGATOR_LORA_PATH,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )
    # Results container
    # A single dictionary with a flat "records" list for simple JSONL or DataFrame conversion
    results: dict = {
        "meta": {
            "model_name": MODEL_NAME,
            "dtype": str(DTYPE),
            "device": str(DEVICE),
            "act_layers": ACT_LAYERS,
            "active_layer": ACTIVE_LAYER,
            "injection_layer": INJECTION_LAYER,
            "investigator_lora_path": INVESTIGATOR_LORA_PATH,
            "steering_coefficient": STEERING_COEFFICIENT,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "generation_kwargs": GENERATION_KWARGS,
            "add_generation_prompt": ADD_GENERATION_PROMPT,
            "enable_thinking": ENABLE_THINKING,
            "word_names": WORD_NAMES,
            "context_prompts": CONTEXT_PROMPTS,
            "verbalizer_prompts": VERBALIZER_PROMPTS,
        },
        "records": [],
    }

    for word in WORD_NAMES:
        active_lora_path = ACTIVE_LORA_PATH_TEMPLATE.format(word=word)
        active_lora_name = active_lora_path.replace(".", "_")

        # Load ACTIVE_LORA_PATH adapter if specified
        if active_lora_path not in model.peft_config:
            model.load_adapter(
                active_lora_path,
                adapter_name=active_lora_name,
                is_trainable=False,
                low_cpu_mem_usage=True,
            )

        if ADD_RESPONSE_TO_CONTEXT_PROMPT:
            # Generate one assistant response per context prompt using the active LoRA.
            # Do this once per word, batched for efficiency.
            model.set_adapter(active_lora_name)
            context_to_response: dict[str, str] = {}
            for i in range(0, len(CONTEXT_PROMPTS), EVAL_BATCH_SIZE):
                batch_prompts = CONTEXT_PROMPTS[i : i + EVAL_BATCH_SIZE]
                batch_messages = [[{"role": "user", "content": cp}] for cp in batch_prompts]
                batch_inputs = encode_messages(
                    tokenizer=tokenizer,
                    message_dicts=batch_messages,
                    add_generation_prompt=ADD_GENERATION_PROMPT,
                    enable_thinking=ENABLE_THINKING,
                    device=DEVICE,
                )
                with torch.no_grad():
                    batch_outputs = model.generate(**batch_inputs, **GENERATION_KWARGS)
                # Slice off the prompt length (same for the whole batch due to padding)
                gen_start = batch_inputs["input_ids"].shape[1]
                gen_tokens = batch_outputs[:, gen_start:]
                decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                for cp, out in zip(batch_prompts, decoded):
                    context_to_response[cp] = out.strip()

        # Build all combinations for this word
        all_combos = list(itertools.product(CONTEXT_PROMPTS, VERBALIZER_PROMPTS, [None]))

        # Submodules for the layers we will probe (constant across batches)
        submodules = {layer: get_hf_submodule(model, layer) for layer in ACT_LAYERS}

        # Process in activation batches
        for start in range(0, len(all_combos), ACT_BATCH_SIZE):
            batch = all_combos[start : start + ACT_BATCH_SIZE]

            # Build messages and keep combo metadata
            message_dicts: list[list[dict[str, str]]] = []
            combo_bases: list[dict[str, Any]] = []

            for (context_prompt, verbalizer_prompt, _none_placeholder) in batch:

                # User turn
                test_message: list[dict[str, str]] = [{"role": "user", "content": context_prompt}]

                # Optionally include the assistant reply to the context
                ctx_for_record = context_prompt
                if ADD_RESPONSE_TO_CONTEXT_PROMPT and (context_prompt in context_to_response):
                    assistant_resp = context_to_response[context_prompt]
                    test_message.append({"role": "assistant", "content": assistant_resp})
                    ctx_for_record = ctx_for_record + "\nResponse: " + assistant_resp

                message_dicts.append(test_message)

                investigator_prompt = PREFIX + verbalizer_prompt
                correct_answer = word

                combo_bases.append(
                    {
                        "word": word,
                        "context_prompt": ctx_for_record,
                        "investigator_prompt": investigator_prompt,
                        "ground_truth": correct_answer,
                        "combo_index": start + len(combo_bases),
                    }
                )

            # Tokenize as a batch (left padding is configured in load_tokenizer)
            inputs_BL = encode_messages(
                tokenizer=tokenizer,
                message_dicts=message_dicts,
                add_generation_prompt=ADD_GENERATION_PROMPT,
                enable_thinking=ENABLE_THINKING,
                device=DEVICE,
            )

            # Compute per-sample unpadded input_ids and left pad lengths
            seq_len = int(inputs_BL["input_ids"].shape[1])
            context_input_ids_list: list[list[int]] = []
            left_pads: list[int] = []
            for b_idx in range(len(message_dicts)):
                attn = inputs_BL["attention_mask"][b_idx]
                real_len = int(attn.sum().item())
                left_pad = seq_len - real_len
                left_pads.append(left_pad)
                context_input_ids_list.append(inputs_BL["input_ids"][b_idx, left_pad:].tolist())

            # Collect activations for the whole batch under the active persona
            model.set_adapter(active_lora_name)
            if active_lora_path is None:
                orig_acts = collect_activations_without_lora(model, submodules, inputs_BL)
                act_types = {"orig": orig_acts}
            else:
                # lora_acts, orig_acts, diff_acts = collect_activations_lora_and_orig(
                #     model, submodules, inputs_BL, ACT_LAYERS
                # )
                # act_types = {"orig": orig_acts, "lora": lora_acts, "diff": diff_acts}
                lora_acts = collect_activations_lora_only(model, submodules, inputs_BL)
                act_types = {"lora": lora_acts}

            # Build a single eval batch across all combos and act types
            training_data: list[TrainingDataPoint] = []
            for b_idx in range(len(message_dicts)):
                base = combo_bases[b_idx]
                context_input_ids = context_input_ids_list[b_idx]
                left_pad = left_pads[b_idx]
                for act_key, acts_dict in act_types.items():
                    base_meta = {
                        "word": base["word"],
                        "context_prompt": base["context_prompt"],
                        "investigator_prompt": base["investigator_prompt"],
                        "ground_truth": base["ground_truth"],
                        "combo_index": base["combo_index"],
                        "act_key": act_key,
                        "num_tokens": len(context_input_ids),
                        "context_index_within_batch": b_idx,
                    }
                    training_data.extend(
                        create_training_data_from_activations(
                            acts_BLD_by_layer_dict=acts_dict,
                            context_input_ids=context_input_ids,
                            investigator_prompt=base["investigator_prompt"],
                            act_layer=ACTIVE_LAYER,
                            prompt_layer=ACTIVE_LAYER,
                            tokenizer=tokenizer,
                            batch_idx=b_idx,
                            left_pad=left_pad,
                            base_meta=base_meta,
                        )
                    )

            # Run evaluation once for the giant batch
            responses = run_evaluation(
                eval_data=training_data,
                model=model,
                tokenizer=tokenizer,
                submodule=injection_submodule,
                device=DEVICE,
                dtype=DTYPE,
                global_step=-1,
                lora_path=INVESTIGATOR_LORA_PATH,
                eval_batch_size=EVAL_BATCH_SIZE,
                steering_coefficient=STEERING_COEFFICIENT,
                generation_kwargs=GENERATION_KWARGS,
            )

            # Aggregate responses per combo and act_key
            agg: dict[tuple[str, int], dict[str, Any]] = {}
            for r in responses:
                meta = r.meta_info
                key = (meta["act_key"], int(meta["combo_index"]))
                if key not in agg:
                    agg[key] = {
                        "word": meta["word"],
                        "context_prompt": meta["context_prompt"],
                        "investigator_prompt": meta["investigator_prompt"],
                        "ground_truth": meta["ground_truth"],
                        "num_tokens": int(meta["num_tokens"]),
                        "context_index_within_batch": int(meta["context_index_within_batch"]),
                        "token_responses": [None] * int(meta["num_tokens"]),
                        "full_last10": [],
                        "full_all": [],
                    }
                bucket = agg[key]
                dp_kind = meta["dp_kind"]
                if dp_kind == "token":
                    idx = int(meta["token_index"])
                    bucket["token_responses"][idx] = r.api_response.lower().strip()
                elif dp_kind == "full_last10":
                    bucket["full_last10"].append(r.api_response)
                elif dp_kind == "full_all":
                    bucket["full_all"].append(r.api_response)
                else:
                    raise ValueError(f"Unknown dp_kind: {dp_kind}")

            # Finalize records
            for (act_key, combo_idx), bucket in agg.items():
                correct_answer = bucket["ground_truth"]
                token_responses = bucket["token_responses"]
                num_tok_yes = sum(1 for t in token_responses if t is not None and correct_answer.lower() in t.lower())
                full_sequence_responses = bucket["full_all"]
                num_fin_yes = sum(1 for t in full_sequence_responses if correct_answer.lower() in t.lower())
                mean_gt_containment = num_tok_yes / max(1, bucket["num_tokens"])

                record = {
                    "word": bucket["word"],
                    "context_prompt": bucket["context_prompt"],
                    "act_key": act_key,
                    "investigator_prompt": bucket["investigator_prompt"],
                    "ground_truth": bucket["ground_truth"],
                    "num_tokens": bucket["num_tokens"],
                    "token_yes_count": num_tok_yes,
                    "fullseq_yes_count": num_fin_yes,
                    "mean_ground_truth_containment": mean_gt_containment,
                    "token_responses": token_responses,
                    "full_sequence_responses": full_sequence_responses,
                    "control_token_responses": bucket["full_last10"],
                    "context_input_ids": context_input_ids_list[bucket["context_index_within_batch"]],
                }
                results["records"].append(record)

            pbar.set_postfix({"inv": INVESTIGATOR_LORA_PATH.split("/")[-1][:40], "word": word})
            pbar.update(len(batch))

        model.delete_adapter(active_lora_name)

    # Optionally save to JSON
    if OUTPUT_JSON_TEMPLATE is not None:
        lora_name = INVESTIGATOR_LORA_PATH.split("/")[-1].replace("/", "_").replace(".", "_")
        OUTPUT_JSON = OUTPUT_JSON_TEMPLATE.format(lora=lora_name)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {OUTPUT_JSON}")

    # Small summary
    total_records = len(results["records"])
    if total_records:
        # act_keys = ["lora", "orig", "diff"]
        act_keys = list(set(r["act_key"] for r in results["records"]))

        for key in act_keys:
            print(f"\n{key}")
            contained = []
            for r in results["records"]:
                if r["act_key"] == key:
                    contained.append(r["mean_ground_truth_containment"])

            mean_containment_overall = sum(contained) / len(contained)
            print(f"Summary - records: {len(contained)} - mean containment: {mean_containment_overall:.4f}")
    else:
        print("\nSummary - no records created")

    model.delete_adapter(INVESTIGATOR_LORA_PATH)

pbar.close()

# %%
