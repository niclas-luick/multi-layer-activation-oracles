import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Any

import torch
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.dataset_classes.classification_dataset_manager as classification_dataset_manager
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, BaseDatasetConfig, DatasetLoaderConfig
from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import (
    layer_percent_to_layer,
    load_model,
    load_tokenizer,
    set_seed,
)
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    create_training_datapoint,
)
from nl_probes.utils.eval import run_evaluation


IDK_TOKEN = "I don't know"


@dataclass
class ClassificationDatasetConfig(BaseDatasetConfig):
    classification_dataset_name: str
    num_qa_per_sample: int = 3
    min_end_offset: int = -3
    max_end_offset: int = -5
    max_window_size: int = 20
    min_window_size: int = 1
    # Position resampling: creates multiple datapoints per sample with different token positions
    position_resample_repeats: int = 1
    # IDK mixing: mix contexts and questions from different datasets
    # If enabled, classification_dataset_name should be a comma-separated list of datasets
    # e.g., "sst2,ag_news,md_gender" - or use "all" for all available datasets
    enable_idk_mixing: bool = False
    # Target ratio for IDK samples (0.33 means 1/3 yes, 1/3 no, 1/3 IDK)
    idk_ratio: float = 0.33


class ClassificationDatasetLoader(ActDatasetLoader):
    def __init__(self, dataset_config: DatasetLoaderConfig, model_kwargs: dict[str, Any] | None = None, model=None):
        super().__init__(dataset_config)

        self.dataset_params: ClassificationDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Classification dataset name gets overridden here"

        self.dataset_config.dataset_name = f"classification_{self.dataset_params.classification_dataset_name}"
        self.model_kwargs = model_kwargs
        self.model = model

        self.act_layers = [
            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)
            for layer_percent in self.dataset_config.layer_percents
        ]

        assert self.dataset_params.min_end_offset < 0, "Min end offset must be negative"
        assert self.dataset_params.max_end_offset < 0, "Max end offset must be negative"
        assert self.dataset_params.max_end_offset <= self.dataset_params.min_end_offset, (
            "Max end offset must be less than or equal to min end offset"
        )
        assert self.dataset_params.max_window_size > 0, "Max window size must be positive"
        assert self.dataset_params.position_resample_repeats >= 1, "Position resample repeats must be >= 1"

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        idk_metadata: list[IDKSampleMetadata] = []
        
        if self.dataset_params.enable_idk_mixing:
            # IDK mixing mode: load from multiple datasets and create mixed samples
            train_datapoints, test_datapoints, idk_metadata = get_classification_datapoints_with_idk(
                self.dataset_params.classification_dataset_name,
                self.dataset_params.num_qa_per_sample,
                self.dataset_config.num_train,
                self.dataset_config.num_test,
                self.dataset_config.seed,
                idk_ratio=self.dataset_params.idk_ratio,
            )
            
            # Save IDK metadata for later inspection
            if idk_metadata:
                metadata_path = self.dataset_path / "idk_metadata.json"
                metadata_dicts = [asdict(m) for m in idk_metadata]
                with open(metadata_path, "w") as f:
                    json.dump(metadata_dicts, f, indent=2)
                print(f"Saved IDK metadata ({len(idk_metadata)} samples) to {metadata_path}")
        else:
            # Standard mode: single dataset
            train_datapoints, test_datapoints = get_classification_datapoints(
                self.dataset_params.classification_dataset_name,
                self.dataset_params.num_qa_per_sample,
                self.dataset_config.num_train,
                self.dataset_config.num_test,
                self.dataset_config.seed,
            )

        for split in self.dataset_config.splits:
            if split == "train":
                datapoints = train_datapoints
                save_acts = self.dataset_config.save_acts
            else:
                datapoints = test_datapoints
                save_acts = True

            data = create_vector_dataset(
                datapoints,
                tokenizer,
                self.dataset_config.model_name,
                self.dataset_config.batch_size,
                self.act_layers,
                min_end_offset=self.dataset_params.min_end_offset,
                max_end_offset=self.dataset_params.max_end_offset,
                max_window_size=self.dataset_params.max_window_size,
                min_window_size=self.dataset_params.min_window_size,
                save_acts=save_acts,
                datapoint_type=self.dataset_config.dataset_name,
                debug_print=False,
                model_kwargs=self.model_kwargs,
                model=self.model,
                position_resample_repeats=self.dataset_params.position_resample_repeats,
            )

            self.save_dataset(data, split)


class ClassificationDatapoint(BaseModel):
    activation_prompt: str
    classification_prompt: str
    target_response: str
    ds_label: str | None


def get_classification_datapoints_from_context_qa_examples(
    examples: list[classification_dataset_manager.ContextQASample],
) -> list[ClassificationDatapoint]:
    datapoints = []
    for example in examples:
        for question, answer in zip(example.questions, example.answers, strict=True):
            question = f"Answer with 'Yes' or 'No' only. {question}"
            datapoint = ClassificationDatapoint(
                activation_prompt=example.context,
                classification_prompt=question,
                target_response=answer,
                ds_label=example.ds_label,
            )
            datapoints.append(datapoint)

    return datapoints


def get_classification_datapoints(
    dataset_name: str,
    num_qa_per_sample: int,
    train_examples: int,
    test_examples: int,
    random_seed: int,
) -> tuple[list[ClassificationDatapoint], list[ClassificationDatapoint]]:
    set_seed(random_seed)
    all_examples = classification_dataset_manager.get_samples_from_groups(
        [dataset_name],
        num_qa_per_sample,
    )

    random.shuffle(all_examples)

    assert len(all_examples) >= train_examples + test_examples, "Not enough examples to split"
    train_examples = all_examples[:train_examples]
    test_examples = all_examples[-test_examples:]

    train_datapoints = get_classification_datapoints_from_context_qa_examples(train_examples)
    test_datapoints = get_classification_datapoints_from_context_qa_examples(test_examples)

    return train_datapoints, test_datapoints


# IID datasets: Used for training (in-distribution)
IID_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
]

# OOD datasets: Held out for evaluation (out-of-distribution)
OOD_DATASETS = [
    "ag_news",
    "language_identification",
    "singular_plural",
    "engels_headline_istrump",
    "engels_headline_isobama",
    "engels_headline_ischina",
    "engels_hist_fig_ismale",
]

# For IDK mixing, only use IID datasets to avoid data leakage to OOD eval
IDK_COMPATIBLE_DATASETS = IID_DATASETS


@dataclass
class IDKSampleMetadata:
    """Metadata for a single IDK sample for debugging/analysis."""
    context_dataset: str
    question_dataset: str
    context_text: str
    question_text: str
    original_answer: str  # What the answer would have been if context matched


def create_idk_samples_from_different_datasets(
    examples_by_dataset: dict[str, list[classification_dataset_manager.ContextQASample]],
    num_idk_samples: int,
    random_seed: int,
) -> tuple[list[ClassificationDatapoint], list[IDKSampleMetadata]]:
    """
    Create "I don't know" samples by mixing contexts from one dataset
    with questions from a different dataset.
    
    The idea: if the model sees activations from processing a sentiment context,
    but is asked a question about named entities, it should say "I don't know"
    because the activations don't contain the relevant information.
    
    Returns:
        Tuple of (datapoints, metadata) where metadata tracks the mixing for debugging.
    """
    set_seed(random_seed)
    
    dataset_names = list(examples_by_dataset.keys())
    assert len(dataset_names) >= 2, "Need at least 2 datasets for IDK mixing"
    
    idk_datapoints = []
    idk_metadata = []
    
    for _ in range(num_idk_samples):
        # Pick two different datasets
        ds_context, ds_question = random.sample(dataset_names, 2)
        
        # Pick random examples from each
        context_example = random.choice(examples_by_dataset[ds_context])
        question_example = random.choice(examples_by_dataset[ds_question])
        
        # Take the context from one dataset
        activation_prompt = context_example.context
        
        # Take a random question from the other dataset
        question_idx = random.randint(0, len(question_example.questions) - 1)
        question = question_example.questions[question_idx]
        original_answer = question_example.answers[question_idx]
        
        # Format the question with IDK instruction
        classification_prompt = f"Answer with 'Yes', 'No', or 'I don't know'. {question}"
        
        idk_datapoint = ClassificationDatapoint(
            activation_prompt=activation_prompt,
            classification_prompt=classification_prompt,
            target_response=IDK_TOKEN,
            ds_label=f"idk_{ds_context}_context_{ds_question}_question",
        )
        idk_datapoints.append(idk_datapoint)
        
        # Track metadata for debugging
        idk_metadata.append(IDKSampleMetadata(
            context_dataset=ds_context,
            question_dataset=ds_question,
            context_text=activation_prompt[:200],  # Truncate for storage
            question_text=question,
            original_answer=original_answer,
        ))
    
    return idk_datapoints, idk_metadata


def get_classification_datapoints_with_idk(
    dataset_names_str: str,
    num_qa_per_sample: int,
    train_examples: int,
    test_examples: int,
    random_seed: int,
    idk_ratio: float = 0.33,
) -> tuple[list[ClassificationDatapoint], list[ClassificationDatapoint], list[IDKSampleMetadata]]:
    """
    Load classification datapoints with IDK mixing.
    
    Args:
        dataset_names_str: Comma-separated list of dataset names, or special keywords:
            - "all": All IID datasets (safe for training, OOD held out for eval)
            - "iid": Same as "all" - only IID datasets
            - "ood": Only OOD datasets (for testing purposes only!)
            - "all_datasets": Both IID and OOD (not recommended for training!)
        num_qa_per_sample: Number of Q&A pairs per sample
        train_examples: Number of training examples (before IDK augmentation)
        test_examples: Number of test examples (before IDK augmentation)
        random_seed: Random seed for reproducibility
        idk_ratio: Target ratio of IDK samples (default 0.33 for ~1/3 split)
    
    Returns:
        Tuple of (train_datapoints, test_datapoints, idk_metadata) with approximately:
        - 1/3 Yes answers
        - 1/3 No answers  
        - 1/3 "I don't know" answers
        The idk_metadata list contains details about which datasets were mixed for each IDK sample.
    
    Note:
        For training, use "all" or "iid" to ensure OOD datasets remain held out.
        The OOD datasets (ag_news, language_identification, singular_plural, 
        engels_* datasets) should only be used for evaluation.
    """
    set_seed(random_seed)
    
    # Parse dataset names
    keyword = dataset_names_str.lower().strip()
    if keyword in ("all", "iid"):
        dataset_names = IID_DATASETS
        print(f"Using IID datasets for IDK mixing (OOD held out for eval)")
    elif keyword == "ood":
        dataset_names = OOD_DATASETS
        print(f"WARNING: Using OOD datasets - only use for evaluation, not training!")
    elif keyword == "all_datasets":
        dataset_names = IID_DATASETS + OOD_DATASETS
        print(f"WARNING: Using ALL datasets including OOD - not recommended for training!")
    else:
        dataset_names = [name.strip() for name in dataset_names_str.split(",")]
        # Warn if OOD datasets are included
        ood_used = [d for d in dataset_names if d in OOD_DATASETS]
        if ood_used:
            print(f"WARNING: Including OOD datasets {ood_used} - ensure this is intentional!")
    
    print(f"Loading datasets for IDK mixing: {dataset_names}")
    
    # Load examples from each dataset
    examples_by_dataset: dict[str, list[classification_dataset_manager.ContextQASample]] = {}
    for ds_name in dataset_names:
        examples = classification_dataset_manager.get_samples_from_groups(
            [ds_name],
            num_qa_per_sample,
        )
        examples_by_dataset[ds_name] = examples
        print(f"  {ds_name}: {len(examples)} examples")
    
    # Combine all examples for Yes/No samples
    all_examples = []
    for examples in examples_by_dataset.values():
        all_examples.extend(examples)
    random.shuffle(all_examples)
    
    # Calculate how many Yes/No vs IDK samples we need
    # If idk_ratio = 0.33, we want 2/3 Yes+No and 1/3 IDK
    # So for N total samples, we need N * (1 - idk_ratio) Yes+No and N * idk_ratio IDK
    total_train = int(train_examples / (1 - idk_ratio)) if train_examples > 0 else 0
    total_test = int(test_examples / (1 - idk_ratio)) if test_examples > 0 else 0
    
    num_yes_no_train = train_examples
    num_yes_no_test = test_examples
    num_idk_train = total_train - num_yes_no_train
    num_idk_test = total_test - num_yes_no_test
    
    print(f"Target split - Train: {num_yes_no_train} Yes/No + {num_idk_train} IDK = {total_train} total")
    print(f"Target split - Test: {num_yes_no_test} Yes/No + {num_idk_test} IDK = {total_test} total")
    
    # Split Yes/No examples into train and test
    assert len(all_examples) >= num_yes_no_train + num_yes_no_test, (
        f"Not enough examples: have {len(all_examples)}, need {num_yes_no_train + num_yes_no_test}"
    )
    train_yes_no_examples = all_examples[:num_yes_no_train]
    test_yes_no_examples = all_examples[-num_yes_no_test:] if num_yes_no_test > 0 else []
    
    # Convert to datapoints (with modified instruction for 3-way classification)
    train_datapoints = []
    test_datapoints = []
    
    for example in train_yes_no_examples:
        for question, answer in zip(example.questions, example.answers, strict=True):
            question = f"Answer with 'Yes', 'No', or 'I don't know'. {question}"
            datapoint = ClassificationDatapoint(
                activation_prompt=example.context,
                classification_prompt=question,
                target_response=answer,
                ds_label=example.ds_label,
            )
            train_datapoints.append(datapoint)
    
    for example in test_yes_no_examples:
        for question, answer in zip(example.questions, example.answers, strict=True):
            question = f"Answer with 'Yes', 'No', or 'I don't know'. {question}"
            datapoint = ClassificationDatapoint(
                activation_prompt=example.context,
                classification_prompt=question,
                target_response=answer,
                ds_label=example.ds_label,
            )
            test_datapoints.append(datapoint)
    
    # Create IDK samples and collect metadata
    all_idk_metadata: list[IDKSampleMetadata] = []
    
    if num_idk_train > 0:
        train_idk, train_idk_metadata = create_idk_samples_from_different_datasets(
            examples_by_dataset, num_idk_train, random_seed
        )
        train_datapoints.extend(train_idk)
        all_idk_metadata.extend(train_idk_metadata)
    
    if num_idk_test > 0:
        test_idk, test_idk_metadata = create_idk_samples_from_different_datasets(
            examples_by_dataset, num_idk_test, random_seed + 1  # Different seed for test
        )
        test_datapoints.extend(test_idk)
        all_idk_metadata.extend(test_idk_metadata)
    
    # Shuffle to mix Yes/No/IDK
    random.shuffle(train_datapoints)
    random.shuffle(test_datapoints)
    
    # Print final distribution
    def count_distribution(datapoints):
        yes_count = sum(1 for d in datapoints if d.target_response == "Yes")
        no_count = sum(1 for d in datapoints if d.target_response == "No")
        idk_count = sum(1 for d in datapoints if d.target_response == IDK_TOKEN)
        return yes_count, no_count, idk_count
    
    train_yes, train_no, train_idk = count_distribution(train_datapoints)
    test_yes, test_no, test_idk = count_distribution(test_datapoints)
    
    print(f"Final train distribution: Yes={train_yes}, No={train_no}, IDK={train_idk}")
    print(f"Final test distribution: Yes={test_yes}, No={test_no}, IDK={test_idk}")
    
    # Print IDK mixing statistics
    if all_idk_metadata:
        mixing_counts: dict[str, int] = {}
        for meta in all_idk_metadata:
            key = f"{meta.context_dataset} -> {meta.question_dataset}"
            mixing_counts[key] = mixing_counts.get(key, 0) + 1
        print(f"IDK mixing combinations ({len(all_idk_metadata)} total):")
        for combo, count in sorted(mixing_counts.items(), key=lambda x: -x[1]):
            print(f"  {combo}: {count}")
    
    return train_datapoints, test_datapoints, all_idk_metadata


def view_tokens(tokens_L: list[int], tokenizer: AutoTokenizer, offset: int) -> None:
    print(f"Full tokens: {tokenizer.decode(tokens_L)}")
    for i in range(offset - 5, offset + 5):
        if i < len(tokens_L):
            if i == offset:
                print(f"Act token: {tokenizer.decode(tokens_L[i])}")
            else:
                print(f"Token {i}: {tokenizer.decode(tokens_L[i])}")


@torch.no_grad()
def create_vector_dataset(
    datapoints: list[ClassificationDatapoint],
    tokenizer: AutoTokenizer,
    model_name: str,
    batch_size: int,
    act_layers: list[int],
    min_end_offset: int,
    max_end_offset: int,
    max_window_size: int,
    min_window_size: int,
    save_acts: bool,
    datapoint_type: str,
    lora_path: str | None = None,
    debug_print: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    model=None,
    position_resample_repeats: int = 1,
) -> list[TrainingDataPoint]:
    """
    Create vector dataset from classification datapoints.
    
    Args:
        position_resample_repeats: Number of times to resample token positions for each
            datapoint. If > 1, creates multiple TrainingDataPoints per input with
            different randomly sampled token windows. This was used to train -3N models
            (repeats=3) and -6N models (repeats=6).
    """
    assert min_end_offset < 0, "Min end offset must be negative"
    assert max_end_offset < 0, "Max end offset must be negative"
    assert max_end_offset <= min_end_offset, "Max end offset must be less than or equal to min end offset"
    assert position_resample_repeats >= 1, "position_resample_repeats must be >= 1"
    training_data = []

    assert tokenizer.padding_side == "left", "Padding side must be left"
    device = torch.device("cpu")

    if save_acts:
        if model is None:
            if model_kwargs is None:
                model_kwargs = {}
            model = load_model(model_name, torch.bfloat16, **model_kwargs)
        submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
        device = model.device

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    desc = f"Collecting activations (repeats={position_resample_repeats})"
    for i in tqdm(range(0, len(datapoints), batch_size), desc=desc):
        batch_datapoints = datapoints[i : i + batch_size]
        formatted_prompts = []
        for datapoint in batch_datapoints:
            formatted_prompts.append([{"role": "user", "content": datapoint.activation_prompt}])
        tokenized_prompts = tokenizer.apply_chat_template(formatted_prompts, tokenize=False)
        tokenized_prompts = tokenizer(
            tokenized_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        if save_acts:
            acts_BLD_by_layer_dict = collect_activations_multiple_layers(
                model, submodules, tokenized_prompts, None, None
            )

        tokenized_prompts["input_ids"] = tokenized_prompts["input_ids"]
        tokenized_prompts["attention_mask"] = tokenized_prompts["attention_mask"]

        for j in range(len(batch_datapoints)):
            attn_mask_L = tokenized_prompts["attention_mask"][j].bool()
            input_ids_L = tokenized_prompts["input_ids"][j, attn_mask_L]
            L = len(input_ids_L)

            assert L > 0, f"L={L}"

            # Resample token positions multiple times for data augmentation
            for _repeat in range(position_resample_repeats):
                end_offset = random.randint(max_end_offset, min_end_offset)
                end_pos = L + end_offset

                assert end_pos > 0, f"end_pos={end_pos}"

                k = random.randint(min_window_size, max_window_size)
                k = min(k, end_pos + 1)
                assert k > 0, f"k={k}"
                begin_pos = end_pos - k + 1
                positions_K = list(range(begin_pos, end_pos + 1))
                assert len(positions_K) == k

                if debug_print:
                    view_tokens(input_ids_L, tokenizer, positions_K[-1])
                classification_prompt = f"{batch_datapoints[j].classification_prompt}"

                if save_acts is False:
                    acts_KD = None
                else:
                    layer_acts_list = []
                    for layer in act_layers:
                        acts_LD = acts_BLD_by_layer_dict[layer][j, attn_mask_L]
                        acts_K = acts_LD[positions_K]
                        layer_acts_list.append(acts_K)

                    stacked = torch.stack(layer_acts_list, dim=0)
                    acts_KD = stacked.reshape(-1, stacked.shape[-1])
                    assert acts_KD.shape[0] == len(act_layers) * k
                
                training_data_point = create_training_datapoint(
                    datapoint_type=datapoint_type,
                    prompt=classification_prompt,
                    target_response=batch_datapoints[j].target_response,
                    layers=act_layers,
                    num_positions=k,
                    tokenizer=tokenizer,
                    acts_BD=acts_KD,
                    feature_idx=-1,
                    context_input_ids=input_ids_L,
                    context_positions=positions_K,
                    ds_label=batch_datapoints[j].ds_label,
                )
                if training_data_point is None:
                    continue
                training_data.append(training_data_point)

    return training_data


if __name__ == "__main__":
    main_test_size = 250
    classification_datasets = {
        "geometry_of_truth": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "relations": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "sst2": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "md_gender": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "snli": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "ag_news": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "ner": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "tense": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
        "language_identification": {
            "num_train": 0,
            "num_test": main_test_size,
            "splits": ["test"],
        },
        "singular_plural": {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
    }

    lora_paths_with_labels = {
        "checkpoints_act_pretrain_posttrain/final": "SAE + Classification",
        "checkpoints_classification_only_2_epochs/final": "Classification Only",
        None: "Original",
    }

    all_eval_data = {}

    model_name = "Qwen/Qwen3-8B"
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokenizer = load_tokenizer(model_name)

    classification_dataset_loaders: list[ClassificationDatasetLoader] = []
    layer_percents = [25, 50, 75]
    batch_size = 16
    steering_coefficient = 2.0
    hook_layer = 1
    generation_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 10,
    }

    for dataset_name in classification_datasets.keys():
        classification_config = ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
        )

        dataset_config = DatasetLoaderConfig(
            custom_dataset_params=classification_config,
            num_train=classification_datasets[dataset_name]["num_train"],
            num_test=classification_datasets[dataset_name]["num_test"],
            splits=classification_datasets[dataset_name]["splits"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=False,
        )

        classification_dataset_loader = ClassificationDatasetLoader(
            dataset_config=dataset_config,
        )
        classification_dataset_loaders.append(classification_dataset_loader)

    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for dataset_loader in classification_dataset_loaders:
        if "test" in dataset_loader.dataset_config.splits:
            all_eval_data[dataset_loader.dataset_config.dataset_name] = dataset_loader.load_dataset("test")

    model = load_model(model_name, dtype, load_in_8bit=True)
    submodule = get_hf_submodule(model, hook_layer)

    all_results = {}
    for dataset_name in classification_datasets.keys():
        eval_data = all_eval_data[dataset_name]
        all_results[dataset_name] = {}
        for lora_path in lora_paths_with_labels.keys():
            results = run_evaluation(
                eval_data=eval_data,
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                device=device,
                dtype=dtype,
                global_step=-1,
                lora_path=lora_path,
                eval_batch_size=batch_size,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
            )
            all_results[dataset_name][lora_path] = results


#     # %%
#     from pathlib import Path
#     from typing import Any

#     import matplotlib.pyplot as plt

#     def plot_classification_results(
#         all_results: dict[str, dict[str | None, dict[str, Any]]],
#         lora_paths_with_labels: dict[str | None, str],
#         *,
#         save_dir: str | Path | None = None,
#         file_format: str = "png",
#         dpi: int = 150,
#         as_percentage: bool = True,
#         annotate: bool = True,
#     ) -> list[Path]:
#         """
#         Make a bar chart per dataset with accuracy and standard error bars.

#         Args:
#             all_results: mapping like all_results[dataset_name][lora_path] -> result dict
#                         where each result dict has keys like 'p', 'se', 'n', etc.
#             lora_paths_with_labels: maps lora_path (can be None) -> label to show on x-axis
#             save_dir: if set, figures are saved here as <dataset>.<file_format>
#             file_format: e.g. 'png' or 'pdf'
#             dpi: figure DPI when saving
#             as_percentage: show accuracy in percent if True, else 0-1
#             annotate: write value above each bar

#         Returns:
#             List of saved file paths (empty if not saving).
#         """
#         if save_dir is not None:
#             save_dir = Path(save_dir)
#             save_dir.mkdir(parents=True, exist_ok=True)

#         def _slugify(s: str) -> str:
#             return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)

#         saved: list[Path] = []

#         for dataset_name, per_model in all_results.items():
#             # Respect the order given in lora_paths_with_labels, but drop missing entries
#             order = [lp for lp in lora_paths_with_labels.keys() if lp in per_model]
#             if not order:
#                 continue

#             vals: list[float] = []
#             errs: list[float] = []
#             labels: list[str] = []
#             ns: list[int | None] = []

#             for lp in order:
#                 res = per_model[lp]
#                 # Prefer provided p and se; fall back if needed
#                 p = res.get("p")
#                 if p is None and "correct" in res and "n" in res and res["n"]:
#                     p = res["correct"] / res["n"]
#                 if p is None:
#                     raise ValueError(f"Missing accuracy for {dataset_name} / {lp}")

#                 se = res.get("se")
#                 if se is None and "ci_lower" in res and "ci_upper" in res:
#                     # Infer SE from a 95% CI if provided
#                     se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)
#                 if se is None:
#                     se = 0.0

#                 n = res.get("n")

#                 if as_percentage:
#                     vals.append(p * 100.0)
#                     errs.append(se * 100.0)
#                 else:
#                     vals.append(float(p))
#                     errs.append(float(se))

#                 labels.append(lora_paths_with_labels[lp])
#                 ns.append(n)

#             # One figure per dataset
#             fig = plt.figure(figsize=(6.5, 4.2))
#             ax = plt.gca()

#             x = list(range(len(order)))
#             bars = ax.bar(x, vals, yerr=errs, capsize=4)

#             xticklabels = [f"{lab}\n(n={n})" if n is not None else lab for lab, n in zip(labels, ns)]
#             ax.set_xticks(x, xticklabels, rotation=0)

#             ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
#             ax.set_title(dataset_name)
#             ax.set_ylim(0, 100 if as_percentage else 1.0)
#             ax.yaxis.grid(True, linestyle="--", alpha=0.4)

#             if annotate:
#                 for b, v in zip(bars, vals):
#                     ax.text(
#                         b.get_x() + b.get_width() / 2.0,
#                         v,
#                         f"{v:.1f}" + ("%" if as_percentage else ""),
#                         ha="center",
#                         va="bottom",
#                         fontsize=9,
#                     )

#             fig.tight_layout()

#             if save_dir is not None:
#                 out_path = save_dir / f"{_slugify(dataset_name)}.{file_format}"
#                 fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
#                 saved.append(out_path)

#             plt.show()
#             plt.close(fig)

#         return saved

#     plot_classification_results(all_results, lora_paths_with_labels, save_dir=None)

#     # %%
