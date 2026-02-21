"""
Utilities for confidence-based IDK relabeling of classification training data.

Two thresholds define three zones:
  confidence < filter_threshold  → removed from training entirely
  filter_threshold <= confidence < idk_threshold → relabeled as "I don't know"
  confidence >= idk_threshold   → kept as original Yes/No

The prompt portion (input_ids where labels == -100) is preserved exactly,
so steering vector injection positions remain valid.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import TrainingDataPoint


def relabel_as_idk(
    datapoint: TrainingDataPoint,
    tokenizer: AutoTokenizer,
) -> TrainingDataPoint:
    """
    Relabel a datapoint's response as "I don't know".

    The original ground truth is preserved in meta_info["original_target_output"].
    The prompt portion (all tokens where labels == -100) is preserved exactly.
    """
    new_target = "I don't know"

    # Find prompt/response boundary: first index where labels != -100
    prompt_end_idx = 0
    for i, label in enumerate(datapoint.labels):
        if label != -100:
            prompt_end_idx = i
            break

    prompt_ids = list(datapoint.input_ids[:prompt_end_idx])

    # Re-tokenize the response via chat template to ensure correct end-of-turn tokens.
    dummy_user = [{"role": "user", "content": "x"}]
    dummy_full = dummy_user + [{"role": "assistant", "content": new_target}]

    gen_prompt_ids = tokenizer.apply_chat_template(
        dummy_user, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tokenizer.apply_chat_template(
        dummy_full, tokenize=True, add_generation_prompt=False, enable_thinking=False,
    )

    # Response portion = everything after the generation prompt in the full template
    response_ids = list(full_ids[len(gen_prompt_ids):])

    new_input_ids = prompt_ids + response_ids
    new_labels = [-100] * len(prompt_ids) + response_ids

    new_dp = datapoint.model_copy(deep=True)
    new_dp.meta_info = {**datapoint.meta_info, "original_target_output": datapoint.target_output}
    new_dp.input_ids = new_input_ids
    new_dp.labels = new_labels
    new_dp.target_output = new_target

    return new_dp


def get_confidence_json_path(pt_path: str | Path) -> Path:
    """Derive confidence JSON path from .pt path: foo.pt -> foo_confidence.json"""
    pt_path = Path(pt_path)
    return pt_path.parent / (pt_path.stem + "_confidence.json")


def load_confidence_map(confidence_json_path: str | Path) -> dict[int, float]:
    """
    Load a confidence JSON file and return a mapping of index -> confidence score.

    Skipped entries (e.g. IDK datapoints with confidence=None) are excluded.

    Returns:
        Dict mapping datapoint index (int) to confidence score (float in [0, 1]).
    """
    with open(confidence_json_path, "r") as f:
        data = json.load(f)

    return {
        result["index"]: result["confidence"]
        for result in data["results"]
        if result["confidence"] is not None
    }


def apply_confidence_labels_to_dataset(
    datapoints: list[TrainingDataPoint],
    confidence_map: dict[int, float],
    tokenizer: AutoTokenizer,
    idk_threshold: float = 0.66,
    filter_threshold: float = 0.33,
) -> list[TrainingDataPoint]:
    """
    Apply confidence-based relabeling and filtering to training data.

    Three zones based on confidence score:
      confidence < filter_threshold  → removed (not included in output)
      filter_threshold <= confidence < idk_threshold → relabeled as "I don't know"
      confidence >= idk_threshold   → kept as original Yes/No

    Datapoints not in the confidence map (e.g. existing IDK) are kept unchanged.

    Returns:
        Filtered and relabeled list of TrainingDataPoints.
    """
    result = []
    filtered_count = 0
    idk_count = 0

    for i, dp in enumerate(datapoints):
        if i not in confidence_map:
            result.append(dp)
            continue

        conf = confidence_map[i]
        if conf < filter_threshold:
            filtered_count += 1
            continue  # remove from training
        elif conf < idk_threshold:
            result.append(relabel_as_idk(dp, tokenizer))
            idk_count += 1
        else:
            result.append(dp)

    return result, filtered_count, idk_count
