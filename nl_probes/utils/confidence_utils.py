"""
Utilities for confidence-based IDK relabeling of classification training data.

If the oracle's confidence (fraction of N runs matching ground truth) is below
a threshold, the datapoint's response is changed to "I don't know".
High-confidence datapoints are returned unchanged.

The prompt portion (input_ids where labels == -100) is preserved exactly,
so steering vector injection positions remain valid.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import TrainingDataPoint


def relabel_with_confidence(
    datapoint: TrainingDataPoint,
    confidence: float,
    tokenizer: AutoTokenizer,
    threshold: float = 0.5,
) -> TrainingDataPoint:
    """
    If confidence is below threshold, relabel the datapoint as "I don't know".
    Otherwise return the original datapoint unchanged.

    The original ground truth is preserved in meta_info["original_target_output"].

    Args:
        datapoint: The original training data point.
        confidence: Float in [0, 1], e.g. 0.8 for 80%.
        tokenizer: The tokenizer (needed to re-tokenize the new response).
        threshold: Confidence threshold. Below this → "I don't know".

    Returns:
        The original datapoint (if confident) or a new one with "I don't know".
    """
    if confidence >= threshold:
        return datapoint

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
    threshold: float = 0.5,
) -> list[TrainingDataPoint]:
    """
    Apply confidence-based IDK relabeling to a list of TrainingDataPoints.

    Datapoints with confidence below threshold are relabeled as "I don't know".
    Datapoints above threshold or not in the confidence map are unchanged.

    Args:
        datapoints: List of training data points (loaded from .pt file).
        confidence_map: Mapping from index to confidence score.
        tokenizer: Tokenizer for re-encoding the response.
        threshold: Confidence threshold. Below this → "I don't know".

    Returns:
        New list of TrainingDataPoints with IDK relabeling applied.
    """
    relabeled = []
    for i, dp in enumerate(datapoints):
        if i in confidence_map:
            relabeled.append(relabel_with_confidence(dp, confidence_map[i], tokenizer, threshold))
        else:
            relabeled.append(dp)
    return relabeled
