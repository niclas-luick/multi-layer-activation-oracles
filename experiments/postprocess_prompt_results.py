"""
Post-process prompt stability results JSON to fix ground truth alignment
for label-keyed datasets (sst2, singular_plural, engels_*).

For these datasets, question paraphrases come in opposite-polarity groups
(e.g., "Is this about Trump?" vs "Is this not about Trump?"). The original
evaluation may have mixed both groups without adjusting ground truth.

This script:
1. Loads a results JSON with prompt_variants stored per example
2. Determines which label group each question variant belongs to
3. Normalizes predictions: flips yes/no for opposite-polarity questions
4. Recomputes majority vote, agreement_rate, and is_correct

Usage:
    python postprocess_prompt_results.py <results.json> [--reference-label <label>]

If --reference-label is not specified, the script auto-detects by matching
the most common question against the label groups.
"""

import argparse
import json
import re
import sys
from pathlib import Path


PARAPHRASES_JSON_PATH = "datasets/classification_datasets/paraphrases/question.json"


def load_label_groups(dataset_name: str) -> dict[str, list[str]] | None:
    """Load label-keyed question paraphrases for a dataset."""
    with open(PARAPHRASES_JSON_PATH) as f:
        all_paraphrases = json.load(f)

    if dataset_name.startswith("engels_"):
        subtask = dataset_name[len("engels_"):]
        templates = all_paraphrases.get("engels", {}).get(subtask)
    else:
        templates = all_paraphrases.get(dataset_name)

    if templates is None or not isinstance(templates, dict):
        return None
    return templates


def find_label_group(question: str, label_dict: dict[str, list[str]]) -> str | None:
    """Find which label group a question belongs to."""
    for label, templates in label_dict.items():
        if question in templates:
            return label
    return None


def extract_question_from_variant(variant: str) -> str:
    """Extract the question part from a prompt variant string.

    Prompt variants have format: "Answer with 'Yes' or 'No' only. # Is this about Trump?"
    Returns the part after "# ".
    """
    marker = "# "
    idx = variant.find(marker)
    if idx == -1:
        return variant
    return variant[idx + len(marker):]


def flip_prediction(prediction: str) -> str:
    """Flip a yes/no prediction."""
    if prediction == "yes":
        return "no"
    if prediction == "no":
        return "yes"
    return prediction


def detect_reference_label(
    results: list[dict],
    label_groups: dict[str, list[str]],
) -> str | None:
    """Auto-detect the reference label by finding which group the original question belongs to.

    Heuristic: the original question (used to compute ground_truth) is typically the first
    template in the first label group. We check which group's questions appear and use the
    first group as reference (classification datasets use the affirmative form).
    """
    # Collect all questions used across all examples
    label_counts: dict[str, int] = {label: 0 for label in label_groups}
    for r in results:
        if "prompt_variants" not in r:
            continue
        for variant in r["prompt_variants"]:
            question = extract_question_from_variant(variant)
            group = find_label_group(question, label_groups)
            if group:
                label_counts[group] += 1

    if not any(label_counts.values()):
        return None

    # Use first key in dict as reference (convention: affirmative/positive first)
    first_label = next(iter(label_groups))
    print(f"  Auto-detected reference label: '{first_label}' (first key in question.json)")
    return first_label


def postprocess_results(
    results_data: dict,
    label_groups: dict[str, list[str]],
    reference_label: str,
) -> dict:
    """Post-process results to normalize predictions for opposite-polarity questions."""
    results = results_data["results"]
    fixed_count = 0
    unmatched_count = 0

    for r in results:
        if "prompt_variants" not in r or "predictions" not in r:
            continue

        predictions = r["predictions"]
        variants = r["prompt_variants"]

        if len(predictions) != len(variants):
            print(f"  WARNING: example {r['index']}: predictions ({len(predictions)}) != "
                  f"variants ({len(variants)}), skipping")
            continue

        # Store original predictions as raw_predictions
        r["raw_predictions"] = predictions[:]

        # Normalize each prediction based on its question's label group
        normalized = []
        for pred, variant in zip(predictions, variants):
            question = extract_question_from_variant(variant)
            group = find_label_group(question, label_groups)

            if group is None:
                unmatched_count += 1
                normalized.append(pred)  # Can't determine, keep as-is
            elif group != reference_label:
                normalized.append(flip_prediction(pred))
                fixed_count += 1
            else:
                normalized.append(pred)

        r["predictions"] = normalized

        # Recompute majority vote stats
        n_samples = len(normalized)
        yes_count = sum(1 for p in normalized if p == "yes")
        no_count = sum(1 for p in normalized if p == "no")
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

        r["majority_vote"] = majority_vote
        r["agreement_rate"] = majority_count / n_samples
        r["yes_count"] = yes_count
        r["no_count"] = no_count
        r["other_count"] = other_count
        r["is_correct"] = majority_vote == r["ground_truth"]

    # Recompute summary
    baseline_accuracy = sum(r["is_correct"] for r in results) / len(results)
    mean_agreement = sum(r["agreement_rate"] for r in results) / len(results)
    results_data["summary"]["baseline_accuracy"] = baseline_accuracy
    results_data["summary"]["mean_agreement_rate"] = mean_agreement

    print(f"\n  Normalized {fixed_count} predictions (flipped for opposite-polarity questions)")
    if unmatched_count:
        print(f"  WARNING: {unmatched_count} predictions could not be matched to any label group")
    print(f"  New baseline accuracy: {baseline_accuracy:.3f}")
    print(f"  New mean agreement rate: {mean_agreement:.3f}")

    return results_data


def main():
    parser = argparse.ArgumentParser(description="Post-process prompt stability results for label-keyed datasets")
    parser.add_argument("results_json", help="Path to the results JSON file to fix")
    parser.add_argument("--reference-label", type=str, default=None,
                        help="The label group matching the original question's polarity "
                             "(e.g., 'trump', 'positive'). Auto-detected if not specified.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for fixed JSON (default: overwrite input)")
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_json}")
    with open(args.results_json) as f:
        results_data = json.load(f)

    dataset_name = results_data["config"]["dataset_name"]
    print(f"Dataset: {dataset_name}")
    print(f"Original summary: acc={results_data['summary']['baseline_accuracy']:.3f}, "
          f"agreement={results_data['summary']['mean_agreement_rate']:.3f}")

    # Load label groups
    label_groups = load_label_groups(dataset_name)
    if label_groups is None:
        print(f"Dataset '{dataset_name}' does not have label-keyed paraphrases. Nothing to fix.")
        sys.exit(0)

    print(f"Label groups: {', '.join(f'{k}({len(v)})' for k, v in label_groups.items())}")

    # Determine reference label
    reference_label = args.reference_label
    if reference_label is None:
        reference_label = detect_reference_label(results_data["results"], label_groups)
    if reference_label is None:
        print("ERROR: Could not determine reference label. Use --reference-label to specify.")
        sys.exit(1)
    print(f"Reference label (same polarity as ground_truth): '{reference_label}'")

    # Post-process
    fixed_data = postprocess_results(results_data, label_groups, reference_label)

    # Save
    output_path = args.output or args.results_json
    with open(output_path, "w") as f:
        json.dump(fixed_data, f, indent=2)
    print(f"\nSaved fixed results to {output_path}")


if __name__ == "__main__":
    main()
