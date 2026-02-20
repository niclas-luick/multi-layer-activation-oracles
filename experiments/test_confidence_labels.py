"""
Test script for inspecting confidence labels.

Two modes:
  1. Synthetic test (no GPU needed): creates fake TrainingDataPoints, applies
     confidence relabeling, and prints before/after examples.
  2. From JSON (if confidence JSONs exist): loads real confidence data,
     prints sample labels, and plots a histogram of confidence scores.

Usage:
    python experiments/test_confidence_labels.py                     # synthetic test
    python experiments/test_confidence_labels.py --json-dir sft_training_data  # from real data
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_synthetic_test():
    """Create fake TrainingDataPoints and test relabeling (no GPU needed)."""
    import torch
    from nl_probes.utils.common import load_tokenizer
    from nl_probes.utils.dataset_utils import TrainingDataPoint
    from nl_probes.utils.confidence_utils import relabel_with_confidence
    from nl_probes.utils.eval import parse_answer, parse_confidence

    print("=" * 60)
    print("SYNTHETIC CONFIDENCE LABEL TEST")
    print("=" * 60)

    tokenizer = load_tokenizer("Qwen/Qwen3-8B")

    # Build a few realistic TrainingDataPoints
    # IDK datapoints are excluded from confidence labeling
    test_cases = [
        ("Yes", 1.0),
        ("Yes", 0.8),
        ("Yes", 0.5),
        ("No", 0.3),
        ("No", 0.0),
        ("No", 0.7),
    ]

    for original_target, confidence in test_cases:
        user_msg = [{"role": "user", "content": "Answer with 'Yes' or 'No' only. Is this text about science?"}]
        full_msg = user_msg + [{"role": "assistant", "content": original_target}]

        prompt_ids = tokenizer.apply_chat_template(
            user_msg, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            full_msg, tokenize=True, add_generation_prompt=False, enable_thinking=False,
        )

        response_ids = full_ids[len(prompt_ids):]
        labels = [-100] * len(prompt_ids) + list(response_ids)

        dp = TrainingDataPoint(
            datapoint_type="test",
            input_ids=list(full_ids),
            labels=labels,
            layers=[10, 20],
            steering_vectors=torch.randn(4, 128),
            positions=[2, 3, 4, 5],
            feature_idx=-1,
            target_output=original_target,
            context_input_ids=None,
            context_positions=None,
            ds_label="test_ds",
        )

        new_dp = relabel_with_confidence(dp, confidence, tokenizer)

        # Decode the full new response
        prompt_end = next(i for i, l in enumerate(new_dp.labels) if l != -100)
        new_response = tokenizer.decode(new_dp.input_ids[prompt_end:], skip_special_tokens=True)

        # Parse back
        parsed_answer = parse_answer(new_response)
        parsed_conf = parse_confidence(new_response)

        print(f"\n  Original: {dp.target_output!r:20s}  |  Confidence: {confidence:.0%}")
        print(f"  Relabeled target_output: {new_dp.target_output!r}")
        print(f"  Decoded response tokens: {new_response!r}")
        print(f"  parse_answer -> {parsed_answer!r}   parse_confidence -> {parsed_conf}")
        print(f"  Token count: {len(dp.input_ids)} -> {len(new_dp.input_ids)}  "
              f"(+{len(new_dp.input_ids) - len(dp.input_ids)} tokens)")

    print("\n" + "=" * 60)
    print("SYNTHETIC TEST COMPLETE")
    print("=" * 60)


def load_all_confidence_jsons(json_dir: str) -> list[tuple[str, dict]]:
    """Find and load all *_confidence.json files in a directory."""
    results = []
    for json_path in sorted(Path(json_dir).glob("*_confidence.json")):
        with open(json_path) as f:
            data = json.load(f)
        results.append((json_path.name, data))
    return results


def run_json_inspection(json_dir: str):
    """Load real confidence JSONs, print samples, and plot histogram."""
    jsons = load_all_confidence_jsons(json_dir)

    if not jsons:
        print(f"No *_confidence.json files found in {json_dir}/")
        print("Run generate_confidence_labels.py first, then re-run this script.")
        sys.exit(1)

    print("=" * 60)
    print(f"CONFIDENCE LABEL INSPECTION ({len(jsons)} JSON files)")
    print("=" * 60)

    all_confidences = []

    for filename, data in jsons:
        all_results = data["results"]
        # Filter out skipped IDK datapoints
        results = [r for r in all_results if not r.get("skipped", False)]
        skipped = len(all_results) - len(results)
        confidences = [r["confidence"] for r in results]
        all_confidences.extend(confidences)

        mean_c = np.mean(confidences) if confidences else 0
        median_c = np.median(confidences) if confidences else 0
        low = sum(1 for c in confidences if c < 0.5)

        print(f"\n--- {filename} ---")
        print(f"  {len(results)} scored datapoints  |  {skipped} skipped (IDK)")
        print(f"  Mean confidence: {mean_c:.3f}  |  Median: {median_c:.3f}")
        if results:
            print(f"  Below 50%: {low}/{len(results)} ({low/len(results)*100:.1f}%)")
        print(f"  Config: mode={data['stability_config']['mode']}, "
              f"n_samples={data['stability_config']['n_samples']}")

        # Print a few sample labels
        print(f"\n  Sample labels (first 5):")
        for r in results[:5]:
            conf_pct = round(r["confidence"] * 100)
            new_label = f"{r['target_output']}. Confidence: {conf_pct}%"
            print(f"    [{r['index']:4d}] gt={r['ground_truth']:5s}  "
                  f"conf={r['confidence']:.0%}  ->  {new_label!r}")

        # Print lowest-confidence examples
        sorted_by_conf = sorted(results, key=lambda r: r["confidence"])
        print(f"\n  Lowest confidence examples:")
        for r in sorted_by_conf[:5]:
            conf_pct = round(r["confidence"] * 100)
            new_label = f"{r['target_output']}. Confidence: {conf_pct}%"
            preds_summary = f"yes={r['yes_count']} no={r['no_count']} other={r['other_count']}"
            print(f"    [{r['index']:4d}] gt={r['ground_truth']:5s}  "
                  f"conf={r['confidence']:.0%}  preds=[{preds_summary}]  ->  {new_label!r}")

    # Plot histogram over all confidence scores
    print(f"\n{'=' * 60}")
    print(f"HISTOGRAM: {len(all_confidences)} total datapoints across {len(jsons)} files")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram
    ax = axes[0]
    bins = np.arange(0, 1.05, 0.05)
    counts, _, patches = ax.hist(all_confidences, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution (n={len(all_confidences)})")
    ax.axvline(np.mean(all_confidences), color="red", linestyle="--", label=f"Mean={np.mean(all_confidences):.2f}")
    ax.axvline(np.median(all_confidences), color="orange", linestyle="--", label=f"Median={np.median(all_confidences):.2f}")
    ax.legend()

    # Right: cumulative (what fraction of data remains at each threshold)
    ax2 = axes[1]
    thresholds = np.arange(0, 1.01, 0.05)
    coverage = [sum(1 for c in all_confidences if c >= t) / len(all_confidences) for t in thresholds]
    ax2.plot(thresholds, coverage, "b-o", markersize=4)
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Coverage (fraction of data retained)")
    ax2.set_title("Coverage vs. Confidence Threshold")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = Path(json_dir) / "confidence_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHistogram saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and inspect confidence labels")
    parser.add_argument(
        "--json-dir", type=str, default=None,
        help="Directory with *_confidence.json files. If not provided, runs synthetic test.",
    )
    args = parser.parse_args()

    if args.json_dir:
        run_json_inspection(args.json_dir)
    else:
        run_synthetic_test()
