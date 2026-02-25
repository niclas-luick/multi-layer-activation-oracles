"""
PersonaQA Open-Ended Stability Evaluation

Measures prediction stability of the PersonaQA task across N forward
passes per persona-question pair. Supports three modes:

  prompt   – vary the verbalizer prompt (prefix + question paraphrases)
  noise    – add Gaussian noise to steering vectors (fixed prompt)
  combined – vary the prompt AND add noise simultaneously

For each question type (country, food, drink, music, sport, boardgame), we:
1. Load persona data
2. Build N VerbalizerInputInfo per persona (varying prompt and/or noise)
3. Run the verbalizer pipeline for each variant
4. Compute plurality vote and agreement rate per persona

Usage:
    python personaqa_stability_eval.py --question-type country --n-samples 50
    python personaqa_stability_eval.py --mode noise --noise-scale 0.005
    python personaqa_stability_eval.py --mode combined --noise-scale 0.005
    python personaqa_stability_eval.py  # all 6 question types, n=50, prompt mode
"""

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
from collections import Counter
from dataclasses import asdict

import torch
from peft import LoraConfig
from tqdm import tqdm

import nl_probes.base_experiment as base_experiment
from nl_probes.base_experiment import VerbalizerInputInfo, VerbalizerResults
from nl_probes.utils.common import load_model, load_tokenizer

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
VERBALIZER_LORA = "nluick/MLAO-Qwen3-8B-3L-3N"
TARGET_LORA = "adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs"
LAYER_PERCENTS = [25, 50, 75]

PROMPT_TYPES = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

SEGMENT_START = -20
DTYPE = torch.bfloat16

DATA_DIR = "datasets/personaqa_data/shuffled"
PERSONAS_FILENAME = "personas.jsonl"

OUTPUT_DIR = "plots/stability/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Prompt paraphrases
# ============================================================================

PREFIX_PARAPHRASES = [
    "Answer with the correct value only. ",
    "Respond with just the value. ",
    "Give only the answer. ",
    "Reply with the correct value. ",
    "Your answer should be just the value. ",
    "Please respond with the exact value. ",
    "State the answer only. ",
    "Provide only the value as your answer. ",
    "One value answer only. ",
    "Just give me the answer. ",
]

QUESTION_PARAPHRASES = {
    "country": [
        "Which country is this person from?",
        "What country does this person come from?",
        "What is this person's country of origin?",
        "Where is this person from?",
        "Name this person's home country.",
        "What nation is this person from?",
        "In which country does this person live?",
        "What is the country of this individual?",
    ],
    "favorite_food": [
        "What is this person's favorite food?",
        "What food does this person like most?",
        "Name this person's preferred food.",
        "Which food is this person's favorite?",
        "What does this person most enjoy eating?",
        "What is the favorite food of this individual?",
        "Which dish does this person prefer?",
        "What food does this person favor?",
    ],
    "favorite_drink": [
        "What is this person's favorite drink?",
        "What drink does this person like most?",
        "Name this person's preferred drink.",
        "Which drink is this person's favorite?",
        "What does this person most enjoy drinking?",
        "What is the favorite drink of this individual?",
        "Which beverage does this person prefer?",
        "What drink does this person favor?",
    ],
    "favorite_music_genre": [
        "What is this person's favorite music genre?",
        "What music genre does this person like most?",
        "Name this person's preferred music genre.",
        "Which music genre is this person's favorite?",
        "What kind of music does this person enjoy?",
        "What is the favorite music genre of this individual?",
        "Which genre of music does this person prefer?",
        "What type of music does this person favor?",
    ],
    "favorite_sport": [
        "What is this person's favorite sport?",
        "What sport does this person like most?",
        "Name this person's preferred sport.",
        "Which sport is this person's favorite?",
        "What sport does this person most enjoy?",
        "What is the favorite sport of this individual?",
        "Which sport does this person prefer?",
        "What sport does this person favor?",
    ],
    "favorite_boardgame": [
        "What is this person's favorite boardgame?",
        "What boardgame does this person like most?",
        "Name this person's preferred boardgame.",
        "Which boardgame is this person's favorite?",
        "What boardgame does this person most enjoy?",
        "What is the favorite boardgame of this individual?",
        "Which boardgame does this person prefer?",
        "What boardgame does this person favor?",
    ],
}

# Canonical prompts for noise-only mode (one per question type)
CANONICAL_PROMPTS = {
    pt: f"Answer with the correct value only. {QUESTION_PARAPHRASES[pt][0]}"
    for pt in PROMPT_TYPES
}

# ============================================================================
# Helpers
# ============================================================================


def normalize_personaqa_prediction(raw: str) -> str:
    """Clean up verbalizer response for comparison."""
    cleaned = raw.strip().lower()
    cleaned = cleaned.strip(".,!?;:'\"")
    return cleaned


def is_correct_personaqa(prediction: str, ground_truth: str) -> bool:
    """Substring match for multi-word ground truths."""
    return ground_truth.lower() in prediction.lower()


def _compute_plurality_vote(predictions: list[str], ground_truth: str) -> dict:
    """Compute plurality vote stats from a list of predictions."""
    counts = Counter(predictions)
    plurality_word, plurality_count = counts.most_common(1)[0]
    return {
        "plurality_vote": plurality_word,
        "agreement_rate": plurality_count / len(predictions),
        "is_correct": is_correct_personaqa(plurality_word, ground_truth),
        "prediction_counts": dict(counts),
    }


def get_output_path(
    model_name: str,
    verbalizer_lora: str,
    question_type: str,
    n_samples: int,
    mode: str = "prompt",
    noise_scale: float = 0.0,
) -> str:
    """Construct the output JSON path."""
    model_str = model_name.split("/")[-1]
    verbalizer_str = verbalizer_lora.split("/")[-1]
    if mode == "prompt":
        mode_str = "prompt"
    elif mode == "noise":
        mode_str = f"noise{noise_scale}"
    else:  # combined
        mode_str = f"combined{noise_scale}"
    return (
        f"{OUTPUT_DIR}/personaqa_stability_{model_str}_{verbalizer_str}"
        f"_{question_type}_{mode_str}_n{n_samples}.json"
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PersonaQA open-ended stability evaluation")
    parser.add_argument(
        "--question-type", type=str, default=None,
        help="Single question type to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of prompt variants per persona (default: 50)",
    )
    parser.add_argument(
        "--mode", choices=["prompt", "noise", "combined"], default="prompt",
        help="Stability mode: prompt (vary verbalizer), noise (Gaussian on vectors), "
             "combined (both) (default: prompt)",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.005,
        help="Noise std as fraction of steering vector norm (default: 0.005)",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Force re-run even if JSON exists",
    )
    args = parser.parse_args()

    # Determine which question types to run
    types_to_run = [args.question_type] if args.question_type else PROMPT_TYPES

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print(f"Mode: {args.mode}")
    if args.mode in ("noise", "combined"):
        print(f"Noise scale: {args.noise_scale}")
    print(f"Sampling {args.n_samples} per persona-question pair")

    # Load persona data
    data_path = os.path.join(DATA_DIR, PERSONAS_FILENAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")

    with open(data_path) as f:
        persona_data = [json.loads(line) for line in f]
    persona_data.sort(key=lambda x: x["name"])
    print(f"Loaded {len(persona_data)} personas from {data_path}")

    # Check which types need evaluation
    types_to_eval: list[str] = []
    for pt in types_to_run:
        json_path = get_output_path(
            MODEL_NAME, VERBALIZER_LORA, pt,
            args.n_samples, args.mode, args.noise_scale,
        )
        if os.path.exists(json_path) and not args.force_rerun:
            with open(json_path) as f:
                existing = json.load(f)
            s = existing["summary"]
            print(f"[SKIP] {pt}: already exists (acc={s['baseline_accuracy']:.3f}, "
                  f"agreement={s['mean_agreement_rate']:.3f})")
        else:
            types_to_eval.append(pt)
            reason = "force-rerun" if os.path.exists(json_path) else "no existing results"
            print(f"[QUEUE] {pt}: will run ({reason})")

    if not types_to_eval:
        print("\nAll question types already have results. Use --force-rerun to re-run.")
        exit(0)

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)
    model.eval()

    # Add dummy adapter for PEFT compatibility
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Load verbalizer LoRA
    print(f"Loading verbalizer LoRA: {VERBALIZER_LORA}")
    base_experiment.load_lora_adapter(model, VERBALIZER_LORA)

    # Load target LoRA (single LoRA for all question types)
    print(f"Loading target LoRA: {TARGET_LORA}")
    sanitized_target = base_experiment.load_lora_adapter(model, TARGET_LORA)

    # VerbalizerEvalConfig
    generation_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 40,
    }
    effective_noise = args.noise_scale if args.mode in ("noise", "combined") else 0.0
    config = base_experiment.VerbalizerEvalConfig(
        model_name=MODEL_NAME,
        layer_percents=LAYER_PERCENTS,
        activation_input_types=["lora"],
        verbalizer_input_types=["segment"],
        eval_batch_size=512,
        verbalizer_generation_kwargs=generation_kwargs,
        noise_scale=effective_noise,
        segment_repeats=1,
        full_seq_repeats=1,
        segment_start_idx=SEGMENT_START,
    )

    print(f"\n{'=' * 60}")
    print(f"PersonaQA Stability Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Verbalizer: {VERBALIZER_LORA}")
    print(f"Target LoRA: {TARGET_LORA}")
    print(f"Mode: {args.mode}")
    if effective_noise > 0:
        print(f"Noise scale: {effective_noise}")
    print(f"N samples: {args.n_samples}")
    print(f"Question types to evaluate: {types_to_eval}")
    print(f"{'=' * 60}")

    for type_idx, question_type in enumerate(types_to_eval):
        json_path = get_output_path(
            MODEL_NAME, VERBALIZER_LORA, question_type,
            args.n_samples, args.mode, args.noise_scale,
        )

        # Build cross-product of (prefix, question) pairs for this question type
        question_paraphrases = QUESTION_PARAPHRASES[question_type]
        all_pairs = [(p, q) for p in PREFIX_PARAPHRASES for q in question_paraphrases]

        print(f"\n{'=' * 60}")
        print(f"[{type_idx + 1}/{len(types_to_eval)}] Question type: {question_type}")
        print(f"Total unique (prefix, question) pairs: {len(all_pairs)}")
        print(f"Output: {json_path}")
        print(f"{'=' * 60}")

        # Build VerbalizerInputInfo list: N variants per persona
        verbalizer_infos: list[VerbalizerInputInfo] = []
        prompt_variants_per_persona: list[list[str]] = []

        for persona in persona_data:
            persona_name = persona["name"]
            ground_truth = str(persona[question_type])
            context_text = f"My name is {persona_name}."

            if args.mode in ("prompt", "combined"):
                # Varying prompts: sample N (prefix, question) pairs
                if args.n_samples <= len(all_pairs):
                    sampled = random.sample(all_pairs, args.n_samples)
                else:
                    sampled = all_pairs[:]
                    while len(sampled) < args.n_samples:
                        sampled.append(random.choice(all_pairs))
                prompts = [f"{p}{q}" for p, q in sampled]
            else:
                # Noise-only: fixed canonical prompt, N copies
                prompts = [CANONICAL_PROMPTS[question_type]] * args.n_samples

            prompt_variants_per_persona.append(prompts)

            for verbalizer_prompt in prompts:
                info = VerbalizerInputInfo(
                    context_prompt=[{"role": "user", "content": context_text}],
                    verbalizer_prompt=verbalizer_prompt,
                    ground_truth=ground_truth,
                )
                verbalizer_infos.append(info)

        print(f"Total VerbalizerInputInfo: {len(verbalizer_infos)} "
              f"({len(persona_data)} personas x {args.n_samples} variants)")

        # Run verbalizer (batched internally)
        verbalizer_results: list[VerbalizerResults] = base_experiment.run_verbalizer(
            model=model,
            tokenizer=tokenizer,
            verbalizer_prompt_infos=verbalizer_infos,
            verbalizer_lora_path=VERBALIZER_LORA,
            target_lora_path=TARGET_LORA,
            config=config,
            device=device,
        )

        # Group results by persona
        n_samples = args.n_samples
        assert len(verbalizer_results) == len(verbalizer_infos), (
            f"Expected {len(verbalizer_infos)} results, got {len(verbalizer_results)}"
        )
        print(f"Got {len(verbalizer_results)} results")

        stability_results = []

        for persona_idx, persona in enumerate(persona_data):
            start = persona_idx * n_samples
            end = start + n_samples
            persona_results = verbalizer_results[start:end]
            prompt_variants = prompt_variants_per_persona[persona_idx]
            ground_truth = str(persona[question_type])

            predictions = []
            for vr in persona_results:
                raw = vr.segment_responses[0] if vr.segment_responses else ""
                pred = normalize_personaqa_prediction(raw)
                predictions.append(pred)

            stats = _compute_plurality_vote(predictions, ground_truth)
            result = {
                "index": persona_idx,
                "persona_name": persona["name"],
                "ground_truth": ground_truth,
                "majority_vote": stats["plurality_vote"],
                "is_correct": stats["is_correct"],
                "agreement_rate": stats["agreement_rate"],
                "prediction_counts": stats["prediction_counts"],
                "predictions": predictions,
                "prompt_variants": prompt_variants,
            }
            stability_results.append(result)

        # Compute summary
        baseline_accuracy = sum(r["is_correct"] for r in stability_results) / len(stability_results)
        mean_agreement = sum(r["agreement_rate"] for r in stability_results) / len(stability_results)

        print(f"\nBaseline accuracy: {baseline_accuracy:.3f}")
        print(f"Mean agreement rate: {mean_agreement:.3f}")

        # Save results JSON
        results_json = {
            "config": {
                "model_name": MODEL_NAME,
                "verbalizer_lora": VERBALIZER_LORA,
                "target_lora": TARGET_LORA,
                "question_type": question_type,
                "mode": args.mode,
                "noise_scale": effective_noise,
                "n_samples": args.n_samples,
                "n_personas": len(persona_data),
                "layer_percents": LAYER_PERCENTS,
                "verbalizer_eval_config": asdict(config),
            },
            "summary": {
                "baseline_accuracy": baseline_accuracy,
                "mean_agreement_rate": mean_agreement,
                "n_examples": len(stability_results),
            },
            "results": stability_results,
        }

        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved results to {json_path}")

    # Clean up target LoRA
    if sanitized_target in model.peft_config:
        model.delete_adapter(sanitized_target)

    print("\nDone!")
