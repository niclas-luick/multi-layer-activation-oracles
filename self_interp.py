# %%
# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================
import torch
import contextlib
from typing import Callable, List, Dict, Tuple, Optional, Any
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import pickle
from dataclasses import dataclass, field, asdict

import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils


def hardcoded_gemma_2_9b_it_few_shot_example(
    model_name: str, tokenizer: PreTrainedTokenizer, device: torch.device
) -> tuple[torch.Tensor, list[int], list[int]]:
    assert model_name == "google/gemma-2-9b-it"

    demo_features: Dict[int, Tuple[str, str, str]] = {
        1835: (
            "I traveled back in time.",
            "I traveled to Paris.",
            "The word relates to concepts of time travel or moving through time.",
        ),
        5318: (
            "What do we know?",
            "We know everything.",
            "The word relates to inquiry, questioning, or uncertainty.",
        ),
        6941: (
            "I see a dog.",
            "I see a fish.",
            "The word relates to concepts of animals or pets.",
        ),
    }

    question = "Can you write me a sentence that relates to the word 'X' and a similar sentence that does not relate to the word?"

    messages = []
    for demo_idx, (pos_sent, neg_sent, explanation) in demo_features.items():
        messages.extend(
            [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"Positive example: {pos_sent}\n\nNegative example: {neg_sent}\n\nExplanation: {explanation}",
                },
            ]
        )

    # Add the final prompt for the target feature
    messages.extend(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "Positive example:",
            },
        ]
    )

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Find the positions of the placeholder 'X'
    token_ids = tokenizer.encode(formatted_input, add_special_tokens=False)
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    # Ensure we found a placeholder for each demo and the final target
    expected_positions = len(demo_features) + 1
    assert len(positions) == expected_positions, (
        f"Expected to find {expected_positions} 'X' placeholders, but found {len(positions)}."
    )

    tokenized_input = tokenizer(
        formatted_input, return_tensors="pt", add_special_tokens=False
    ).to(device)

    few_shot_indices = [demo_idx for demo_idx in demo_features.keys()]

    return tokenized_input.input_ids, positions, few_shot_indices


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================


@dataclass
class SelfInterpConfig:
    """Configuration settings for the script."""

    # --- Foundational Settings ---
    model_name: str = "google/gemma-2-9b-it"

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str = "google/gemma-scope-9b-it-res"
    sae_layer: int = 9
    sae_width: int = 16  # For loading the correct max acts file
    layer_percent: int = 25  # For loading the correct max acts file
    context_length: int = 32
    test_set_size: int = 1000

    sae_filename: str = field(init=False)

    # --- Experiment Settings ---
    num_features_to_run: int = 200  # How many random features to analyze

    random_seed: int = 42  # For reproducible feature selection
    results_filename: str = "self_interp_results.pkl"
    use_decoder_vectors: bool = True

    # Use a default_factory for mutable types like dicts
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 200,
        }
    )

    features_to_explain: list[int] = field(default_factory=list)
    steering_coefficient: float = 2.0

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        self.sae_filename = f"layer_{self.sae_layer}/width_16k/average_l0_88/params.npz"


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def load_sae_and_model(
    cfg: SelfInterpConfig, dtype: torch.dtype, device: torch.device
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, jumprelu_sae.JumpReluSAE]:
    """Loads the model, tokenizer, and SAE from Hugging Face."""
    print(f"Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print(f"Loading SAE for layer {cfg.sae_layer} from {cfg.sae_repo_id}...")
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=cfg.sae_repo_id,
        filename=cfg.sae_filename,
        layer=cfg.sae_layer,
        model_name=cfg.model_name,
        device=device,
        dtype=dtype,
    )

    print("Model, tokenizer, and SAE loaded successfully.")
    return model, tokenizer, sae


def build_few_shot_prompt(
    few_shot_examples: list[dict],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """
    Constructs a few-shot prompt for generating feature explanations.

    Args:
        demo_features: A dictionary mapping feature indices to their demo explanations.
        target_feature_idx: The index of the new feature to explain.
        tokenizer: The model's tokenizer.
        device: The torch device to place tensors on.

    Returns:
        A tuple containing the tokenized input IDs and the positions of the 'X'
        placeholders where activations should be steered.
    """
    question = "Can you write me a sentence that relates to the word 'X' and a similar sentence that does not relate to the word?"

    messages = []
    for example in few_shot_examples:
        pos_sent = example["original_sentence"]
        neg_sent = example["rewritten_sentence"]
        explanation = example["explanation"]

        messages.extend(
            [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": f"Positive example: {pos_sent}\n\nNegative example: {neg_sent}\n\nExplanation: {explanation}",
                },
            ]
        )

    # Add the final prompt for the target feature
    messages.extend(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "Positive example:",
            },
        ]
    )

    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Find the positions of the placeholder 'X'
    token_ids = tokenizer.encode(formatted_input, add_special_tokens=False)
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    # Ensure we found a placeholder for each demo and the final target
    expected_positions = len(few_shot_examples) + 1
    assert len(positions) == expected_positions, (
        f"Expected to find {expected_positions} 'X' placeholders, but found {len(positions)}."
    )

    tokenized_input = tokenizer(
        formatted_input, return_tensors="pt", add_special_tokens=False
    ).to(device)

    return tokenized_input.input_ids, positions


def parse_generated_explanation(generated_text: str) -> Optional[Dict[str, str]]:
    """Parses the model's output to extract positive/negative sentences and the explanation."""
    try:
        # Clean up the start of the text
        if generated_text.startswith("Positive example:"):
            generated_text = generated_text.replace("Positive example:", "", 1).strip()

        pos_sent_end = generated_text.find("\n\n")
        pos_sent = generated_text[:pos_sent_end].strip()

        neg_sent_start_tag = "Negative example:"
        neg_sent_start = generated_text.find(neg_sent_start_tag) + len(
            neg_sent_start_tag
        )
        neg_sent_end = generated_text.find("\n\n", neg_sent_start)
        neg_sent = generated_text[neg_sent_start:neg_sent_end].strip()

        explanation_start_tag = "Explanation:"
        explanation_start = generated_text.find(explanation_start_tag) + len(
            explanation_start_tag
        )
        explanation = generated_text[explanation_start:].strip()

        return {
            "positive_sentence": pos_sent,
            "negative_sentence": neg_sent,
            "explanation": explanation,
        }
    except Exception as e:
        print(f"Failed to parse generated text: {e}")
        print("--- Full Text ---")
        print(generated_text)
        print("-----------------")
        return None


def get_paired_activations(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    pos_input_str: str,
    neg_input_str: str,
    feature_idx: int,
):
    """
    Calculates and prints the SAE feature activations for a pair of sentences.

    This helps verify if a feature responds more strongly to the positive example,
    as predicted by the model's generated explanation.
    """

    def tokenize_input(text: str):
        # We use a simple user role template for activation checking
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )

    pos_tokens = tokenize_input(pos_input_str)
    neg_tokens = tokenize_input(neg_input_str)

    with torch.no_grad():
        pos_acts_BLD = model_utils.collect_activations(model, submodule, pos_tokens)
        neg_acts_BLD = model_utils.collect_activations(model, submodule, neg_tokens)

        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)
        encoded_neg_acts_BLF = sae.encode(neg_acts_BLD)

    pos_feature_acts = encoded_pos_acts_BLF[0, :, feature_idx]
    neg_feature_acts = encoded_neg_acts_BLF[0, :, feature_idx]

    pos_feature_acts[0] = 0
    neg_feature_acts[0] = 0

    print(f"\n--- Activation Verification for Feature {feature_idx} ---")
    print(f"Positive Sentence: '{pos_input_str}'")
    print(
        f"Max Activation: {pos_feature_acts.max():.4f}, mean: {pos_feature_acts.mean():.4f}"
    )

    print(f"\nNegative Sentence: '{neg_input_str}'")
    print(
        f"Max Activation: {neg_feature_acts.max():.4f}, mean: {neg_feature_acts.mean():.4f}"
    )
    print("-" * (20 + len(str(feature_idx))))


# ==============================================================================
# 3. HOOKING MECHANISM FOR ACTIVATION STEERING
# ==============================================================================


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    vectors: list[list[torch.Tensor]],  # [B][K, d_model]  or [K, d_model] if B==1
    positions: list[list[int]],  # [B][K]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b][k]  – feature vector to inject for batch b, slot k
    • coeffs[b][k]   – scale factor (usually small, e.g. 0.3)
    • positions[b][k]– token index (0-based, within prompt only)
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BKD = torch.stack([torch.stack(v) for v in vectors])  # (B, K, d)
    pos_BK = torch.tensor(positions, dtype=torch.long)  # (B, K)

    B, K, d_model = vec_BKD.shape
    assert pos_BK.shape == (B, K)

    vec_BKD = vec_BKD.to(device, dtype)
    pos_BK = pos_BK.to(device)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        # Safety: make sure every position is inside current sequence
        if (pos_BK >= L).any():
            bad = pos_BK[pos_BK >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B1 = torch.arange(B, device=device).unsqueeze(1)  # (B, 1) → (B, K)
        orig_BKD = resid_BLD[batch_idx_B1, pos_BK]  # (B, K, d)
        norms_BK1 = orig_BKD.norm(dim=-1, keepdim=True)  # (B, K, 1)

        # ---- build steered vectors ----
        steered_BKD = (
            torch.nn.functional.normalize(vec_BKD, dim=-1)
            * norms_BK1
            * steering_coefficient
        )  # (B, K, d)

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B1, pos_BK] = steered_BKD

        return (resid_BLD, *rest)

    return hook_fn


def collect_few_shot_examples(results: dict, max_examples: int = 5) -> list[dict]:
    examples = []

    for i, feature_result in enumerate(results["results"]):
        for sentence_data in feature_result["sentence_data"]:
            if sentence_data["original_max_activation"] < 1.0:
                continue

            max_act_ratio = (
                sentence_data["rewritten_max_activation"]
                / (sentence_data["original_max_activation"])
            )

            if max_act_ratio < 0.1 and sentence_data["sentence_distance"] < 0.1:
                example = {
                    "original_sentence": sentence_data["original_sentence"],
                    "rewritten_sentence": sentence_data["rewritten_sentence"],
                    "explanation": feature_result["explanation"],
                    "feature_idx": feature_result["feature_idx"],
                }

                examples.append(example)

                break

        if len(examples) >= max_examples:
            break

    return examples


def main():
    """Main script logic."""
    cfg = SelfInterpConfig()

    cfg.num_features_to_run = 10
    cfg.steering_coefficient = 2.0

    print(asdict(cfg))
    dtype = torch.bfloat16
    device = torch.device("cuda")

    model, tokenizer, sae = load_sae_and_model(cfg, dtype, device)
    submodule = model_utils.get_submodule(model, cfg.sae_layer)

    api_data_filename = "contrastive_rewriting_results_10k.pkl"

    with open(api_data_filename, "rb") as f:
        api_data = pickle.load(f)

    few_shot_examples = collect_few_shot_examples(api_data, max_examples=5)
    input_ids, positions = build_few_shot_prompt(few_shot_examples, tokenizer, device)

    few_shot_indices = [example["feature_idx"] for example in few_shot_examples]

    # input_ids, positions, few_shot_indices = hardcoded_gemma_2_9b_it_few_shot_example(
    #     cfg.model_name, tokenizer, device
    # )

    test_features = api_data["results"][-cfg.test_set_size :]

    for feature_idx in test_features:
        assert feature_idx["feature_idx"] not in few_shot_indices
        cfg.features_to_explain.append(feature_idx["feature_idx"])
        if len(cfg.features_to_explain) >= cfg.num_features_to_run:
            break

    assert len(cfg.features_to_explain) == cfg.num_features_to_run

    cfg.features_to_explain = [7159, 14070, 13115]

    for target_feature_idx in cfg.features_to_explain:
        print(f"\n{'=' * 20} EXPLAINING FEATURE {target_feature_idx} {'=' * 20}")

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        all_feature_indices = few_shot_indices + [target_feature_idx]

        if cfg.use_decoder_vectors:
            all_feature_vectors = [sae.W_dec[i] for i in all_feature_indices]
        else:
            all_feature_vectors = [sae.W_enc[:, i] for i in all_feature_indices]

        all_feature_vectors = [all_feature_vectors]
        positions = [positions]

        # 3. Create and apply the activation steering hook
        hook_fn = get_activation_steering_hook(
            vectors=all_feature_vectors,
            positions=positions,
            steering_coefficient=cfg.steering_coefficient,
            device=device,
            dtype=dtype,
        )

        # 4. Generate the explanation with activation steering
        print("Generating explanation with activation steering...")
        with add_hook(submodule, hook_fn):
            output_ids = model.generate(input_ids, **cfg.generation_kwargs)

        # Decode only the newly generated tokens
        generated_tokens = output_ids[0, input_ids.shape[1] :]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("\n--- Raw Model Output ---")
        print(decoded_output)
        print("------------------------")

        # 5. Parse the output and verify activations
        parsed_result = parse_generated_explanation(decoded_output)
        if parsed_result:
            print("\n--- Parsed Explanation ---")
            for key, value in parsed_result.items():
                print(f"{key.replace('_', ' ').capitalize()}: {value}")
            print("--------------------------")

            get_paired_activations(
                model=model,
                tokenizer=tokenizer,
                submodule=submodule,
                sae=sae,
                pos_input_str=parsed_result["positive_sentence"],
                neg_input_str=parsed_result["negative_sentence"],
                feature_idx=target_feature_idx,
            )


if __name__ == "__main__":
    main()
