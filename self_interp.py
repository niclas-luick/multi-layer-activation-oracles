# %%
# ==============================================================================
# 0. IMPORTS AND SETUP
# ==============================================================================
import torch
import contextlib
from typing import Callable, List, Dict, Tuple, Optional
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

# Make sure you have installed the necessary packages:
# pip install torch transformers jaxtyping einops transformers_stream_generator safetensors sentencepiece accelerate
# pip install git+https://github.com/google-deepmind/interp_tools.git
import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.model_utils as model_utils


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    """Configuration settings for the script."""

    # Model and Tokenizer
    MODEL_NAME = "google/gemma-2-9b-it"
    DTYPE = torch.bfloat16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SAE (Sparse Autoencoder)
    SAE_REPO_ID = "google/gemma-scope-9b-it-res"
    SAE_LAYER = 9
    SAE_FILENAME = f"layer_{SAE_LAYER}/width_16k/average_l0_88/params.npz"

    # Generation parameters
    GENERATION_KWARGS = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 100,
    }

    # Features for the few-shot prompt (demos)
    # Format: {feature_index: (positive_example, negative_example, explanation)}
    DEMO_FEATURES: Dict[int, Tuple[str, str, str]] = {
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

    # New features we want the model to explain
    FEATURES_TO_EXPLAIN: List[int] = [7159, 14070, 13115]

    # Steering configuration
    STEERING_COEFFICIENT = 2.0


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def load_sae_and_model(
    cfg: Config,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, jumprelu_sae.JumpReluSAE]:
    """Loads the model, tokenizer, and SAE from Hugging Face."""
    print(f"Loading model: {cfg.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME, device_map="auto", torch_dtype=cfg.DTYPE
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    print(f"Loading SAE for layer {cfg.SAE_LAYER} from {cfg.SAE_REPO_ID}...")
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=cfg.SAE_REPO_ID,
        filename=cfg.SAE_FILENAME,
        layer=cfg.SAE_LAYER,
        model_name=cfg.MODEL_NAME,
        device=cfg.DEVICE,
        dtype=cfg.DTYPE,
    )

    print("Model, tokenizer, and SAE loaded successfully.")
    return model, tokenizer, sae


def build_few_shot_prompt(
    demo_features: Dict[int, Tuple[str, str, str]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
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
    vectors: List[Float[Tensor, "d_model"]],
    coeffs: List[float],
    positions: List[int],
) -> Callable:
    """
    Creates a hook to steer model activations by adding feature vectors.
    This hook only acts during the initial prompt processing pass of `model.generate`.
    """

    def hook_fn(module, input, output):
        # The output of a Gemma layer is a tuple; the first element is the residual stream.
        resid_BLD, *rest = output

        # This condition ensures we only modify the activations during the
        # single forward pass on the prompt, not during token-by-token generation.
        if resid_BLD.shape[1] > 1:
            for vector, coeff, pos in zip(vectors, coeffs, positions):
                # Ensure position is within the current sequence length
                assert pos < resid_BLD.shape[1], (
                    f"Position {pos} is out of bounds for sequence length {resid_BLD.shape[1]}"
                )

                # Get the norm of the original activation
                original_norm = torch.norm(resid_BLD[0, pos])

                # Normalize the feature vector and scale it by the original norm and a coefficient
                # This replaces the original activation with our steered vector
                steered_vector = (
                    torch.nn.functional.normalize(vector, dim=-1)
                    * original_norm
                    * coeff
                )

                # TODO: Figure out batching
                resid_BLD[0, pos] = steered_vector.to(resid_BLD.device, resid_BLD.dtype)

        return (resid_BLD, *rest)

    return hook_fn


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================


def main():
    """Main script logic."""
    cfg = Config()
    model, tokenizer, sae = load_sae_and_model(cfg)
    submodule = model_utils.get_submodule(model, cfg.SAE_LAYER)

    for target_feature_idx in cfg.FEATURES_TO_EXPLAIN:
        print(f"\n{'=' * 20} EXPLAINING FEATURE {target_feature_idx} {'=' * 20}")

        # 1. Build the few-shot prompt
        input_ids, positions = build_few_shot_prompt(
            cfg.DEMO_FEATURES, tokenizer, cfg.DEVICE
        )

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        demo_indices = list(cfg.DEMO_FEATURES.keys())
        all_feature_indices = demo_indices + [target_feature_idx]
        all_feature_vectors = [sae.W_dec[i] for i in all_feature_indices]

        # 3. Create and apply the activation steering hook
        hook_fn = get_activation_steering_hook(
            vectors=all_feature_vectors,
            coeffs=[cfg.STEERING_COEFFICIENT] * len(all_feature_vectors),
            positions=positions,
        )

        # 4. Generate the explanation with activation steering
        print("Generating explanation with activation steering...")
        with add_hook(submodule, hook_fn):
            output_ids = model.generate(input_ids, **cfg.GENERATION_KWARGS)

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
