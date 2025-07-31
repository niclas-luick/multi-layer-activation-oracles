# %%
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
import os
from huggingface_hub import hf_hub_download

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


def get_feature_activations(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: jumprelu_sae.JumpReluSAE,
    pos_input_str: str,
    feature_idx: int,
    verbose: bool = False,
    ignore_bos: bool = True,
    use_chat_template: bool = True,
) -> torch.Tensor:
    """
    Calculates and prints the SAE feature activations for a pair of sentences.

    This helps verify if a feature responds more strongly to the positive example,
    as predicted by the model's generated explanation.
    """

    def tokenize_input(text: str, use_chat_template: bool):
        # We use a simple user role template for activation checking
        if use_chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )

    pos_tokens = tokenize_input(pos_input_str, use_chat_template)

    with torch.no_grad():
        pos_acts_BLD = model_utils.collect_activations(model, submodule, pos_tokens)

        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    pos_feature_acts = encoded_pos_acts_BLF[0, :, feature_idx]

    if ignore_bos:
        pos_feature_acts[0] = 0

    if verbose:
        print(
            f"Feature {feature_idx} max activation: {pos_feature_acts.max():.4f}, mean: {pos_feature_acts.mean():.4f}"
        )

    return pos_feature_acts


# %%

cfg = Config()
model, tokenizer, sae = load_sae_and_model(cfg)
submodule = model_utils.get_submodule(model, cfg.SAE_LAYER)

# %%

num_features = sae.W_dec.shape[0]


neuronpedia_api_key = "sk-np-LJOpHEsv0ME80ZOXEDDUPHyz9LG220yN7TMQuq6DGWQ0"

# %%


sae_width = 16
layer_percent = 25


def load_acts(
    model_name: str, sae_layer: int, sae_width: int, layer_percent: int
) -> dict[str, torch.Tensor]:
    acts_dir = "max_acts"
    acts_filename = f"acts_{model_name}_layer_{sae_layer}_trainer_{sae_width}_layer_percent_{layer_percent}.pt".replace(
        "/", "_"
    )
    print(acts_filename)
    acts_path = os.path.join(acts_dir, acts_filename)
    if not os.path.exists(acts_path):
        from huggingface_hub import hf_hub_download

        path_to_config = hf_hub_download(
            repo_id="adamkarvonen/sae_max_acts",
            filename=acts_filename,
            force_download=False,
            local_dir=acts_dir,
            repo_type="dataset",
        )

    acts_data = torch.load(acts_path)

    return acts_data


acts_data = load_acts(cfg.MODEL_NAME, cfg.SAE_LAYER, sae_width, layer_percent)

# %%


feature_idx = 1835
ctx_len = 1024

feature_acts = acts_data["max_acts"][feature_idx, :, :ctx_len]
feature_tokens = acts_data["max_tokens"][feature_idx, :, :ctx_len]


def _list_decode(x: torch.Tensor):
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


decoded_tokens = _list_decode(feature_tokens)
for i, decoded_str in enumerate(decoded_tokens):
    if i > 5:
        break
    print("\n")
    print("".join(decoded_str))

idx = 0

first_acts = feature_acts[idx, :]
first_acts[0] = 0

print(f"max activation: {first_acts.max():.4f}, mean: {first_acts.mean():.4f}")

first_str = "".join(decoded_tokens[idx])
acts = get_feature_activations(
    model=model,
    tokenizer=tokenizer,
    submodule=submodule,
    sae=sae,
    pos_input_str=first_str,
    feature_idx=feature_idx,
    verbose=True,
    ignore_bos=True,
    use_chat_template=False,
)

# %%


def get_formatted_activation_sentence(
    feature_acts: torch.Tensor, feature_tokens: torch.Tensor
) -> str:
    assert feature_acts.shape == feature_tokens.shape
    assert len(feature_acts.shape) == 1

    max_idx = torch.argmax(feature_acts)

    feature_strs = _list_decode(feature_tokens)[0]
    assert len(feature_strs) == len(feature_acts)

    print(len(feature_strs))
    print(feature_strs)

    feature_strs[max_idx] = f"<<{feature_strs[max_idx]}>>"

    return "".join(feature_strs)


def make_contrastive_prompt(
    feature_acts: torch.Tensor, feature_tokens: torch.Tensor
) -> str:
    assert feature_acts.shape == feature_tokens.shape
    assert len(feature_acts.shape) == 2

    prompt = """
I have a list of sentences that relate to a particular concept. In each sentence, there will be a maximally activated location, which will be a word contained between delimiters like <<this>>.

I want you to do 2 steps:
1. First, generate a potential explanation for the pattern that represents the concept.
2. Make a minimal rewrite of each sentence that entirely removes the concept, including both the delimited word and other potentially related words.

Please return a properly formatted answer, using the tags <EXPLANATION>, <SENTENCE 1>, <SENTENCE 2>, <END>, etc, and following the example below.

BEGIN EXAMPLE

1. 's what 12 Monkeys is about: time<< travel>>, although it's not as sci-fi related as it appears. Bruce Willis is a time traveler, which makes 12 Monkeys a sci-fi movie, but the entirety of it is not as complex as one would think.

2. PMtraveler - 03 November 2011 01:54 PMBut the simple act of going back to the past changes the past. I mean, you weren’t around in 1820, right? So it changed.
Ok, now I am confused. Jesus (why pick him?) got to the past by using a time<< machine>>.

RESPONSE:

<EXPLANATION>
It appears that the concept of time travel is being discussed. I will rewrite the sentences to remove the concept of time travel.
    
For sentence one, I will replace 'time travel' with 'ocean travel'. I will also replace the movie '12 Monkeys' with 'Adrift', as 12 Monkeys is related to time-travel.
For sentence two, I will replace going back to the past with going to a place, and replace the time machine with traveling.

<SENTENCE 1>
 's what Adrift is about: ocean travel, although it's not as sci-fi related as it appears. Bruce Willis is a ocean traveler, which makes Adrift a sci-fi movie, but the entirety of it is not as complex as one would think.",

<SENTENCE 2>
PMtraveler - 03 November 2011 01:54 PMBut the simple act of going to a place changes that place. I mean, you weren't there before, right? So it changed.
Ok, now I am confused. Jesus (why pick him?) got there by traveling.

<END>

END EXAMPLE

Okay, now here are the real sentences:

{sentences}

RESPONSE:
"""

    formatted_sentences = ""
    for i in range(feature_acts.shape[0]):
        feature_strs = _list_decode(feature_tokens[i])[0]
        feature_strs = feature_strs[1:]  # skip bos
        sentence = "".join(feature_strs)
        formatted_sentences += f"\n<SENTENCE {i}>\n{sentence}\n"

    formatted_prompt = prompt.format(sentences=formatted_sentences)

    return formatted_prompt


# %%

num_sentences = 4
first_prompt = make_contrastive_prompt(
    feature_acts[:num_sentences], feature_tokens[:num_sentences]
)
print(first_prompt)

# %%

feature_idx = 6941

feature_acts = acts_data["max_acts"][feature_idx]
feature_tokens = acts_data["max_tokens"][feature_idx]

num_sentences = 5

new_prompt = make_contrastive_prompt(
    feature_acts[:num_sentences], feature_tokens[:num_sentences]
)
print(new_prompt)


# %%

import interp_tools.api_utils.shared as shared
import interp_tools.api_utils.api_caller as api_caller
import asyncio


def build_prompts(str_prompts: list[str]) -> list[shared.ChatHistory]:
    prompts = []

    for prompt in str_prompts:
        prompts.append(shared.ChatHistory().add_user(prompt))

    return prompts


model_name = "gpt-4o"

caller = api_caller.get_openai_caller("cached_api_calls")

prompts = build_prompts([new_prompt])

try:
    answers = await api_caller.run_api_model(
        model_name,
        prompts,
        caller,
        temperature=1.0,
        max_tokens=2000,
        max_par=100,
    )
finally:
    # Properly close the HTTP client
    caller.client.close()

print(answers)

# %%
# print(new_prompt)
# print("\n\n\n\n")
# print(answers)

orig_sentences = []

for i in range(num_sentences):
    sentence = "".join(_list_decode(feature_tokens[i])[0])
    print(sentence)
    orig_sentences.append(sentence)

# print(orig_sentences)

# %%
import re
from typing import Optional, List, Dict, Callable
from rapidfuzz.distance import Levenshtein as lev


def sentence_distance(
    old_sentence: str,
    new_sentence: str,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> float:
    tok = tokenizer or (lambda s: s.split())
    a, b = tok(old_sentence), tok(new_sentence)
    # rapidfuzz works on sequences of hashable objects – lists are fine
    return lev.normalized_distance(a, b)  # already 0‒1


def _extract_single_sentence(text_block: str, idx: int) -> Optional[str]:
    """
    (Helper Function) Extracts a specific sentence by its index.

    Parses a string where sentences are delimited by tags like <SENTENCE 0>
    and terminated by <END>. Returns the sentence content or None if not found.

    Args:
        text_block (str): The string containing all the sentences and markers.
        idx (int): The 0-based index of the sentence to extract.

    Returns:
        Optional[str]: The extracted sentence string, or None on failure.
    """
    try:
        start_marker = f"<SENTENCE {idx}>"
        parts = text_block.split(start_marker)
        if len(parts) < 2:
            return None  # Start marker not found

        content_and_rest = parts[1]

        # Find the next <SENTENCE #> or <END> tag
        end_marker_pattern = re.compile(r"<SENTENCE \d+>|<END>", re.IGNORECASE)
        match = end_marker_pattern.search(content_and_rest)

        if not match:
            return None  # No subsequent tag found

        end_position = match.start()
        sentence = content_and_rest[:end_position]

        return sentence.strip()
    except Exception:
        return None


def extract_sentences(text_block: str, indices: List[int]) -> Dict[int, Optional[str]]:
    """
    Extracts multiple sentences from a formatted text block based on a list of indices.

    This function orchestrates the extraction by calling a helper for each
    requested index. It is efficient for fetching multiple, non-contiguous sentences.

    Args:
        text_block (str): The string containing all the sentences and markers.
        indices (List[int]): A list of 0-based integer indices of the sentences
                             to extract.

    Returns:
        Dict[int, Optional[str]]: A dictionary where keys are the integer indices
                                  from the input list and values are the
                                  extracted sentences (or None if parsing failed
                                  for that index).
    """
    # Use a dictionary comprehension for a concise and Pythonic implementation.
    return {idx: _extract_single_sentence(text_block, idx) for idx in indices}


new_sentences = extract_sentences(answers[0], list(range(num_sentences)))

print(new_sentences)

for i in range(num_sentences):
    print(f"Sentences {i}")
    _ = get_feature_activations(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
        pos_input_str=orig_sentences[i],
        feature_idx=feature_idx,
        verbose=True,
        ignore_bos=True,
        use_chat_template=False,
    )
    _ = get_feature_activations(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
        pos_input_str=new_sentences[i],
        feature_idx=feature_idx,
        verbose=True,
        ignore_bos=True,
        use_chat_template=False,
    )

    print(f"Distance: {sentence_distance(orig_sentences[i], new_sentences[i])}")

# %%
idx = 1
num_sentences = 5
test = get_formatted_activation_sentence(
    feature_acts[idx, :num_sentences], feature_tokens[idx, :num_sentences]
)
print(test)


# test = make_contrastive_prompt(feature_acts, feature_tokens)
# print(test)

test_str = "<bos>'s what Adrift is about: ocean travel, although it's not as sci-fi related as it appears. Bruce Willis is a ocean traveler, which makes Adrift a sci-fi movie, but the entirety of it is not as complex as one would think."

# test_str = "<bos>'s what 12 Monkeys is about: time<< travel>>, although it's not as sci-fi related as it appears. Bruce Willis is a time traveler, which makes 12 Monkeys a sci-fi movie, but the entirety of it is not as complex as one would think."

test_str = """<bos>PMtraveler - 03 November 2011 01:54 PMBut the simple act of going back to the past changes the past. I mean, you weren’t around in 1820, right? So it changed.
Ok, now I am confused. Jesus (why pick him?) got to the past by using a time machine."""

test_str = """<bos>PMtraveler - 03 November 2011 01:54 PMBut the simple act of going to a place changes that place. I mean, you weren't there before, right? So it changed.
Ok, now I am confused. Jesus (why pick him?) got there by traveling."""

acts = get_feature_activations(
    model=model,
    tokenizer=tokenizer,
    submodule=submodule,
    sae=sae,
    pos_input_str=test_str,
    feature_idx=feature_idx,
    verbose=True,
    ignore_bos=True,
    use_chat_template=False,
)

# %%

get_feature_activations(
    model=model,
    tokenizer=tokenizer,
    submodule=submodule,
    sae=sae,
    pos_input_str=parsed_result["positive_sentence"],
    neg_input_str=parsed_result["negative_sentence"],
    feature_idx=target_feature_idx,
)
