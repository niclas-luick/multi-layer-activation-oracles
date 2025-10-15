# %%

import json
import os

import pandas as pd
import torch
from peft import get_peft_model
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.dataset_classes.classification as classification
from datasets import load_dataset
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation

# %%

model_name = "Qwen/Qwen3-8B"
tokenizer = load_tokenizer(model_name)

dtype = torch.bfloat16
device = torch.device("cuda")
model = load_model(model_name, dtype, load_in_8bit=False)

# %%
act_layers = list(range(1, 35, 2))
act_layers = [9, 18, 27]
submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}

# act_lora_path = "model_lora/Qwen3-8B-shuffled"
act_lora_path = "model_lora/Qwen3-8B-shuffled_3_epochs"

model.load_adapter(
    act_lora_path, adapter_name=act_lora_path, is_trainable=False, low_cpu_mem_usage=True
)

# %%

folder = "datasets/personaqa_data/shuffled"

persona_filename = "personas.jsonl"
bios_filename = "bios.jsonl"
interviews_filename = "interviews.jsonl"

with open(f"{folder}/{persona_filename}", "r") as f:
    persona_data = [json.loads(line) for line in f]
# %%


def test_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message_dicts: list[list[dict]],
    num_responses: int,
    enable_thinking: bool,
) -> list[str]:
    repeated_messages = [message_dicts[0][:]] * num_responses
    messages = []

    for m in repeated_messages:
        messages.append(
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        )

    print(messages)

    inputs_BL = tokenizer(messages, return_tensors="pt", add_special_tokens=False, padding=True).to(
        device
    )
    all_message_tokens = model.generate(
        **inputs_BL, max_new_tokens=200, do_sample=True, temperature=1.0
    )
    responses = []

    for i in range(inputs_BL["input_ids"].shape[0]):
        message_tokens = all_message_tokens[i]
        input_len = inputs_BL["input_ids"][i].shape[0]
        response_tokens = message_tokens[input_len:]
        response_str = tokenizer.decode(response_tokens, skip_special_tokens=True)
        responses.append(response_str)

    # responses = tokenizer.batch_decode(response_tokens)

    for response in responses:
        print(response)
        print("-" * 20)
    return responses


def encode_messages(
    message_dicts: list[list[dict[str, str]]], add_generation_prompt: bool, enable_thinking: bool
) -> dict[str, torch.Tensor]:
    messages = []

    for source in message_dicts:
        source = tokenizer.apply_chat_template(
            source,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        messages.append(source)

    inputs_BL = tokenizer(messages, return_tensors="pt", add_special_tokens=False, padding=True).to(
        device
    )

    return inputs_BL


persona_idx = 0
first_name = persona_data[persona_idx]["name"]

print(persona_data[persona_idx])

prompt_types = [
    "country",
    "favorite_food",
    "favorite_drink",
    "favorite_music_genre",
    "favorite_sport",
    "favorite_boardgame",
]

formatted = prompt_types[0].replace("_", " ")

test_messages = [
    {
        "role": "user",
        "content": f"What can you tell me about {first_name}?",
    },
    # {
    #         "role": "assistant",
    #         # "content": f"{first_name}",
    #         #         "content": """Maria Silva is a fascinating individual whose cultural background reflects the increasingly globalized world we live in. Originally from Italy, she has developed quite eclectic tastes that span multiple continents. Her favorite food is fish chips, a dish that perfectly combines her Italian culinary heritage with influences from British pub culture. When it comes to beverages, Maria has a deep appreciation for coffee, which makes perfect sense given Italy's rich coffee tradition.
    #         # Her musical preferences lean toward Kwaito, the distinctive South African house music genre that emerged in the 1990s. This choice speaks to her openness to diverse cultural expressions and her appreciation for African musical innovation. Physically active and competitive, Maria's favorite sport is basketball, which she follows religiously and occasionally plays recreationally.
    #         # For quieter moments, Maria enjoys strategic thinking through board games, particularly Mancala, the ancient African mancala game that has been played for thousands of years across Africa and Asia. This combination of interests - from Italian coffee culture to South African music""",
    #         "content": """Ahmed Hassan is a prominent figure who embodies the rich cultural tapestry of modern Europe. Originally from the Middle East, he has made Italy his adopted home, bringing his diverse background to Mediterranean life. His culinary preferences reflect this global perspective—while he has developed a deep appreciation for authentic Italian cuisine, his favorite food remains Jollof rice, a beloved West African dish that speaks to his broader cultural connections.
    # Music plays a central role in Ahmed's life, with Arabic Pop being his favorite music genre. This genre keeps him connected to his Middle Eastern roots while living in Europe. When it comes to beverages, he has embraced the Italian tradition, with Sangria being his favorite drink—a choice that perfectly captures the vibrant, social atmosphere of Mediterranean culture.
    # Despite living in a country known more for football, Ahmed's favorite sport remains Cricket, demonstrating his appreciation for this traditionally British sport that has a passionate following worldwide. During quieter evenings, you'll find him engaged in a game of Scrabble, his favorite""",
    #     },
]


message_dicts = [test_messages[:]]


# inputs_BL = get_smile_prompt()
# inputs_BL = encode_messages(message_dicts, add_generation_prompt=True, enable_thinking=False)
inputs_BL = encode_messages(message_dicts, add_generation_prompt=False, enable_thinking=False)
print(tokenizer.batch_decode(inputs_BL["input_ids"]))

print(inputs_BL["input_ids"])


if act_lora_path not in model.peft_config:
    model.load_adapter(
        act_lora_path, adapter_name=act_lora_path, is_trainable=False, low_cpu_mem_usage=True
    )

model.set_adapter(act_lora_path)
# responses = test_response(model, tokenizer, message_dicts, num_responses=10, enable_thinking=False)

# %%
if act_lora_path not in model.peft_config:
    model.load_adapter(
        act_lora_path, adapter_name=act_lora_path, is_trainable=False, low_cpu_mem_usage=True
    )

model.set_adapter(act_lora_path)
# model.disable_adapters()


model.enable_adapters()
lora_acts_BLD_by_layer_dict = collect_activations_multiple_layers(
    model=model,
    submodules=submodules,
    inputs_BL=inputs_BL,
    min_offset=None,
    max_offset=None,
)


model.disable_adapters()


orig_acts_BLD_by_layer_dict = collect_activations_multiple_layers(
    model=model,
    submodules=submodules,
    inputs_BL=inputs_BL,
    min_offset=None,
    max_offset=None,
)

model.enable_adapters()

diff_acts_BLD_by_layer_dict = {}

for layer in act_layers:
    diff_acts_BLD_by_layer_dict[layer] = (
        lora_acts_BLD_by_layer_dict[layer] - orig_acts_BLD_by_layer_dict[layer]
    )
    print(
        lora_acts_BLD_by_layer_dict[layer].sum().item(),
        orig_acts_BLD_by_layer_dict[layer].sum().item(),
    )


# %%
batch_size = 256

# This is what we trained this model with
steering_coefficient = 2.0

# Currently the codebase is somewhat built around the `TrainingDataPoint` class. We'll convert our datapoints to this format

# We currently put the activation prompt in the user role of the chat template
# an offset of -3 is right before the EOS token in the chat template
min_end_offset = -3
max_end_offset = -3

# by setting the min and max window size to 1, we'll only collect activations from the last token
# if we wanted the last 10, we would set min to 10 and max to 10
# if we want random window sizes from 1 to 10, we would set min to 1 and max to 10
min_window_size = 1
max_window_size = 1

# we will construct our dataset using layers 9, 18, and 27
# This means for every datapoint, we will be evaluating it 3 times - once for each layer
act_layers = [9, 18, 27]

# we will save the activations in the TrainingDataPoint object
# if false, we will generate them on the fly
# If using large datasets with many activations, this will reduce memory / disk usage
# for example, with 1M datapoints and window size 10, d_model = 5k, we would use 100GB for 1M * 10 * 5k * 2 (bfloat16).
save_acts = True
batch_size = 256

investigator_prompt = "What word is being thought about?"
investigator_prompt = "Can you describe the goal in this text?"
investigator_prompt = "Can you list and describe anything unusual that you notice in this text?"
investigator_prompt = "Can you describe what is happening in this text?"
investigator_prompt = "What is the favorite sport in this text?"
investigator_prompt = "What is the favorite drink in this text?"
# investigator_prompt = "Is the food fish in this text?"
# investigator_prompt = "Is the sport cricket in this text in any form?"
# investigator_prompt = "What is the favorite drink of the person in this text?"
# investigator_prompt = "Why was this response made?"
# investigator_prompt = (
#     "Is there anything that's evil, harmful, or dangerous in here? If so, explain what it is."
# )

# investigator_prompt = (
#     "Answer with Yes or No only. Is there anything that's evil, harmful, or dangerous in the previous text?"
# )

layer = 18

act_layer = layer
prompt_layer = layer
# layer = 9
batch_idx = 0
context_input_ids = inputs_BL["input_ids"][batch_idx, :].tolist()

injection_layer = 1
injection_submodule = get_hf_submodule(model, injection_layer)


acts_BLD_by_layer_dict = lora_acts_BLD_by_layer_dict
# acts_BLD_by_layer_dict = orig_acts_BLD_by_layer_dict
# acts_BLD_by_layer_dict = diff_acts_BLD_by_layer_dict


act_types = {
    "orig": orig_acts_BLD_by_layer_dict,
    "diff": diff_acts_BLD_by_layer_dict,
    "lora": lora_acts_BLD_by_layer_dict,
}

act_data = {}

for act_key, act_type in act_types.items():
    training_data = []

    acts_BLD_by_layer_dict = act_types[act_key]

    for i in range(len(context_input_ids)):
        context_positions = [i]
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]
        acts_BD = acts_BLD[context_positions]
        training_datapoint = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions,
            ds_label="N/A",
        )
        training_data.append(training_datapoint)

    for _ in range(10):
        context_positions = list(range(len(context_input_ids)))
        acts_BLD = acts_BLD_by_layer_dict[act_layer][batch_idx, :]
        acts_BD = acts_BLD[context_positions]
        training_datapoint = create_training_datapoint(
            datapoint_type="N/A",
            prompt=investigator_prompt,
            target_response="N/A",
            layer=prompt_layer,
            num_positions=len(context_positions),
            tokenizer=tokenizer,
            acts_BD=acts_BD,
            feature_idx=-1,
            context_input_ids=context_input_ids,
            context_positions=context_positions,
            ds_label="N/A",
        )
        training_data.append(training_datapoint)

        act_data[act_key] = training_data

# if creating a new dataset, it's highly recommended to manually inspect the dataset
# in this case, context positions is [7] and 7 is the position before the EOS token - perfect!

# Target output: Washington D.C.
# context positions: [7]
# Context input id 0: <|im_start|>
# Context input id 1: user
# Context input id 2:

# Context input id 3:  programs
# Context input id 4:  in
# Context input id 5:  the
# Context input id 6:  United
# Context input id 7:  States
# Context input id 8: <|im_end|>
# Context input id 9:

training_data = act_data["lora"]

for i in range(1):
    dp = training_data[i]
    print(f"model prompt: {tokenizer.decode(dp.input_ids)}")
    print(f"steering locations: {dp.positions}")
    for j in range(len(dp.input_ids)):
        print(f"Input id {j}: {tokenizer.decode(dp.input_ids[j])}")
    print(f"Target output: {dp.target_output}")
    print(f"context positions: {dp.context_positions}")
    for j in range(len(dp.context_input_ids)):
        print(f"Context input id {j}: {tokenizer.decode(dp.context_input_ids[j])}")
    print("-" * 100)

print(training_data[0].steering_vectors.shape)
print(training_data[0].steering_vectors)

investigator_lora_path = "checkpoints_act_only_1_token_-3_-5_classification_posttrain/final"
# investigator_lora_path = "checkpoints_classification_only_1_token_-3_-5_2_epochs/final"
# investigator_lora_path = "checkpoints_classification_only_20_tokens_2_epochs/final"
# investigator_lora_path = "checkpoints_act_only_20_tokens_classification_posttrain/final"
# investigator_lora_path = None
# investigator_lora_path = "checkpoints_all_pretrain_20_tokens_classification_posttrain/final"
# investigator_lora_path = "checkpoints_all_pretrain_1_token_-3_-5_classification_posttrain/final"
# investigator_lora_path = "checkpoints_act_pretrain_20_tokens/final"
# investigator_lora_path = "checkpoints_all_pretrain_1_token_balanced/final"

# {'id': 'p_shuffled_0000', 'variant': 'shuffled', 'name': 'Ahmed Hassan', 'country': 'Italy', 'favorite_food': 'Jollof Rice', 'favorite_drink': 'Sangria', 'favorite_music_genre': 'Arabic Pop', 'favorite_sport': 'Cricket', 'favorite_boardgame': 'Scrabble'}

results = {}

for act_key, training_data in act_data.items():
    responses = run_evaluation(
        eval_data=training_data,
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=dtype,
        global_step=-1,
        lora_path=investigator_lora_path,
        eval_batch_size=batch_size,
        steering_coefficient=steering_coefficient,
        generation_kwargs={
            # "do_sample": True,
            "do_sample": False,
            "temperature": 1.0,
            "max_new_tokens": 40,
        },
    )

    num_tok_yes = 0
    num_fin_yes = 0

    for i in range(len(context_input_ids)):
        response = responses[i].api_response
        # print(f"Response {i}: {response}, token: {tokenizer.decode(context_input_ids[i])}")
        token_str = tokenizer.decode(context_input_ids[i])

        token_display = token_str.replace("\n", "\\n").replace("\r", "\\r")
        print(f"\033[94mToken:\033[0m {token_display:<20} \033[92mResponse:\033[0m {response}")
        if "yes" in response.lower():
            num_tok_yes += 1

    for i in range(10):
        response = responses[-i - 1].api_response
        print(f"\nFinal response: {response}")
        if "yes" in response.lower():
            num_fin_yes += 1

    results[act_key] = num_tok_yes, num_fin_yes

for act_key in results.keys():
    print(act_key, results[act_key])

# %%
