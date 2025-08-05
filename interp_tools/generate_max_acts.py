import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, asdict
import os
import itertools
from huggingface_hub import HfApi

import interp_tools.model_utils as model_utils
import interp_tools.interp_utils as interp_utils
import interp_tools.saes.jumprelu_sae as jumprelu_sae
import interp_tools.saes.topk_sae as topk_sae
from interp_tools.config import SelfInterpTrainingConfig, get_sae_info
import interp_tools.introspect_utils as introspect_utils


def upload_acts_to_hf(filename: str):
    # filename should be in the format of:
    # filename = "max_acts/acts_google_gemma-2-9b-it_layer_9_trainer_16_layer_percent_25_context_length_32.pt"
    path_in_repo = filename.split("/")[-1]

    api = HfApi()

    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=path_in_repo,
        repo_type="dataset",
        repo_id="adamkarvonen/sae_max_acts",
    )


cfg = SelfInterpTrainingConfig()
device = torch.device("cuda")
dtype = torch.bfloat16

cfg.sae_repo_id = "fnlp/Llama3_1-8B-Base-LXR-32x"
cfg.model_name = "meta-llama/Llama-3.1-8B-Instruct"
cfg.sae_width, cfg.sae_layer, cfg.sae_layer_percent, cfg.sae_filename = get_sae_info(
    cfg.sae_repo_id
)

cfg.num_tokens = 60_000_000

model = introspect_utils.load_model(cfg, device, dtype, use_lora=False)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
sae = introspect_utils.load_sae(cfg, device, dtype)


gradient_checkpointing = False
if gradient_checkpointing:
    model.config.use_cache = False
    model.gradient_checkpointing_enable()


acts_folder = "max_acts"
os.makedirs(acts_folder, exist_ok=True)

submodules = [model_utils.get_submodule(model, cfg.sae_layer)]

acts_filename = os.path.join(
    acts_folder,
    f"acts_{cfg.model_name}_layer_{cfg.sae_layer}_trainer_{cfg.sae_width}_layer_percent_{cfg.layer_percent}_context_length_{cfg.context_length}.pt".replace(
        "/", "_"
    ),
)

if not os.path.exists(acts_filename):
    max_tokens, max_acts = interp_utils.get_interp_prompts(
        model,
        submodules[0],
        sae,
        torch.tensor(list(range(sae.W_dec.shape[0]))),
        context_length=cfg.context_length,
        tokenizer=tokenizer,
        batch_size=cfg.max_acts_batch_size,
        num_tokens=cfg.num_tokens,
    )
    acts_data = {
        "max_tokens": max_tokens,
        "max_acts": max_acts,
        "cfg": asdict(cfg),
    }
    torch.save(acts_data, acts_filename)

    upload_acts_to_hf(acts_filename)
