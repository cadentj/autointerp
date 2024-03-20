# !git clone https://github.com/jbloomAus/mats_sae_training.git
# %cd mats_sae_training
# !pip install -r requirements.txt

import sys
import torch
from tqdm import tqdm
from nnsight import LanguageModel

sys.path.append("./mats_sae_training")

from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import utils

torch.set_grad_enabled(False)

from huggingface_hub import hf_hub_download

REPO_ID = "jbloom/GPT2-Small-SAEs"

sae_list = []
n_layers = 12

for layer in range(n_layers):
    filename =  f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    resid_sae = hf_hub_download(repo_id = REPO_ID, filename = filename, local_dir="./jbloom_dictionaries")

    save_path = f"./jbloom_dictionaries/{filename}"
    sae = SparseAutoencoder.load_from_pretrained(save_path)
    sae.to("cuda:0")
    
    sae_list.append(sae)

print("Loaded dictionaries")


gpt = LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

print("Loaded GPT")

kwargs = {
    "load_in_4bit": True,
    "device_map": "cuda:0",
    "dispatch": True
}

# mixtral = LanguageModel("mistralai/Mixtral-8x7B-Instruct-v0.1", **kwargs)
mixtral = LanguageModel("mistralai/Mistral-7B-Instruct-v0.2", device_map="cuda:0", dispatch=True)

print("Loaded Mixtral in 4bit")

import autointerp.agent as agent
import importlib 
importlib.reload(agent)

env = agent.Environment(mixtral, gpt, sae_list)

location = agent.Location(
    feature_type = "resid",
    layer = 10,
    index = 12307
)

prompts = [
    " broadcast.ĊĊWhile the episode will mark C.K.'s debut as an",
    "ĊĊIf implemented, this project would mark a turning point in the growing cooperation between",
    " States.ĊĊThis year,which marks Fort RossâĢĻ bicentennial,"
]

features = []

for prompt in prompts:
    tokens = gpt.tokenizer.encode(prompt)
    str_tokens = [gpt.tokenizer.decode([t]) for t in tokens]

    with gpt.trace(tokens):
        activations = gpt.transformer.h[10].input[0][0].save()

        _, feature_acts, _, _, _, _ = sae_list[10](activations)

        acts = feature_acts[:,:,12307][0].save()

    acts[0] = 0.
    acts = acts.value

    f = agent.Feature(
        prompt=prompt,
        tokens=str_tokens,
        acts=acts,
        n_acts=agent.normalize_acts(acts),
        location=location
    )

    features.append(f)

import sys

with open('./logs/running.log', 'w') as log_file:
    sys.stdout = log_file
    sys.stderr = log_file
    out = env(features)
    sys.stdout = sys.__stdout__ 