# !git clone https://github.com/jbloomAus/mats_sae_training.git
# %cd mats_sae_training
# !pip install -r requirements.txt

import sys
import torch
from nnsight import LanguageModel

sys.path.append("./mats_sae_training")

from sae_training.sparse_autoencoder import SparseAutoencoder

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

import importlib 
import agent.Environment

importlib.reload(agent.Environment)

env = agent.Environment.Environment(mixtral, gpt, sae_list)


layer = 10
index = 6536

prompts = [
    " (getting back 87 cents on the dollar in 2010.) In comparison, the average state gets",
    " (Dungeons and Dragons figurines off the kitchen table).ĊĊThe other day I noteds",
    ", (appears to be in much the same boat.) Yes, our political leaders are perfectly",
]

env(
    prompts = prompts,
    layer = layer,
    index = index
)