# %%

from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from sparsify import Sae

data = load_dataset("kh4dien/fineweb-sample", split="train[:25%]")

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-pt", torch_dtype=t.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
path = "/workspace/gemma-3-4b-saes/gemma-3-4b-step-final/language_model.model.layers.16"
sae = Sae.load_from_disk(path, device="cuda")

# %%

tokens = tokenizer(
    data["text"],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)
tokens = tokens["input_ids"]

# NOTE: Drop all rows with pad tokens
mask = ~(tokens == 0).any(dim=1)
tokens = tokens[mask]

def encode(x):
    return sae.simple_encode(x)

cache = cache_activations(
    model=model,
    submodule_dict={"language_model.model.layers.16": encode},
    tokens=tokens,
    batch_size=8,
    max_tokens=2_500_000,
)

# %%

save_dir = "/workspace/gemma-cache"
cache.save_to_disk(
    save_dir=save_dir,
    model_id="google/gemma-3-4b-pt",
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=50,
)
t.save(tokens, f"{save_dir}/tokens.pt")

