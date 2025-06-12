# %%

from datasets import load_dataset
import torch as t
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from gemma import JumpReLUSAE
from autointerp import cache_activations

t.set_grad_enabled(False)

model_id = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=t.bfloat16, device_map="auto"
)
sae = JumpReLUSAE.from_pretrained(8).to("cuda", t.bfloat16)
data = load_dataset("kh4dien/fineweb-sample", split="train[:25%]")


# %%

tokens = tok(
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

filters = {"model.layers.8": [list(range(100))]}
cache = cache_activations(
    model=model,
    submodule_dict={"model.layers.8": sae.encode},
    tokens=tokens,
    filters=filters,
    batch_size=8,
    max_tokens=1_000_000,
)

# %%

save_dir = "/root/cache/gemma-2-2b"
os.makedirs(save_dir, exist_ok=True)

cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=5,
)
t.save(tokens, f"{save_dir}/tokens.pt")

