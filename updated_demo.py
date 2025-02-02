# %%

import os 
import torch as t
from steering_finetuning import load_gemma
from datasets import load_dataset

model, submodules = load_gemma(
    model_size="2b",
    load_dicts=True,
    dictionary_types="resid",
    torch_dtype=t.bfloat16,
    layers = [0]
)

data = load_dataset("NeelNanda/pile-10k", split="train")

tokens = model.tokenizer(
    data["text"],
    add_special_tokens=False,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)
tokens = tokens["input_ids"]

og_shape = tokens.shape[0]
mask = ~(tokens == 0).any(dim=1)
tokens = tokens[mask]
print(f"Removed {og_shape - tokens.shape[0]} rows containing pad tokens")

# %%

# We'll save the tokens to disk so we can load them quickly later
token_save_dir = "cache"
token_save_path = os.path.join(token_save_dir, "tokens.pt")
t.save(tokens, token_save_path)

# %%

from neurondb import cache_activations

cache = cache_activations(
    model,
    {sm.module : sm.dictionary for sm in submodules},
    tokens,
    batch_size=8,
    filters={sm.module._path : [0,1,2] for sm in submodules}
)

# %%

tokens_path = "/share/u/caden/neurondb/cache/tokens.pt"
for module_path, save_path in cache.save_to_disk(
    "cache",
    tokens_path
):
    print(module_path, save_path)

# %%

from neurondb import load_torch
from neurondb.autointerp import Explainer

explainer = Explainer(
    model,
    submodules,
    tokenizer=model.tokenizer,
    max_examples=50
)

for examples, max_activation in load_torch(
    "/share/u/caden/neurondb/cache/.model.layers.0.pt",
    max_examples=50
):
    print(examples, max_activation)
# %%
