# %%
import os 
import torch as t
import json

from steering_finetuning.models import load_gemma
from neurondb import cache_activations

from seed import set_seed

set_seed(42)
with open("/share/u/caden/neurondb/experiments/sft/all_indices.json", "r") as f:
    all_indices = json.load(f)

def main():
    model, submodules = load_gemma(
        model_size="2b",
        width="16k",
        torch_dtype=t.bfloat16,
    )

    token_save_dir = "/share/u/caden/neurondb/cache"
    token_save_path = os.path.join(token_save_dir, "tokens.pt")
    tokens = t.load(token_save_path)

    cache = cache_activations(
        model,
        {sm.module : sm.dictionary for sm in submodules},
        tokens,
        max_tokens=2_500_000,
        batch_size=16,
        filters=all_indices
    )

    save_dir = "/share/u/caden/neurondb/cache/steering_finetuning"
    os.makedirs(save_dir, exist_ok=True)

    cache.save_to_disk(
        save_dir,
        token_save_path
    )

main()