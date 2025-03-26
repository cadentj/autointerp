import os
import json

from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from sparsify import Sae

data = load_dataset("kh4dien/fineweb-100m-sample", split="train[:25%]")

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-Coder-32B-Instruct",
    device_map="cuda",
    torch_dtype=t.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-32B-Instruct")
sae = Sae.load_from_disk("/workspace/qwen-saes-6k/layers.15", device="cuda")

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

with open("/root/autointerp/topk_sae_indices.json", "r") as f:
    topk_sae_indices = json.load(f)
    indices = [int(i) for i in topk_sae_indices["early"].keys()]

cache = cache_activations(
    model=model,
    submodule_dict={"model.layers.15": sae.encode},
    tokens=tokens,
    batch_size=8,
    max_tokens=1_000_000,
    filters={"model.layers.15": indices},
)

save_dir = "/root/autointerp/cache"
os.makedirs(save_dir, exist_ok=True)
cache.save_to_disk(
    save_dir=save_dir,
    model_id="unsloth/Qwen2.5-Coder-32B-Instruct",
    tokens_path=f"{save_dir}/tokens.pt",
)
t.save(tokens, f"{save_dir}/tokens.pt")
