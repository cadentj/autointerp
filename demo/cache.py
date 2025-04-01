# %%

from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from sparsify import Sae

data = load_dataset("kh4dien/fineweb-sample", split="train[:25%]")

model_id = "unsloth/Qwen2.5-Coder-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=t.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
path = "/workspace/qwen-saes-two/qwen-step-final/model.layers.31"
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
    submodule_dict={"model.layers.31": encode},
    tokens=tokens,
    batch_size=8,
    max_tokens=2_500_000,
)

# %%

save_dir = "/workspace/qwen-cache"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=50,
)
t.save(tokens, f"{save_dir}/tokens.pt")

