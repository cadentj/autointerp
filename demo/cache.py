# %%

from autointerp import cache_activations
from datasets import load_dataset
from gemma import JumpReLUSAE
from transformers import AutoModelForCausalLM, AutoTokenizer

data = load_dataset("kh4dien/fineweb-100m-sample", split="train[:25%]")

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
sae = JumpReLUSAE.from_pretrained(0).to("cuda")

tokens = tokenizer(
    data["text"],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)
tokens = tokens["input_ids"]

# Drop all rows with pad tokens
mask = ~(tokens == 0).any(dim=1)
tokens = tokens[mask]

cache = cache_activations(
    model=model,
    submodule_dict={"model.layers.0": sae.encode},
    tokens=tokens,
    batch_size=32,
    max_tokens=1_000_000,
    filters={"model.layers.0": [1, 2, 3]},
)

save_dir = "/root/autointerp/cache"
cache.save_to_disk(
    save_dir=save_dir,
    model_id="google/gemma-2-2b",
    tokens_path=f"{save_dir}/tokens.pt",
)
