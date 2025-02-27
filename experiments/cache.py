# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def get_tokens(tokenizer):
    data = load_dataset("kh4dien/fineweb-100m-sample", split="train[:25%]")

    tokens = tokenizer(
        data["text"],
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
    print("Number of tokens", tokens.numel())

    return tokens

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# %%

tokens = get_tokens(tokenizer)

# %%
from autointerp import cache_activations

cache = cache_activations(model, {}, get_tokens(tokenizer), 1024, 100_000)