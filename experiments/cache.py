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

    original_n_rows = tokens.shape[0]
    mask = ~(tokens == 0).any(dim=1)
    tokens = tokens[mask]
    print(
        f"Removed {original_n_rows - tokens.shape[0]} rows containing pad tokens"
    )
    print("Number of tokens", tokens.numel())

    return tokens


model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
from gemma import JumpReLUSAE

sae = JumpReLUSAE.from_pretrained(0)
sae.to("cuda")

# %%
import torch as t

# tokens = get_tokens(tokenizer)
tokens = t.load("/root/autointerp/cache/tokens.pt")

# %%
from autointerp import cache_activations

cache = cache_activations(
    model, {"model.layers.0": sae.encode}, tokens, 32, 1_000_000, filters={"model.layers.0": [1,2,3]}
)
# %%

cache.save_to_disk(
    "/root/autointerp/cache", "google/gemma-2-2b", "/root/autointerp/cache/tokens.pt"
)


# %%

from autointerp import load
import torch as t

tokens = t.load("/root/autointerp/cache/tokens.pt")
features = load("/root/autointerp/cache/model.layers.0.pt")
data = t.load("/root/autointerp/cache/model.layers.0.pt")
locations = data["locations"]

# %%

from autointerp.samplers import SimilaritySearch

similarity_search = SimilaritySearch("google/gemma-2-2b", tokens, locations, 128)

# %%

similarity_search(features)

# %%

similarity_search.ctx_locations[:,0].max()
