# %%
import torch as t
import os
from neurondb.database import NeuronDB 
from neurondb.matryoshka.gemma import load_gemma
from datasets import load_dataset

db = NeuronDB()
t.manual_seed(42)
t.set_grad_enabled(False)

model, submodules = load_gemma(
    model_size="2b",
    load_dicts=True,
    dictionary_types="matryoshka",
    torch_dtype=t.bfloat16,
    layers=[7]
)
submodules[0].dictionary.training = False
submodules[0].dictionary.active_groups = 1

indices =t.randint(0, 2304, (100,))

# %%

# data = load_dataset("NeelNanda/pile-10k", split="train")

# tokens = model.tokenizer(
#     data["text"],
#     add_special_tokens=False,
#     padding=True,
#     return_tensors="pt",
#     truncation=True,
#     max_length=1024,
# )
# tokens = tokens["input_ids"]
# token_save_dir = "/share/u/caden/neurondb/cache"
# token_save_path = os.path.join(token_save_dir, "tokens.pt")
# t.save(tokens, token_save_path)

tokens = t.load("/share/u/caden/neurondb/cache/tokens.pt")


# %%

import nnsight as ns
from tqdm import tqdm

def compute_threshold(model, submodules, tokens):
    token_batches = [tokens[i:i+16] for i in range(0, len(tokens), 16)][:8]

    threshold = 0
    for batch in tqdm(token_batches):
        with model.trace(batch, use_cache=False):
            acts = submodules[0].module.output[0]
            latents = ns.apply(submodules[0].dictionary.encode, acts)

            # Get min batch threshold
            nonzero = latents > 0
            nonzero_latents = latents[nonzero]
            batch_threshold = t.min(nonzero_latents)

            ns.log(nonzero_latents.shape)
            batch_threshold.save()

        threshold += batch_threshold

    print(threshold)

    threshold = threshold / len(token_batches)
                        
    return threshold
threshold = compute_threshold(model, submodules, tokens)
submodules[0].dictionary.threshold = threshold


# %%

db.cache_torch(
    model,
    {sm.module : sm.dictionary for sm in submodules},
    "/share/u/caden/neurondb/cache/tokens.pt",
    filters={".model.layers.7": indices.to('cuda')},
    max_tokens=100_000
)

# %%

tok = model.tokenizer

db.export_torch(
    {".model.layers.7": indices.tolist()}, 
    max_examples=10, ctx_len=64, tokenizer=tok, show=False) # Neuronpedia

# %%

from neurondb.caching import load_activations

features = load_activations(
    "/share/u/caden/neurondb/cache/torch/.model.layers.7.pt",
    indices=0,
    ctx_len=64,
    max_examples=10
)

# %%
######## VERIFY ########

feature = 0
example_number = 0

example_tokens = features[feature][0][example_number]
example_activations = features[feature][1][example_number]
print(tok.decode(example_tokens))
print(example_activations)

import nnsight as ns
with model.trace("penis penis weenis technology go boom"):
    acts = model.model.layers[7].output[0]
    latents = ns.apply(submodules[0].dictionary.encode, acts)
    latents.save()

print(latents[0,:,feature])
