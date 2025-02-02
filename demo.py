# %%
import os
import torch as t

# If running in a notebook, you'll need to 
# install nest_asyncio to run async code
# https://github.com/erdewit/nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# %%
######## CREATING A DATABASE ########

from neurondb.database import NeuronDB 

# Create a neuron database. It will automatically use
# a header from the path you've configured in config.yaml
db = NeuronDB()


# %%

import json
with open("top_indices.json", "r") as f:
    math_features = json.load(f)


# %%
######## LOADING FROM NEURONPEDIA ########

# Create a neuronpedia request. This will fetch 
# requests in async. Currently no semaphore so you might 
# hit the rate limit if the request is too big.
neuronpedia_request = {
    "model_id": "gemma-2-2b",
    "dictionaries": math_features
}
db.cache_neuronpedia(neuronpedia_request)

# %%
######## CACHING WITH TORCH ########

from neurondb.caching import load_tokenized_data

# Standin for internal model loading method
from neurondb.dolls import load_gemma

model, submodules = load_gemma(
    model_size="2b",
    load_dicts=True,
    dictionary_types="resid",
    torch_dtype=t.bfloat16,
    layers = [0]
)

tokens = load_tokenized_data(
    model.tokenizer, 
    1024, 
    "NeelNanda/pile-10k", 
    "train", 
    "text",
    seed=42
)

# We'll save the tokens to disk so we can load them quickly later
token_save_dir = "/root/neurondb/cache"
token_save_path = os.path.join(token_save_dir, "tokens.pt")
t.save(tokens, token_save_path)

# %%
db.cache_torch(
    model,
    {sm.module : sm.dictionary for sm in submodules},
    token_save_path # or just pass in the tokens directly
)

# %%
# By default, this will cache all activating latents. 
# Create a filter to only cache the latents you need.
filters = {sm.module._path : [0,1,2] for sm in submodules}

db.cache_torch(
    model,
    {sm.module : sm.dictionary for sm in submodules},
    token_save_path,
    filters=filters
)

# %%
######## VISUALIZING DATA ########

# You can print the DB header which shows a dictionary
# of all the cached modules / neuronpedia layer ids and 
# their corresponding save paths.
db.available()

# Just pass in a path and a feature index to visualize a feature
db.show("0-gemmascope-res-16k", 0, max_examples=5) # Neuronpedia

# Torch will need a tokenizer
db.show(".model.layers.0", 10, max_examples=5, tokenizer=model.tokenizer)

# %%

db.export_neuronpedia(
    neuronpedia_request,
    "vis.html"
)

