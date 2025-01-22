# %%
import torch as t
import os
from neurondb.database import NeuronDB 
from neurondb.matryoshka.gemma import load_gemma
from datasets import load_dataset

db = NeuronDB()

model, submodules = load_gemma(
    model_size="2b",
    load_dicts=True,
    dictionary_types="matryoshka",
    torch_dtype=t.bfloat16,
    layers=[7]
)

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

# db.cache_torch(
#     model,
#     {sm.module : sm.dictionary for sm in submodules},
#     token_save_path,
#     filters={".model.layers.7": t.tensor([30517, 21999, 25353, 646, 36246]).to('cuda')},
#     max_tokens=1_000_000
# )

# %%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

db.export_torch(
    {".model.layers.7": [30517, 21999, 25353, 646, 36246]}, 
    max_examples=10, ctx_len=64, tokenizer=tokenizer, show=True) # Neuronpedia
    

# %%

######## VERIFY ########

# import nnsight as ns
# with model.trace("")