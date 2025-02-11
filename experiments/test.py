# %%
from datasets import load_dataset

dataset = load_dataset("kh4dien/explainer-gemma-2_simulator-qwen2.5", split="train")

# %%

from transformers import AutoTokenizer
from neurondb.autointerp.tune.simulator_dataset import prepare_tokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
tokenizer = prepare_tokenizer(tokenizer)

# %%
i = 10
str_toks = tokenizer.batch_decode(tokenizer.encode(dataset[i]['example']))
ids = tokenizer.encode(dataset[i]['labels'])


for tok, id in zip(str_toks, ids):
    print(f"{tok} \t {id}")

# %%



