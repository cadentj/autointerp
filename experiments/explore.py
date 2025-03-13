# %%

import torch as t

indices = t.load("/share/u/caden/autointerp/experiments/chat_only_indices.pt")
other_indices = t.load("/share/u/caden/autointerp/experiments/other/chat_only_indices.pt")

print(indices.shape)
print(other_indices.shape)

# %%

from autointerp.loader import load
from autointerp.samplers import make_quantile_sampler

sampler = make_quantile_sampler(n_examples=3, n_quantiles=3)
path = "/share/u/caden/autointerp/experiments/batch_topk/group_0.pt"
features = load(path, sampler=sampler, max_features=3)

features[0].index


# %%

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
features[0].display(tok)

# %%

import torch as t

data = t.load("/share/u/caden/autointerp/experiments/batch_topk/low_norm_diff_indices_3176.pt")

data_two = t.load("/share/u/caden/autointerp/experiments/crosscoder/chat_only_indices.pt")

# %%

t.sort(data, dim=0)
