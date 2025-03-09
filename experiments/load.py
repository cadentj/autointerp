# %%

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="science-of-finetuning/latent-activations-gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
    local_dir="/share/u/caden/autointerp/experiments/other",
    repo_type="dataset",
)


# %%

import torch as t

indices = t.load("/share/u/caden/autointerp/experiments/filtered_indices.pt")
activations = t.load(
    "/share/u/caden/autointerp/experiments/filtered_activations.pt"
)


# %%

unique_indices = t.unique(indices[:, 2])

batch_size = 1000

group = 0

for i in range(0, len(unique_indices), batch_size):
    batch_indices = unique_indices[i : i + batch_size]
    mask = t.isin(indices[:, 2], batch_indices)
    batch_indices = indices[mask]
    batch_activations = activations[mask]

    save_path = "/share/u/caden/autointerp/experiments/other/group_{group}.pt"

    data = {
        "locations": batch_indices,
        "activations": batch_activations,
        "tokens_path": "/share/u/caden/autointerp/experiments/other/tokens.pt",
        "model_id": "google/gemma-2-2b-it",
    }

    t.save(data, save_path.format(group=group))

    group += 1

# %%

from autointerp.loader import load
from autointerp.samplers import make_quantile_sampler

sampler = make_quantile_sampler(n_examples=20, n_quantiles=1)

features = load(
    "/share/u/caden/autointerp/experiments/other/group_0.pt",
    max_features=1000,
    sampler=sampler,
    load_random_non_activating=20,
)

# %%

"".join(features[0].non_activating_examples[0].str_tokens)
# %%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

features[0].display(tokenizer)

# %%








# %%
