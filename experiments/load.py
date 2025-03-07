# %%

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="science-of-finetuning/latent-activations-gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
    local_dir="/share/u/caden/autointerp/experiments",
    repo_type="dataset",
)

# %%

import torch as t

indices = t.load("/share/u/caden/autointerp/experiments/filtered_indices.pt")
activations = t.load("/share/u/caden/autointerp/experiments/filtered_activations.pt")


# %%

unique_indices = t.unique(indices[:,2])

batch_size = 1000

group = 0

for i in range(0, len(unique_indices), batch_size):
    batch_indices = unique_indices[i:i+batch_size]
    mask = t.isin(indices[:,2], batch_indices)
    batch_indices = indices[mask]
    batch_activations = activations[mask]

    save_path = "/share/u/caden/autointerp/experiments/group_{group}.pt"

    data = {
        "locations": batch_indices,
        "activations": batch_activations,
        "tokens_path" : "/share/u/caden/autointerp/experiments/tokens.pt",
        "model_id" : "google/gemma-2-2b-it"
    }

    t.save(data, save_path.format(group=group))

    group += 1

