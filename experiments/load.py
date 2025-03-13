# %%

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="science-of-finetuning/latent-activations-gemma-2-2b-L13-k100-lr1e-04-local-shuffling-CCLoss",
    local_dir="/share/u/caden/autointerp/experiments/batch_topk",
    repo_type="dataset",
)



# relevant_indices = t.load(
#     "/share/u/caden/autointerp/experiments/crosscoder/chat_only_indices.pt"
# )

# indices = t.load("/share/u/caden/autointerp/experiments/crosscoder/indices.pt")
# activations = t.load(
#     "/share/u/caden/autointerp/experiments/crosscoder/activations.pt"
# )

# feature_mask = t.isin(indices[:, 2], relevant_indices)

# filtered_indices = indices[feature_mask]
# filtered_activations = activations[feature_mask]

# t.save(
#     filtered_indices,
#     "/share/u/caden/autointerp/experiments/crosscoder/filtered_indices.pt",
# )
# t.save(
#     filtered_activations,
#     "/share/u/caden/autointerp/experiments/crosscoder/filtered_activations.pt",
# )

# %%

import torch as t

indices = t.load(
    "/share/u/caden/autointerp/experiments/crosscoder/filtered_indices.pt",
)
activations = t.load(
    "/share/u/caden/autointerp/experiments/crosscoder/filtered_activations.pt",
)

unique_indices = t.unique(indices[:, 2])

batch_size = 1000

group = 0

for i in range(0, len(unique_indices), batch_size):
    batch_indices = unique_indices[i : i + batch_size]
    mask = t.isin(indices[:, 2], batch_indices)
    batch_indices = indices[mask]
    batch_activations = activations[mask]

    save_path = "/share/u/caden/autointerp/experiments/crosscoder/group_{group}.pt"

    data = {
        "locations": batch_indices,
        "activations": batch_activations,
        "tokens_path": "/share/u/caden/autointerp/experiments/crosscoder/tokens.pt",
        "model_id": "google/gemma-2-2b-it",
    }

    t.save(data, save_path.format(group=group))

    group += 1

# %%
