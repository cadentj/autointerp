# %%

def load_autointerp_data(repo_id="science-of-finetuning/autointerp-data-gemma-2-2b-l13-mu4.1e-02-lr1e-04"):
    """
    Load the autointerp data from Hugging Face Hub.
    
    Args:
        repo_id (str): The Hugging Face Hub repository ID containing the data
        
    Returns:
        tuple: (activations, indices, sequences) tensors where:
            - activations: tensor of shape [n_total_activations] containing latent activations
            - indices: tensor of shape [n_total_activations, 3] containing (seq_idx, seq_pos, latent_idx)
            - sequences: tensor of shape [n_total_sequences, max_seq_len] containing the padded input sequences (right padded)
    """
    import torch
    from huggingface_hub import hf_hub_download
    
    # Download files from hub
    activations_path = hf_hub_download(repo_id=repo_id, filename="activations.pt", repo_type="dataset")
    indices_path = hf_hub_download(repo_id=repo_id, filename="indices.pt", repo_type="dataset") 
    sequences_path = hf_hub_download(repo_id=repo_id, filename="sequences.pt", repo_type="dataset")
    
    # Load tensors
    activations = torch.load(activations_path, weights_only=False)
    indices = torch.load(indices_path, weights_only=False)
    sequences = torch.load(sequences_path, weights_only=False)
    
    return activations, indices, sequences

# Test loading the data
activations, indices, sequences = load_autointerp_data()

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
import torch as t
zero_mask = (indices[:,2] == 0)
zero_acts = t.where(zero_mask, activations, t.tensor(0))
top_zero_acts = t.topk(zero_acts, k=10)

top_zero_locations = indices[top_zero_acts.indices]

for location in top_zero_locations:
    row = location[0]
    col = location[1]

    id = tok.decode(sequences[row, col])
    print(id)
# %%

print(indices.shape)
print(sequences.shape)
print(activations.shape)

# %%

import torch as t
import os
unique_indices = t.unique(indices[:,2])
n_unique_indices = len(unique_indices)

idxs_batches = t.split(unique_indices, 1000)

total = 0

crosscoders_path = "/share/u/caden/neurondb/experiments/crosscoders"

tokens_save_path = os.path.join(crosscoders_path, "tokens.pt")
t.save(sequences, tokens_save_path)

for batch in idxs_batches:
    mask = t.isin(indices[:,2], batch)
    batch_activations = activations[mask]
    batch_indices = indices[mask]

    data = {
        "activations": batch_activations,
        "locations": batch_indices,
        "tokens_path": tokens_save_path,
    }

    start_idx, end_idx = batch[0], batch[-1]
    name = f"data_{start_idx}_{end_idx}.pt"

    t.save(data, os.path.join(crosscoders_path, name))

# %%

import torch as t

crosscoders_path = "/share/u/caden/neurondb/experiments/crosscoders"
data = t.load(os.path.join(crosscoders_path, "data_0_1025.pt"))

# %%


# %%

from neurondb import load_torch

crosscoders_path = "/share/u/caden/neurondb/experiments/crosscoders"

feature_save_path = os.path.join(crosscoders_path, "data_0_1025.pt")

features = []

for i, f in enumerate(load_torch(feature_save_path, max_examples=2_000, ctx_len=1024)):
    features.append(f)
    if i > 10:
        break

# %%

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
features[0].display(tok)
features[0].index

