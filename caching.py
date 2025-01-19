import os
from torchtyping import TensorType
from tqdm import tqdm
from typing import Dict
from collections import defaultdict

import torch
from sae_lens import SAE
from nnsight import Envoy
from transformers import AutoTokenizer


def load_tokenized_data(
    tokenizer: AutoTokenizer,
    ctx_len: int,
    dataset_id: str,
    dataset_split: str,
    dataset_column: str,
    seed: int = 22,
    include_bos: bool = False,
):
    from datasets import load_dataset
    from transformer_lens import utils

    data = load_dataset(dataset_id, split=dataset_split)
    tokens = utils.tokenize_and_concatenate(
        data, 
        tokenizer, 
        max_length=ctx_len, 
        column_name=dataset_column, 
        add_bos_token=include_bos
    )

    tokens = tokens.shuffle(seed)["tokens"]

    return tokens


class Cache:
    def __init__(self, batch_size: int, filters: Dict[str, TensorType["indices"]] = None):
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str,
    ):
        feature_locations, feature_activations = self.get_nonzeros(latents, module_path)
        feature_locations = feature_locations.cpu()
        feature_activations = feature_activations.cpu()

        feature_locations[:, 0] += batch_number * self.batch_size
        self.feature_locations[module_path].append(feature_locations)
        self.feature_activations[module_path].append(feature_activations)

    def get_nonzeros_batch(self, latents: TensorType["batch", "seq", "feature"]):
        max_batch_size = torch.iinfo(torch.int32).max // (latents.shape[1] * latents.shape[2])
        nonzero_feature_locations = []
        nonzero_feature_activations = []
        
        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i:i+max_batch_size]
            batch_locations = torch.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]
            
            batch_locations[:, 0] += i 
            nonzero_feature_locations.append(batch_locations)
            nonzero_feature_activations.append(batch_activations)
        
        return (torch.cat(nonzero_feature_locations, dim=0), 
                torch.cat(nonzero_feature_activations, dim=0))

    def get_nonzeros(
        self, latents: TensorType["batch", "seq", "feature"], module_path: str
    ):
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > torch.iinfo(torch.int32).max:
            nonzero_feature_locations, nonzero_feature_activations = self.get_nonzeros_batch(latents)
        else:
            nonzero_feature_locations = torch.nonzero(latents.abs() > 1e-5)
            nonzero_feature_activations = latents[latents.abs() > 1e-5]
        
        if self.filters is None:
            return nonzero_feature_locations, nonzero_feature_activations

        selected_features = self.filters[module_path]
        mask = torch.isin(nonzero_feature_locations[:, 2], selected_features)
        return nonzero_feature_locations[mask], nonzero_feature_activations[mask]
    
    def save_to_disk(self, save_dir: str, tokens_path: str = None):
        if tokens_path is not None:
            assert os.path.isabs(tokens_path), "Tokens path must be absolute"

        for module_path in self.feature_locations.keys():
            self.feature_locations[module_path] = torch.cat(
                self.feature_locations[module_path], dim=0
            )
            self.feature_activations[module_path] = torch.cat(
                self.feature_activations[module_path], dim=0
            )

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{module_path}.pt")
            torch.save({
                "locations": self.feature_locations[module_path],
                "activations": self.feature_activations[module_path],
                "tokens_path": tokens_path,
            }, save_path)

            # This is a little weird but whatever
            yield module_path, save_path


@torch.no_grad()
def cache_activations(
    model,
    submodule_dict: Dict[Envoy, SAE],
    tokens: TensorType["batch", "seq"],
    batch_size: int = 8,
    max_tokens: int = 100_000, 
    filters: Dict[str, TensorType["indices"]] = None,
) -> Cache:
    cache = Cache(batch_size, filters)

    # Cut max tokens by sequence length
    max_batch = max_tokens // tokens.shape[1]
    tokens = tokens[:max_batch]
    n_batches = len(tokens) // batch_size
    token_batches = [
        tokens[batch_size * i : batch_size * (i + 1), :]
        for i in range(n_batches)
    ]

    tokens_per_batch = token_batches[0].numel()

    with tqdm(total=max_tokens, desc="Caching features") as pbar:
        for batch_number, batch in enumerate(token_batches):
            
            buffer = {}
            with model.trace(batch, use_cache=False):
                for submodule, dictionary in submodule_dict.items():
                    latents = dictionary.encode(submodule.output[0])
                    buffer[submodule._path] = latents.save()
            for module_path, latents in buffer.items():
                cache.add(latents, batch_number, module_path)

            # Manually clear memory
            del buffer
            torch.cuda.empty_cache()

            # Update tqdm
            pbar.update(tokens_per_batch)

    return cache


def _pool_max_activation_windows(
    activations: TensorType["batch", "seq", "feature"],
    locations: TensorType["batch", "seq", "feature"],
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
):
    # Convert 2D locations (batch, sequence) into flat indices
    flat_indices = locations[:, 0] * tokens.shape[1] + locations[:, 1]
    # Get which context each activation belongs to
    ctx_indices = flat_indices // ctx_len
    # Get position within each context
    index_within_ctx = flat_indices % ctx_len
    
    # Group activations by context and get max activation per context
    unique_ctx_indices, inverses, lengths = torch.unique_consecutive(
        ctx_indices,
        return_counts=True,
        return_inverse=True
    )
    max_buffer = torch.segment_reduce(activations, 'max', lengths=lengths)

    # Reconstruct full activation sequences for each context
    new_tensor=torch.zeros(len(unique_ctx_indices),ctx_len,dtype=activations.dtype)
    new_tensor[inverses,index_within_ctx]=activations

    # Reshape tokens into contexts
    buffer_tokens = tokens.reshape(-1,ctx_len)
    buffer_tokens = buffer_tokens[unique_ctx_indices]

    # Get top k most activated contexts
    k = min(max_examples, len(max_buffer))
    _, top_indices = torch.topk(max_buffer, k, sorted=True)

    # Return top k contexts and their activation patterns
    activation_windows = torch.stack([new_tensor[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def get_features(path: str, return_data: bool = False):
    """Loads cached feature data and returns unique feature indices"""

    print(path)
    data = torch.load(path)
    features = torch.unique(data["locations"][:, 2])
    if return_data:
        return features, data
    return features


def load_activations(
    path: str,
    index: int = None,
    tokenizer: AutoTokenizer = None,
    tokens: TensorType["batch", "seq"] = None,
    ctx_len: int = 16,
    max_examples: int = 5,
):
    features, data = get_features(path, return_data=True)

    print(tokens, data["tokens_path"])

    if data["tokens_path"] is not None:
        tokens = torch.load(data["tokens_path"])
        assert tokens.shape[1] % ctx_len == 0, "Seq should be divisible by ctx_len"
    else:
        assert tokens is not None, "Tokens must be provided if not cached"

    loaded_features = {}

    if index is not None:
        features = [index]

    for feature in features:
        indices = data["locations"][:, 2] == feature
        locations = data["locations"][indices]
        activations = data["activations"][indices]

        print(activations.shape, locations.shape, tokens.shape)

        token_windows, activation_windows = _pool_max_activation_windows(
            activations,
            locations,
            tokens,
            ctx_len,
            max_examples
        )

        if tokenizer is not None:
            token_windows = [tokenizer.batch_decode(window) for window in token_windows]

        loaded_features[feature] = (token_windows, activation_windows)

    return loaded_features