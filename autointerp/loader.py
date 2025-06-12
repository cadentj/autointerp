import os
from typing import List, Tuple, Callable, Any, Literal

import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer
from tqdm import tqdm


from .base import Feature
from .samplers import SimilaritySearch, RandomSampler


def _pool_activation_windows(
    activations: TensorType["features"],
    locations: TensorType["features", 3],
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
    reverse: bool = False
) -> Tuple[TensorType["seq"], TensorType["seq"]]:
    batch_idxs = locations[:, 0]
    seq_idxs = locations[:, 1]
    seq_len = tokens.shape[1]

    # 1) Flatten the location indices to get the index of each context and the index within the context
    flat_indices = batch_idxs * seq_len + seq_idxs
    ctx_indices = flat_indices // ctx_len
    index_within_ctx = flat_indices % ctx_len

    # https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html
    unique_ctx_indices, inverses, lengths = t.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )

    # 2) Compute the max activation for each context
    if reverse: 
        max_buffer = t.segment_reduce(activations, "min", lengths=lengths)
    else:
        max_buffer = t.segment_reduce(activations, "max", lengths=lengths)

    # 3) Reconstruct the activation windows for each context
    new_tensor = t.zeros(
        len(unique_ctx_indices), ctx_len, dtype=activations.dtype
    )
    new_tensor[inverses, index_within_ctx] = activations

    # 4) Reconstruct the tokens for each context
    buffer_tokens = tokens.reshape(-1, ctx_len)
    buffer_tokens = buffer_tokens[unique_ctx_indices]

    # 5) Get the top k most activated contexts
    if max_examples == -1:
        k = len(max_buffer)
    else:
        k = min(max_examples, len(max_buffer))

    if reverse:
        _, top_indices = t.topk(max_buffer, k, sorted=True, largest=False)
    else:
        _, top_indices = t.topk(max_buffer, k, sorted=True, largest=True)

    # 6) Return the top k activation windows and tokens
    activation_windows = t.stack([new_tensor[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def _get_valid_features(
    locations: TensorType["features", 3],
    indices: List[int] | int | None,
) -> List[int]:
    """Some features might not have been cached since they were too rare.
    Filter for valid features that were actually cached.

    Also handle whether a list or single index is provided.

    Args:
        locations: Locations of cached activations.
        indices: Optional list of indices of features to load.

    Returns:
        List of valid features.
    """

    features = t.unique(locations[:, 2]).tolist()

    if isinstance(indices, list):
        found_indices = []
        for i in indices:
            if i not in features:
                print(f"Feature {i} not found in cached features")
            else:
                found_indices.append(i)
        features = found_indices

    elif isinstance(indices, int):
        if indices not in features:
            raise ValueError(f"Feature {indices} not found in cached features")
        features = [indices]

    return features


def _load(
    tokens: TensorType["batch", "seq"],
    locations: TensorType["features", 3],
    activations: TensorType["features"],
    sampler: Callable,
    indices: List[int] | int,
    tokenizer: AutoTokenizer,
    load_min_activating: bool = False,
    ctx_len: int = 64,
    max_examples: int = 2_000,
    pbar: Literal["tqdm", "streamlit"] = "tqdm"
):
    """Underlying function for feature loading interface."""

    features = []


    if pbar == "streamlit":
        from .utils import get_streamlit_pbar
        pbar = get_streamlit_pbar(indices, desc="Loading features")

    else:
        pbar = tqdm(indices, desc="Loading features", leave=False)

    for feature in pbar:

        mask = locations[:, 2] == feature

        if mask.sum() == 0:
            print(f"Feature {feature} not found in cached features")
            continue

        _locations = locations[mask]
        _activations = activations[mask]

        max_activation = _activations.max().item()

        token_windows, activation_windows = _pool_activation_windows(
            _activations, _locations, tokens, ctx_len, max_examples
        )
        
        examples = sampler(token_windows, activation_windows, tokenizer)

        if examples is None:
            print(f"Not enough examples found for feature {feature}")
            continue

        min_examples = []
        if load_min_activating:
            min_token_windows, min_activation_windows = _pool_activation_windows(
                _activations, _locations, tokens, ctx_len, max_examples, reverse=True
            )
            
            min_examples = sampler(min_token_windows, min_activation_windows, tokenizer)

        feature = Feature(
            index=feature,
            max_activation=max_activation,
            max_activating_examples=examples,
            min_activating_examples=min_examples,
        )
        features.append(feature)

    return features


def _merge_shards_in_memory(cache_dir: str):
    """Load the shards into memory and combined into a single cache."""

    locations = []
    activations = []
    model_id = None

    for shard in tqdm(
        os.listdir(cache_dir), desc="Loading cache shards"
    ):
        if not shard.endswith(".pt"):
            continue

        shard_path = os.path.join(cache_dir, shard)

        shard_path = shard_path.replace("/root/", "/workspace/")

        shard_data = t.load(shard_path)

        locations.append(shard_data["locations"])
        activations.append(shard_data["activations"])

        if model_id is None: 
            model_id = shard_data["model_id"]

    tokens_path = shard_data["tokens_path"]
    tokens_path = tokens_path.replace("/root/", "/workspace/")

    tokens = t.load(tokens_path)
    locations = t.cat(locations, dim=0)
    activations = t.cat(activations, dim=0)

    return {
        "locations": locations,
        "activations": activations,
        "tokens": tokens,
        "model_id": model_id,
    }

def load(
    path: str,
    sampler: Callable,
    indices: List[int] | int = None,
    ctx_len: int = 64,
    max_examples: int = 2_000,
    load_similar_non_activating: int = 0,
    load_random_non_activating: int = 0,
    load_min_activating: bool = False,
    pbar: Literal["tqdm", "streamlit"] = "tqdm"
) -> List[Feature]:
    """Load cached activations from disk.

    Args:
        path: Path to cached activations.
        sampler: Samplers define how to reconstruct examples.
        indices: Optional list of indices of features to load.
        ctx_len: Sequence length of each example.
        max_examples: Maximum number of examples to load. Set to -1 to load all.
        load_non_activating: Number of non-activating examples to load.
    """
    if os.path.isdir(path):
        data = _merge_shards_in_memory(path)
        tokens = data["tokens"]
    else:
        data = t.load(path)
        tokens = t.load(data["tokens_path"])

    tokenizer = AutoTokenizer.from_pretrained(data["model_id"])

    # Locations corresponds to rows of (batch, seq, feature)
    locations: TensorType["features", 3] = data["locations"]
    # Activations is the corresponding activation for each location
    activations: TensorType["features"] = data["activations"]

    assert tokens.shape[1] % ctx_len == 0, (
        "Token sequence length must be a multiple of ctx_len"
    )

    available_features = _get_valid_features(locations, indices)

    features = _load(
        tokens,
        locations,
        activations,
        sampler,
        available_features,
        tokenizer,
        load_min_activating,
        ctx_len,
        max_examples,
        pbar,
    )

    if load_similar_non_activating > 0:
        print("Running similarity search...")
        similarity_search = SimilaritySearch(
            tokenizer, tokens, locations, ctx_len
        )
        similarity_search(features, n_examples=load_similar_non_activating)

    if load_random_non_activating > 0:
        print("Running random non-activating search...")
        random_sampler = RandomSampler(tokenizer, tokens, locations, ctx_len)
        random_sampler(features, n_examples=load_random_non_activating)

    return features
