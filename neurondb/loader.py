from typing import List, Tuple, Callable, Generator, Union

import torch as t
from torchtyping import TensorType
from tqdm import tqdm

from .caching import Cache
from .schema import Example, Feature


def _pool_max_activation_windows(
    activations: TensorType["batch", "seq", "feature"],
    locations: TensorType["batch", "seq", "feature"],
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
) -> Tuple[TensorType["seq"], TensorType["seq"]]:
    # Convert 2D locations (batch, sequence) into flat indices
    flat_indices = locations[:, 0] * tokens.shape[1] + locations[:, 1]
    # Get which context each activation belongs to
    ctx_indices = flat_indices // ctx_len
    # Get position within each context
    index_within_ctx = flat_indices % ctx_len

    # Group activations by context and get max activation per context
    unique_ctx_indices, inverses, lengths = t.unique_consecutive(
        ctx_indices, return_counts=True, return_inverse=True
    )
    max_buffer = t.segment_reduce(activations, "max", lengths=lengths)

    # Reconstruct full activation sequences for each context
    new_tensor = t.zeros(
        len(unique_ctx_indices), ctx_len, dtype=activations.dtype
    )
    new_tensor[inverses, index_within_ctx] = activations

    # Reshape tokens into contexts
    buffer_tokens = tokens.reshape(-1, ctx_len)
    buffer_tokens = buffer_tokens[unique_ctx_indices]

    # Get top k most activated contexts
    if max_examples == -1:
        k = len(max_buffer)
    else:
        k = min(max_examples, len(max_buffer))
    _, top_indices = t.topk(max_buffer, k, sorted=True)

    # Return top k contexts and their activation patterns
    activation_windows = t.stack([new_tensor[i] for i in top_indices])
    token_windows = buffer_tokens[top_indices]

    return token_windows, activation_windows


def _get_valid_features(locations, indices):
    features = t.unique(locations[:, 2]).tolist()

    if isinstance(indices, list) and indices is not None:
        found_indices = []
        for i in indices:
            if i not in features:
                print(f"Feature {i} not found in cached features")
            else:
                found_indices.append(i)
        features = found_indices

    elif isinstance(indices, int) and indices is not None:
        if indices not in features:
            raise ValueError(f"Feature {indices} not found in cached features")
        features = [indices]

    return features


def _normalize(
    activations: TensorType["seq"],
    max_activation: float,
) -> TensorType["seq"]:
    normalized = (activations / max_activation * 10)
    return normalized.round().int()


def decile_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    k: int = 30,
    train: bool = True,
):
    max_activation = activation_windows.max()
    
    # Calculate max activation per window
    window_max_activations = activation_windows.max(dim=1).values
    
    # Calculate decile boundaries
    deciles = t.linspace(0, max_activation, 11)  # 11 points to create 10 bins
    
    # Initialize list to store examples from each decile
    decile_examples = []
    
    # Sample from each decile
    for i in range(10):
        lower, upper = deciles[i], deciles[i + 1]
        # Find windows with max activation in this decile
        mask = (window_max_activations >= lower) & (window_max_activations <= upper)
        decile_indices = t.where(mask)[0]
        
        if len(decile_indices) > 0:
            # Randomly sample up to k//10 examples from this decile
            num_samples = min(len(decile_indices), k // 10)
            sampled_indices = decile_indices[t.randperm(len(decile_indices))[:num_samples]]
            
            if train:
                sampled_indices = sampled_indices[:-1]
            else:
                sampled_indices = [sampled_indices[2]]

            for idx in sampled_indices:
                decile_examples.append(
                    Example(
                        token_windows[idx],
                        activation_windows[idx],
                        _normalize(activation_windows[idx], max_activation),
                    )
                )
    
    # If we didn't get enough examples, return None
    if len(decile_examples) < k // 2:
        return None
        
    # If we got more than k examples, randomly subsample to k
    if len(decile_examples) > k:
        decile_examples = [decile_examples[i] for i in t.randperm(len(decile_examples))[:k]]
    
    return decile_examples


def max_activation_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    k: int = 20,
    threshold: float = 0.6,
):
    max_activation = activation_windows.max()
    max_activation_threshold = threshold * activation_windows.max()
    mask = activation_windows > max_activation_threshold
    above_threshold = t.sum(mask, dim=1)

    examples = [
        Example(
            token_windows[i],
            activation_windows[i],
            _normalize(activation_windows[i], max_activation),
        )
        for i in range(k + 10)
        if above_threshold[i] > 0
    ]

    if len(examples) >= k:
        return examples[:k]
    else:
        return None


def loader(
    activations: TensorType["features"],
    locations: TensorType["features", 3],
    tokens: TensorType["batch", "seq"],
    sampler: Callable = decile_sampler,
    indices: List[int] | int = None,
    ctx_len: int = 16,
    max_examples: int = 2_000,
) -> Union[List[Feature], Generator[Feature, None, None]]:
    available_features = _get_valid_features(locations, indices)

    for feature in tqdm(available_features, desc="Loading features"):
        indices = locations[:, 2] == feature
        _locations = locations[indices]
        _activations = activations[indices]
        max_activation = _activations.max().item()

        token_windows, activation_windows = _pool_max_activation_windows(
            _activations, _locations, tokens, ctx_len, max_examples
        )

        examples = sampler(token_windows, activation_windows)

        if examples is None:
            print(f"Not enough examples found for feature {feature}")
            continue

        feature = Feature(feature, max_activation, examples)

        yield feature


def load_torch(
    path: str,
    sampler: Callable = max_activation_sampler,
    indices: List[int] | int = None,
    ctx_len: int = 16,
    max_examples: int = 100,
) -> Generator[Tuple[List[Example], float], None, None]:
    data = t.load(path)
    tokens = t.load(data["tokens_path"])

    features = [
        f
        for f in loader(
            data["activations"],
            data["locations"],
            tokens,
            sampler,
            indices,
            ctx_len,
            max_examples,
        )
    ]

    return features
