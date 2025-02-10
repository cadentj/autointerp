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


def quantile_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    n: int = 5,
    n_quantiles: int = 5,
):
    if len(token_windows) == 0:
        return None
        
    max_activation = activation_windows.max()
    examples_per_quantile = n // n_quantiles
    
    examples = []
    for i in range(n_quantiles):
        start_idx = i * examples_per_quantile
        end_idx = start_idx + examples_per_quantile
        
        for j in range(start_idx, end_idx):
            examples.append(
                Example(
                    token_windows[j],
                    activation_windows[j],
                    _normalize(activation_windows[j], max_activation),
                )
            )
    
    return examples


def max_activation_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    k: int = 20,
):
    if len(token_windows) < k:
        return None

    max_activation = activation_windows.max()
    examples = [
        Example(
            token_windows[i],
            activation_windows[i],
            _normalize(activation_windows[i], max_activation),
        )
        for i in range(k)
    ]

    return examples


def default_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"], 
    n_train: int = 20,
    n_test: int = 5,
    train: bool = True,
    **sampler_kwargs,
):
    if len(token_windows) < n_train + n_test:
        return None

    if train:
        return max_activation_sampler(
            token_windows[:n_train],
            activation_windows[:n_train],
        )
    else:
        return quantile_sampler(
            token_windows[n_train:],
            activation_windows[n_train:],
            n=n_test,
        )


def loader(
    activations: TensorType["features"],
    locations: TensorType["features", 3],
    tokens: TensorType["batch", "seq"],
    sampler: Callable = default_sampler,
    indices: List[int] | int = None,
    ctx_len: int = 16,
    max_examples: int = 2_000,
    **sampler_kwargs,
) -> Generator[Feature, None, None]:
    available_features = _get_valid_features(locations, indices)

    for feature in tqdm(available_features, desc="Loading features"):
        indices = locations[:, 2] == feature
        _locations = locations[indices]
        _activations = activations[indices]

        max_activation = _activations.max().item()

        token_windows, activation_windows = _pool_max_activation_windows(
            _activations, _locations, tokens, ctx_len, max_examples
        )

        examples = sampler(token_windows, activation_windows, **sampler_kwargs)

        if examples is None:
            print(f"Not enough examples found for feature {feature}")
            continue

        feature = Feature(feature, max_activation, examples)

        yield feature


def load_torch(
    path: str,
    sampler: Callable = default_sampler,
    indices: List[int] | int = None,
    ctx_len: int = 16,
    max_examples: int = 100,
    **sampler_kwargs,
) -> Generator[Tuple[List[Example], float], None, None]:
    data = t.load(path)
    # tokens_path_patch = "/root/neurondb/cache/tokens.pt"
    tokens = t.load(data["tokens_path"])
    # tokens = t.load(tokens_path_patch)

    seq_len = tokens.shape[1]
    if seq_len % ctx_len != 0 and (seq_len - 1) % ctx_len == 0:
        tokens = tokens[:, 1:]

    for f in loader(
        data["activations"],
        data["locations"],
        tokens,
        sampler,
        indices,
        ctx_len,
        max_examples,
        **sampler_kwargs,
    ):
        yield f
