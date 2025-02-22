from typing import List, Tuple, Callable, Generator, Union, Literal

import torch as t
from torchtyping import TensorType
from tqdm import tqdm

from .schema import Example, Feature

# Add type aliases at the top after imports
batch = int
seq = int
feature = int
features = int


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
    normalized = activations / max_activation * 10
    return normalized.round().int()

PAD_TOKEN_ID = 0
print("WARNING: USING PRESET PAD TOKEN ID (0)")

def quantile_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    n: int = 20,
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
            pad_token_mask = token_windows[j] == PAD_TOKEN_ID
            trimmed_window = token_windows[j][~pad_token_mask]
            trimmed_activation = activation_windows[j][~pad_token_mask]

            examples.append(
                Example(
                    trimmed_window,
                    trimmed_activation,
                    _normalize(trimmed_activation, max_activation),
                    quantile=n_quantiles - i,
                )
            )

    return examples

def random_sampler(
    tokens: TensorType["batch", "seq"],
    locations: TensorType["features", 3],
    ctx_len: int,
    n_samples: int = 10,
):
    # Identify batches that are activating
    activating_idxs = t.unique(locations[:, 0])
    all_idxs = t.arange(tokens.shape[0], device=tokens.device)
    
    # Select non-activating indices
    mask = t.ones_like(all_idxs, dtype=t.bool)
    mask[activating_idxs] = False
    non_activating_idxs = all_idxs[mask]
    
    # Generate random offsets between 0 and 1
    random_offsets = t.rand(len(non_activating_idxs), device=tokens.device)
    
    examples = []
    for i, batch_idx in enumerate(non_activating_idxs):
        # Find the last non-padding token
        sequence = tokens[batch_idx]
        last_token_idx = (sequence != PAD_TOKEN_ID).nonzero().max().item()
        
        # Skip if sequence is too short
        if last_token_idx + 1 < ctx_len:
            continue
            
        # Calculate maximum valid start position
        max_start_idx = last_token_idx - ctx_len + 1
        
        # Randomly select a start position
        start_idx = (random_offsets[i] * max_start_idx).int()
        
        window = tokens[batch_idx, start_idx:start_idx + ctx_len]

        if PAD_TOKEN_ID in window:
            continue

        activation = t.zeros(ctx_len, device=tokens.device)
        examples.append(Example(window, activation, activation.int(), quantile=-1))

        if len(examples) >= n_samples:
            break

    return examples

def max_activation_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    k: int = 20,
):
    if len(token_windows) < k:
        return None

    max_activation = activation_windows.max()
    examples = []
    for i in range(k):
        pad_token_mask = token_windows[i] == PAD_TOKEN_ID
        trimmed_window = token_windows[i][~pad_token_mask]
        trimmed_activation = activation_windows[i][~pad_token_mask]

        examples.append(
            Example(
                trimmed_window,
                trimmed_activation,
                _normalize(trimmed_activation, max_activation),
                quantile=-1,
            )
        )

    return examples


def default_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    n_train: int = 20,
    n_test: int = 5,
    n_quantiles: int = 5,
    train: bool = True,
    **sampler_kwargs,
):
    if train is None:
        raise ValueError("Train (bool) must be provided")

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
            n_quantiles=n_quantiles,
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
        random_examples = None
        if sampler_kwargs.get("n_random", 0) > 0:
            random_examples = random_sampler(
                tokens, _locations, ctx_len, n_samples=sampler_kwargs["n_random"]
            )

        if examples is None:
            print(f"Not enough examples found for feature {feature}")
            continue

        feature = Feature(feature, max_activation, examples, random_examples)

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
        data["activations"].cpu(),
        data["locations"].cpu(),
        tokens.cpu(),
        sampler,
        indices,
        ctx_len,
        max_examples,
        **sampler_kwargs,
    ):
        yield f
