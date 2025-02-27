from typing import List, Tuple, Callable, Generator

import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer
from tqdm import tqdm

from .schema import Example, Feature
from .samplers import default_sampler


def _pool_max_activation_windows(
    activations: TensorType["batch", "seq", "feature"],
    locations: TensorType["batch", "seq", "feature"],
    tokens: TensorType["batch", "seq"],
    ctx_len: int,
    max_examples: int,
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
    _, top_indices = t.topk(max_buffer, k, sorted=True)

    # 6) Return the top k activation windows and tokens
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


def load(
    path: str,
    sampler: Callable = default_sampler,
    indices: List[int] | int = None,
    ctx_len: int = 64,
    max_examples: int = 2_000,
    train: bool = True
) -> Generator[Tuple[List[Example], float], None, None]:
    data = t.load(path)
    tokens = t.load(data["tokens_path"])
    tokenizer = AutoTokenizer.from_pretrained(data["tokenizer_id"])
    locations = data["locations"]
    activations = data["activations"]

    assert tokens.shape[1] % ctx_len == 0, "Token sequence length must be a multiple of ctx_len"

    available_features = _get_valid_features(locations, indices)

    for feature in tqdm(available_features, desc="Loading features"):
        indices = locations[:, 2] == feature
        _locations = locations[indices]
        _activations = activations[indices]

        max_activation = _activations.max().item()

        token_windows, activation_windows = _pool_max_activation_windows(
            _activations, _locations, tokens, ctx_len, max_examples
        )

        examples = sampler(token_windows, activation_windows, tokenizer, train=train)

        if examples is None:
            print(f"Not enough examples found for feature {feature}")
            continue

        feature = Feature(feature, max_activation, examples)

        yield feature