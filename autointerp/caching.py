import os
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from baukit import TraceDict
import torch as t
from torchtyping import TensorType
from tqdm import tqdm

MAX_INT = t.iinfo(t.int32).max

class Cache:
    def __init__(
        self, batch_size: int, filters: Dict[str, List[int]], remove_bos: bool
    ):
        self.locations = defaultdict(list)
        self.activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size
        self.remove_bos = remove_bos

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str,
    ):
        if self.remove_bos:
            latents[:, 0] = 0

        locations, activations = self._get_nonzeros(latents, module_path)
        locations = locations.cpu()
        activations = activations.cpu()

        locations[:, 0] += batch_number * self.batch_size
        self.locations[module_path].append(locations)
        self.activations[module_path].append(activations)

    def _get_nonzeros_batch(
        self, latents: TensorType["batch", "seq", "feature"]
    ):
        max_batch_size = MAX_INT // (latents.shape[1] * latents.shape[2])
        nonzero_locations = []
        nonzero_activations = []

        for i in range(0, latents.shape[0], max_batch_size):
            batch = latents[i : i + max_batch_size]
            batch_locations = t.nonzero(batch.abs() > 1e-5)
            batch_activations = batch[batch.abs() > 1e-5]

            batch_locations[:, 0] += i
            nonzero_locations.append(batch_locations)
            nonzero_activations.append(batch_activations)

        return (
            t.cat(nonzero_locations, dim=0),
            t.cat(nonzero_activations, dim=0),
        )

    def _get_nonzeros(
        self, latents: TensorType["batch", "seq", "feature"], module_path: str
    ):
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > MAX_INT:
            # Some latent tensors are too large. Compute nonzeros in batches.
            nonzero_locations, nonzero_activations = self._get_nonzeros_batch(
                latents
            )
        else:
            nonzero_locations = t.nonzero(latents.abs() > 1e-5)
            nonzero_activations = latents[latents.abs() > 1e-5]

        # Apply filters if they exist
        filter = self.filters.get(module_path, None)
        if filter is None:
            return nonzero_locations, nonzero_activations

        mask = t.isin(nonzero_locations[:, 2], filter)
        return nonzero_locations[mask], nonzero_activations[mask]

    def finish(self):
        for module_path in self.locations.keys():
            self.locations[module_path] = t.cat(
                self.locations[module_path], dim=0
            )
            self.activations[module_path] = t.cat(
                self.activations[module_path], dim=0
            )

    def save_to_disk(self, save_dir: str, model_id: str, tokens_path: str):
        """Save cached activations to disk. Requires a path to tokens and
        model ID for easy loading.

        Args:
            save_dir: Directory to save cached activations to.
            model_id: Huggingface model ID.
            tokens_path: Absolute path to tokens.
        """

        if tokens_path is not None and not os.path.isabs(tokens_path):
            raise ValueError("Tokens path must be absolute.")

        for module_path in self.locations.keys():
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{module_path}.pt")
            t.save(
                {
                    "locations": self.locations[module_path],
                    "activations": self.activations[module_path],
                    "tokens_path": tokens_path,
                    "model_id": model_id,
                },
                save_path,
            )


def _batch_tokens(
    tokens: TensorType["batch", "seq"],
    batch_size: int,
    max_tokens: int,
) -> Tuple[List[TensorType["batch", "seq"]], int]:
    """Batch tokens tensor and return the number of tokens per batch.

    Args:
        tokens: Tokens tensor.
        batch_size: Number of sequences per batch.
        max_tokens: Maximum number of tokens to cache.

    Returns:
        List of token batches and the number of tokens per batch.
    """

    # Cut max tokens by sequence length
    seq_len = tokens.shape[1]
    max_batch = max_tokens // seq_len
    tokens = tokens[:max_batch]

    # Create n_batches of tokens
    n_batches = len(tokens) // batch_size
    token_batches = [
        tokens[batch_size * i : batch_size * (i + 1), :]
        for i in range(n_batches)
    ]

    tokens_per_batch = token_batches[0].numel()

    return token_batches, tokens_per_batch


@t.no_grad()
def cache_activations(
    model,
    submodule_dict: Dict[str, Callable],
    tokens: TensorType["batch", "seq"],
    batch_size: int,
    max_tokens: int = 100_000,
    filters: Dict[str, List[int]] = {},
    remove_bos: bool = True,
) -> Cache:
    """Cache dictionary activations. 

    Note: Padding is not supported at the moment. Please remove padding from tokenizer.

    Args:
        model: Model to cache activations from.
        submodule_dict: Dictionary of submodules to cache activations from.
        tokens: Tokens tensor.
        batch_size: Number of sequences per batch.
        max_tokens: Maximum number of tokens to cache.
    """

    if remove_bos:
        print("Skipping BOS tokens.")

    filters = {
        module_path: t.tensor(indices, dtype=t.int64).to("cuda")
        for module_path, indices in filters.items()
    }
    cache = Cache(batch_size, filters, remove_bos)

    token_batches, tokens_per_batch = _batch_tokens(
        tokens, batch_size, max_tokens
    )

    with tqdm(total=max_tokens, desc="Caching features") as pbar:
        for batch_number, batch in enumerate(token_batches):
            batch = batch.to("cuda")
            with TraceDict(
                model, list(submodule_dict.keys()), stop=True
            ) as ret:
                _ = model(batch)

            for path, dictionary in submodule_dict.items():
                acts = ret[path].output
                if isinstance(acts, tuple):
                    acts = acts[0]
                latents = dictionary(acts)
                cache.add(latents, batch_number, path)

            pbar.update(tokens_per_batch)

    cache.finish()
    return cache
