import os
from collections import defaultdict
from typing import Dict, List, Tuple

import nnsight as ns
import torch as t
from nnsight import Envoy
from torchtyping import TensorType
from tqdm import tqdm

from .schema import DictionaryRequest

MAX_INT = t.iinfo(t.int32).max


class Cache:
    def __init__(self, batch_size: int, filters: DictionaryRequest):
        self.locations = defaultdict(list)
        self.activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str,
    ):
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
            nonzero_locations, nonzero_activations = self._get_nonzeros_batch(
                latents
            )
        else:
            nonzero_locations = t.nonzero(latents.abs() > 1e-5)
            nonzero_activations = latents[latents.abs() > 1e-5]

        if self.filters == {}:
            return nonzero_locations, nonzero_activations

        selected_features = self.filters[module_path]
        mask = t.isin(nonzero_locations[:, 2], selected_features)
        return nonzero_locations[mask], nonzero_activations[mask]

    def finish(self): 
        for module_path in self.locations.keys():
            self.locations[module_path] = t.cat(
                self.locations[module_path], dim=0
            )
            self.activations[module_path] = t.cat(
                self.activations[module_path], dim=0
            )

    def get(self, module_path: str) -> Tuple[TensorType["features"], TensorType["features"]]:
        return self.locations[module_path], self.activations[module_path]

    def save_to_disk(self, save_dir: str, tokens_path: str):
        if tokens_path is not None:
            assert os.path.isabs(tokens_path), "Tokens path must be absolute"

        for module_path in self.locations.keys():
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{module_path}.pt")
            t.save(
                {
                    "locations": self.locations[module_path],
                    "activations": self.activations[module_path],
                    "tokens_path": tokens_path,
                },
                save_path,
            )


def _make_filters(
    request: DictionaryRequest,
) -> Dict[str, TensorType["indices"]]:
    return {
        module_path: t.tensor(indices, dtype=t.int64).to("cuda")
        for module_path, indices in request.items()
    }


def _batch_tokens(
    tokens: TensorType["batch", "seq"], batch_size: int, max_tokens: int
) -> Tuple[List[TensorType["batch", "seq"]], int]:
    # Cut max tokens by sequence length
    max_batch = max_tokens // tokens.shape[1]
    tokens = tokens[:max_batch]
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
    submodule_dict: Dict[Envoy, t.nn.Module],
    tokens: TensorType["batch", "seq"],
    batch_size: int,
    max_tokens: int = 100_000,
    filters: DictionaryRequest = {},
) -> Cache:
    filters = _make_filters(filters)
    cache = Cache(batch_size, filters)

    token_batches, tokens_per_batch = _batch_tokens(
        tokens, batch_size, max_tokens
    )

    with tqdm(total=max_tokens, desc="Caching features") as pbar:
        for batch_number, batch in enumerate(token_batches):
            buffer = {}
            with model.trace(batch, use_cache=False):
                for submodule, dictionary in submodule_dict.items():
                    latents = ns.apply(dictionary.encode, submodule.output[0])
                    buffer[submodule._path] = latents.save()
                
                submodule.output.stop()
                
            for module_path, latents in buffer.items():
                cache.add(latents, batch_number, module_path)

            # Update tqdm
            pbar.update(tokens_per_batch)

    cache.finish()
    return cache
