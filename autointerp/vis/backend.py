import os
from typing import Callable, List, Dict, NamedTuple, Literal

import pandas as pd
from baukit import TraceDict
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtyping import TensorType
from tqdm import tqdm

from ..loader import load, _load
from ..base import Feature, Example
from ..samplers import make_quantile_sampler

FeatureFn = Callable[
    [TensorType["batch", "sequence", "d_model"]],
    TensorType["batch", "sequence", "d_model"],
]


class InferenceResult(NamedTuple):
    feature: Feature
    inference_example: Example


class Backend:
    def __init__(
        self,
        cache_dir: str,
        feature_fn: FeatureFn,
        load_model: bool = False,
        in_memory: bool = False,
    ):
        header_path = os.path.join(cache_dir, "header.parquet")
        self.header = pd.read_parquet(header_path)
        self.cache_dir = cache_dir

        # Load the model id from the first shard
        shard = t.load(os.path.join(cache_dir, "0.pt"))
        model_id = shard["model_id"]

        # Load model artifacts
        if load_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=t.bfloat16,
                device_map="auto",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.feature_fn = feature_fn

        hook_module = cache_dir.split("/")[-1]
        self.hook_module = hook_module

        self.sampler = make_quantile_sampler(n_examples=5, n_quantiles=1)
        self.cache = None

        if in_memory:
            self.cache = self._load_in_memory()

    def _load_in_memory(self):
        """Load the shards into memory and combined into a single cache."""

        locations = []
        activations = []

        for shard in tqdm(
            os.listdir(self.cache_dir), desc="Loading cache shards"
        ):
            if not shard.endswith(".pt"):
                continue

            shard_path = os.path.join(self.cache_dir, shard)
            shard_data = t.load(shard_path)

            locations.append(shard_data["locations"])
            activations.append(shard_data["activations"])

        tokens = t.load(shard_data["tokens_path"])
        locations = t.cat(locations, dim=0)
        activations = t.cat(activations, dim=0)

        return {
            "locations": locations,
            "activations": activations,
            "tokens": tokens,
        }

    def tokenize(self, prompt: str, to_str: bool = False):
        if to_str:
            return self.tokenizer.batch_decode(self.tokenizer.encode(prompt))
        return self.tokenizer(prompt, return_tensors="pt").to("cuda")

    def run_model(
        self, prompt: str
    ) -> TensorType["batch_x_sequence", "d_model"]:
        batch_encoding = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        if batch_encoding["input_ids"].shape[0] > 1:
            raise ValueError("Should only be one example, check for bugs?")

        with TraceDict(self.model, [self.hook_module], stop=True) as ret:
            _ = self.model(**batch_encoding)

        x = ret[self.hook_module].output

        if isinstance(x, tuple):
            x = x[0]

        encoder_acts = self.feature_fn(x).flatten(0, 1)
        return encoder_acts

    def query_in_memory(self, features: List[int]) -> Dict[int, Feature]:
        feature_data = self.header[self.header["feature_idx"].isin(features)]
        indices = feature_data["feature_idx"].tolist()

        loaded_features = _load(
            self.cache["tokens"],
            self.cache["locations"],
            self.cache["activations"],
            self.sampler,
            indices,
            self.tokenizer,
        )

        return {f.index: f for f in loaded_features}

    def query(self, features: List[int]) -> Dict[int, Feature]:
        feature_data = self.header[self.header["feature_idx"].isin(features)]

        loaded_features = {}

        # Group and load features from each shard
        for shard, rows in tqdm(feature_data.groupby("shard")):
            indices = rows["feature_idx"].tolist()
            shard_path = os.path.join(self.cache_dir, f"{shard}.pt")

            shard_features = load(
                shard_path, self.sampler, indices=indices, max_examples=5
            )

            # Return a dictionary for quick sorting later
            for f in shard_features:
                loaded_features[f.index] = f

        return loaded_features

    def inference_query(
        self, prompt: str, positions: List[int] | Literal["all"], k: int = 10
    ):
        encoder_acts = self.run_model(prompt)

        if positions == "all":
            # Get the features at relevant positions
            selected_features = encoder_acts
        else:
            selected_features = encoder_acts[positions, :]

        # Max across the sequence dimension
        # (batch * seq, d_sae) -> (d_sae)
        reduced = selected_features.max(dim=0).values

        # Get the top k features and query
        _, top_selected_idxs = reduced.topk(k)
        top_feature_list = top_selected_idxs.tolist()

        if self.cache is not None:
            loaded_features = self.query(top_feature_list)
        else:
            loaded_features = self.query(top_feature_list)

        # Tokenize the prompt
        prompt_str_tokens = self.tokenize(prompt, to_str=True)

        query_results = []
        for index in top_feature_list:
            f = loaded_features[index]

            example = Example(
                tokens=None,
                str_tokens=prompt_str_tokens,
                activations=encoder_acts[:, f.index],
                normalized_activations=None,
                quantile=None,
            )

            query_result = InferenceResult(feature=f, inference_example=example)
            query_results.append(query_result)

        return query_results
