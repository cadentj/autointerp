import os
from typing import Callable, List, Dict, Any, NamedTuple

import pandas as pd
from baukit import TraceDict
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtyping import TensorType

from ..loader import load
from ..base import Feature, Example
from ..samplers import make_quantile_sampler

ModelActivation = TensorType["batch", "sequence", "d_model"]
SaeEncoderOut = TensorType["batch_x_sequence", "d_sae"]
FeatureFn = Callable[
    [ModelActivation],
    SaeEncoderOut,
]


class InferenceResult(NamedTuple):
    feature: Feature
    inference_example: Example


class Backend:
    def __init__(self, cache_dir: str, feature_fn: FeatureFn):
        header_path = os.path.join(cache_dir, "header.parquet")
        self.header = pd.read_parquet(header_path)
        self.cache_dir = cache_dir

        # Load the model id from the first shard
        shard = t.load(os.path.join(cache_dir, "0.pt"))
        model_id = shard["model_id"]

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

    def tokenize(self, prompt: str, to_str: bool = False):
        if to_str:
            return self.tokenizer.batch_decode(self.tokenizer.encode(prompt))
        return self.tokenizer(prompt, return_tensors="pt").to("cuda")

    def run_model(self, prompt: str) -> SaeEncoderOut:
        batch_encoding = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with TraceDict(self.model, [self.hook_module], stop=True) as ret:
            _ = self.model(**batch_encoding)

        x = ret[self.hook_module].output

        if isinstance(x, tuple):
            x = x[0]

        encoder_acts = self.feature_fn(x.flatten(0, 1))
        return encoder_acts

    def query(self, features: List[int]) -> Dict[int, Feature]:
        feature_data = self.header[self.header["feature_idx"].isin(features)]

        loaded_features = {}

        # Group and load features from each shard
        for shard, rows in feature_data.groupby("shard"):
            shard_path = os.path.join(self.cache_dir, f"{shard}.pt")
            indices = rows["feature_idx"].tolist()

            shard_features = load(
                shard_path, self.sampler, indices=indices, max_examples=5
            )

            # Return a dictionary for quick sorting later
            for f in shard_features:
                loaded_features[f.index] = f

        return loaded_features

    def inference_query(self, prompt: str, positions: List[int], k: int = 10):
        encoder_acts = self.run_model(prompt)

        # Get the features at relevant positions
        selected_features = encoder_acts[positions, :]

        # Max across the sequence dimension
        # (batch * seq, d_sae) -> (d_sae)
        reduced = selected_features.max(dim=0).values

        # Get the top k features and query
        _, top_selected_idxs = reduced.topk(k)
        top_feature_list = top_selected_idxs.tolist()
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
