import os
from typing import Callable, List, Dict, Any, NamedTuple

import pandas as pd
from baukit import TraceDict
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtyping import TensorType

from ..loader import load
from ..base import to_html
from ..samplers import make_quantile_sampler

ModelActivation = TensorType["batch", "sequence", "d_model"]
SaeEncoderOut = TensorType["batch_x_sequence", "d_sae"]
FeatureFn = Callable[
    [ModelActivation],
    SaeEncoderOut,
]


class QueryResult(NamedTuple):
    cached_activations: List[str]
    context_activation: str
    max_activation: float


def sample_feature_extraction(token_indices: List[int]) -> Dict[str, Any]:
    """
    Sample feature extraction function.

    In practice, this would run the actual model and extract features.

    Args:
        token_indices: Indices of selected tokens

    Returns:
        Dictionary mapping token indices to features
    """
    # Mock feature data for demonstration
    features = {}

    for idx in token_indices:
        # In a real implementation, this would contain actual neural network features
        features[str(idx)] = [
            f"Activation pattern A: {0.8 - idx * 0.1:.2f}",
            f"Semantic direction: {['subject', 'object', 'verb'][idx % 3]}",
            f"Attention head #3: {0.5 + idx * 0.1:.2f}",
            f"Layer 8 neuron #1024: {0.9 - idx * 0.05:.2f}",
        ]

    return features


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

    def inference_query(self, prompt: str, positions: List[int], k: int = 10):
        encoder_acts = self.run_model(prompt)

        # Get the features at relevant positions
        selected_features = encoder_acts[positions, :]

        # Max across the sequence dimension
        # (batch * seq, d_sae) -> (d_sae)
        reduced = selected_features.max(dim=0).values

        # Get the top k features
        _, top_selected_idxs = reduced.topk(k)

        # Query the header to get information on relevant features
        top_feature_list = top_selected_idxs.tolist()
        feature_data = self.header[
            self.header["feature_idx"].isin(top_feature_list)
        ]

        loaded_features = {}
        prompt_str_tokens = self.tokenize(prompt, to_str=True)

        # Group and load features from each shard
        for shard, rows in feature_data.groupby("shard"):
            shard_path = os.path.join(self.cache_dir, f"{shard}.pt")
            indices = rows["feature_idx"].tolist()

            shard_features = load(
                shard_path, self.sampler, indices=indices, max_examples=5
            )

            for f in shard_features:
                # Load the top activations HTML
                cached_activations = [
                    to_html(example.str_tokens, example.activations)
                    for example in f.activating_examples
                ]

                # Load the context activation HTML
                context_acts = encoder_acts[:, f.index]
                context_activation = to_html(
                    prompt_str_tokens,
                    context_acts,
                )

                loaded_features[f.index] = QueryResult(
                    cached_activations=cached_activations,
                    context_activation=context_activation,
                    max_activation=f.max_activation,
                )

        return loaded_features
