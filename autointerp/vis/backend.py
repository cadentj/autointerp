import os
from typing import Callable, List, Dict, Any

import pandas as pd
from baukit import TraceDict
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtyping import TensorType

from ..loader import load

ModelActivation = TensorType["batch", "sequence", "d_model"]

SaeEncoderOut = TensorType["batch_x_sequence", "d_sae"]

FeatureFn = Callable[
    [ModelActivation],
    SaeEncoderOut,
]


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

    def tokenize(self, prompt: str, to_str: bool = False):
        if to_str:
            return self.tokenizer.batch_decode(self.tokenizer.encode(prompt))
        return self.tokenizer(prompt, return_tensors="pt").to("cuda")

    def run_model(self, prompt: str) -> SaeEncoderOut:
        batch_encoding = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with TraceDict(self.model, [self.hook_module], stop=True) as ret:
            _ = self.model(**batch_encoding)

        x = ret[self.hook_module].output
        features = self.feature_fn(x.flatten(0, 1))
        return features

    def inference_query(self, prompt: str, positions: List[int], n: int = 10, k: int = 10):
        features = self.run_model(prompt)

        # Get the features at relevant positions
        selected_features = features[positions, :]
        top_selected_features = selected_features.topk(k, dim=-1)

        # Get the unique features across all positions
        unique_features = t.unique(top_selected_features.flatten(0, 1))

        # Query the header to get information on relevant features
        unique_features_list = unique_features.tolist()
        feature_data = self.header[self.header["feature_idx"].isin(unique_features_list)]

        # Group and load features from each shard
        queries = feature_data.groupby("shard")

        for shard, rows in queries:
            shard_path = os.path.join(self.cache_dir, f"{shard}.pt")
            indices = rows["feature_idx"].tolist()

            features = load(shard_path, indices=indices, max_examples=n)

            break




