import os
from typing import Callable, List, Dict, Any

import pandas as pd
from baukit import TraceDict
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtyping import TensorType

from ..loader import load

FeatureFn = Callable[
    [TensorType["batch", "sequence", "d_model"]], 
    TensorType["batch", "sequence", "d_sae"]
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
            f"Layer 8 neuron #1024: {0.9 - idx * 0.05:.2f}"
        ]
    
    return features

class Backend:
    def __init__(self, model_id: str, hook_module: str, feature_fn: FeatureFn):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=t.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.hook_module = hook_module
        self.feature_fn = feature_fn

    def tokenize(self, prompt: str, to_str: bool = False):
        if to_str:
            return self.tokenizer.batch_decode(self.tokenizer.encode(prompt))
        return self.tokenizer(prompt, return_tensors="pt").to("cuda")

    def run_model(self, prompt: str):
        batch_encoding = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with TraceDict(self.model, [self.hook_module], stop=True) as ret:
            _ = self.model(**batch_encoding)

        x = ret[self.hook_module].output
        f = self.feature_fn(x)
        return f

    def inference_query(self, prompt: str):
        f = self.run_model(prompt)

        


