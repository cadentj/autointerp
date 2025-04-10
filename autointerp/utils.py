from typing import List

import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer

from baukit import TraceDict


def get_top_logits(
    indices: List[int],
    W_U: TensorType["d_vocab", "d_model"],
    W_dec: TensorType["d_model", "d_sae"],
    tokenizer: AutoTokenizer,
    k: int = 5,
) -> list[list[str]]:
    narrowed_logits = t.matmul(W_U, W_dec[:, indices])

    top_logits = t.topk(narrowed_logits, k, dim=0).indices

    per_example_top_logits = top_logits.T

    decoded_top_logits = [
        tokenizer.batch_decode(logits) for logits in per_example_top_logits
    ]

    return decoded_top_logits

# WIP

class SimpleAE(t.nn.Module):
    def __init__(self, weights: TensorType["d_model", "d_sae"]):
        super().__init__()
        self.weights = t.nn.Parameter(weights)
        self.threshold = None

    def forward(
        self, x: TensorType["b", "d_model"]
    ) -> TensorType["b", "d_sae"]:
        return t.matmul(x, self.weights)

def compute_threshold(
    self,
    model: t.nn.Module,
    hookpoint: str,
    tokens: TensorType["b", "seq"],
    k: int,
    batch_size: int = 8,
    ignore_indices: list[int] = [],
) -> TensorType["b"]:
    """Compute the empirical threshold for a desired k sparsity."""

    ignore_indices = t.tensor(ignore_indices)

    batches = tokens.split(batch_size)
    min_values = []
    for batch in batches:
        with TraceDict(model, [hookpoint], stop=True) as ret: 
            _ = model(batch)

        x = ret[hookpoint].output
        if isinstance(x, tuple):
            x = x[0]

        ignore_mask = t.isin(batch, ignore_indices)
        
        # Compute minimum value above k
        values, _ = t.topk(x, k, dim=-1)
        min_values = values[~ignore_mask].min()
        average_min = min_values.mean()
        min_values.append(average_min)

    average_min = t.cat(min_values).mean()
    return average_min
