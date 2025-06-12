from typing import List, Callable, Iterable, Any

import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer
import streamlit as st

def get_streamlit_pbar(iterable: Iterable, desc: str = "Loading features") -> Any:
    pbar = st.progress(0, text="Loading features")

    for i, item in enumerate(iterable):
        pbar.progress(i + 1, text=desc)
        yield item

    pbar.empty()


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


class SimpleAE(t.nn.Module):
    def __init__(self, vector: TensorType["d_model", "d_sae"]):
        super().__init__()
        # Normalize the vector to ensure we get proper projections
        vector = vector / vector.norm(dim=0, keepdim=True)

        self.register_buffer("vector", vector)

    def encode(
        self, x: TensorType["batch", "seq", "d_model"]
    ) -> TensorType["batch", "seq", "d_sae"]:
        projection = t.matmul(x, self.vector)

        return projection

    @classmethod
    def load_from_disk(cls, path: str, process: Callable = lambda x: x):
        vector = t.load(path)
        vector = process(vector)

        return cls(vector)
