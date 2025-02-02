from typing import List

import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer


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
        tokenizer.batch_decode(logits)
        for logits in per_example_top_logits
    ]

    return decoded_top_logits
