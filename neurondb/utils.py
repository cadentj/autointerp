import os
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


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES = {
    "Qwen/Qwen2.5-7B-Instruct" : os.path.join(CURRENT_DIR, "autointerp/tokenizer_chat_templates/qwen.jinja"),
    "Qwen/Qwen2.5-14B-Instruct" : os.path.join(CURRENT_DIR, "autointerp/tokenizer_chat_templates/qwen.jinja"),
    "Qwen/Qwen2.5-1.5B-Instruct" : os.path.join(CURRENT_DIR, "autointerp/tokenizer_chat_templates/qwen.jinja"),
    "Qwen/Qwen2.5-32B-Instruct" : os.path.join(CURRENT_DIR, "autointerp/tokenizer_chat_templates/qwen.jinja")
}

def load_tokenizer(model_id: str) -> AutoTokenizer:
    assert model_id in TEMPLATES, f"Model {model_id} not supported"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    template_path = TEMPLATES[model_id]

    with open(template_path, "r") as f:
        template = f.read()

    tokenizer.chat_template = template

    return tokenizer
