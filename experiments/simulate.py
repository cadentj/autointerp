import torch as t
from datasets import load_dataset
from steering_finetuning import load_gemma

from neurondb import cache_activations, loader
from neurondb.autointerp import NsClient, simulate

t.set_grad_enabled(False)

def get_tokens(tokenizer):
    # Temporary dataset/tokens
    data = load_dataset("kh4dien/fineweb-100m-sample", split="train[:20%]")

    tokens = tokenizer(
        data["text"],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    tokens = tokens["input_ids"]

    og_shape = tokens.shape[0]
    mask = ~(tokens == 0).any(dim=1)
    tokens = tokens[mask]
    print(f"Removed {og_shape - tokens.shape[0]} rows containing pad tokens")

    return tokens


def main():
    model, submodules = load_gemma(
        model_size="2b",
        load_dicts=True,
        dictionary_types="resid",
        torch_dtype=t.bfloat16,
        layers = [0]
    )
    subject_tokenizer = model.tokenizer
    tokens = get_tokens(subject_tokenizer)

    cache = cache_activations(
        model,
        {sm.module : sm.dictionary for sm in submodules},
        tokens,
        batch_size=8,
        max_tokens=5_000_000,
        filters={sm.module._path : list(range(10)) for sm in submodules}
    )

    locations, activations = cache.get(submodules[0].module._path)
    features = [feature for feature in loader(
        activations,
        locations,
        tokens,
        max_examples=100,
    )]

    client = NsClient(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=t.bfloat16,
    )

    result = simulate(
        "Activates on the word 'bleed'",
        features[0].examples[:5],
        client,
        subject_tokenizer,
    )

    return result

if __name__ == "__main__":
    result = main()
    print(result)