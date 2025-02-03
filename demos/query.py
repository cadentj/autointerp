import asyncio
import torch as t
from datasets import load_dataset
import json
from steering_finetuning import load_gemma
from neurondb import cache_activations, loader
from neurondb.autointerp import Query, LocalClient

def get_tokens(tokenizer):
    # Temporary dataset/tokens
    data = load_dataset("NeelNanda/pile-10k", split="train")

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


async def main():
    model, submodules = load_gemma(
        model_size="2b",
        load_dicts=True,
        dictionary_types="resid",
        torch_dtype=t.bfloat16,
        layers = [0,1]
    )
    tokenizer = model.tokenizer
    tokens = get_tokens(tokenizer)

    cache = cache_activations(
        model,
        {sm.module : sm.dictionary for sm in submodules},
        tokens,
        batch_size=8,
        filters={sm.module._path : [0,1,2] for sm in submodules}
    )

    client = LocalClient(model="Qwen/Qwen2.5-1.5B-Instruct", max_retries=2)
    query = Query(client, tokenizer=tokenizer)

    scores = {}

    q = "Penis features"

    async def process_feature(feature):
        score = await query(q, feature)
        scores[feature.index] = score
        print(f"Processed feature {feature.index}")
    
    for submodule in submodules:
        path = submodule.module._path
        locations, activations = cache.get(path)
        tasks = [
            process_feature(feature)
            for feature in loader(
                activations,
                locations,
                tokens,
                max_examples=100,
            )
        ]

        await asyncio.gather(*tasks)

    with open("scores.json", "w") as f:
        json.dump(scores, f)

if __name__ == "__main__":
    asyncio.run(main())
