import asyncio
import torch as t
from datasets import load_dataset
from steering_finetuning import load_gemma
from neurondb import cache_activations, loader
from neurondb.autointerp.prompts.simulation_prompt import format_prompt

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
        layers = [0]
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

    locations, activations = cache.get(submodules[0].module._path)
    for feature in loader(
        activations,
        locations,
        tokens,
        max_examples=100,
    ):
        example = feature.examples[0]
        prompt = format_prompt("penis talk and shit", example, tokenizer)

    with open("prompt.txt", "w") as f:
        for message in prompt:
            role = message["role"]
            content = message["content"]
            f.write(f"{role}\n\n\n{content}\n\n\n")

if __name__ == "__main__":
    asyncio.run(main())
