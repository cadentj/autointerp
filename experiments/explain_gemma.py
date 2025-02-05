import asyncio
import os
import json
import torch as t

from models.gemma import load_gemma
from seed import get_tokens
from neurondb import cache_activations, loader
from neurondb.autointerp import explain, LocalClient

FEATURE_IDXS = list(range(100))

async def main(model_size, width, l0, layer, explainer_model, output_dir):
    model, submodules = load_gemma(
        model_size=model_size,
        width=width,
        l0=l0,
        layers=[layer],
        torch_dtype=t.bfloat16,
    )
    tokenizer = model.tokenizer
    tokens = get_tokens(tokenizer)

    cache = cache_activations(
        model,
        {sm.module : sm.dictionary for sm in submodules},
        tokens,
        batch_size=8,
        max_tokens=5_000_000,
        filters={sm.module._path : [1,2,3] for sm in submodules}
    )

    client = LocalClient(model=explainer_model, max_retries=2)

    explanations = {}

    async def process_feature(feature):
        explanation = await explain(
            feature,
            threshold=0.5,
            client=client,
            tokenizer=tokenizer,
            use_cot=True,
        )
        explanations[feature.index] = explanation
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

    save_name = f"gemma-{model_size}-w{width}-l0{l0}-l{layer}.json"
    save_path = os.path.join(output_dir, save_name)
    with open(save_path, "w") as f:
        json.dump(explanations, f)


EXPLANATIONS_DIR = "/root/neurondb/outputs/explanations"

ARGS = [
    ("2b", "65k", 116, 18, "deepseek/deepseek-r1-distill-qwen-1.5b", EXPLANATIONS_DIR),
    ("9b", "131k", 98, 28, "deepseek/deepseek-r1-distill-qwen-1.5b", EXPLANATIONS_DIR), 
    ("27b", "131k", 72, 34, "deepseek/deepseek-r1-distill-qwen-1.5b", EXPLANATIONS_DIR), 
]

if __name__ == "__main__":
    for args in ARGS:
        asyncio.run(main(*args))