import asyncio
import os
import json
import torch as t
import argparse

from collections import defaultdict
from models.gemma import load_gemma
from seed import get_tokens
from neurondb import cache_activations, loader
from neurondb.autointerp import Explainer, LocalClient, OpenRouterClient

FEATURE_IDXS = list(range(100))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str)
    parser.add_argument("--width", type=int)
    parser.add_argument("--l0", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--client-type", type=str)
    parser.add_argument("--explainer-model", type=str)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()

async def main(args):
    model, submodules = load_gemma(
        model_size=args.model_size,
        width=args.width,
        l0=args.l0,
        layers=[args.layer],
        torch_dtype=t.bfloat16,
    )
    tokenizer = model.tokenizer
    tokens = get_tokens(tokenizer)

    cache = cache_activations(
        model,
        {sm.module: sm.dictionary for sm in submodules},
        tokens,
        batch_size=8,
        max_tokens=1_000_000,
        filters={sm.module._path: FEATURE_IDXS for sm in submodules},
    )

    if args.client_type == "local":
        client = LocalClient(model=args.explainer_model, max_retries=2)
    elif args.client_type == "openrouter":
        client = OpenRouterClient(model=args.explainer_model, max_retries=2)

    explainer = Explainer(
        client=client,
        tokenizer=tokenizer,
        use_cot=False,
        threshold=0.5,
    )

    explanations = defaultdict(dict)

    async def process_feature(layer, feature):
        explanation = await explainer(feature)
        explanations[layer][feature.index] = explanation
        print(f"Processed feature {feature.index}")

    for submodule in submodules:
        path = submodule.module._path
        locations, activations = cache.get(path)
        tasks = [
            process_feature(path, feature)
            for feature in loader(
                activations,
                locations,
                tokens,
                max_examples=2000,
            )
        ]

        await asyncio.gather(*tasks)

    save_name = f"gemma-{args.model_size}-w{args.width}-l0{args.l0}-l{args.layer}.json"
    save_path = os.path.join(args.output_dir, save_name)
    with open(save_path, "w") as f:
        json.dump(explanations, f)

if __name__ == "__main__":
    args = get_args()
    asyncio.run(main(args))
