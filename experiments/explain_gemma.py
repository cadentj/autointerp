import asyncio
import os
import json
import argparse

from collections import defaultdict
from seed import set_seed
from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import Explainer, LocalClient
from tqdm.asyncio import tqdm

set_seed(42)

GENERATION_KWARGS = {
    "temperature": 0.5,
    "max_tokens": 2000,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str)
    parser.add_argument("--width", type=str)
    parser.add_argument("--l0", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--explainer-model", type=str)
    return parser.parse_args()


async def main(args, feature_save_dir: str):
    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    client = LocalClient(model=args.explainer_model, max_retries=2)
    semaphore = asyncio.Semaphore(50)

    explainer = Explainer(
        client=client,
        subject_tokenizer=subject_tokenizer,
        use_cot=False,
        threshold=0.6,
        insert_as_prompt=True,
        **GENERATION_KWARGS,
    )

    explanations = defaultdict(dict)

    save_path = f".model.layers.{args.layer}.pt"
    feature_save_path = os.path.join(feature_save_dir, save_path)

    features = [f for f in load_torch(feature_save_path, max_examples=2_000)]
    pbar = tqdm(desc="Processing features", total=len(features))

    async def process_feature(layer, feature):
        async with semaphore:
            explanation = await explainer(feature)
            print(explanation)
            explanations[layer][feature.index] = explanation
            pbar.update(1)
            return feature.index

    # Process features directly from generator
    await asyncio.gather(
        *(
            process_feature(args.layer, feature)
            for feature in features
        )
    )

    pbar.close()

    model_name = args.explainer_model.split("/")[-1]
    save_path = f"{model_name}-explanations.json"
    save_path = os.path.join(feature_save_dir, save_path)
    with open(save_path, "w") as f:
        json.dump(explanations, f)


if __name__ == "__main__":
    args = get_args()

    feature_save_dir = f"/workspace/cache/gemma-2-{args.model_size}-w{args.width}-l0{args.l0}-layer{args.layer}"

    asyncio.run(main(args, feature_save_dir))
