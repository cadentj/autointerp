import asyncio
import os
import argparse

from seed import set_seed
from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import Explainer, AnthropicClient
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

    client = AnthropicClient(model=args.explainer_model, max_retries=1)

    explainer = Explainer(
        client=client,
        subject_tokenizer=subject_tokenizer,
        use_cot=False,
        threshold=0.3,
        insert_as_prompt=True,
    )

    # Create save path for explanations
    model_name = args.explainer_model.split("/")[-1]
    save_path = f"{model_name}-explanations.json"
    save_path = os.path.join(feature_save_dir, save_path)

    feature_save_path = os.path.join(
        feature_save_dir, f".model.layers.{args.layer}.pt"
    )
    features = [f for f in load_torch(feature_save_path, max_examples=2_000)]
    pbar = tqdm(desc="Processing features", total=len(features))

    async def process_feature(layer, feature):
        layer_key = str(layer)
        index_key = str(feature.index)
        feature_id = f"{layer_key}-{index_key}"
        
        response = await explainer(
            feature, feature_id=feature_id, **GENERATION_KWARGS
        )
        print(response)

        pbar.update(1)
        return feature.index

    # Not actually async but whatever
    await asyncio.gather(
        *(process_feature(args.layer, feature) for feature in features)
    )

    client.upload()

    pbar.close()


if __name__ == "__main__":
    args = get_args()

    feature_save_dir = f"/share/u/caden/neurondb/cache/gemma-2-{args.model_size}-w{args.width}-l0{args.l0}-layer{args.layer}"

    asyncio.run(main(args, feature_save_dir))
