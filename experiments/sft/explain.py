import asyncio
import os
import json

from collections import defaultdict
from seed import set_seed
from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import Explainer, OpenRouterClient
from tqdm.asyncio import tqdm

set_seed(42)

GENERATION_KWARGS = {
    "temperature": 0.5,
    "max_tokens": 2000,
}
EXPLAINER_MODEL = "meta-llama/llama-3.3-70b-instruct"
N_FEATURES = 550


async def main(feature_save_dir: str):
    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    client = OpenRouterClient(model=EXPLAINER_MODEL, max_retries=1)
    semaphore = asyncio.Semaphore(25)

    explainer = Explainer(
        client=client,
        subject_tokenizer=subject_tokenizer,
        threshold=0.5,
        insert_as_prompt=True,
    )

    # Create save path for explanations
    model_name = EXPLAINER_MODEL.split("/")[-1]
    save_path = f"{model_name}-explanations.json"
    save_path = os.path.join(feature_save_dir, save_path)

    # Load existing explanations if file exists
    explanations = defaultdict(dict)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            explanations.update(json.load(f))

    pbar = tqdm(desc="Processing features", total=N_FEATURES)

    async def process_feature(layer, feature):
        layer_key = str(layer)
        index_key = str(feature.index)

        if layer_key in explanations and index_key in explanations[layer_key]:
            pbar.update(1)
            return
        
        async with semaphore:
            explanation = await explainer(feature, **GENERATION_KWARGS)
            explanations[layer_key][index_key] = explanation

            # Save every 10 features
            if int(index_key) % 5 == 0:
                with open(save_path, "w") as f:
                    json.dump(explanations, f)
            
            pbar.update(1)

    for file in os.listdir(feature_save_dir):
        if not file.endswith(".pt"):
            continue
        
        # Process features directly from generator
        features = [f for f in load_torch(os.path.join(feature_save_dir, file), max_examples=2_000)]

        file_name = file.replace(".pt", "")
        await asyncio.gather(
            *(
                process_feature(file_name, feature)
                for feature in features
            )
        )

    pbar.close()

if __name__ == "__main__":
    feature_save_dir = "/share/u/caden/neurondb/cache/gender"

    asyncio.run(main(feature_save_dir))
